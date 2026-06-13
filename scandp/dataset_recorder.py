#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import cv2
import diffusion_policy_3d.module.gridmap as gridmap
import h5py
import message_filters
import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image, PointCloud2
from termcolor import cprint
from tqdm import tqdm


def image_msg_to_numpy(img_msg, desired_encoding="bgr8"):
    """sensor_msgs/Image -> numpy.ndarray

    サポートしているエンコーディング (このスクリプトで扱う代表的なもの):
      - 'rgb8', 'bgr8', 'rgba8', 'bgra8'  -> uint8 HxWxC
      - 'mono8' / '8UC1'                   -> uint8 HxW
      - '16UC1' / 'mono16'                 -> uint16 HxW
      - '32FC1'                            -> float32 HxW
      - 'passthrough'                      -> try uint16 fallback

    注意: msg.step (行バイト長) や is_bigendian による byteswap を考慮します。
    常にコピーを返すので、元データ参照の Read-only 配列問題も回避します。
    """
    enc = img_msg.encoding
    h = img_msg.height
    w = img_msg.width

    # ほとんどの ROS Image は連続配列(step == w * channels) で来るため簡便に扱う
    b = img_msg.data

    # normalize encoding string to common patterns
    e_low = enc.lower()

    try:
        if e_low in ("rgb8", "bgr8"):
            channels = 3
            arr = np.frombuffer(b, dtype=np.uint8)
            arr = arr.reshape((h, w, channels))
            if e_low == "rgb8" and desired_encoding == "bgr8":
                arr = arr[..., ::-1]
            return arr.copy()

        if e_low in ("rgba8", "bgra8"):
            channels = 4
            arr = np.frombuffer(b, dtype=np.uint8)
            arr = arr.reshape((h, w, channels))
            # 4ch -> 3ch に落とす（アルファを無視）
            arr = arr[..., :3]
            # rgb -> bgr 変換
            if e_low == "rgba8" and desired_encoding == "bgr8":
                arr = arr[..., ::-1]
            elif e_low == "bgra8" and desired_encoding == "bgr8":
                # already bgr order, nothing
                pass
            return arr.copy()

        if e_low in ("mono8", "8uc1"):
            arr = np.frombuffer(b, dtype=np.uint8).reshape((h, w))
            return arr.copy()

        if e_low in ("16uc1", "mono16"):
            arr = np.frombuffer(b, dtype=np.uint16).reshape((h, w))
            if img_msg.is_bigendian:
                arr = arr.byteswap()
            return arr.copy()

        if e_low == "32fc1":
            arr = np.frombuffer(b, dtype=np.float32).reshape((h, w))
            if img_msg.is_bigendian:
                arr = arr.byteswap()
            return arr.copy()

        if e_low == "passthrough":
            # RealSense などで深度が来ることがある。まず uint16 を試す
            try:
                arr = np.frombuffer(b, dtype=np.uint16).reshape((h, w))
                if img_msg.is_bigendian:
                    arr = arr.byteswap()
                return arr.copy()
            except Exception:
                raise ValueError(
                    f"Unsupported passthrough depth format for message: h={h}, w={w}")

        raise ValueError(f"Unsupported image encoding: {enc}")

    except Exception as e:
        # 変換失敗時はエラー詳細をわかりやすくする
        raise RuntimeError(
            f"Failed to convert Image (encoding={enc}, size={w}x{h}): {e}")


class RTABMapRecorder:
    def __init__(self, hdf5_out_path, seq_length=500, use_cuda_if_available=True, show_window=False):
        self.seq_length = seq_length
        self.IMAGE_SIZE = None
        self.lock = False
        self.show_window = show_window
        cprint(self.seq_length, "cyan")

        # device selection
        self.device_name = "cuda" if (
            use_cuda_if_available and torch.cuda.is_available()) else "cpu"

        # Occupancy grid (pass "cuda" or "cpu")
        self.mapper = gridmap.LocalMap3D(
            X_lim=np.array([0.1, 0.9]),
            Y_lim=np.array([-0.4, 0.4]),
            Z_lim=np.array([0.1, 0.9]),
            resolution=0.01,
            p=0.5,
            device=("cuda" if self.device_name == "cuda" else "cpu")
        )

        # buffers (初期化)
        self.img_array = None
        self.depth_array = None
        self.cloud_array = None
        self.pose_array = None
        self.gridmap_array = None
        self._buffer_count = 0
        self.expected_gridmap_shape = None
        self.pcd_total = o3d.geometry.PointCloud()

        # 出力先のディレクトリを作る
        odir = os.path.dirname(hdf5_out_path)
        if odir and not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)

        self.hdf5_out_path = hdf5_out_path

        # --- ROS Subscribers ---
        color_sub = message_filters.Subscriber(
            "/camera/color/image_raw", Image)
        depth_sub = message_filters.Subscriber(
            "/camera/depth/image_rect_raw", Image)
        # cloud_sub = message_filters.Subscriber("/rtabmap/cloud_map", PointCloud2)
        cloud_sub = message_filters.Subscriber("/scan_cloud", PointCloud2)
        pose_sub = message_filters.Subscriber(
            "/rtabmap/localization_pose", PoseWithCovarianceStamped)

        ats = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, cloud_sub, pose_sub],
            queue_size=10,
            slop=0.05
        )
        ats.registerCallback(self.callback)

    def callback(self, rgb_msg, depth_msg, cloud_msg, pose_msg):
        if self.lock:
            return

        # RGB
        try:
            rgb_img = image_msg_to_numpy(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert rgb image: {e}")
            return

        # optional: resize to IMAGE_SIZE to save memory
        if self.IMAGE_SIZE:
            h, w = rgb_img.shape[:2]
            if (h, w) != (self.IMAGE_SIZE, self.IMAGE_SIZE):
                rgb_img = cv2.resize(
                    rgb_img, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        # Depth: make consistent float32 (meters) if possible
        try:
            depth_img = image_msg_to_numpy(depth_msg)
            if depth_img is None:
                depth_arr = np.zeros(
                    (rgb_img.shape[0], rgb_img.shape[1]), dtype=np.float32)
            else:
                if depth_img.dtype == np.uint16:
                    # many cameras provide depth in millimeters as uint16 -> convert to meters
                    depth_arr = depth_img.astype(np.float32) / 1000.0
                else:
                    depth_arr = depth_img.astype(np.float32)
                # resize depth to match rgb if needed
                if depth_arr.shape != rgb_img.shape[:2]:
                    depth_arr = cv2.resize(
                        depth_arr, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            rospy.logwarn(f"Depth conversion failed: {e}")
            depth_arr = np.zeros(
                (rgb_img.shape[0], rgb_img.shape[1]), dtype=np.float32)

        # Pose (unchanged)
        pos = np.array([pose_msg.pose.pose.position.x,
                        pose_msg.pose.pose.position.y,
                        pose_msg.pose.pose.position.z], dtype=np.float32)
        quat = np.array([pose_msg.pose.pose.orientation.x,
                         pose_msg.pose.pose.orientation.y,
                         pose_msg.pose.pose.orientation.z,
                         pose_msg.pose.pose.orientation.w], dtype=np.float32)
        pose_vec = np.concatenate([pos, quat])

        # PointCloud -> points using Open3D, pad/truncate to (4096,3)
        try:
            # Convert ROS PointCloud2 to numpy array using Open3D
            pc = o3d.geometry.PointCloud()
            # Convert PointCloud2 to Open3D point cloud
            xyz = []
            for p in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                xyz.append([p[0], p[1], p[2]])
            if len(xyz) == 0:
                raise ValueError("No valid points in PointCloud2 message.")
            pc.points = o3d.utility.Vector3dVector(
                np.array(xyz, dtype=np.float64))
            self.pcd_total += pc

            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pc, mesh_frame])

            # Downsample if too many points
            if len(pc.points) > 4096:
                pc = pc.random_down_sample(4096 / len(pc.points))
            points = np.asarray(pc.points, dtype=np.float32)
        except Exception as e:
            rospy.logwarn(f"PointCloud parsing (Open3D) failed: {e}")
            points = np.zeros((4096, 3), dtype=np.float32)
        # Pad or truncate to (4096, 3)
        if points.shape[0] < 4096:
            pad = np.zeros((4096 - points.shape[0], 3), dtype=np.float32)
            points = np.vstack((points, pad))
        else:
            points = points[:4096]

        # Gridmap update (device-aware)
        try:
            torch_points = torch.from_numpy(points).to(self.device_name)
            self.mapper.update(torch.tensor(pos[0], device=self.device_name),
                               torch.tensor(pos[1], device=self.device_name),
                               torch.tensor(pos[2], device=self.device_name),
                               torch_points,
                               p_free=0.4,
                               p_occ=0.7)
            occ_grid_map = self.mapper.to_prob_occ_map()
            if isinstance(occ_grid_map, torch.Tensor):
                occ_np = occ_grid_map.cpu().numpy()
            else:
                occ_np = np.array(occ_grid_map)
            # 初回フレームで形状を固定する
            if self.expected_gridmap_shape is None:
                self.expected_gridmap_shape = occ_np.shape
            elif occ_np.shape != self.expected_gridmap_shape:
                rospy.logwarn(
                    f"Gridmap shape changed: expected {self.expected_gridmap_shape}, got {occ_np.shape}. Skipping frame gridmap.")
                occ_np = np.zeros(self.expected_gridmap_shape,
                                  dtype=occ_np.dtype)
        except Exception as e:
            rospy.logwarn(f"Gridmap update failed: {e}")
            occ_np = np.zeros(
                self.expected_gridmap_shape if self.expected_gridmap_shape is not None else (1,), dtype=np.uint8)

        # バッファへの追加（初回到達時に preallocate する例）
        if self.img_array is None:
            # preallocate based on first frame shapes
            H, W = rgb_img.shape[:2]
            self.img_array = np.zeros(
                (self.seq_length, H, W, 3), dtype=np.uint8)
            self.depth_array = np.zeros(
                (self.seq_length, H, W), dtype=np.float32)
            self.cloud_array = np.zeros(
                (self.seq_length, 4096, 3), dtype=np.float32)
            self.pose_array = np.zeros((self.seq_length, 7), dtype=np.float32)
            self.gridmap_array = np.zeros(
                (self.seq_length, ) + occ_np.shape, dtype=occ_np.dtype)

        idx = self._buffer_count
        self.img_array[idx] = rgb_img
        self.depth_array[idx] = depth_arr
        self.cloud_array[idx] = points
        self.pose_array[idx] = pose_vec
        self.gridmap_array[idx] = occ_np
        self._buffer_count += 1
        cprint(f"Buffer count: {self._buffer_count}", "cyan")

        if self._buffer_count >= self.seq_length:
            # save only up to _buffer_count（全埋めでない可能性を考慮）
            self.save_hdf5()
            self.lock = True
            rospy.signal_shutdown("Data collection complete.")

    def save_hdf5(self):
        # Save numpy arrays directly
        with h5py.File(self.hdf5_out_path, "w") as f:
            f.create_dataset("color", data=self.img_array, compression="gzip")
            f.create_dataset("depth", data=self.depth_array,
                             compression="gzip")
            f.create_dataset("cloud", data=self.cloud_array,
                             compression="gzip")
            f.create_dataset("pose", data=self.pose_array, compression="gzip")
            f.create_dataset(
                "gridmap", data=self.gridmap_array, compression="gzip")

        # gridmap debug
        o3d.visualization.draw_geometries([self.pcd_total])
        gridmap_tensor_path = "/home/cvl/ROS_i611/data/gridmap_tensor.pt"
        torch.save(self.mapper.to_prob_occ_map(), gridmap_tensor_path)

        cprint(f"Saved HDF5 dataset: {self.hdf5_out_path}", "yellow")
        cprint(f"color shape: {self.img_array.shape}", "yellow")
        cprint(f"depth shape: {self.depth_array.shape}", "yellow")
        cprint(f"cloud shape: {self.cloud_array.shape}", "yellow")
        cprint(f"pose shape: {self.pose_array.shape}", "yellow")
        cprint(f"gridmap shape: {self.gridmap_array.shape}", "yellow")


if __name__ == "__main__":
    rospy.init_node("rtabmap_hdf5_recorder", anonymous=True)
    cprint("RTAB-Map HDF5 Recorder Node Started.", "green")

    hdf5_out_path = rospy.get_param("~hdf5_out_path", "/tmp/default.h5")
    seq_length = rospy.get_param("~seq_length", 100)
    cprint(f"Using hdf5_out_path: {hdf5_out_path}", "cyan")
    cprint(f"Using seq_length: {seq_length}", "cyan")

    recorder = RTABMapRecorder(
        hdf5_out_path=hdf5_out_path,
        seq_length=seq_length
    )
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cprint("Shutting down (KeyboardInterrupt)", "yellow")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
