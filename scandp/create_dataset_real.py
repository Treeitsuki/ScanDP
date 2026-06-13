#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import sys

import cv2
# Custom imports
# import diffusion_policy_3d.module.gridmap as gridmap
import diffusion_policy_3d.module.gridmap_feat as gridmap
import h5py
import message_filters
import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import torch
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from termcolor import cprint
from tqdm import tqdm


class RTABMapRecorder:
    def __init__(self, hdf5_out_path, seq_length=500, use_cuda=True):
        self.seq_length = seq_length
        self.hdf5_out_path = hdf5_out_path
        self.buffer_count = 0
        self.is_recording_complete = False
        self.pbar = tqdm(
            total=self.seq_length,
            desc="Recording",
            dynamic_ncols=True,
            leave=True,
        )

        # --- Configs ---
        self.num_points = 4096

        # --- Camera Intrinsics ---
        self.camera_intrinsics = None
        self.intrinsics_received = False

        # --- Device ---
        self.device = "cuda:1" if use_cuda and torch.cuda.is_available() else "cpu"
        cprint(f"Mapping Device: {self.device}", "cyan")

        # --- GridMap ---
        self.mapper = gridmap.LocalMap3D(
            X_lim=np.array([-0.4, 0.4]),
            Y_lim=np.array([-0.4, 0.4]),
            Z_lim=np.array([0.0, 0.8]),
            resolution=0.02,
            p=0.5,
            device=self.device
        )

        # --- Buffers ---
        self.img_array = None
        self.depth_array = None
        self.scan_cloud_array = None
        self.depth_cloud_array = None
        self.pose_array = None
        self.gridmap_array = None
        self.expected_gridmap_shape = None

        # --- Output dir ---
        odir = os.path.dirname(hdf5_out_path)
        if odir and not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.world_frame = "map"          # or "odom"
        self.camera_frame = "camera_link"

        # --- Subscribers ---
        self.info_sub = rospy.Subscriber(
            "/camera/color/camera_info",
            CameraInfo,
            self.camera_info_callback
        )

        color_sub = message_filters.Subscriber(
            "/camera/color/image_raw", Image)
        depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image)
        cloud_sub = message_filters.Subscriber(
            "/camera/depth/color/points", PointCloud2)

        self.last_msg_time = rospy.Time.now()

        # !: queue_size and slop may need to be adjusted
        self.ats = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, cloud_sub],
            queue_size=200,
            slop=1.0
        )
        self.ats.registerCallback(self.callback)

        self.monitor_timer = rospy.Timer(
            rospy.Duration(1.0), self.check_topic_status)

        cprint("Recorder initialized.", "green")
        cprint(
            f"  Pose source: TF ({self.world_frame} -> {self.camera_frame})", "green")

    def camera_info_callback(self, msg):
        if self.intrinsics_received:
            return
        w, h = msg.width, msg.height
        fx, fy = msg.K[0], msg.K[4]
        cx, cy = msg.K[2], msg.K[5]
        self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            w, h, fx, fy, cx, cy)
        self.intrinsics_received = True
        self.info_sub.unregister()
        cprint("Camera intrinsics received.", "green")

    def check_topic_status(self, event):
        if self.is_recording_complete:
            return
        dt = (rospy.Time.now() - self.last_msg_time).to_sec()
        if dt > 2.0:
            rospy.logwarn_throttle(
                5.0, f"No synchronized messages for {dt:.1f} sec")

    def image_msg_to_numpy(self, msg):
        dtype = np.uint16 if "16" in msg.encoding else np.uint8
        img = np.frombuffer(msg.data, dtype=dtype)
        if len(img) == msg.height * msg.width * 3:
            img = img.reshape(msg.height, msg.width, 3)
        else:
            img = img.reshape(msg.height, msg.width)
        return img.copy()

    def process_pointcloud(self, points):
        if len(points) == 0:
            return np.zeros((self.num_points, 3), np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if len(pcd.points) > self.num_points:
            pcd = pcd.random_down_sample(self.num_points / len(pcd.points))

        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[0] < self.num_points:
            pad = np.zeros((self.num_points - pts.shape[0], 3), np.float32)
            pts = np.vstack([pts, pad])
        return pts[:self.num_points]

    def callback(self, rgb_msg, depth_msg, cloud_msg):
        if self.is_recording_complete:
            return
        self.last_msg_time = rospy.Time.now()

        # --- RGB ---
        rgb = self.image_msg_to_numpy(rgb_msg)
        h, w = rgb.shape[:2]

        # --- Depth ---
        depth_raw = self.image_msg_to_numpy(depth_msg)
        depth = depth_raw.astype(np.float32) / 1000.0

        # --- Depth Cloud ---
        depth_cloud = np.zeros((self.num_points, 3), np.float32)
        if self.intrinsics_received:
            o3d_depth = o3d.geometry.Image(depth)
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                o3d_depth, self.camera_intrinsics,
                depth_scale=1.0, depth_trunc=10.0, stride=2)
            depth_cloud = self.process_pointcloud(
                np.asarray(pcd.points))

        # --- Scan Cloud ---
        gen = pc2.read_points(cloud_msg, ("x", "y", "z"), skip_nans=True)
        scan_xyz = np.array(list(gen), dtype=np.float32)
        scan_cloud = self.process_pointcloud(scan_xyz)

        # --- Pose (TF) ---
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                rgb_msg.header.stamp,
                rospy.Duration(0.1)
            )
            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation
            pose = np.array(
                [t.x, t.y, t.z, q.x, q.y, q.z, q.w], np.float32)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"TF lookup failed: {e}")
            pose = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)

        # --- GridMap ---
        torch_pts = torch.from_numpy(scan_cloud).to(self.device)
        self.mapper.update(
            torch.tensor(pose[0], device=self.device),
            torch.tensor(pose[1], device=self.device),
            torch.tensor(pose[2], device=self.device),
            torch_pts,
            p_free=0.4,
            p_occ=0.7
        )
        grid = self.mapper.to_prob_occ_map().cpu().numpy()

        self._add_to_buffer(rgb, depth, scan_cloud,
                            depth_cloud, pose, grid)

    def _add_to_buffer(self, rgb, depth, scan, depth_cloud, pose, grid):
        if self.img_array is None:
            H, W = rgb.shape[:2]
            self.img_array = np.zeros(
                (self.seq_length, H, W, 3), np.uint8)
            self.depth_array = np.zeros(
                (self.seq_length, H, W), np.float32)
            self.scan_cloud_array = np.zeros(
                (self.seq_length, self.num_points, 3), np.float32)
            self.depth_cloud_array = np.zeros(
                (self.seq_length, self.num_points, 3), np.float32)
            self.pose_array = np.zeros((self.seq_length, 7), np.float32)
            self.gridmap_array = np.zeros(
                (self.seq_length,) + grid.shape, grid.dtype)

        i = self.buffer_count
        self.img_array[i] = rgb
        self.depth_array[i] = depth
        self.scan_cloud_array[i] = scan
        self.depth_cloud_array[i] = depth_cloud
        self.pose_array[i] = pose
        self.gridmap_array[i] = grid

        self.buffer_count += 1
        if self.pbar is not None:
            self.pbar.update(1)

        if self.buffer_count >= self.seq_length:
            self.is_recording_complete = True
            if self.pbar is not None:
                self.pbar.close()
                self.pbar = None
            self.save_hdf5()
            rospy.signal_shutdown("Collection complete")

    def save_hdf5(self):
        cprint(f"Saving HDF5: {self.hdf5_out_path}", "cyan")
        with h5py.File(self.hdf5_out_path, "w") as f:
            n = self.buffer_count
            f.create_dataset("color", data=self.img_array[:n])
            f.create_dataset("depth", data=self.depth_array[:n])
            f.create_dataset("cloud", data=self.scan_cloud_array[:n])
            f.create_dataset("depth_cloud", data=self.depth_cloud_array[:n])
            f.create_dataset("pose", data=self.pose_array[:n])
            f.create_dataset("gridmap", data=self.gridmap_array[:n])
        cprint("Save complete.", "green")


if __name__ == "__main__":
    rospy.init_node("rtabmap_hdf5_recorder")

    out_path = rospy.get_param("~hdf5_out_path", "")
    seq_len = rospy.get_param("~seq_length", 150)

    if out_path == "":
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(os.getcwd(), "data", f"record_{now}.h5")

    RTABMapRecorder(out_path, seq_len)
    rospy.spin()
