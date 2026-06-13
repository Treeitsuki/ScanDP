import os
import pathlib
import signal
import sys
import threading

import cv2
# import diffusion_policy_3d.module.gridmap as gridmap
import diffusion_policy_3d.module.gridmap_feat as gridmap
import diffusion_policy_3d.module.vis_trajectory as vis_trajectory
import hydra
import message_filters
import numpy as np
import open3d as o3d
import pandas as pd
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import torch
import torch.nn.functional as F
import tqdm
from cv_bridge import CvBridge
from diffusion_policy_3d.module.extractor_dinov2 import DINOv2FeatureExtractor
from diffusion_policy_3d.module.publisher import PublisherHelper
from diffusion_policy_3d.module.waypoint_extraction.extract_waypoints import \
    dp_waypoint_selection
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped
from ground_truth.coverage_gt import eval_inference
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sklearn.decomposition import PCA
from std_msgs.msg import Bool, Float32, String
from termcolor import cprint

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


os.environ['WANDB_SILENT'] = "True"
OmegaConf.register_new_resolver("eval", eval, replace=True)

DEPLOY_LENGTH = 500
INIT_POS = [0, -1, 0.7]
TARGET = "bunny"  # "bunny", "spot", "armadillo", "teapot", "dragon", "bust", "bike", "happy"
NAME = "spconv"  # "dp", "spconv"
SCALE = 1
use_waypoints = True
WAIT_FOR_ARM_DONE = True
ARM_DONE_TIMEOUT_S = 30
ARM_DONE_EXPECTED = "any"  # "all" or "any"
ABORT_ON_SYNC_FAILURE = False

_stop_event = threading.Event()


def _handle_shutdown_signal(signum, frame):
    _stop_event.set()
    try:
        rospy.signal_shutdown(f"signal {signum}")
    except Exception:
        pass


def _should_stop():
    return _stop_event.is_set() or rospy.is_shutdown()


class RealRobotNode:
    def __init__(self, obs_horizon=2,
                 action_horizon=8,
                 device="gpu",
                 use_image=True,
                 use_point_cloud=False,
                 #  img_size=224,
                 #  use_gridmap=True,
                 use_gridmap=False,
                 ):

        self.node_name = 'diffusion_policy_node'
        rospy.init_node(self.node_name, anonymous=True, disable_signals=True)

        # --- ROS Utilities ---
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_lookup_timeout_s = rospy.get_param("~tf_lookup_timeout_s", 2.0)
        self.tf_use_image_stamp = rospy.get_param("~tf_use_image_stamp", False)
        self.tf_max_stamp_skew_s = rospy.get_param("~tf_max_stamp_skew_s", 1.0)

        # --- Publishers ---
        # Next immediate target pose
        self.pose_pub = rospy.Publisher(
            '/policy/pose', PoseStamped, queue_size=1)
        # Full planned path
        self.path_pub = rospy.Publisher(
            '/policy/path', PoseArray, queue_size=1)
        # Full planned poses (publish actions)
        self.poses_pub = rospy.Publisher(
            '/policy/poses/', PoseArray, queue_size=1)
        # Debug: Visualizing the gridmap or pointcloud used for inference
        self.debug_pc_pub = rospy.Publisher(
            '/debug/pointcloud_feat', PointCloud2, queue_size=1)
        # Path length publisher (meters)
        self.path_sum_topic = rospy.get_param(
            "~path_sum_topic", "/policy/path_sum"
        )
        self.path_sum_pub = rospy.Publisher(
            self.path_sum_topic, Float32, queue_size=1
        )

        # --- Subscribers ---
        # Using message_filters to sync RGB and Depth
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber(
            '/camera/aligned_depth_to_color/image_raw', Image)
        if use_point_cloud:
            cloud_sub = message_filters.Subscriber(
                '/camera/depth/color/points', PointCloud2)

        # ApproximateTimeSynchronizer is robust for real-world network delays
        # ![IMPORTANT] Adjust the queue size and slop as needed.　
        # ApproximateTimeSynchronizer([self.temp_sub, self.fluid_sub], queue_size, max_delay)
        sync_sources = [rgb_sub, depth_sub]
        if use_point_cloud:
            sync_sources.append(cloud_sub)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            sync_sources, 200, 1.0)
        self.ts.registerCallback(self.data_callback)

        # self.camera_info_sub = rospy.Subscriber(
        #     '/camera/camera_info', CameraInfo, self.cam_info_cb)
        self.camera_intrinsics = None
        self.camera_K = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.latest_img = None
        self.latest_depth = None
        self.latest_cloud = None
        self.latest_stamp = None

        cprint("Node Ready. Waiting for data...", "green")

        # # Publishers
        # Publisher helper
        # self.publisher = PublisherHelper(
        #     self, self.rgb_pub, self.depth_pub, self.pcd_pub,
        #     self.camera_info_pub, self.bridge
        # )

        # obs/action
        self.use_image = use_image
        self.use_point_cloud = use_point_cloud
        self.use_gridmap = use_gridmap

        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # inference device
        if device == "gpu":
            # set current CUDA device and keep a torch.device object for moves
            torch.cuda.set_device(0)
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.dinov2_extractor = DINOv2FeatureExtractor(
            model_name="dinov2_vitl14",
            device=self.device,
            batch_size=8,
        )

        self.mapper = gridmap.LocalMap3D(
            X_lim=np.array([-0.8, 0.8]),
            Y_lim=np.array([-0.8, 0.8]),
            Z_lim=np.array([0.0, 1.6]),
            resolution=0.02,
            p=0.5,
            device="cuda"
        )
        self.depth_scale = rospy.get_param('~depth_scale', 1000.0)
        self.min_depth_m = rospy.get_param('~min_depth_m', 0.05)
        self.max_depth_m = rospy.get_param('~max_depth_m', 5.0)
        self.num_points = rospy.get_param('~num_points', 4096)

        self.pcd_total = o3d.geometry.PointCloud()
        self.csv_list = []
        self.path_sum = 0

        cprint("Node Ready. Waiting for data...", "green")

        self.camera_info_sub = rospy.Subscriber(
            '/camera/color/camera_info', CameraInfo, self.cam_info_cb)

        # --- Arm done / MoveIt failure sync ---
        self.arm_done_topic = rospy.get_param(
            "~arm_done_topic", "/arm/move_done"
        )
        self.moveit_failure_topic = rospy.get_param(
            "~moveit_failure_topic", "/moveit/failure"
        )
        self.arm_done_count = 0
        self.last_arm_done_time = None
        self.last_moveit_failure = None
        self.last_moveit_failure_time = None
        rospy.Subscriber(self.arm_done_topic, Bool, self._arm_done_cb)
        rospy.Subscriber(self.moveit_failure_topic,
                         String, self._moveit_failure_cb)

    # MARK:

    def data_callback(self, rgb_msg, depth_msg, cloud_msg=None):
        try:
            rgb = self.bridge.imgmsg_to_cv2(
                rgb_msg, desired_encoding="passthrough")
            if hasattr(rgb_msg, "encoding") and rgb_msg.encoding:
                enc = rgb_msg.encoding.lower()
                if enc.startswith("bgr"):
                    rgb = rgb[..., ::-1]
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            rospy.logwarn(f"Failed to decode RGB/Depth images: {exc}")
            return
        self.latest_img = rgb
        self.latest_depth = depth
        self.latest_stamp = rgb_msg.header.stamp
        if cloud_msg is not None:
            try:
                gen = pc2.read_points(
                    cloud_msg, ("x", "y", "z"), skip_nans=True)
                points = np.array(list(gen), dtype=np.float32)
                self.latest_cloud = self._process_pointcloud(points)
            except Exception as exc:
                rospy.logwarn(f"Failed to decode point cloud: {exc}")
                self.latest_cloud = None

    def cam_info_cb(self, msg):
        if self.camera_intrinsics is None:
            K = np.array(msg.K).reshape(3, 3)
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
            self.camera_intrinsics = msg
            self.camera_info_sub.unregister()

    def _arm_done_cb(self, msg):
        if bool(msg.data):
            self.arm_done_count += 1
            self.last_arm_done_time = rospy.Time.now()

    def _moveit_failure_cb(self, msg):
        self.last_moveit_failure = msg.data
        self.last_moveit_failure_time = rospy.Time.now()

    def wait_for_arm_done(self, expected_done, timeout_s=300.0):
        start_count = self.arm_done_count
        start_time = rospy.Time.now()
        rate = rospy.Rate(20)
        while not _should_stop():
            if self.last_moveit_failure_time is not None:
                if self.last_moveit_failure_time >= start_time:
                    return False, "moveit_failure"
            if (self.arm_done_count - start_count) >= expected_done:
                return True, "done"
            if timeout_s > 0:
                if (rospy.Time.now() - start_time).to_sec() >= timeout_s:
                    return False, "timeout"
            rate.sleep()
        return False, "shutdown"

    def wait_for_arm_done_count(self, target_count, timeout_s=30):
        start_time = rospy.Time.now()
        rate = rospy.Rate(20)
        while not _should_stop():
            if self.last_moveit_failure_time is not None:
                if self.last_moveit_failure_time >= start_time:
                    return False, "moveit_failure"
            if self.arm_done_count >= target_count:
                return True, "done"
            if timeout_s > 0:
                if (rospy.Time.now() - start_time).to_sec() >= timeout_s:
                    return False, "timeout"
            rate.sleep()
        return False, "shutdown"

    def _get_latest_observations(self):
        if self.latest_img is None or self.latest_depth is None or (
                self.use_point_cloud and self.latest_cloud is None):
            while not _should_stop():
                try:
                    rgb_msg = rospy.wait_for_message(
                        '/camera/color/image_raw', Image, timeout=0.5)
                    depth_msg = rospy.wait_for_message(
                        '/camera/aligned_depth_to_color/image_raw',
                        Image,
                        timeout=0.5,
                    )
                    cloud_msg = None
                    if self.use_point_cloud:
                        cloud_msg = rospy.wait_for_message(
                            '/camera/depth/color/points', PointCloud2, timeout=0.5)
                    self.data_callback(rgb_msg, depth_msg, cloud_msg)
                    break
                except rospy.ROSException:
                    continue
            if _should_stop():
                raise KeyboardInterrupt("shutdown requested")
        return self.latest_img, self.latest_depth, self.latest_cloud

    def _process_pointcloud(self, points):
        if points is None or len(points) == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if len(pcd.points) > self.num_points:
            pcd = pcd.farthest_point_down_sample(self.num_points)

        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[0] < self.num_points:
            pad = np.zeros((self.num_points - pts.shape[0], 3),
                           dtype=np.float32)
            pts = np.vstack([pts, pad])
        return pts[:self.num_points]

    def _depth_to_pointcloud(self, depth):
        if self.fx is None or self.fy is None:
            rospy.logwarn_throttle(5.0, "Camera intrinsics not received yet.")
            return np.zeros((0, 3), dtype=np.float32), None

        if depth.dtype == np.uint16 or depth.dtype == np.int16:
            depth_m = depth.astype(np.float32) / float(self.depth_scale)
        else:
            depth_m = depth.astype(np.float32)

        valid_depth = (depth_m > self.min_depth_m) & (
            depth_m < self.max_depth_m)
        v, u = np.where(valid_depth)
        if len(v) == 0:
            return np.zeros((0, 3), dtype=np.float32), valid_depth

        z = depth_m[v, u]
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        pc = np.stack([x, y, z], axis=-1).astype(np.float32)
        return pc, valid_depth

    def publish_action_list(self, action_list, frame_id="map"):
        msg = PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        for act in action_list:
            pose = Pose()
            pose.position.x = float(act[0])
            pose.position.y = float(act[1])
            pose.position.z = float(act[2])
            pose.orientation.x = float(act[3])
            pose.orientation.y = float(act[4])
            pose.orientation.z = float(act[5])
            pose.orientation.w = float(act[6])
            msg.poses.append(pose)
        self.poses_pub.publish(msg)

    def _select_tf_stamp(self):
        if not self.tf_use_image_stamp or self.latest_stamp is None:
            return rospy.Time(0), "latest"
        now = rospy.Time.now()
        try:
            skew_s = abs((now - self.latest_stamp).to_sec())
        except Exception:
            skew_s = float("inf")
        if skew_s <= self.tf_max_stamp_skew_s:
            return self.latest_stamp, "image"
        rospy.logwarn_throttle(
            5.0,
            "TF stamp skew too large (%.3fs). Using latest TF instead.",
            skew_s,
        )
        return rospy.Time(0), "latest"

    def _get_camera_link_pose(self, target_frame="map", source_frame="ee_state"):
        stamp, mode = self._select_tf_stamp()
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                rospy.Duration(self.tf_lookup_timeout_s),
            )
        except Exception as exc:
            if mode != "latest":
                try:
                    trans = self.tf_buffer.lookup_transform(
                        target_frame,
                        source_frame,
                        rospy.Time(0),
                        rospy.Duration(self.tf_lookup_timeout_s),
                    )
                    rospy.logwarn_throttle(
                        5.0,
                        "TF lookup failed at image stamp. Fell back to latest TF.",
                    )
                except Exception as exc2:
                    rospy.logwarn(
                        f"TF lookup failed {target_frame}->{source_frame}: {exc2}"
                    )
                    return None, None
            else:
                rospy.logwarn(
                    f"TF lookup failed {target_frame}->{source_frame}: {exc}"
                )
                return None, None
        t = trans.transform.translation
        r = trans.transform.rotation
        pos = torch.tensor(
            [t.x, t.y, t.z], dtype=torch.float32, device=self.device)
        quat = torch.tensor([r.x, r.y, r.z, r.w],
                            dtype=torch.float32, device=self.device)
        return pos, quat

    def step(self, action_list, arm_done_start_count=None):

        if arm_done_start_count is None:
            arm_done_start_count = self.arm_done_count

        for action_id in range(self.action_horizon):
            if _should_stop():
                raise KeyboardInterrupt("shutdown requested")
            if WAIT_FOR_ARM_DONE:
                target_count = arm_done_start_count + action_id + 1
                if self.arm_done_count < target_count:
                    ok, reason = self.wait_for_arm_done_count(
                        target_count=target_count,
                        timeout_s=ARM_DONE_TIMEOUT_S,
                    )
                    if not ok:
                        cprint(
                            f"Arm sync failed at action {action_id}: {reason}",
                            "red",
                        )
                        if ABORT_ON_SYNC_FAILURE:
                            raise KeyboardInterrupt("arm sync failed")
            # self.scene.step()
            act = action_list[action_id]
            self.action_array.append(act)

            # print(f"Action: {act}")
            # self.drone.set_pos(torch.tensor([act[:3]]))
            # self.drone.set_quat(torch.tensor([act[3:]]))

            img, depth, cloud = self._get_latest_observations()
            if depth.shape[:2] != img.shape[:2]:
                depth = cv2.resize(
                    depth,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            seg_mask = np.ones(img.shape[:2], dtype=bool)
            seg_mask_t = torch.from_numpy(seg_mask).to(self.device)

            img_safe = np.array(img, copy=True)
            img_torch = torch.from_numpy(img_safe).to(
                self.device)    # torch.Size([256, 256, 3])
            feats, ph, pw = self.dinov2_extractor.extract_features_batch(
                img_torch.unsqueeze(0))

            mask_float = seg_mask_t.unsqueeze(0).unsqueeze(0).float()
            resized_area = F.interpolate(
                mask_float, size=(ph, pw), mode='area')

            patch_mask_area = resized_area.squeeze().bool()
            patch_mask_area = ~patch_mask_area  # Invert mask

            feats_flat = feats[0]   # torch.Size([ph*pw, C])
            patch_mask_flat = patch_mask_area.view(-1)  # torch.Size([ph*pw])
            # torch.Size([ph*pw, C])
            feats_masked = feats_flat[~patch_mask_flat]

            if isinstance(feats_masked, torch.Tensor) and feats_masked.shape[0] > 0:
                # Determine a safe number of PCA components (<= samples and features)
                feats_np = feats_masked.cpu().numpy()
                n_samples, n_features = feats_np.shape
                n_components_safe = min(3, n_samples, n_features)

                # Fit PCA with a valid component count
                pca = PCA(n_components=n_components_safe)
                selected_pca = pca.fit_transform(feats_np)    # (N, k)

                # If PCA returned fewer than 3 channels, pad with zeros so the
                # downstream code (which expects 3 channels) continues to work.
                if n_components_safe < 3:
                    padded = np.zeros((selected_pca.shape[0], 3),
                                      dtype=selected_pca.dtype)
                    padded[:, :n_components_safe] = selected_pca
                    selected_pca = padded

                pca_feats = torch.zeros((ph, pw, 3), device=self.device)
                patch_mask_area = patch_mask_area.clone().to(self.device)

                # create tensor directly on the target device to avoid CPU/CUDA
                # device mismatch during assignment with a CUDA boolean mask
                pca_feats[~patch_mask_area] = torch.as_tensor(
                    selected_pca, device=self.device, dtype=torch.float32)

                # normalize robustly (avoid division by zero)
                min_v = pca_feats.min()
                max_v = pca_feats.max()
                if (max_v - min_v) > 1e-8:
                    pca_rgb = (pca_feats - min_v) / (max_v - min_v)
                else:
                    pca_rgb = torch.zeros_like(pca_feats)
            else:
                # No valid features — use zeros so rest of pipeline can run.
                pca_feats = torch.zeros((ph, pw, 3), device=self.device)
                pca_rgb = torch.zeros_like(pca_feats)

            # resize patch-level features back to image resolution
            img_h, img_w = img.shape[:2]
            pca_rgb_resized = F.interpolate(
                pca_rgb.permute(2, 0, 1).unsqueeze(0),
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            pca_feats_resized = F.interpolate(
                pca_feats.permute(2, 0, 1).unsqueeze(0),
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

            # pc, _ = self.cam.render_pointcloud()
            pc_masked, valid_mask = self._depth_to_pointcloud(depth)
            if valid_mask is None:
                valid_mask = np.zeros(img.shape[:2], dtype=bool)

            if pc_masked.shape[0] == 0:
                rospy.logwarn_throttle(5.0, "No valid depth points.")
                continue

            if not isinstance(pc_masked, torch.Tensor):
                pc_masked_t = torch.as_tensor(
                    pc_masked, device=self.device, dtype=torch.float32)
            else:
                pc_masked_t = pc_masked.to(
                    device=self.device, dtype=torch.float32)

            pca_rgb_flat = pca_rgb_resized.reshape(-1,
                                                   pca_rgb_resized.shape[2])
            pca_feats_flat = pca_feats_resized.reshape(
                -1, pca_feats_resized.shape[2])  # (H*W, C)

            # Ensure mask_bool is a torch boolean tensor
            if not isinstance(valid_mask, torch.Tensor):
                mask_t = torch.as_tensor(
                    valid_mask, device=self.device, dtype=torch.bool).view(-1)
            else:
                mask_t = valid_mask.to(device=self.device,
                                       dtype=torch.bool).view(-1)

            colors_t = pca_rgb_flat[mask_t].to(
                dtype=torch.float32)  # (N,3)
            pc_color = torch.cat((pc_masked_t, colors_t), dim=1)  # (N,6)
            pc_feat = torch.cat(
                (pc_masked_t, pca_feats_flat[mask_t]), dim=1)    # (N,6)

            cam_pos, cam_quat = self._get_camera_link_pose()
            print(f"Camera Pose: pos={cam_pos}, quat={cam_quat}")
            if cam_pos is None:
                continue

            self.mapper.update(
                cam_pos[0],
                cam_pos[1],
                cam_pos[2],
                pc_feat,
                p_free=0.3,
                p_occ=0.7,
            )

            ogm_pc = self.mapper.voxel_centers_with_rgb(threshold=0.9)

            timestamp = rospy.Time.now()

            # self.camera_info_msg.header.stamp = timestamp
            # self.camera_info_pub.publish(self.camera_info_msg)
            # self.publisher.publish_pointcloud2(pc_color, timestamp)

            # self.publisher.publish_rgb_image(img, timestamp)
            # self.publisher.publish_depth_image(depth, timestamp)
            # self.publisher.publish_pointcloud2_to(
            #     ogm_pc,
            #     timestamp,
            #     self.ogm_pub,
            #     frame_id='map'
            # )

            prev_pos = torch.tensor(self.env_qpos_array[-1][:3]).cuda()
            distance = torch.linalg.norm(cam_pos - prev_pos)
            self.path_sum += distance
            path_sum_value = self.path_sum
            try:
                if hasattr(path_sum_value, "item"):
                    path_sum_value = float(path_sum_value.item())
                else:
                    path_sum_value = float(path_sum_value)
                self.path_sum_pub.publish(Float32(data=path_sum_value))
            except Exception:
                pass

            self.color_array.append(img)
            self.depth_array.append(depth)
            if self.use_point_cloud:
                if cloud is None:
                    cloud = np.zeros(
                        (self.num_points, 3), dtype=np.float32)
                self.cloud_array.append(cloud)

            self.env_qpos_array.append(np.concatenate(
                [cam_pos.cpu().view(-1), cam_quat.cpu().view(-1)]))
            self.gridmap_array.append(self.mapper.to_tensor_prob().cpu())

        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)

        obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)
        if self.use_point_cloud:
            obs_cloud = np.stack(
                self.cloud_array[-self.obs_horizon:], axis=0)
        obs_gridmap = np.stack(self.gridmap_array[-self.obs_horizon:], axis=0)

        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_image:
            # build directly on the proper device
            obs_dict['image'] = torch.as_tensor(
                obs_img, device=self.device).permute(0, 3, 1, 2).unsqueeze(0)
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.as_tensor(
                obs_cloud, device=self.device).unsqueeze(0)
        if self.use_gridmap:
            obs_dict['gridmap'] = torch.as_tensor(
                obs_gridmap, device=self.device).unsqueeze(0)  # torch.Size([2, 25, 25, 25])
        return obs_dict

    # MARK:
    def reset(self, first_init=True):
        # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
        self.gridmap_array = []

        if first_init:
            # self.drone.set_pos(torch.tensor([INIT_POS]))
            # self.drone.set_quat([1, 0, 0, 0])
            # self.drone.set_quat(xyz_to_quat(torch.tensor([[90, 0, 0]])))
            # cam_transform = np.array([
            #     [1.0, 0.0, 0.0, 0.0],
            #     [0.0, 0.0, -1.0, 0.0],
            #     [0.0, 1.0, 0.0, 0.0],
            #     [0.0, 0.0, 0.0, 1.0],
            # ])
            # self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu(
            # ).numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform)

            # self.cam.set_pose(
            #     pos=self.drone.get_pos()[0].cpu().numpy(),
            #     lookat=(0, 0, 0.7)
            # )

            img, depth, cloud = self._get_latest_observations()
            # Set all values in depth that are exactly 100 to 0
            depth = np.where(depth == 100, 0, depth)

        self.color_array.append(img)
        if self.use_point_cloud:
            if cloud is None:
                cloud = np.zeros(
                    (self.num_points, 3), dtype=np.float32)
            self.cloud_array.append(cloud)
        self.gridmap_array.append(
            self.mapper.to_tensor_prob().cpu())
        # self.gridmap_array.append(
        #     self.update_occupancy_gridmap(depth, segmentation).cpu())
        # action = np.zeros(7)
        print("Ready!")

        cam_pos, cam_quat = self._get_camera_link_pose()
        cprint(f"Initial Camera Pose: pos={cam_pos}, quat={cam_quat}", "green")
        if cam_pos is None:
            cam_pos = torch.zeros(3, dtype=torch.float32, device=self.device)
            cam_quat = torch.tensor([0.0, 0.0, 0.0, 1.0],
                                    dtype=torch.float32, device=self.device)
        env_qpos = np.concatenate(
            [cam_pos.cpu().view(-1), cam_quat.cpu().view(-1)])

        self.env_qpos_array.append(env_qpos)
        agent_pos = np.stack([self.env_qpos_array[-1]]
                             * self.obs_horizon, axis=0)

        obs_img = np.stack([self.color_array[-1]]*self.obs_horizon, axis=0)
        if self.use_point_cloud:
            obs_cloud = np.stack([self.cloud_array[-1]] *
                                 self.obs_horizon, axis=0)
        obs_gridmap = np.stack([self.gridmap_array[-1]]
                               * self.obs_horizon, axis=0)

        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_image:
            obs_dict['image'] = torch.as_tensor(
                obs_img, device=self.device).permute(0, 3, 1, 2).unsqueeze(0)
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.as_tensor(
                obs_cloud, device=self.device).unsqueeze(0)
        if self.use_gridmap:
            obs_dict['gridmap'] = torch.as_tensor(
                obs_gridmap, device=self.device).unsqueeze(0)  # torch.Size([2, 25, 25, 25])
        return obs_dict


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config')),
    version_base=None
)
# MARK:
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    if workspace.__class__.__name__ == 'DPWorkspace':
        use_image = True
        use_point_cloud = False
    else:
        use_image = False
        use_point_cloud = True

    # fetch policy model
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1

    roll_out_length = DEPLOY_LENGTH

    img_size = 224
    num_points = 4096
    first_init = True
    record_data = True

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    env = RealRobotNode(
        obs_horizon=2,
        action_horizon=action_horizon,
        device="gpu",
        use_image=use_image,
        use_point_cloud=use_point_cloud,
        # img_size=img_size,
    )

    obs_dict = env.reset(first_init=first_init)
    env.csv_list.append((0, 0, 0))

    step_count = 0
    try:
        while step_count < roll_out_length and not _should_stop():
            with torch.no_grad():
                action = policy(obs_dict)[-1]   # torch.Size([15, 7])
                if use_waypoints:
                    action_torch = torch.stack(
                        [act.detach().clone().to(dtype=torch.float32, device="cuda") for act in action])
                    action_list_pos = [act[:3] for act in action]
                    action_torch_pos = torch.stack([act.detach().clone().to(
                        dtype=torch.float32, device="cuda") for act in action_list_pos])

                    waypoints = dp_waypoint_selection(
                        action_torch_pos,
                        gt_states=action_torch_pos,
                        err_threshold=0.01,
                        pos_only=True,
                    )
                    selected_points = action_torch[waypoints]
                    action_list = [point.cpu().numpy()
                                   for point in selected_points]
                    if len(action_list) < 15:
                        while len(action_list) < 15:
                            action_list.append(action_list[-1])
                else:
                    action_list = [act.numpy() for act in action]

            arm_done_start_count = env.arm_done_count
            env.publish_action_list(action_list)
            if WAIT_FOR_ARM_DONE:
                if ARM_DONE_EXPECTED == "all":
                    expected = len(action_list)
                else:
                    expected = 1
                ok, reason = env.wait_for_arm_done(
                    expected_done=expected,
                    timeout_s=ARM_DONE_TIMEOUT_S,
                )
                if not ok:
                    cprint(f"Arm sync failed: {reason}", "red")
                    if ABORT_ON_SYNC_FAILURE:
                        break
            obs_dict = env.step(
                action_list,
                arm_done_start_count=arm_done_start_count,
            )
            step_count += action_horizon
            print(f"step_count: {step_count}")

            cprint(f"Path length: {env.path_sum:.2f} meters", "green")
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        cprint("Shutdown requested. Exiting main loop.", "yellow")
        return

    if _should_stop():
        cprint("Shutdown requested. Skipping post-processing.", "yellow")
        return

    # For the final result
    final_cover_ratio, pcd_final = eval_inference(
        TARGET,
        env.pcd_total,
        scale=SCALE,
        threshold_icp=1,
        threshold_dis=0.002,
        visualize=True,
        noise_remove=False,
    )
    cprint(f"Final Coverage ratio: {final_cover_ratio:.4f}", "yellow")

    # Save coverage data to a CSV file using pandas
    csv_save_path = os.path.join(os.getcwd(
    ), f"data/logs/csv_{INIT_POS}_{SCALE}", f"{TARGET}_{NAME}_{DEPLOY_LENGTH}.csv")
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    coverage_df = pd.DataFrame(env.csv_list, columns=[
        "step", "coverage", "path_length"])
    path_sum_value = env.path_sum
    try:
        if hasattr(path_sum_value, "item"):
            path_sum_value = float(path_sum_value.item())
        else:
            path_sum_value = float(path_sum_value)
    except Exception:
        pass
    coverage_df["path_sum"] = path_sum_value
    coverage_df.to_csv(csv_save_path, index=False)
    print(f"Coverage data saved to {csv_save_path}")

    # Save the accumulated point cloud
    pcd_save_path = os.path.join(os.getcwd(), "data", "pointcloud.ply")
    o3d.io.write_point_cloud(pcd_save_path, pcd_final)

    choice = input("whether to rename of ply: y/n")
    if choice == "y":
        renamed = input("file rename of ply:")
        os.rename(src=pcd_save_path, dst=pcd_save_path.replace(
            "pointcloud.ply", renamed+'.ply'))
        new_name = pcd_save_path.replace("pointcloud.ply", renamed+'.ply')
        cprint(f"save data at step: {roll_out_length} in {new_name}", "yellow")
    else:
        cprint(
            f"save data at step: {roll_out_length} in {pcd_save_path}", "yellow")

    if record_data:
        import h5py
        root_dir = os.getcwd()
        save_dir = os.path.join(root_dir, "deploy_dir")
        os.makedirs(save_dir, exist_ok=True)

        record_file_name = f"{save_dir}/demo.h5"
        color_array = np.array(env.color_array)
        depth_array = np.array(env.depth_array)
        cloud_array = np.array(env.cloud_array)
        qpos_array = np.array(env.env_qpos_array)
        with h5py.File(record_file_name, 'w') as f:
            f.create_dataset("color_array", data=color_array)
            f.create_dataset("depth_array", data=depth_array)
            f.create_dataset("cloud_array", data=cloud_array)
            f.create_dataset("qpos_array", data=qpos_array)

        choice = input("whether to rename: y/n")
        if choice == "y":
            renamed = input("file rename:")
            os.rename(src=record_file_name, dst=record_file_name.replace(
                "demo.h5", renamed+'.h5'))
            record_file_name = record_file_name.replace(
                "demo.h5", renamed+'.h5')
            cprint(
                f"save data at step: {roll_out_length} in {record_file_name}", "yellow")
        else:
            cprint(
                f"save data at step: {roll_out_length} in {record_file_name}", "yellow")

    # plot trajectory
    vis_trajectory.plot_trajectory_and_obj(
        hdf5_file=record_file_name,
        obj_path=os.path.join(env.asset_dir, env.mesh_cfg["file"]),
        scale=env.mesh_cfg["scale"]*SCALE,
        pos=env.mesh_cfg["pos"],
        quat=env.mesh_cfg.get("quat", None),
        show_background=False,
        show_axes=False,
    )


if __name__ == "__main__":
    main()
