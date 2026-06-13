import os
import pathlib
import sys

# import diffusion_policy_3d.module.gridmap as gridmap
import diffusion_policy_3d.module.gridmap_feat as gridmap
import diffusion_policy_3d.module.vis_trajectory as vis_trajectory
import genesis as gs
import hydra
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import rclpy
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
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat
from geometry_msgs.msg import TransformStamped
from ground_truth.coverage_gt import eval_inference
from omegaconf import OmegaConf
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sklearn.decomposition import PCA
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


class PCPublisherNode(Node):
    def __init__(self, obs_horizon=2,
                 action_horizon=8,
                 device="gpu",
                 use_image=True, img_size=224,
                 use_gridmap=True,
                 ):
        super().__init__('pcl_publisher_node')

        # Publishers
        self.pcd_pub = self.create_publisher(PointCloud2, 'pointcloud', 10)
        # ogm-style voxel centers (for rviz2 /ogm_point_cloud_centers)
        self.ogm_pub = self.create_publisher(
            PointCloud2, 'ogm_point_cloud_centers', 10)
        self.rgb_pub = self.create_publisher(
            Image, 'camera/image_raw', 10)
        self.depth_pub = self.create_publisher(
            Image, 'camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, 'camera/camera_info', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # CvBridge for image conversion
        self.bridge = CvBridge()

        # Publisher helper
        self.publisher = PublisherHelper(
            self, self.rgb_pub, self.depth_pub, self.pcd_pub,
            self.camera_info_pub, self.bridge
        )

        # obs/action
        self.use_image = use_image
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

        self.IMAGE_SIZE_W = 256
        self.IMAGE_SIZE_H = 256
        self.median_filter = True
        self.add_noise = False
        self.pcd_total = o3d.geometry.PointCloud()
        self.csv_list = []
        self.path_sum = 0

        gs.init(backend=gs.gpu)

        self.scene = gs.Scene(
            vis_options=gs.options.VisOptions(
                show_world_frame=False,   # Show xyz axes
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=True,
                ambient_light=(0.1, 0.1, 0.1),
            ),
        )

        self.cam = self.scene.add_camera(
            res=(self.IMAGE_SIZE_W, self.IMAGE_SIZE_H),
            pos=(0.0, 1.0, 0.7),
            lookat=(0, 0, 0.7),
            fov=45,
            GUI=True,
            env_idx=0,
        )

        # MARK: target
        self.plane = self.scene.add_entity(
            gs.morphs.Plane()
        )

        # WARNING: You must add bunny before drone
        self.target_map = {
            "bunny": dict(
                file="stanfordbunny.stl", pos=(0, 0, 0.8), scale=3.0, quat=(1, 0, 0, 0)
            ),
            "spot": dict(
                file="spot.obj", pos=(0, 0.05, 0.8), scale=0.28, quat=(1, 1, 0, 0)
            ),
            "armadillo": dict(
                file="armadillo.obj", pos=(0, 0, 0.8), scale=0.0025, quat=(1, 1, 0, 0)
            ),
            "teapot": dict(
                file="teapot.obj", pos=(0, 0, 0.55), scale=0.07, quat=(1, 1, 0, 0)
            ),
            "dragon": dict(
                file="dragon.obj", pos=(0, 0, 0.8), scale=0.5, quat=(1, 1, 0, 0)
            ),
            "bust": dict(
                file="bust.obj", pos=(0, 0, 0.8), scale=0.002, quat=(1, 0, 0, 0)
            ),
            "bike": dict(
                file="bike.obj", pos=(0.0, 0.11, 0.45), scale=0.00023, quat=(1, 1, 0, 0)
            ),
            "happy": dict(
                file="happy.obj", pos=(0, 0, 0.4), scale=3, quat=(1, 1, 0, 0)
            ),
        }
        # mesh_cfg = target_map[self.cfg.target_mesh.name]
        self.mesh_cfg = self.target_map[TARGET]
        self.asset_dir = "/home/user/workspace/scandp/diffusion_policy_3d/assets/"

        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file=os.path.join(self.asset_dir, self.mesh_cfg["file"]),
                pos=self.mesh_cfg["pos"],
                scale=self.mesh_cfg["scale"]*SCALE,
                fixed=True,
                quat=self.mesh_cfg.get("quat", None)
            ),
            surface=gs.surfaces.Default(color=(0.82, 0.82, 0.82))
        )

        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file="/home/user/workspace/scandp/diffusion_policy_3d/assets/drones/cf2x.urdf",
                pos=(0.0, -1.0, 0.02),
                visualization=False,
            ),
        )

        self.scene.build(n_envs=1, env_spacing=(2.0, 2.0))
        self.cam.render(depth=True, segmentation=False)

        # prepare CameraInfo message for publishing
        self.camera_info_msg = self.create_camera_info_msg()

    # MARK:

    def step(self, action_list):

        for action_id in range(self.action_horizon):
            self.scene.step()
            act = action_list[action_id]
            self.action_array.append(act)

            print(f"Action: {act}")
            self.drone.set_pos(torch.tensor([act[:3]]))
            self.drone.set_quat(torch.tensor([act[3:]]))

            cam_transform = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            # self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu(
            # ).numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform,)

            self.cam.set_pose(
                pos=self.drone.get_pos()[0].cpu().numpy(),
                lookat=(0, 0, 0.7)
            )

            img, depth, seg, _ = self.cam.render(
                depth=True, segmentation=True)
            seg_mask = (seg == 2)
            seg_mask_t = torch.from_numpy(seg_mask).to(self.device)

            img_torch = torch.from_numpy(np.ascontiguousarray(img)).to(
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

            pca_rgb_resized = (
                pca_rgb
                .repeat_interleave(ph, dim=0)
                .repeat_interleave(pw, dim=1)
            )   # torch.size([H, W, 3])
            pca_feats_resized = (
                pca_feats
                .repeat_interleave(ph, dim=0)
                .repeat_interleave(pw, dim=1)
            )   # torch.size([H, W, 3])

            pc, _ = self.cam.render_pointcloud()
            mask_bool = (seg == 2).astype(bool)
            pc_masked = pc[mask_bool]    # torch.Size([N, 3])

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
            if not isinstance(mask_bool, torch.Tensor):
                mask_t = torch.as_tensor(
                    mask_bool, device=self.device, dtype=torch.bool).view(-1)
            else:
                mask_t = mask_bool.to(device=self.device,
                                      dtype=torch.bool).view(-1)

            colors_t = pca_rgb_flat[mask_t].to(
                dtype=torch.float32)  # (N,3)
            pc_color = torch.cat((pc_masked_t, colors_t), dim=1)  # (N,6)
            pc_feat = torch.cat(
                (pc_masked_t, pca_feats_flat[mask_t]), dim=1)    # (N,6)

            self.mapper.update(
                torch.tensor(self.cam.pos[0], device=self.device),
                torch.tensor(self.cam.pos[1], device=self.device),
                torch.tensor(self.cam.pos[2], device=self.device),
                pc_feat,
                p_free=0.3,
                p_occ=0.7,
            )

            ogm_pc = self.mapper.voxel_centers_with_rgb(threshold=0.9)

            timestamp = self.get_clock().now().to_msg()

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = 'camera_link'
            pos = self.cam.get_pos()
            quat = self.cam.get_quat()
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            t.transform.rotation.w = float(quat[0])
            t.transform.rotation.x = float(quat[1])
            t.transform.rotation.y = float(quat[2])
            t.transform.rotation.z = float(quat[3])
            self.tf_broadcaster.sendTransform(t)

            self.camera_info_msg.header.stamp = timestamp
            self.camera_info_pub.publish(self.camera_info_msg)
            self.publisher.publish_pointcloud2(pc_color, timestamp)

            # self.publisher.publish_rgb_image(img, timestamp)
            # self.publisher.publish_depth_image(depth, timestamp)
            self.publisher.publish_pointcloud2_to(
                ogm_pc,
                timestamp,
                self.ogm_pub,
                frame_id='map'
            )

            prev_pos = torch.tensor(self.env_qpos_array[-1][:3]).cuda()
            distance = torch.linalg.norm(self.drone.get_pos() - prev_pos)
            self.path_sum += distance

            self.color_array.append(img)
            self.depth_array.append(depth)

            self.env_qpos_array.append(np.concatenate(
                [self.drone.get_pos().cpu().view(-1), self.drone.get_quat().cpu().view(-1)]))
            self.gridmap_array.append(self.mapper.to_tensor_prob().cpu())

        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)

        obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)
        obs_gridmap = np.stack(self.gridmap_array[-self.obs_horizon:], axis=0)

        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_image:
            # build directly on the proper device
            obs_dict['image'] = torch.as_tensor(
                obs_img, device=self.device).permute(0, 3, 1, 2).unsqueeze(0)
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
            self.drone.set_pos(torch.tensor([INIT_POS]))
            self.drone.set_quat([1, 0, 0, 0])
            # self.drone.set_quat(xyz_to_quat(torch.tensor([[90, 0, 0]])))
            cam_transform = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            # self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu(
            # ).numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform)

            self.cam.set_pose(
                pos=self.drone.get_pos()[0].cpu().numpy(),
                lookat=(0, 0, 0.7)
            )

            img, depth, segmentation, _ = self.cam.render(
                depth=True, segmentation=True)
            # Set all values in depth that are exactly 100 to 0
            depth = np.where(depth == 100, 0, depth)

        self.color_array.append(img)
        self.gridmap_array.append(
            self.mapper.to_tensor_prob().cpu())
        # self.gridmap_array.append(
        #     self.update_occupancy_gridmap(depth, segmentation).cpu())
        # action = np.zeros(7)
        print("Drone ready!")

        env_qpos = np.concatenate(
            [self.drone.get_pos().cpu().view(-1), self.drone.get_quat().cpu().view(-1)])

        self.env_qpos_array.append(env_qpos)
        agent_pos = np.stack([self.env_qpos_array[-1]]
                             * self.obs_horizon, axis=0)

        obs_img = np.stack([self.color_array[-1]]*self.obs_horizon, axis=0)
        obs_gridmap = np.stack([self.gridmap_array[-1]]
                               * self.obs_horizon, axis=0)

        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_image:
            obs_dict['image'] = torch.as_tensor(
                obs_img, device=self.device).permute(0, 3, 1, 2).unsqueeze(0)
        if self.use_gridmap:
            obs_dict['gridmap'] = torch.as_tensor(
                obs_gridmap, device=self.device).unsqueeze(0)  # torch.Size([2, 25, 25, 25])
        return obs_dict

    def create_camera_info_msg(self):
        """CameraInfoメッセージを作成（固定値）"""
        msg = CameraInfo()
        msg.header.frame_id = 'camera_link'
        msg.width = self.cam.res[0]
        msg.height = self.cam.res[1]

        K = self.cam.intrinsics.cpu().numpy() if hasattr(
            self.cam.intrinsics, 'cpu') else self.cam.intrinsics
        msg.k = K.flatten().tolist()

        # 投影行列 P = K[I|0]
        P = np.zeros((3, 4))
        P[:3, :3] = K
        msg.p = P.flatten().tolist()

        # 歪み係数（歪みなしと仮定）
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.distortion_model = "plumb_bob"

        # 回転行列（単位行列）
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        return msg


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config')),
    version_base=None
)
# MARK:
def main(cfg: OmegaConf):
    rclpy.init()
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

    env = PCPublisherNode(
        obs_horizon=2,
        action_horizon=action_horizon,
        device="gpu",
        use_image=use_image,
        img_size=img_size,
    )

    obs_dict = env.reset(first_init=first_init)
    env.csv_list.append((0, 0, 0))

    step_count = 0
    while step_count < roll_out_length:
        with torch.no_grad():
            action = policy(obs_dict)[-1]   # torch.Size([15, 7])
            if use_waypoints:
                action_torch = torch.stack(
                    [torch.tensor(act, dtype=torch.float32, device="cuda") for act in action])
                action_list_pos = [act[:3] for act in action]
                action_torch_pos = torch.stack([torch.tensor(
                    act, dtype=torch.float32, device="cuda") for act in action_list_pos])

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

        obs_dict = env.step(action_list)
        # allow rclpy to process pending callbacks (publishers / TF)
        try:
            rclpy.spin_once(env, timeout_sec=0)
        except Exception:
            # ignore spin errors here to let rollout continue
            pass
        step_count += action_horizon
        print(f"step_count: {step_count}")

        path_length = env.path_sum.item()
        cprint(f"Path length: {env.path_sum:.2f} meters", "green")
        # cover_ratio, _ = eval_inference(
        #     TARGET,
        #     env.pcd_total,
        #     scale=SCALE,
        #     threshold_icp=1,
        #     threshold_dis=0.002,
        #     visualize=False,
        #     noise_remove=False,
        # )
        # coverage = cover_ratio * 100
        # cprint(f"Coverage ratio: {coverage:.4f}", "green")
        # env.csv_list.append((step_count, coverage, path_length))

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
