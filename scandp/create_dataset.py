
import argparse
import os
import time

import genesis as gs
import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy.spatial.transform as tf
# ROS2 imports
# import rclpy
# import tf2_ros
import torch
import torch.nn.functional as F
from cprint import cprint
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat
# from cv_bridge import CvBridge
# from geometry_msgs.msg import TransformStamped
# from rclpy.node import Node
# from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sklearn.decomposition import PCA
from tqdm import tqdm


# Custom imports
import diffusion_policy_3d.module.gridmap_feat as gridmap
from diffusion_policy_3d.module.extractor_dinov2 import DINOv2FeatureExtractor
from diffusion_policy_3d.module.publisher import PublisherHelper


class Teleoperator():
    def __init__(self, hdf5_file_path=None, save_dir="save_dir", enable_rviz=False):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.IMAGE_SIZE = 256
        self.pos = np.zeros(3)
        self.quat = np.zeros(4)

        self.img_array = []
        self.depth_array = []
        self.cloud_array = []
        self.gridmap_array = []

        self.hdf5_file = hdf5_file_path
        # directory where generated demo .h5 will be saved
        self.save_dir = save_dir
        with h5py.File(self.hdf5_file, 'r') as f:
            self.action_array = f["action"][:]
            print(f"load data from {self.hdf5_file}")
        print(f"action", self.action_array.shape)

        # Genesis/scene/camera初期化
        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=True,
                ambient_light=(0.1, 0.1, 0.1),
            ),
        )
        self.cam = self.scene.add_camera(
            res=(256, 256),
            pos=(0.0, -1.0, 0.7),
            lookat=(0, 0, 0.7),
            fov=45,
            # The aperture size of the camera, controlling depth of field.
            aperture=10,
            GUI=True,
            env_idx=0,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="/home/user/workspace/scandp/diffusion_policy_3d/assets/stanfordbunny.stl",
                pos=(0, 0, 0.7),
                scale=3.0,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.72, 0.72, 0.72),
            ),
        )
        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file="/home/user/workspace/scandp/diffusion_policy_3d/assets/drones/cf2x.urdf",
                pos=(0.0, -1.0, 0.02),
                visualization=False,
            ),
        )
        self.scene.build(n_envs=1, env_spacing=(8.0, 8.0))
        self.cam.render(depth=True, segmentation=False)

        self.i = 0

        self.dinov2 = DINOv2FeatureExtractor(
            model_name='dinov2_vitb14',
            device=self.device,
            batch_size=8,
        )   # patch_size=14

        self.mapper = gridmap.LocalMap3D(
            X_lim=np.array([-0.4, 0.4]),
            Y_lim=np.array([-0.4, 0.4]),
            Z_lim=np.array([0.2, 1.0]),
            resolution=0.02,
            p=0.5,
            device="cuda"
        )

        # RViz / ROS2 publishing (optional)
        self.enable_rviz = enable_rviz
        self.rclpy_node = None
        self.publisher_helper = None
        if self.enable_rviz:
            try:
                import rclpy
                from cv_bridge import CvBridge
                from rclpy.node import Node
                from sensor_msgs.msg import CameraInfo, Image, PointCloud2

                # Initialize rclpy (idempotent guard)
                try:
                    rclpy.init()
                except Exception:
                    # Already initialized in the process
                    pass

                node = Node('create_dataset_rviz_publisher')
                # create publishers matching PublisherHelper expectations
                rgb_pub = node.create_publisher(Image, 'camera/image_raw', 10)
                depth_pub = node.create_publisher(
                    Image, 'camera/depth/image_raw', 10)
                pcd_pub = node.create_publisher(PointCloud2, 'pointcloud', 10)
                camera_info_pub = node.create_publisher(
                    CameraInfo, 'camera/camera_info', 10)
                ogm_pub = node.create_publisher(
                    PointCloud2, 'ogm_point_cloud_centers', 10)

                bridge = CvBridge()
                self.rclpy_node = node
                self.publisher_helper = PublisherHelper(
                    node, rgb_pub, depth_pub, pcd_pub, camera_info_pub, bridge
                )
                # keep ogm publisher so we can publish voxel centers separately
                self.ogm_pub = ogm_pub
                # attach rclpy for later cleanup
                self._rclpy = rclpy
            except Exception as e:
                print(
                    f"Warning: RViz integration disabled due to import error: {e}")
                self.enable_rviz = False

    def main(self):
        if True:
            #### some parameters ####
            self.save_img = True
            self.save_depth = True
            # length of demo
            length = 500
            demo_dir = self.save_dir
            demo_name = os.path.basename(self.hdf5_file).replace(".h5", "")

            #### initialize demo saving ####
            os.makedirs(demo_dir, exist_ok=True)
            record_file_name = os.path.join(demo_dir, demo_name + ".h5")
            env_qpos_array = []
            action_array = []
            pos_array = []
            quat_array = []

        for i in tqdm(range(length), desc="Processing frames"):

            self.scene.step()
            x, y, z = self.action_array[i, :3]
            quat = self.action_array[i, 3:]

            self.drone.set_pos(
                pos=torch.tensor([[x, y, z]]),
            )
            self.drone.set_quat(torch.tensor([quat]))

            cam_transform = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu(
            ).numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform)

            img, depth, seg, normal = self.cam.render(
                depth=True, segmentation=True, normal=False)
            seg_mask = (seg == 2)  # shape (H, W) boolean mask
            seg_mask_t = torch.from_numpy(seg_mask).to(self.device)

            img_torch = torch.from_numpy(np.ascontiguousarray(img)).to(
                self.device)    # torch.Size([256, 256, 3])
            feats, ph, pw = self.dinov2.extract_features_batch(
                img_torch.unsqueeze(0))

            mask_float = seg_mask_t.float().unsqueeze(0).unsqueeze(0)
            resized_area = F.interpolate(
                mask_float, size=(ph, pw), mode='area')

            # torch.Size([ph, pw])
            patch_mask_area = resized_area.squeeze().bool()
            # invert mask: foreground=False, background=True
            patch_mask_area = ~patch_mask_area

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
                pca_feats[~patch_mask_area] = torch.from_numpy(
                    selected_pca).to(self.device)

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

            # keep original pc (H, W, 3) for later masking
            pc, _ = self.cam.render_pointcloud()
            pc_flat_orig = pc.reshape(-1, 3)  # [H*W, 3]

            # FPS sampling to 4096 points using Open3D (fallback to repeat if not enough points)
            num_samples = 4096
            pts = pc_flat_orig.astype(np.float64)  # Open3D prefers float64

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)

            if pts.shape[0] >= num_samples:
                # farthest point sampling (available in recent Open3D)
                sampled_pcd = pcd.farthest_point_down_sample(num_samples)
                sampled_np = np.asarray(sampled_pcd.points)
            else:
                # fallback: repeat indices to reach desired count
                n = pts.shape[0]
                reps = np.repeat(np.arange(n), int(
                    np.ceil(num_samples / n)))[:num_samples]
                sampled_np = pts[reps]

            # final pc_flat is the sampled point cloud in numpy (float32)
            pc_flat = sampled_np.astype(np.float32)
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

            # colors_t = img_flat[mask_t].to(dtype=torch.float32) / 255.0  # (N,3)
            colors_t = pca_rgb_flat[mask_t].to(
                dtype=torch.float32)  # (N,3)
            pc_color = torch.cat((pc_masked_t, colors_t), dim=1)  # (N,6)
            pc_feat = torch.cat(
                (pc_masked_t, pca_feats_flat[mask_t]), dim=1)    # (N,6)

            self.mapper.update(
                torch.tensor(self.cam.pos[0], device="cuda"),
                torch.tensor(self.cam.pos[1], device="cuda"),
                torch.tensor(self.cam.pos[2], device="cuda"),
                pc_feat,
                p_free=0.3,
                p_occ=0.7,
            )

            ogm_pc = self.mapper.voxel_centers_with_rgb(threshold=0.9)
            gridmap_feat = self.mapper.to_tensor_prob()

            # Optionally publish to RViz via ROS2
            if self.enable_rviz and self.publisher_helper is not None:
                try:
                    timestamp = self.rclpy_node.get_clock().now().to_msg()
                    # publish colored pointcloud (camera points + PCA colors)
                    self.publisher_helper.publish_pointcloud2(
                        pc_color, timestamp)
                    # publish OGM voxel centers to a separate topic
                    self.publisher_helper.publish_pointcloud2_to(
                        ogm_pc, timestamp, self.ogm_pub)
                except Exception as e:
                    print(f"RViz publish error: {e}")

            self.write_data(img, depth, pc_flat, gridmap_feat.cpu())
            cam_pos = self.cam.get_pos().cpu().view(-1)
            cam_quat = self.cam.get_quat().cpu().view(-1)

            action = torch.cat((cam_pos, cam_quat), dim=0).numpy()
            action_array.append(action)
            env_qpos_array.append(action)

        # save the data
        with h5py.File(record_file_name, "w") as f:
            seq_length = len(action_array)
            img_array = np.array(self.img_array)[:seq_length]
            depth_array = np.array(self.depth_array)[:seq_length]
            cloud_array = np.array(self.cloud_array)[:seq_length]
            env_qpos_array = np.array(env_qpos_array)[:seq_length]
            action_array = np.array(action_array)[:seq_length]
            gridmap_array = np.array(self.gridmap_array)[:seq_length]

            f.create_dataset("color", data=img_array[:])
            f.create_dataset("depth", data=depth_array[:])
            f.create_dataset("cloud", data=cloud_array[:])
            f.create_dataset("env_qpos_proprioception", data=env_qpos_array[:])
            f.create_dataset("action", data=action_array[:])
            f.create_dataset(
                "gridmap", data=gridmap_array[:], compression='gzip')

            # cprint(f"color shape: {img_array.shape}", "yellow")
            # cprint(f"depth shape: {depth_array.shape}", "yellow")
            # cprint(f"cloud shape: {cloud_array.shape}", "yellow")
            # cprint(f"action shape: {action_array.shape}", "yellow")
            # cprint(f"env_qpos shape: {env_qpos_array.shape}", "yellow")
            # cprint(f"gridmap shape: {gridmap_array.shape}", "yellow")
            # cprint(
            #     f"save data at step: {seq_length} in {record_file_name}", "yellow")

        # Clean up rclpy node if we created one
        if self.enable_rviz and getattr(self, 'rclpy_node', None) is not None:
            try:
                self.rclpy_node.destroy_node()
                # shutdown rclpy
                try:
                    self._rclpy.shutdown()
                except Exception:
                    pass
            except Exception as e:
                print(f"RViz cleanup error: {e}")

    def write_data(self, img, depth, point_cloud, occ_grid_map):
        self.img_array.append(img)
        self.depth_array.append(depth)
        self.cloud_array.append(point_cloud)
        self.gridmap_array.append(occ_grid_map)

    def process_hdf5_file(input_dir, output_dir):
        """Processes all HDF5 files in the given directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.h5'):
                file_path = os.path.join(input_dir, file_name)
                teleoperator = Teleoperator(hdf5_file_path=file_path)
                # `main` is a synchronous method. Call it directly instead of
                # using asyncio.run which expects a coroutine.
                teleoperator.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process a single HDF5 demo and save an output demo file.')
    parser.add_argument('input', help='Path to input HDF5 file to process')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Directory to save generated demos (overrides default)')
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir is not None else 'save_dir'
    teleoperator = Teleoperator(hdf5_file_path=args.input, save_dir=out_dir)
    # main is synchronous
    teleoperator.main()
