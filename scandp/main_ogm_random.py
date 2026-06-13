
import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
# ROS2 imports
import rclpy
import tf2_ros
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sklearn.decomposition import PCA

# Custom imports
import diffusion_policy_3d.module.gridmap_feat as gridmap
from diffusion_policy_3d.module.extractor_dinov2 import DINOv2FeatureExtractor
from diffusion_policy_3d.module.publisher import PublisherHelper

# export CUDA_VISIBLE_DEVICES=1


class PCPublisherNode(Node):
    def __init__(self):
        super().__init__('pcl_publisher_node')

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

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
                file="meshes/bunny.obj",
                # file="meshes/dragon.obj",
                pos=(0, 0, 0.7),
                fixed=True,
            )
        )
        self.scene.build(n_envs=1, env_spacing=(8.0, 8.0))
        self.cam.render(depth=True, segmentation=False)

        self.camera_info_msg = self.create_camera_info_msg()

        self.i = 0
        self.timer = self.create_timer(0.02, self.step)

        self.dinov2 = DINOv2FeatureExtractor(
            model_name='dinov2_vitb14',
            device=self.device,
            batch_size=8,
        )   # patch_size=14

        self.mapper = gridmap.LocalMap3D(
            X_lim=np.array([-0.8, 0.8]),
            Y_lim=np.array([-0.8, 0.8]),
            Z_lim=np.array([0.0, 1.6]),
            resolution=0.02,
            p=0.5,
            device="cuda"
        )

    def step(self):
        # if self.i >= 10_000:
        #     self.get_logger().info('Publishing finished.')
        #     return

        self.scene.step()
    # Sample a random position on a hemisphere surface.
    # Assumption: hemisphere is centered at origin.
    # Camera looks at target z=0.7.
    # We sample cos(theta) in [0,1] to obtain uniform surface distribution.
        # - phi is sampled uniformly in [0, 2pi)
        # - radius is randomized between 1.0 and 2.5 meters to vary distance
        r = float(np.random.uniform(1.0, 2.5))
        cos_theta = float(np.random.uniform(0.0, 1.0))
        phi = float(np.random.uniform(0.0, 2.0 * np.pi))
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        x = r * sin_theta * np.cos(phi)
        y = r * sin_theta * np.sin(phi)
        z = r * cos_theta

        self.cam.set_pose(
            pos=(float(x), float(y), float(z)),
            lookat=(0, 0, 0.7),
        )
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

        patch_mask_area = resized_area.squeeze().bool()  # torch.Size([ph, pw])
        # invert mask: foreground=False, background=True
        patch_mask_area = ~patch_mask_area

        feats_flat = feats[0]   # torch.Size([ph*pw, C])
        patch_mask_flat = patch_mask_area.view(-1)  # torch.Size([ph*pw])
        feats_masked = feats_flat[~patch_mask_flat]   # torch.Size([ph*pw, C])

        pca = PCA(n_components=3)
        selected_pca = pca.fit_transform(
            feats_masked.cpu().numpy())    # (N, 3)

        pca_feats = torch.zeros((ph, pw, 3), device=self.device)

        pca_feats[~patch_mask_area] = torch.from_numpy(
            selected_pca).to(self.device)
        pca_rgb = (pca_feats - pca_feats.min()) / \
            (pca_feats.max() - pca_feats.min())

        # bg_mask, pca_rgb, pca_feats = self.dinov2.pca_segment_gpu(
        #     feats_masked, ph, pw)

        # plt.subplot(1, 3, 1)
        # plt.title("Seg Mask")
        # plt.imshow(seg_mask_t.cpu())
        # plt.subplot(1, 3, 2)
        # plt.title("Mask (Area)")
        # plt.imshow(patch_mask_area.cpu())
        # plt.subplot(1, 3, 3)
        # plt.title("PCA")
        # plt.imshow(pca_rgb.cpu())
        # plt.show()

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
            pc_masked_t = pc_masked.to(device=self.device, dtype=torch.float32)

        pca_rgb_flat = pca_rgb_resized.reshape(-1, pca_rgb_resized.shape[2])
        pca_feats_flat = pca_feats_resized.reshape(
            -1, pca_feats_resized.shape[2])  # (H*W, C)
        # normalize pca_feats_flat to range [-1, 1] per-channel (robust to zero range)
        pca_feats_flat = pca_feats_flat.to(dtype=torch.float32)
        min_vals = pca_feats_flat.min(dim=0, keepdim=True)[0]
        max_vals = pca_feats_flat.max(dim=0, keepdim=True)[0]
        mid = (max_vals + min_vals) / 2.0
        half_range = (max_vals - min_vals) / 2.0
        half_range[half_range == 0.0] = 1.0
        pca_feats_flat = (pca_feats_flat - mid) / half_range
        pca_feats_flat = pca_feats_flat.clamp(-1.0, 1.0)

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

        # timestamp for this step's messages
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

        self.i += 1

    def create_camera_info_msg(self):
        """CameraInfoメッセージを作成（固定値）"""
        msg = CameraInfo()
        msg.header.frame_id = 'camera_link'
        msg.width = self.cam.res[0]
        msg.height = self.cam.res[1]

        # カメラ内部パラメータ
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


def main(args=None):
    rclpy.init(args=args)
    node = PCPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
