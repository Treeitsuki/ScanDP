#!/usr/bin/env python3
import logging
import sys

import numpy as np
import torch

# 事前に bresenham_torch がインストールされているか、パスが通っている必要があります
try:
    from bresenham_torch import bresenham3D
except ImportError:
    print("Error: bresenham_torch module not found. Please ensure it is in your PYTHONPATH.")
    sys.exit(1)

# --- ROS Imports ---
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sensor_msgs
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# -------------------------------------------------------------------


class LocalMap3D:
    def __init__(self, X_lim, Y_lim, Z_lim, resolution, p, device=None):
        self.ratio = 100  # メートルをセンチメートル単位等へ変換して整数計算しやすくするための係数
        self.X_lim = X_lim * self.ratio
        self.Y_lim = Y_lim * self.ratio
        self.Z_lim = Z_lim * self.ratio
        self.resolution = resolution * self.ratio
        self.p = p
        self.device = device if device is not None else torch.device("cpu")

        # グリッド座標の生成
        x = torch.arange(
            start=self.X_lim[0], end=self.X_lim[1], step=self.resolution, device=self.device)
        y = torch.arange(
            start=self.Y_lim[0], end=self.Y_lim[1], step=self.resolution, device=self.device)
        z = torch.arange(
            start=self.Z_lim[0], end=self.Z_lim[1], step=self.resolution, device=self.device)

        self.x_max, self.y_max, self.z_max = len(x), len(y), len(z)

        rospy.loginfo(
            f"Map Size initialized: [{self.x_max}, {self.y_max}, {self.z_max}] grid cells")

        # インデックスエンコード用のストライド（C-contiguous: X, Y, Z）
        self.stride_y = self.z_max
        self.stride_x = self.y_max * self.z_max

        # マップ原点の保持（更新時に毎回生成しない）
        self.origin_tensor = torch.tensor(
            [self.X_lim[0], self.Y_lim[0], self.Z_lim[0]], device=self.device, dtype=torch.float32)

        # (x, y, z, 4): log-odds + f1 + f2 + f3
        self.gridmap = torch.zeros(
            (self.x_max, self.y_max, self.z_max, 4), device=self.device)
        self.gridmap[..., 0] = self.log_odds(p)

    def log_odds(self, p):
        if isinstance(p, float):
            p = torch.tensor(p, device=self.device)
        return torch.log(p / (1 - p))

    def retrieve_p(self, log_map):
        return 1 - 1 / (1 + torch.exp(log_map))

    def is_valid(self, x_idx, y_idx, z_idx):
        return (x_idx >= 0) & (x_idx < self.x_max) & \
               (y_idx >= 0) & (y_idx < self.y_max) & \
               (z_idx >= 0) & (z_idx < self.z_max)

    def update(self, x0, y0, z0, points, p_free, p_occ):
        # 1. 入力チェック
        if points.shape[0] == 0:
            return

        # [N, 3] または [N, 6] の確認
        has_features = (points.shape[1] == 6)

        # 座標変換 (メートル -> グリッド計算用スケール)
        pts_xyz = points[:, :3] * self.ratio
        feats = points[:, 3:] if has_features else None

        # センサー原点 (メートル -> グリッド計算用スケール)
        origin_pos = torch.tensor(
            [x0, y0, z0], device=self.device, dtype=torch.float32) * self.ratio

        # --- Index Calculation ---
        # 始点 (センサー位置) のインデックス: 負の値や範囲外もあり得るが bresenham は許容する
        start_idx = torch.floor(
            (origin_pos - self.origin_tensor) / self.resolution).to(torch.int64)

        # 終点 (ヒット位置) のインデックス
        end_idx = torch.floor(
            (pts_xyz - self.origin_tensor) / self.resolution).to(torch.int64)

        # 2. 有効範囲内のエンドポイントのみを抽出
        valid_mask = self.is_valid(end_idx[:, 0], end_idx[:, 1], end_idx[:, 2])
        end_idx = end_idx[valid_mask]
        if has_features:
            feats = feats[valid_mask]

        if end_idx.shape[0] == 0:
            return

        # 3. Ray Casting (Free Space Update)
        # 修正: map_sizeを対角線長をカバーできる程度に大きく取る (x+y+z)
        # これが短いと、遠くの壁に当たった時に手前の空間がクリアされず、更新がおかしくなる
        ray_length_limit = self.x_max + self.y_max + self.z_max
        free_points = bresenham3D(
            start_idx, end_idx, map_size=ray_length_limit)
        free_points = free_points.to(self.device)

        # 4. 衝突点(Endpoint)をFree空間から除外するためのフィルタリング
        def encode_idx(idx):
            # idx: [N, 3] -> flat index
            return idx[:, 0] * self.stride_x + idx[:, 1] * self.stride_y + idx[:, 2]

        free_encoded = encode_idx(free_points)
        end_encoded = encode_idx(end_idx)

        # エンドポイントと同じ座標にある Free 点を除去 (壁を消さないようにする)
        mask_not_hit = ~torch.isin(free_encoded, end_encoded)
        free_points = free_points[mask_not_hit]

        # 5. マップ範囲外の Free 点を除去
        valid_free = self.is_valid(
            free_points[:, 0], free_points[:, 1], free_points[:, 2])
        free_points = free_points[valid_free]

        # 6. グリッド更新 (Log-Odds)
        # Free Space: 確率を下げる
        # 注意: 同じボクセルを何度も通過する場合、単純加算だと急激に下がる可能性があるが、
        # 元コード(Code1)の挙動に合わせてそのまま加算する。
        self.gridmap[free_points[:, 0], free_points[:, 1],
                     free_points[:, 2], 0] += self.log_odds(p_free)

        # Occupied Space: 確率を上げる
        self.gridmap[end_idx[:, 0], end_idx[:, 1],
                     end_idx[:, 2], 0] += self.log_odds(p_occ)

        # 7. 特徴量の更新 (Occupiedのみ)
        if has_features:
            for i in range(3):
                self.gridmap[end_idx[:, 0], end_idx[:, 1],
                             end_idx[:, 2], i + 1] += feats[:, i]

    def to_prob_occ_map(self):
        return self.retrieve_p(self.gridmap[..., 0])

    def voxel_centers_with_rgb(self, threshold=0.5):
        occ = self.to_prob_occ_map()
        mask = occ >= threshold
        indices = torch.nonzero(mask, as_tuple=False)
        if indices.shape[0] == 0:
            return torch.zeros((0, 6), device=self.device, dtype=torch.float32)

        ix = indices[:, 0].to(torch.float32)
        iy = indices[:, 1].to(torch.float32)
        iz = indices[:, 2].to(torch.float32)

        # グリッドインデックス -> メートル座標
        x0 = (self.X_lim[0] + (ix + 0.5) * self.resolution) / self.ratio
        y0 = (self.Y_lim[0] + (iy + 0.5) * self.resolution) / self.ratio
        z0 = (self.Z_lim[0] + (iz + 0.5) * self.resolution) / self.ratio

        feats = self.gridmap[indices[:, 0], indices[:, 1], indices[:, 2], 1:4]

        # 特徴量の正規化 (RGB可視化用)
        eps = 1e-12
        f_min = feats.min(dim=0).values
        f_max = feats.max(dim=0).values
        denom = (f_max - f_min) + eps
        feats_norm = (feats - f_min) / denom

        centers = torch.stack([x0, y0, z0], dim=1)
        centers_color = torch.cat([centers, feats_norm], dim=1)
        return centers_color


# --- ROS Utility Functions ---

def unpack_pointcloud2_to_torch(msg, device='cpu', use_rgb=True):
    """
    Convert PointCloud2 msg to torch tensor.
    """
    field_names = [f.name for f in msg.fields]
    has_rgb = 'rgb' in field_names or 'rgba' in field_names

    if use_rgb and has_rgb:
        # xyz + rgb
        pts_gen = pc2.read_points(msg, field_names=(
            "x", "y", "z", "rgb"), skip_nans=True)
        try:
            # ジェネレータを一括リスト化
            pts_list = list(pts_gen)
            if not pts_list:
                return torch.zeros((0, 6), device=device)
            pts_np = np.array(pts_list, dtype=np.float32)
        except Exception as e:
            rospy.logwarn(f"Failed to unpack pointcloud: {e}")
            return torch.zeros((0, 6), device=device)

        rgb_float = pts_np[:, 3]
        # float32として解釈されたデータをuint32に戻してビットシフト
        rgb_uint32 = rgb_float.view(np.uint32)

        r = ((rgb_uint32 >> 16) & 0x0000ff).astype(np.float32) / 255.0
        g = ((rgb_uint32 >> 8) & 0x0000ff).astype(np.float32) / 255.0
        b = ((rgb_uint32) & 0x0000ff).astype(np.float32) / 255.0

        xyz = torch.tensor(pts_np[:, :3], device=device, dtype=torch.float32)
        rgb = torch.tensor(np.stack([r, g, b], axis=1),
                           device=device, dtype=torch.float32)

        return torch.cat([xyz, rgb], dim=1)  # [N, 6]

    else:
        # xyz only
        pts_gen = pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)
        try:
            pts_list = list(pts_gen)
            if not pts_list:
                return torch.zeros((0, 3), device=device)
            pts_tensor = torch.tensor(
                pts_list, device=device, dtype=torch.float32)
            return pts_tensor
        except:
            return torch.zeros((0, 3), device=device)


class MapNode:
    def __init__(self):
        rospy.init_node('localmap3d_node', anonymous=True)

        # --- GPU Configuration ---
        gpu_id = rospy.get_param('~gpu_id', 0)
        if torch.cuda.is_available():
            if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
                self.device = torch.device(f'cuda:{gpu_id}')
                rospy.loginfo(
                    f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            else:
                rospy.logwarn(
                    f"Invalid GPU ID {gpu_id}. Available GPUs: {torch.cuda.device_count()}. Using cuda:0")
                self.device = torch.device('cuda:0')
        else:
            rospy.loginfo("CUDA not available, using CPU")
            self.device = torch.device('cpu')
        rospy.loginfo(f"Using device: {self.device}")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Parameters ---
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.sensor_topic = rospy.get_param(
            '~sensor_topic', '/camera/depth/color/points')

        x_min = rospy.get_param('~x_min', -0.8)
        x_max = rospy.get_param('~x_max', 0.8)
        y_min = rospy.get_param('~y_min', -0.8)
        y_max = rospy.get_param('~y_max', 0.8)
        z_min = rospy.get_param('~z_min', 0.0)
        z_max = rospy.get_param('~z_max', 1.6)
        resolution = rospy.get_param('~resolution', 0.02)

        self.p_init = rospy.get_param('~p_init', 0.5)
        self.p_free = rospy.get_param('~p_free', 0.4)
        self.p_occ = rospy.get_param('~p_occ', 0.7)
        self.vis_threshold = rospy.get_param(
            '~vis_threshold', 0.6)  # 可視化閾値を少し下げて確認しやすくする
        self.use_rgb = rospy.get_param('~use_rgb', True)

        rospy.loginfo(f"Map Config: {self.map_frame}, Res: {resolution}")

        self.mapper = LocalMap3D(
            X_lim=np.array([x_min, x_max]),
            Y_lim=np.array([y_min, y_max]),
            Z_lim=np.array([z_min, z_max]),
            resolution=resolution,
            p=self.p_init,
            device=self.device
        )

        self.sub = rospy.Subscriber(
            self.sensor_topic, PointCloud2, self.callback, queue_size=1, buff_size=2**24
        )

        self.pub_centers = rospy.Publisher(
            'ogm_centers', PointCloud2, queue_size=10)

    def callback(self, msg):
        try:
            # 1. Transform to Map Frame
            trans = self.tf_buffer.lookup_transform(
                self.map_frame,
                msg.header.frame_id,
                msg.header.stamp,
                rospy.Duration(1.0)
            )
            cloud_out = tf2_sensor_msgs.do_transform_cloud(msg, trans)

            # 2. Sensor Origin
            sensor_pos = trans.transform.translation
            x0, y0, z0 = sensor_pos.x, sensor_pos.y, sensor_pos.z

            # 3. Convert to Tensor
            points_tensor = unpack_pointcloud2_to_torch(
                cloud_out, self.device, use_rgb=self.use_rgb
            )

            # 4. Update
            self.mapper.update(
                x0, y0, z0,
                points_tensor,
                p_free=self.p_free,
                p_occ=self.p_occ
            )

            # 5. Publish
            self.publish_map(msg.header.stamp)

        except tf2_ros.ExtrapolationException as e:
            # 点群の時刻が新しすぎて、まだTFが届いていない場合などに発生
            rospy.logwarn_throttle(1.0, f"TF Extrapolation Error: {e}")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
            rospy.logwarn_throttle(1.0, f"TF Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error: {e}")

    def publish_map(self, stamp):
        centers = self.mapper.voxel_centers_with_rgb(
            threshold=self.vis_threshold)
        if centers.shape[0] == 0:
            return

        centers_np = centers.cpu().numpy()
        pc2_msg = self.create_pointcloud2(
            centers_np, frame_id=self.map_frame, stamp=stamp)
        self.pub_centers.publish(pc2_msg)

    def create_pointcloud2(self, centers_rgb, frame_id, stamp):
        if centers_rgb.shape[0] == 0:
            return PointCloud2(header=Header(frame_id=frame_id, stamp=stamp))

        xyz = centers_rgb[:, :3]
        rgb = centers_rgb[:, 3:]

        r = (np.clip(rgb[:, 0], 0, 1) * 255).astype(np.uint8)
        g = (np.clip(rgb[:, 1], 0, 1) * 255).astype(np.uint8)
        b = (np.clip(rgb[:, 2], 0, 1) * 255).astype(np.uint8)

        rgb_int = (r.astype(np.uint32) << 16) | (
            g.astype(np.uint32) << 8) | b.astype(np.uint32)
        rgb_float = rgb_int.view(np.float32).reshape(-1, 1)

        data_np = np.hstack([xyz, rgb_float]).astype(np.float32)

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1)
        ]

        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp
        return pc2.create_cloud(header, fields, data_np)


if __name__ == "__main__":
    try:
        node = MapNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
