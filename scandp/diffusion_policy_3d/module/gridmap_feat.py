import logging

import numpy as np
import torch
from diffusion_policy_3d.module.bresenham_torch import bresenham3D

# from bresenham_torch import bresenham3D


class LocalMap3D:
    def __init__(self, X_lim, Y_lim, Z_lim, resolution, p, device=None):
        self.ratio = 100
        self.X_lim = X_lim * self.ratio
        self.Y_lim = Y_lim * self.ratio
        self.Z_lim = Z_lim * self.ratio
        self.resolution = resolution * self.ratio
        self.p = p
        self.device = device if device is not None else torch.device("cpu")
        self.base = (self.X_lim[1] - self.X_lim[0]) / self.resolution

        x = torch.arange(
            start=self.X_lim[0], end=self.X_lim[1], step=self.resolution, device=self.device)
        y = torch.arange(
            start=self.Y_lim[0], end=self.Y_lim[1], step=self.resolution, device=self.device)
        z = torch.arange(
            start=self.Z_lim[0], end=self.Z_lim[1], step=self.resolution, device=self.device)

        self.x_max, self.y_max, self.z_max = len(x), len(y), len(z)

        # (x, y, z, 4): log-odds + f1 + f2 + f3
        self.gridmap = torch.zeros(
            (self.x_max, self.y_max, self.z_max, 4), device=self.device)
        self.gridmap[..., 0] = self.log_odds(p)  # initialize occupancy

    def log_odds(self, p):
        p = torch.tensor(p, device=self.device)
        return torch.log(p / (1 - p))

    def retrieve_p(self, log_map):
        return 1 - 1 / (1 + torch.exp(log_map))

    def is_valid(self, x_idx, y_idx, z_idx):
        return (x_idx >= 0) & (x_idx < self.x_max) & \
               (y_idx >= 0) & (y_idx < self.y_max) & \
               (z_idx >= 0) & (z_idx < self.z_max)

    def discretize(self, points):
        points = points * self.ratio
        x_idx = torch.floor(
            (points[:, 0] - self.X_lim[0]) / self.resolution).to(torch.int64)
        y_idx = torch.floor(
            (points[:, 1] - self.Y_lim[0]) / self.resolution).to(torch.int64)
        z_idx = torch.floor(
            (points[:, 2] - self.Z_lim[0]) / self.resolution).to(torch.int64)
        valid_mask = self.is_valid(x_idx, y_idx, z_idx)
        return x_idx[valid_mask], y_idx[valid_mask], z_idx[valid_mask]

    def update(self, x0, y0, z0, points, p_free, p_occ):
        """
        Update 3D occupancy and feature maps.
        Args:
            x0, y0, z0: sensor origin (floats or tensors)
            points: Tensor [N, 3] = [x, y, z] or [N, 6] = [x, y, z, f1, f2, f3]
            p_free, p_occ: float (probabilities)
        """
        if points.shape[1] < 3:
            raise ValueError("Input points must have at least 3 columns (x,y,z).")

        # Separate geometry and features
        pts_xyz = points[:, :3] * self.ratio
        feats = points[:, 3:] if points.shape[1] > 3 else None
        num_feat_channels = 0 if feats is None else feats.shape[1]

        x0, y0, z0 = (x0 * self.ratio).to(self.device), (y0 *
                                                         self.ratio).to(self.device), (z0 * self.ratio).to(self.device)
        x0_idx = torch.floor(
            (x0 - self.X_lim[0]) / self.resolution).to(torch.int64)
        y0_idx = torch.floor(
            (y0 - self.Y_lim[0]) / self.resolution).to(torch.int64)
        z0_idx = torch.floor(
            (z0 - self.Z_lim[0]) / self.resolution).to(torch.int64)
        start = torch.tensor([x0_idx, y0_idx, z0_idx], device=self.device)

        x_idx = torch.floor(
            (pts_xyz[:, 0] - self.X_lim[0]) / self.resolution).to(torch.int64)
        y_idx = torch.floor(
            (pts_xyz[:, 1] - self.Y_lim[0]) / self.resolution).to(torch.int64)
        z_idx = torch.floor(
            (pts_xyz[:, 2] - self.Z_lim[0]) / self.resolution).to(torch.int64)
        end = torch.stack([x_idx, y_idx, z_idx], dim=1)

        valid_mask = self.is_valid(x_idx, y_idx, z_idx)
        end = end[valid_mask]
        if feats is not None:
            feats = feats[valid_mask]

        if end.shape[0] == 0:
            return

        # Free space update
        free_points = bresenham3D(start, end, map_size=25).to(self.device)
        base = self.base
        free_points_encoded = free_points[:, 0] * base * \
            base + free_points[:, 1] * base + free_points[:, 2]
        end_encoded = end[:, 0] * base * base + end[:, 1] * base + end[:, 2]
        free_points = free_points[~torch.isin(
            free_points_encoded, end_encoded)]

        fx, fy, fz = free_points[:, 0], free_points[:, 1], free_points[:, 2]
        valid_free = self.is_valid(fx, fy, fz)
        fx, fy, fz = fx[valid_free], fy[valid_free], fz[valid_free]

        # Occupancy update (log-odds)
        self.gridmap[fx, fy, fz, 0] += self.log_odds(p_free)
        self.gridmap[end[:, 0], end[:, 1],
                     end[:, 2], 0] += self.log_odds(p_occ)

        # Feature accumulation for occupied voxels
        if num_feat_channels >= 1:
            for i in range(min(3, num_feat_channels)):
                self.gridmap[end[:, 0], end[:, 1], end[:, 2], i + 1] += feats[:, i]

    def to_tensor(self):
        """Return final [X, Y, Z, 4] grid tensor"""
        return self.gridmap

    def to_tensor_prob(self):
        """Return final [X, Y, Z, 4] grid tensor with occupancy probability.

        Channels are ordered as: [p_occ, f1, f2, f3].
        - p_occ: occupancy probability in range [0, 1], computed from log-odds.
        - f1,f2,f3: accumulated feature channels (same units as stored in gridmap).

        Returns:
            Tensor of shape (X, Y, Z, 4) on the same device as the map.
        """
        # occupancy probability from log-odds channel
        prob = self.to_prob_occ_map()

        # features are stored in channels 1..3
        feats = self.gridmap[..., 1:4]

        # ensure the probability channel has an explicit trailing dimension
        prob = prob.unsqueeze(-1)

        # concatenate into final tensor [p, f1, f2, f3]
        tensor = torch.cat([prob, feats], dim=-1)
        return tensor

    def to_prob_occ_map(self):
        """Return occupancy probability (without features)"""
        return self.retrieve_p(self.gridmap[..., 0])

    def voxel_centers_with_rgb(self, threshold=0.5):
        """
        Return voxel centers whose occupancy probability >= threshold with RGB built
        from normalized sum of features (f1+f2+f3).

        Returns:
            Tensor of shape (N, 6): [x, y, z, r, g, b]
            Coordinates are in meters (same frame as input points).
            Colors are floats in range [0, 1].
        """
        # occupancy probability
        occ = self.to_prob_occ_map()

        mask = occ >= threshold
        indices = torch.nonzero(mask, as_tuple=False)
        if indices.shape[0] == 0:
            # return empty tensor
            return torch.zeros((0, 6), device=self.device, dtype=torch.float32)

        # indices: (N,3) -> ix, iy, iz
        ix = indices[:, 0].to(torch.float32)
        iy = indices[:, 1].to(torch.float32)
        iz = indices[:, 2].to(torch.float32)

        # compute centers in scaled units then convert to meters
        x0 = (self.X_lim[0] + (ix + 0.5) * self.resolution) / self.ratio
        y0 = (self.Y_lim[0] + (iy + 0.5) * self.resolution) / self.ratio
        z0 = (self.Z_lim[0] + (iz + 0.5) * self.resolution) / self.ratio

        # gather features f1,f2,f3 for each occupied voxel
        feats = self.gridmap[indices[:, 0], indices[:, 1], indices[:, 2], 1:4]

        # Normalize each channel independently to [0,1] similar to
        # extractor_dinov2.pca_segment_gpu (per-channel min/max scaling).
        eps = 1e-12
        f_min = feats.min(dim=0).values
        f_max = feats.max(dim=0).values
        denom = (f_max - f_min) + eps
        feats_norm = (feats - f_min) / denom

        # Map features to colors per request: R=f1, G=f3, B=f2
        # feats_norm columns are [f1_norm, f2_norm, f3_norm]
        r = feats_norm[:, 0].unsqueeze(1)
        g = feats_norm[:, 1].unsqueeze(1)
        b = feats_norm[:, 2].unsqueeze(1)

        colors = torch.cat([r, g, b], dim=1).to(dtype=torch.float32)

        centers = torch.stack([x0, y0, z0], dim=1)
        centers_color = torch.cat([centers, colors], dim=1)
        return centers_color

    def reset(self):
        self.gridmap[..., 0] = self.log_odds(self.p)
        self.gridmap[..., 1:] = 0.0


if __name__ == "__main__":
    # Simple demo / ROS2 publisher so you can visualize occupied voxel centers in RViz.
    # The code below runs as a node that publishes PointCloud2 on ``ogm_point_cloud_centers``.
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header
    except Exception:
        raise RuntimeError(
            "rclpy/sensor_msgs not available - run in a ROS2 environment to publish to RViz")

    # choose device automatically
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resolution = 0.02
    mapper = LocalMap3D(
        X_lim=np.array([-0.40, 0.40]),
        Y_lim=np.array([-0.40, 0.40]),
        Z_lim=np.array([0.2, 1.0]),
        resolution=resolution,
        p=0.5,
        device=device
    )

    # generate random demo points (x,y,z,f1,f2,f3)
    points = torch.rand((200, 6), dtype=torch.float32, device=device)

    mapper.update(
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        points,
        p_free=0.3,
        p_occ=0.7,
    )

    def create_pointcloud2_from_centers(centers_rgb, frame_id="map", stamp=None):
        """Create a sensor_msgs.msg.PointCloud2 from Nx6 numpy array [x,y,z,r,g,b]

        colors r,g,b expected in range [0,1]. RGB is packed into a single float32 field
        so RViz can display it.
        """
        # centers_rgb: np.ndarray (N,6) float32
        if centers_rgb is None or centers_rgb.shape[0] == 0:
            # empty pointcloud
            pc2 = PointCloud2()
            pc2.header = Header()
            if stamp is not None:
                pc2.header.stamp = stamp
            pc2.header.frame_id = frame_id
            pc2.height = 1
            pc2.width = 0
            pc2.fields = []
            pc2.is_bigendian = False
            pc2.point_step = 0
            pc2.row_step = 0
            pc2.is_dense = True
            pc2.data = b''
            return pc2

        pts = centers_rgb.astype(np.float32)
        xyz = pts[:, :3].astype(np.float32)
        rgbf = pts[:, 3:6]

        # pack rgb (0-1 floats) -> uint8 each -> uint32 packed -> float32 view
        colors_u8 = (np.clip(rgbf, 0.0, 1.0) * 255).astype(np.uint8)
        rgb_uint32 = (
            (colors_u8[:, 0].astype(np.uint32) << 16)
            | (colors_u8[:, 1].astype(np.uint32) << 8)
            | (colors_u8[:, 2].astype(np.uint32))
        )
        # view as float32 for PointCloud2 rgb field
        rgb_as_float = rgb_uint32.view(np.float32)

        N = xyz.shape[0]

        # structured array to make data bytes
        dtype = np.dtype([
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgb", np.float32),
        ])
        arr = np.zeros(N, dtype=dtype)
        arr["x"] = xyz[:, 0]
        arr["y"] = xyz[:, 1]
        arr["z"] = xyz[:, 2]
        arr["rgb"] = rgb_as_float

        data = arr.tobytes()

        header = Header()
        if stamp is not None:
            header.stamp = stamp
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]

        pc2 = PointCloud2()
        pc2.header = header
        pc2.height = 1
        pc2.width = N
        pc2.fields = fields
        pc2.is_bigendian = False
        pc2.point_step = 16  # 4*4 bytes
        pc2.row_step = pc2.point_step * N
        pc2.is_dense = True
        pc2.data = data
        return pc2

    # start ROS2 and publish repeatedly
    rclpy.init()
    node = Node('localmap3d_ogm_publisher')
    pub = node.create_publisher(PointCloud2, 'ogm_point_cloud_centers', 10)

    try:
        print("Publishing occupied voxel centers -> topic 'ogm_point_cloud_centers' (frame='map')")
        # keep alive, we'll manually publish in loop
        rate = node.create_timer(1.0, lambda: None)
        while rclpy.ok():
            # get occupied voxel centers with color
            centers = mapper.voxel_centers_with_rgb(threshold=0.5)
            tensor = mapper.to_tensor_prob()
            print("Occupancy tensor shape:", tensor.shape)

            if isinstance(centers, torch.Tensor):
                centers_np = centers.cpu().numpy()
            else:
                centers_np = np.array(centers, dtype=np.float32)

            stamp = node.get_clock().now().to_msg()
            pc2 = create_pointcloud2_from_centers(
                centers_np, frame_id='map', stamp=stamp)
            pub.publish(pc2)

            # spin once to handle timers and allow ctrl-c
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
