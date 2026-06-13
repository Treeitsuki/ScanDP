import numpy as np
import sensor_msgs.msg
import torch


class PublisherHelper:
    """Helper to publish images and pointclouds efficiently.

    - Avoids creating messages when there are no subscribers.
    - Keeps publishing logic centralized so main code is simpler.
    """

    def __init__(self, node, rgb_pub, depth_pub, pcd_pub,
                 camera_info_pub, bridge):
        self.node = node
        self.rgb_pub = rgb_pub
        self.depth_pub = depth_pub
        self.pcd_pub = pcd_pub
        self.camera_info_pub = camera_info_pub
        self.bridge = bridge

    def _has_subscribers(self, pub):
        # rclpy Publisher has get_subscription_count(). Guard for
        # other runtimes that may not implement it.
        try:
            return pub.get_subscription_count() > 0
        except Exception:
            return True

    def publish_rgb_image(self, img, timestamp):
        """Publish RGB image if there are subscribers."""
        if not self._has_subscribers(self.rgb_pub):
            return
        msg = self.bridge.cv2_to_imgmsg(img, encoding='rgb8')
        msg.header.stamp = timestamp
        msg.header.frame_id = 'camera_link'
        self.rgb_pub.publish(msg)

    def publish_depth_image(self, depth, timestamp):
        """Publish depth image if there are subscribers."""
        if not self._has_subscribers(self.depth_pub):
            return
        # Ensure float32 depth
        depth_msg = self.bridge.cv2_to_imgmsg(
            depth.astype(np.float32), encoding='32FC1')
        depth_msg.header.stamp = timestamp
        depth_msg.header.frame_id = 'camera_link'
        self.depth_pub.publish(depth_msg)

    def publish_pointcloud2(self, pc, timestamp, frame_id='map'):
        """Publish PointCloud2 from numpy or torch tensor (with optional color)

        pc: shape (N,3) or (N,6) torch.Tensor or np.ndarray
            - [:,0:3]: XYZ (float)
            - [:,3:6]: RGB (optional, 0~1 or 0~255)
        """
        if not self._has_subscribers(self.pcd_pub):
            return

        # Convert torch.Tensor -> numpy
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()

        # Flatten
        pc = pc.reshape(-1, pc.shape[-1])
        if pc.shape[0] == 0:
            return

        finite_mask = np.isfinite(pc[:, :3]).all(axis=1)
        pc = pc[finite_mask]

        xyz = pc[:, :3].astype(np.float32)
        has_color = pc.shape[1] >= 6

        msg = sensor_msgs.msg.PointCloud2()
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = xyz.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True

        if has_color:
            msg.fields = [
                sensor_msgs.msg.PointField(
                    name='x', offset=0, datatype=7, count=1),
                sensor_msgs.msg.PointField(
                    name='y', offset=4, datatype=7, count=1),
                sensor_msgs.msg.PointField(
                    name='z', offset=8, datatype=7, count=1),
                sensor_msgs.msg.PointField(
                    name='rgb', offset=12, datatype=7, count=1),
            ]
            msg.point_step = 16
        else:
            msg.fields = [
                sensor_msgs.msg.PointField(
                    name='x', offset=0, datatype=7, count=1),
                sensor_msgs.msg.PointField(
                    name='y', offset=4, datatype=7, count=1),
                sensor_msgs.msg.PointField(
                    name='z', offset=8, datatype=7, count=1),
            ]
            msg.point_step = 12

        msg.row_step = msg.point_step * msg.width

        # --- RGB encoding (if any) ---
        if has_color:
            colors = pc[:, 3:6]
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)

            # Pack RGB into float32
            rgb_uint32 = (colors[:, 0].astype(np.uint32) << 16) | \
                         (colors[:, 1].astype(np.uint32) << 8) | \
                         (colors[:, 2].astype(np.uint32))
            rgb_float = rgb_uint32.view(np.float32)

            data = np.column_stack((xyz, rgb_float))
        else:
            data = xyz

        msg.data = data.astype(np.float32).tobytes()
        self.pcd_pub.publish(msg)

    def publish_pointcloud2_to(self, pc, timestamp, pub, frame_id='map'):
        """Publish PointCloud2 to an arbitrary publisher (same semantics as publish_pointcloud2).

        pc: shape (N,3) or (N,6) torch.Tensor or np.ndarray
        pub: rclpy Publisher for sensor_msgs/PointCloud2
        """
        if not self._has_subscribers(pub):
            return

        # Convert torch.Tensor -> numpy
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()

        # Flatten
        pc = pc.reshape(-1, pc.shape[-1])
        if pc.shape[0] == 0:
            return

        finite_mask = np.isfinite(pc[:, :3]).all(axis=1)
        pc = pc[finite_mask]

        xyz = pc[:, :3].astype(np.float32)
        has_color = pc.shape[1] >= 6

        msg = sensor_msgs.msg.PointCloud2()
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = xyz.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True

        if has_color:
            msg.fields = [
                sensor_msgs.msg.PointField(
                    name='x', offset=0, datatype=7, count=1),
                sensor_msgs.msg.PointField(
                    name='y', offset=4, datatype=7, count=1),
                sensor_msgs.msg.PointField(
                    name='z', offset=8, datatype=7, count=1),
                sensor_msgs.msg.PointField(
                    name='rgb', offset=12, datatype=7, count=1),
            ]
            msg.point_step = 16
        else:
            msg.fields = [
                sensor_msgs.msg.PointField(
                    name='x', offset=0, datatype=7, count=1),
                sensor_msgs.msg.PointField(
                    name='y', offset=4, datatype=7, count=1),
                sensor_msgs.msg.PointField(
                    name='z', offset=8, datatype=7, count=1),
            ]
            msg.point_step = 12

        msg.row_step = msg.point_step * msg.width

        # --- RGB encoding (if any) ---
        if has_color:
            colors = pc[:, 3:6]
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)

            # Pack RGB into float32
            rgb_uint32 = (colors[:, 0].astype(np.uint32) << 16) | \
                         (colors[:, 1].astype(np.uint32) << 8) | \
                         (colors[:, 2].astype(np.uint32))
            rgb_float = rgb_uint32.view(np.float32)

            data = np.column_stack((xyz, rgb_float))
        else:
            data = xyz

        msg.data = data.astype(np.float32).tobytes()
        pub.publish(msg)
