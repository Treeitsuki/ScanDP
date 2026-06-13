#!/usr/bin/env python3
from collections import deque

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import Path
from tf.transformations import (concatenate_matrices, quaternion_from_matrix,
                                quaternion_matrix, translation_from_matrix,
                                translation_matrix)


class EEStateNode:
    def __init__(self):
        rospy.init_node("ee_state_node")

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.ds102_link_frame = rospy.get_param(
            "~ds102_link_frame", "ds102_link"
        )
        self.ee_link = rospy.get_param("~ee_link", "Link6")
        self.ee_trans_frame = rospy.get_param("~ee_trans_frame", "ee_state")
        self.tf_timeout = rospy.get_param("~tf_timeout", 0.5)
        self.rate_hz = rospy.get_param("~rate_hz", 60)
        self.path_topic = rospy.get_param("~path_topic", "/ee_state/path")
        self.path_length = rospy.get_param("~path_length", 100_000_000)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.path_pub = rospy.Publisher(
            self.path_topic, Path, queue_size=1)
        self.path_poses = deque(maxlen=int(self.path_length))
        self.rate = rospy.Rate(self.rate_hz)

    @staticmethod
    def _invert_quat(quat):
        x, y, z, w = quat
        norm = x * x + y * y + z * z + w * w
        if norm <= 0.0:
            return (0.0, 0.0, 0.0, 1.0)
        inv = 1.0 / norm
        return (-x * inv, -y * inv, -z * inv, w * inv)

    def _compute_ee_state(self):
        try:
            ds102_tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.ds102_link_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout),
            )
            ee_tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.ee_link,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ):
            return None

        ds102_pos = (
            ds102_tf.transform.translation.x,
            ds102_tf.transform.translation.y,
            ds102_tf.transform.translation.z,
        )
        ds102_quat = (
            ds102_tf.transform.rotation.x,
            ds102_tf.transform.rotation.y,
            ds102_tf.transform.rotation.z,
            ds102_tf.transform.rotation.w,
        )
        ds102_quat_inv = self._invert_quat(ds102_quat)
        ee_pos = (
            ee_tf.transform.translation.x,
            ee_tf.transform.translation.y,
            ee_tf.transform.translation.z,
        )
        ee_quat = (
            ee_tf.transform.rotation.x,
            ee_tf.transform.rotation.y,
            ee_tf.transform.rotation.z,
            ee_tf.transform.rotation.w,
        )

        rotate_about_ds102 = concatenate_matrices(
            translation_matrix(ds102_pos),
            quaternion_matrix(ds102_quat_inv),
            translation_matrix((-ds102_pos[0], -ds102_pos[1], -ds102_pos[2])),
        )
        ee_mat = concatenate_matrices(
            translation_matrix(ee_pos),
            quaternion_matrix(ee_quat),
        )
        ee_state_mat = concatenate_matrices(rotate_about_ds102, ee_mat)

        ee_state = TransformStamped()
        ee_state.header.stamp = rospy.Time.now()
        ee_state.header.frame_id = self.map_frame
        ee_state.child_frame_id = self.ee_trans_frame
        ee_state_pos = translation_from_matrix(ee_state_mat)
        ee_state_quat = quaternion_from_matrix(ee_state_mat)
        ee_state.transform.translation.x = ee_state_pos[0]
        ee_state.transform.translation.y = ee_state_pos[1]
        ee_state.transform.translation.z = ee_state_pos[2]
        ee_state.transform.rotation = Quaternion(*ee_state_quat)
        return ee_state

    def _publish_path(self, ee_state):
        pose = PoseStamped()
        pose.header.stamp = ee_state.header.stamp
        pose.header.frame_id = ee_state.header.frame_id
        pose.pose.position.x = ee_state.transform.translation.x
        pose.pose.position.y = ee_state.transform.translation.y
        pose.pose.position.z = ee_state.transform.translation.z
        pose.pose.orientation = ee_state.transform.rotation

        self.path_poses.append(pose)

        path = Path()
        path.header.stamp = ee_state.header.stamp
        path.header.frame_id = ee_state.header.frame_id
        path.poses = list(self.path_poses)
        self.path_pub.publish(path)

    def run(self):
        while not rospy.is_shutdown():
            ee_state = self._compute_ee_state()
            if ee_state is None:
                rospy.logwarn_throttle(
                    5.0,
                    "ee_state publish skipped: TF for %s or %s unavailable",
                    self.ds102_link_frame,
                    self.ee_link,
                )
            else:
                self.tf_broadcaster.sendTransform(ee_state)
                self._publish_path(ee_state)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = EEStateNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
