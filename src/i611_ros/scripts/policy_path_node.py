#!/usr/bin/env python3
from collections import deque

import rospy
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Path


class PolicyPathNode:
    def __init__(self):
        rospy.init_node("policy_path_node")

        self.pose_array_topic = rospy.get_param(
            "~pose_array_topic", "/policy/poses"
        )
        self.path_topic = rospy.get_param(
            "~path_topic", "/policy/poses_path"
        )
        self.frame_id = rospy.get_param("~frame_id", "")
        self.latch = rospy.get_param("~latch", True)
        self.path_length = int(rospy.get_param("~path_length", 0))

        if self.path_length > 0:
            self.path_poses = deque(maxlen=self.path_length)
        else:
            self.path_poses = []

        self.path_pub = rospy.Publisher(
            self.path_topic, Path, queue_size=1, latch=self.latch
        )
        rospy.Subscriber(self.pose_array_topic, PoseArray, self._cb)

    def _cb(self, msg):
        if not msg.poses:
            rospy.logwarn_throttle(
                5.0, "Received empty PoseArray on %s", self.pose_array_topic
            )
            return

        frame_id = self.frame_id if self.frame_id else msg.header.frame_id
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

        for pose in msg.poses:
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = frame_id
            ps.pose = pose
            self.path_poses.append(ps)

        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = frame_id
        path.poses = list(self.path_poses)
        self.path_pub.publish(path)


if __name__ == "__main__":
    try:
        node = PolicyPathNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
