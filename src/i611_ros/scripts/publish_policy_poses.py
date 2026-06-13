#!/usr/bin/env python3
import math
import random

import rospy
from geometry_msgs.msg import Pose, PoseArray
from tf.transformations import quaternion_from_matrix


def _normalize(vec):
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(v / norm for v in vec)


def _look_at_quat(from_pos, to_pos, up=(0.0, 0.0, 1.0)):
    direction = (to_pos[0] - from_pos[0], to_pos[1] -
                 from_pos[1], to_pos[2] - from_pos[2])
    x_axis = _normalize(direction)
    if x_axis == (0.0, 0.0, 0.0):
        return (0.0, 0.0, 0.0, 1.0)

    up_axis = _normalize(up)
    dot = x_axis[0] * up_axis[0] + x_axis[1] * \
        up_axis[1] + x_axis[2] * up_axis[2]
    z_axis = (up_axis[0] - dot * x_axis[0], up_axis[1] -
              dot * x_axis[1], up_axis[2] - dot * x_axis[2])
    z_axis = _normalize(z_axis)
    if z_axis == (0.0, 0.0, 0.0):
        up_axis = (0.0, 1.0, 0.0)
        dot = x_axis[0] * up_axis[0] + x_axis[1] * \
            up_axis[1] + x_axis[2] * up_axis[2]
        z_axis = (up_axis[0] - dot * x_axis[0], up_axis[1] -
                  dot * x_axis[1], up_axis[2] - dot * x_axis[2])
        z_axis = _normalize(z_axis)

    y_axis = (
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    )

    rot = [
        [x_axis[0], y_axis[0], z_axis[0], 0.0],
        [x_axis[1], y_axis[1], z_axis[1], 0.0],
        [x_axis[2], y_axis[2], z_axis[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    q = quaternion_from_matrix(rot)
    return (q[0], q[1], q[2], q[3])


def _sample_hemisphere(radius, count, center):
    points = []
    for _ in range(count):
        u = random.random()
        v = random.random()
        theta = 2.0 * math.pi * u
        phi = math.acos(v)

        x = center[0] + radius * math.sin(phi) * math.cos(theta)
        y = center[1] + radius * math.sin(phi) * math.sin(theta)
        z = center[2] + radius * math.cos(phi)
        points.append((x, y, z))
    return points


def main():
    rospy.init_node("publish_policy_poses")

    topic = rospy.get_param("~policy_pose_topic", "/policy/poses")
    frame_id = rospy.get_param("~frame_id", "map")
    radius = rospy.get_param("~radius", 0.5)
    count = rospy.get_param("~count", 8)
    center_x = rospy.get_param("~center_x", 0.0)
    center_y = rospy.get_param("~center_y", 0.0)
    center_z = rospy.get_param("~center_z", 0.1)
    center = (center_x, center_y, center_z)

    pub = rospy.Publisher(topic, PoseArray, queue_size=1, latch=True)
    rospy.sleep(0.5)

    poses = PoseArray()
    poses.header.stamp = rospy.Time.now()
    poses.header.frame_id = frame_id

    for pt in _sample_hemisphere(radius, count, center):
        pose = Pose()
        pose.position.x = pt[0]
        pose.position.y = pt[1]
        pose.position.z = pt[2]

        qx, qy, qz, qw = _look_at_quat(pt, center)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        poses.poses.append(pose)
        print(
            f"Added pose at ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}) with orientation ({qx:.2f}, {qy:.2f}, {qz:.2f}, {qw:.2f})")

    pub.publish(poses)
    rospy.loginfo("Published %d poses to %s", len(poses.poses), topic)


if __name__ == "__main__":
    main()
