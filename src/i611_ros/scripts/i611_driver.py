#!/usr/bin/env python3
import rospy
import actionlib
import xmlrpc.client
import math
from sensor_msgs.msg import JointState
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryResult
)

# ==============================
# 設定
# ==============================
RPC_HOST = "192.168.0.23"
RPC_PORT = 4416
RPC_URL = f"http://{RPC_HOST}:{RPC_PORT}"

JOINT_NAMES = [
    "Joint_1",
    "Joint_2",
    "Joint_3",
    "Joint_4",
    "Joint_5",
    "Joint_6"
]

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0


class i611Driver:
    def __init__(self):
        rospy.init_node("i611_driver")

        # XML-RPC 接続
        rospy.loginfo("Connecting to i611 XML-RPC server...")
        self.rpc = xmlrpc.client.ServerProxy(RPC_URL, allow_none=True)

        # 接続確認
        rospy.loginfo("i611 says: %s", self.rpc.Hello())

        # JointState publisher
        self.joint_state_pub = rospy.Publisher(
            "/joint_states", JointState, queue_size=10)

        # FollowJointTrajectory Action Server
        self.server = actionlib.SimpleActionServer(
            "/follow_joint_trajectory",
            FollowJointTrajectoryAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self.server.start()

        # Timer for joint state update
        rospy.Timer(rospy.Duration(0.1), self.publish_joint_states)

        rospy.loginfo("i611_driver ready.")

    # ==============================
    # JointState publish
    # ==============================
    def publish_joint_states(self, event):
        try:
            jnt_deg = self.rpc.getjnt()   # [deg]
            if not jnt_deg:
                return

            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = JOINT_NAMES
            msg.position = [deg * DEG2RAD for deg in jnt_deg]

            self.joint_state_pub.publish(msg)

        except Exception as e:
            rospy.logwarn("JointState error: %s", e)

    # ==============================
    # Trajectory execution
    # ==============================
    def execute_cb(self, goal):
        rospy.loginfo("Received trajectory with %d points",
                      len(goal.trajectory.points))

        try:
            for point in goal.trajectory.points:
                # MoveIt → rad, i611 → deg
                joint_deg = [
                    pos * RAD2DEG for pos in point.positions
                ]

                rospy.loginfo("Move joint (deg): %s", joint_deg)

                # 非同期移動（i611 側でスレッド）
                ok = self.rpc.move_joint(joint_deg)

                if not ok:
                    raise RuntimeError("i611 move_joint failed")

                # time_from_start を待つ（簡易同期）
                rospy.sleep(point.time_from_start.to_sec())

            result = FollowJointTrajectoryResult()
            result.error_code = FollowJointTrajectoryResult.SUCCESSFUL
            self.server.set_succeeded(result)

        except Exception as e:
            rospy.logerr("Trajectory execution failed: %s", e)
            result = FollowJointTrajectoryResult()
            result.error_code = FollowJointTrajectoryResult.INVALID_GOAL
            self.server.set_aborted(result)


if __name__ == "__main__":
    i611Driver()
    rospy.spin()
