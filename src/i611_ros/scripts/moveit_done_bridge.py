#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from actionlib_msgs.msg import GoalStatus
from moveit_msgs.msg import MoveGroupActionResult, MoveItErrorCodes
from std_msgs.msg import Bool


class MoveItDoneBridge:
    def __init__(self):
        rospy.init_node("moveit_done_bridge", anonymous=False)

        self.result_topic = rospy.get_param(
            "~result_topic", "/move_group/result"
        )
        self.arm_done_topic = rospy.get_param(
            "~arm_done_topic", "/arm/move_done"
        )
        self.publish_on_success_only = rospy.get_param(
            "~publish_on_success_only", True
        )
        self.log_throttle_sec = rospy.get_param(
            "~log_throttle_sec", 2.0
        )

        self._last_goal_id = None

        self._arm_done_pub = rospy.Publisher(
            self.arm_done_topic, Bool, queue_size=10
        )
        rospy.Subscriber(
            self.result_topic, MoveGroupActionResult, self._result_cb, queue_size=10
        )

        rospy.loginfo("MoveItDoneBridge ready")
        rospy.loginfo("result_topic: %s", self.result_topic)
        rospy.loginfo("arm_done_topic: %s", self.arm_done_topic)

    def _result_cb(self, msg):
        goal_id = msg.status.goal_id.id
        if goal_id and goal_id == self._last_goal_id:
            return

        status = msg.status.status
        error = msg.result.error_code.val

        if self.publish_on_success_only:
            if status != GoalStatus.SUCCEEDED:
                rospy.logwarn_throttle(
                    self.log_throttle_sec,
                    "MoveIt result not succeeded (status=%d)",
                    status,
                )
                return
            if error != MoveItErrorCodes.SUCCESS:
                rospy.logwarn_throttle(
                    self.log_throttle_sec,
                    "MoveIt error code not success (error=%d)",
                    error,
                )
                return

        self._last_goal_id = goal_id
        self._arm_done_pub.publish(Bool(True))
        rospy.loginfo("Published %s for goal_id=%s", self.arm_done_topic, goal_id)


if __name__ == "__main__":
    try:
        MoveItDoneBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
