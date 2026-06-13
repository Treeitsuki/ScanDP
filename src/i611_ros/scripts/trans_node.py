#!/usr/bin/env python3
import math

import actionlib
import rospy
import tf2_geometry_msgs
import tf2_ros
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import (PoseArray, PoseStamped, Quaternion,
                               TransformStamped, Vector3)
from moveit_msgs.msg import (Constraints, MoveGroupAction, MoveGroupGoal,
                             MoveItErrorCodes, OrientationConstraint,
                             PositionConstraint)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, Float32, String
from tf.transformations import (concatenate_matrices, quaternion_from_matrix,
                                quaternion_matrix, rotation_matrix,
                                translation_from_matrix, translation_matrix)


class TestNode:
    def __init__(self):
        rospy.init_node("trans_node")

        # TF Setup
        self.dynamic_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Parameters
        self.rate = rospy.Rate(30)
        self.policy_pose_topic = rospy.get_param(
            "~policy_pose_topic", "/policy/poses"
        )
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_link_frame = rospy.get_param("~base_link_frame", "base_link")
        self.moveit_goal_frame = rospy.get_param(
            "~moveit_goal_frame", self.base_link_frame
        )
        self.move_group_name = rospy.get_param("~move_group", "i611")
        self.move_group_action = rospy.get_param(
            "~move_group_action", "/move_group")
        self.ee_link = rospy.get_param("~end_effector_link", "Link6")

        self.goal_pos_tol = rospy.get_param("~goal_position_tolerance", 0.002)
        self.goal_ori_tol = rospy.get_param(
            "~goal_orientation_tolerance", 0.01)
        self.planning_time = rospy.get_param("~planning_time", 5.0)
        self.action_wait_timeout = rospy.get_param(
            "~action_wait_timeout", 180.0)

        self.execute_trajectory = rospy.get_param("~execute_trajectory", True)
        self.max_velocity_scaling = rospy.get_param(
            "~max_velocity_scaling", 1.0)
        self.max_acceleration_scaling = rospy.get_param(
            "~max_acceleration_scaling", 1.0)

        self.go_wait = rospy.get_param("~go_wait", True)
        self.goal_send_interval = rospy.get_param("~goal_send_interval", 1.0)
        self.tf_timeout = rospy.get_param("~tf_timeout", 0.5)

        self.command_angle_topic = rospy.get_param(
            "~command_angle_topic", "/ds102/command_angle"
        )
        self.current_angle_topic = rospy.get_param(
            "~current_angle_topic", "/ds102/current_angle"
        )
        self.move_done_topic = rospy.get_param(
            "~move_done_topic", "/ds102/move_done"
        )
        self.is_moving_topic = rospy.get_param(
            "~is_moving_topic", "/ds102/is_moving"
        )
        self.ds102_move_timeout = rospy.get_param(
            "~ds102_move_timeout", 60
        )
        self.ds102_move_eps = rospy.get_param("~ds102_move_eps", 0.1)
        self.turntable_settle_time = rospy.get_param(
            "~turntable_settle_time", 0.3
        )
        self.turntable_angle_tolerance = rospy.get_param(
            "~turntable_angle_tolerance", self.ds102_move_eps
        )
        self.ds102_angle_sign = rospy.get_param("~ds102_angle_sign", 1.0)
        self.ds102_angle_sign = self._coerce_float_param(
            self.ds102_angle_sign, "~ds102_angle_sign", 1.0
        )
        self.require_turntable_done = rospy.get_param(
            "~require_turntable_done", True
        )
        self.require_moveit_done = rospy.get_param(
            "~require_moveit_done", True
        )
        self.moveit_goal_accept_timeout = rospy.get_param(
            "~moveit_goal_accept_timeout", self.planning_time + 1.0
        )
        self.enforce_arm_reach = rospy.get_param(
            "~enforce_arm_reach", True
        )
        self.abort_on_moveit_failure = rospy.get_param(
            "~abort_on_moveit_failure", True
        )
        self.moveit_failure_topic = rospy.get_param(
            "~moveit_failure_topic", "/moveit/failure"
        )
        self.arm_done_topic = rospy.get_param(
            "~arm_done_topic", "/arm/move_done"
        )

        # Subscriber for input pose array
        self.current_policy_array = None
        self.last_policy_seq = None
        self.last_policy_stamp = None
        self.policy_sub = rospy.Subscriber(
            self.policy_pose_topic, PoseArray, self._policy_callback
        )

        self.command_angle_pub = rospy.Publisher(
            self.command_angle_topic, Float32, queue_size=1
        )
        self.moveit_failure_pub = rospy.Publisher(
            self.moveit_failure_topic, String, queue_size=10
        )
        self.arm_done_pub = rospy.Publisher(
            self.arm_done_topic, Bool, queue_size=10
        )

        self.current_angle_deg = None
        self.is_moving = None
        self.move_done_count = 0
        self.last_move_done_angle = None
        self.last_move_done_time = None
        rospy.Subscriber(self.current_angle_topic,
                         Float32, self._current_angle_cb)
        rospy.Subscriber(self.is_moving_topic, Bool, self._is_moving_cb)
        rospy.Subscriber(self.move_done_topic, Float32, self._move_done_cb)

        # Action Client
        self.move_group_client = actionlib.SimpleActionClient(
            self.move_group_action, MoveGroupAction
        )
        if not self.move_group_client.wait_for_server(rospy.Duration(self.action_wait_timeout)):
            rospy.logwarn(
                f"MoveGroup action server not available: {self.move_group_action}")
            self.move_group_client = None

    def _policy_callback(self, msg):
        """Callback to update the current input poses from /policy/poses (PoseArray)"""
        self.current_policy_array = msg

    def _current_angle_cb(self, msg):
        self.current_angle_deg = float(msg.data)

    def _is_moving_cb(self, msg):
        self.is_moving = bool(msg.data)

    def _move_done_cb(self, msg):
        self.move_done_count += 1
        self.last_move_done_angle = float(msg.data)
        self.last_move_done_time = rospy.Time.now()

    @staticmethod
    def _normalize_quat(q):
        if q.x == 0 and q.y == 0 and q.z == 0 and q.w == 0:
            return (0.0, 0.0, 0.0, 1.0)
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm < 1e-6:
            return (0.0, 0.0, 0.0, 1.0)
        inv = 1.0 / norm
        return (q.x * inv, q.y * inv, q.z * inv, q.w * inv)

    def _coerce_float_param(self, value, name, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            rospy.logwarn("%s must be float", name)
            return float(default)

    @staticmethod
    def _shortest_angle_diff_deg(target, current):
        """Return shortest signed angle difference (deg) in range [-180, 180]."""
        if target is None or current is None:
            return 0.0
        return (target - current + 180.0) % 360.0 - 180.0

    @staticmethod
    def _rotation_to_xz_negative_x(position):
        """位置(x, y)に基づいてZ軸回転行列と角度を計算する"""
        x, y, _ = position
        if x == 0.0 and y == 0.0:
            return rotation_matrix(0.0, (0.0, 0.0, 1.0)), 0.0

        raw_angle = -math.atan2(y, x) + math.pi
        normalized_angle = math.atan2(math.sin(raw_angle), math.cos(raw_angle))

        return rotation_matrix(normalized_angle, (0.0, 0.0, 1.0)), normalized_angle

    def _turntable_state(self, start_count, target_angle, last_change_time, now):
        done_seen = self.move_done_count > start_count
        stopped = done_seen or (self.is_moving is False)

        settle_duration = rospy.Duration(max(0.0, self.turntable_settle_time))
        stable = True
        if settle_duration.to_sec() > 0.0:
            stable = (now - last_change_time) >= settle_duration

        angle_ok = True
        angle_diff = None
        if target_angle is not None and self.current_angle_deg is not None:
            angle_diff = abs(self._shortest_angle_diff_deg(
                target_angle, self.current_angle_deg
            ))
            angle_ok = angle_diff <= self.turntable_angle_tolerance

        return done_seen, stopped, stable, angle_ok, angle_diff

    def run(self):
        # Initial Transform setup
        t_inf = TransformStamped()
        t_inf.child_frame_id = "inf"

        t_ee = TransformStamped()
        t_ee.child_frame_id = "ee"

        rospy.loginfo(
            "Waiting for PoseArray message on %s ...",
            self.policy_pose_topic,
        )

        while not rospy.is_shutdown():
            # Wait until valid data is received
            if self.current_policy_array is None:
                self.rate.sleep()
                continue

            # Check if the array is not empty
            if not self.current_policy_array.poses:
                rospy.logwarn_throttle(
                    5.0, "Received empty PoseArray on /policy/poses")
                self.rate.sleep()
                continue

            policy_msg = self.current_policy_array
            seq = getattr(policy_msg.header, "seq", None)
            stamp = policy_msg.header.stamp
            if seq == self.last_policy_seq and stamp == self.last_policy_stamp:
                self.rate.sleep()
                continue

            self.last_policy_seq = seq
            self.last_policy_stamp = stamp

            policy_frame = policy_msg.header.frame_id or self.map_frame
            poses = list(policy_msg.poses)

            rospy.loginfo("Processing %d poses from %s",
                          len(poses), self.policy_pose_topic)

            for idx, target_pose in enumerate(poses):
                if rospy.is_shutdown():
                    break

                now = rospy.Time.now()

                # 1. Update t_inf from Subscriber Data
                inf_pos = (
                    target_pose.position.x,
                    target_pose.position.y,
                    target_pose.position.z
                )

                # クォータニオンの不正チェック (All zeros 対策)
                q = target_pose.orientation
                inf_quat = self._normalize_quat(q)

                # Apply to t_inf message
                t_inf.header.stamp = now
                t_inf.header.frame_id = policy_frame
                t_inf.transform.translation = Vector3(*inf_pos)
                t_inf.transform.rotation = Quaternion(*inf_quat)

                # 2. Calculate Matrix for t_inf
                t_inf_mat = concatenate_matrices(
                    translation_matrix(inf_pos),
                    quaternion_matrix(inf_quat),
                )

                # 3. Calculate Logic for t_ee (Dependent on t_inf position)
                rot_mat, angle = self._rotation_to_xz_negative_x(inf_pos)

                rospy.loginfo(
                    "Pose %d angle: %.2f deg (%.3f rad)",
                    idx,
                    math.degrees(angle),
                    angle,
                )

                t_ee_mat = concatenate_matrices(rot_mat, t_inf_mat)

                # Apply to t_ee message
                t_ee.header.stamp = now
                t_ee.header.frame_id = policy_frame
                t_ee_trans = translation_from_matrix(t_ee_mat)
                t_ee_rot = quaternion_from_matrix(t_ee_mat)

                t_ee.transform.translation = Vector3(*t_ee_trans)
                t_ee.transform.rotation = Quaternion(*t_ee_rot)

                # Broadcast TF
                self.dynamic_broadcaster.sendTransform([t_inf, t_ee])

                # 4. Send MoveIt Goal (turntable -> arm, with completion checks)
                angle_deg = -math.degrees(angle)

                ee_pose = PoseStamped()
                ee_pose.header.stamp = now
                ee_pose.header.frame_id = policy_frame
                ee_pose.pose.position = t_ee.transform.translation
                ee_pose.pose.orientation = t_ee.transform.rotation

                ee_pose_goal = self._transform_pose_to_goal_frame(ee_pose)
                if ee_pose_goal is None:
                    rospy.logwarn(
                        "Pose %d: TF transform to %s failed",
                        idx,
                        self.moveit_goal_frame,
                    )
                    continue

                start_info = self._start_turntable(angle_deg)
                if start_info == "busy":
                    rospy.logwarn(
                        "Pose %d: Turntable busy/timeout; skip pose",
                        idx,
                    )
                    continue

                start_count, target_angle = start_info
                turntable_ok, turntable_reason = self._wait_for_turntable_done(
                    start_count, target_angle
                )
                if not turntable_ok:
                    rospy.logwarn(
                        "Pose %d: Turntable did not stop in time (%s)",
                        idx,
                        turntable_reason,
                    )
                    if self.require_turntable_done:
                        continue

                wait_for_arm = self.enforce_arm_reach or self.require_moveit_done
                moveit_ok = self._send_move_group_goal(
                    ee_pose_goal, wait=wait_for_arm
                )
                if wait_for_arm and not moveit_ok:
                    rospy.logerr(
                        "Pose %d: MoveIt goal failed; sequence paused",
                        idx,
                    )
                    self._publish_moveit_failure("moveit_goal_failed", idx)
                    if self.move_group_client:
                        self.move_group_client.cancel_goal()
                    if self.abort_on_moveit_failure:
                        break

            self.rate.sleep()

    def _transform_pose_to_goal_frame(self, pose):
        if pose.header.frame_id == self.moveit_goal_frame:
            return pose
        try:
            transform = self.tf_buffer.lookup_transform(
                self.moveit_goal_frame,
                pose.header.frame_id,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ):
            return None
        return tf2_geometry_msgs.do_transform_pose(pose, transform)

    def _start_turntable(self, angle_deg):
        target_angle = angle_deg * self.ds102_angle_sign

        if self.current_angle_deg is not None:
            diff = abs(self._shortest_angle_diff_deg(
                target_angle, self.current_angle_deg))
            if diff <= self.ds102_move_eps:
                return (None, target_angle)

        if self.is_moving:
            ok, _ = self._wait_for_turntable_done(self.move_done_count, None)
            if not ok:
                return "busy"

        start_count = self.move_done_count
        self.command_angle_pub.publish(Float32(target_angle))
        return (start_count, target_angle)

    def _wait_for_turntable_done(self, start_count, target_angle):
        if start_count is None:
            return True, "already_within_eps"
        if start_count == "busy":
            return False, "busy"

        if self.ds102_move_timeout <= 0:
            return True, "timeout_disabled"

        timeout = rospy.Duration(self.ds102_move_timeout)
        end_time = rospy.Time.now() + timeout
        last_angle = self.current_angle_deg
        last_change_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self.current_angle_deg is not None:
                diff = abs(self._shortest_angle_diff_deg(
                    self.current_angle_deg, last_angle))
                if last_angle is None or diff > self.turntable_angle_tolerance:
                    last_angle = self.current_angle_deg
                    last_change_time = now

            done_seen, stopped, stable, angle_ok, _ = self._turntable_state(
                start_count, target_angle, last_change_time, now
            )

            if stopped and stable and angle_ok:
                return True, "done"
            if now >= end_time:
                reason = self._format_turntable_timeout_reason(
                    start_count, target_angle, last_change_time
                )
                return False, reason
            self.rate.sleep()
        return False, "shutdown"

    def _format_turntable_timeout_reason(
        self, start_count, target_angle, last_change_time
    ):
        now = rospy.Time.now()
        done_seen, stopped, stable, angle_ok, angle_diff = self._turntable_state(
            start_count, target_angle, last_change_time, now
        )

        parts = []
        if not done_seen:
            parts.append("move_done_not_received")
        if self.is_moving is None:
            parts.append("is_moving_unknown")
        elif self.is_moving is True:
            parts.append("is_moving_true")
        if not stable:
            parts.append("not_settled")
        if target_angle is not None and self.current_angle_deg is None:
            parts.append("current_angle_unknown")
        if angle_ok is False:
            parts.append(
                "angle_out_of_tolerance"
                + (f"(diff={angle_diff:.3f})" if angle_diff is not None else "")
            )
        if not stopped:
            parts.append("not_stopped")
        if not parts:
            parts.append("unknown")
        return ",".join(parts)

    def _send_move_group_goal(self, ee_pose, wait=None):
        if not self.move_group_client:
            self._publish_moveit_failure("move_group_client_unavailable")
            return False

        goal_pose = PoseStamped()
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.header.frame_id = self.moveit_goal_frame
        goal_pose.pose = ee_pose.pose

        goal = MoveGroupGoal()
        goal.request.group_name = self.move_group_name
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = self.planning_time
        goal.request.max_velocity_scaling_factor = self.max_velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.max_acceleration_scaling
        goal.request.start_state.is_diff = True

        goal.request.goal_constraints.append(
            self._create_constraints(goal_pose))

        goal.planning_options.plan_only = not self.execute_trajectory
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        self.move_group_client.send_goal(goal)

        if not self._wait_for_moveit_goal_acceptance():
            self.move_group_client.cancel_goal()
            self._publish_moveit_failure("goal_not_accepted")
            return False

        if wait is None:
            wait = self.go_wait

        if not wait:
            return True

        return self._wait_for_moveit_result(True)

    def _wait_for_moveit_goal_acceptance(self):
        if self.moveit_goal_accept_timeout <= 0:
            return True

        timeout = rospy.Duration(self.moveit_goal_accept_timeout)
        end_time = rospy.Time.now() + timeout
        while not rospy.is_shutdown():
            state = self.move_group_client.get_state()
            if state in (
                GoalStatus.REJECTED,
                GoalStatus.ABORTED,
                GoalStatus.PREEMPTED,
                GoalStatus.RECALLED,
                GoalStatus.LOST,
            ):
                self._publish_moveit_failure(
                    "goal_rejected_or_lost",
                    state=state,
                )
                return False
            if state in (GoalStatus.ACTIVE, GoalStatus.SUCCEEDED):
                return True
            if rospy.Time.now() >= end_time:
                self._publish_moveit_failure(
                    "goal_accept_timeout",
                    state=state,
                )
                return False
            self.rate.sleep()
        return False

    def _wait_for_moveit_result(self, goal_sent):
        if not goal_sent or not self.move_group_client:
            return False

        finished = self.move_group_client.wait_for_result(
            rospy.Duration(self.action_wait_timeout)
        )
        if not finished:
            self.move_group_client.cancel_goal()
            self._publish_moveit_failure("result_timeout")
            return False

        result = self.move_group_client.get_result()
        if not result:
            self._publish_moveit_failure("empty_result")
            return False
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._publish_moveit_failure(
                "moveit_error_code",
                error_code=result.error_code.val,
            )
            return False
        self.arm_done_pub.publish(Bool(True))
        return True

    def _publish_moveit_failure(self, reason, pose_idx=None, error_code=None, state=None):
        payload = {
            "reason": reason,
            "pose_idx": pose_idx,
            "error_code": error_code,
            "state": state,
            "time": rospy.Time.now().to_sec(),
        }
        msg = String()
        msg.data = str(payload)
        self.moveit_failure_pub.publish(msg)

    def _create_constraints(self, goal_pose):
        constraints = Constraints()

        # Position Constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header = goal_pose.header
        pos_constraint.link_name = self.ee_link
        pos_constraint.target_point_offset = Vector3(0, 0, 0)

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [self.goal_pos_tol]

        pos_constraint.constraint_region.primitives.append(primitive)
        pos_constraint.constraint_region.primitive_poses.append(goal_pose.pose)
        pos_constraint.weight = 1.0

        # Orientation Constraint
        ori_constraint = OrientationConstraint()
        ori_constraint.header = goal_pose.header
        ori_constraint.link_name = self.ee_link
        ori_constraint.orientation = goal_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = self.goal_ori_tol
        ori_constraint.absolute_y_axis_tolerance = self.goal_ori_tol
        ori_constraint.absolute_z_axis_tolerance = self.goal_ori_tol
        ori_constraint.weight = 1.0

        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)
        return constraints


if __name__ == "__main__":
    try:
        node = TestNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
