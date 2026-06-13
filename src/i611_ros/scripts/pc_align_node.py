#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import colorsys
import datetime
import os
import struct
import threading

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, Float32, Header

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except Exception:
    _HAS_OPEN3D = False

try:
    from scipy.spatial import cKDTree
    _HAS_CKDTREE = True
except Exception:
    _HAS_CKDTREE = False


class PointCloudAlignNode:
    def __init__(self):
        rospy.init_node("ds102_pointcloud_align", anonymous=False)
        self.enable_logging = rospy.get_param("~enable_logging", False)
        if not self.enable_logging:
            self._silence_logging()

        # Topics
        self.input_cloud_topic = rospy.get_param(
            "~input_cloud_topic", "/points_in"
        )
        self.output_cloud_topic = rospy.get_param(
            "~output_cloud_topic", "/points_aligned"
        )
        self.ds102_done_topic = rospy.get_param(
            "~ds102_done_topic", "/ds102/move_done"
        )
        self.arm_done_topic = rospy.get_param(
            "~arm_done_topic", "/arm/move_done"
        )

        # Trigger timing
        self.trigger_sync_window = rospy.get_param(
            "~trigger_sync_window", 60
        )
        self.capture_interval = rospy.get_param(
            "~capture_interval", 0.0
        )
        self.cloud_max_age = rospy.get_param(
            "~cloud_max_age", 0.2
        )
        self.cloud_wait_timeout = rospy.get_param(
            "~cloud_wait_timeout", 1.0
        )
        self.max_captures = rospy.get_param(
            "~max_captures", 0
        )
        self.colorize_captures = rospy.get_param(
            "~colorize_captures",
            rospy.get_param("~debug_color", True),
        )
        self.status_period = rospy.get_param(
            "~status_period", 2.0
        )
        self.enable_status_log = rospy.get_param(
            "~enable_status_log", False
        )

        # Reference cloud
        self.reference_mode = rospy.get_param(
            "~reference_mode", "first"
        )
        self.reference_cloud_topic = rospy.get_param(
            "~reference_cloud_topic", ""
        )
        self.reference_cloud_pub_topic = rospy.get_param(
            "~reference_cloud_pub_topic", "/reference_cloud"
        )
        self.reference_pcd_path = rospy.get_param(
            "~reference_pcd_path", "/home/user/workspace/test/millo_sampled.pcd"
        )
        self.reference_center = self._parse_center_param(
            rospy.get_param("~reference_center", [0.0, 0.0, 0.4])
        )
        self.reference_frame_id = rospy.get_param(
            "~reference_frame_id", "map"
        )
        self.reference_wait_timeout = rospy.get_param(
            "~reference_wait_timeout", 2.0
        )
        self.startup_capture = rospy.get_param(
            "~startup_capture", True
        )

        # ICP parameters
        self.icp_max_iter = rospy.get_param(
            "~icp_max_iter", 30
        )
        self.icp_tolerance = rospy.get_param(
            "~icp_tolerance", 1e-4
        )
        self.icp_max_corr_dist = rospy.get_param(
            "~icp_max_corr_dist", 0.02
        )
        self.downsample_voxel = rospy.get_param(
            "~downsample_voxel", 0.01
        )
        self.min_points = rospy.get_param(
            "~min_points", 100
        )
        self.use_open3d = rospy.get_param(
            "~use_open3d", True
        )
        self.use_ransac = rospy.get_param(
            "~use_ransac", True
        )
        self.ransac_voxel = rospy.get_param(
            "~ransac_voxel", 0.01
        )
        self.ransac_distance = rospy.get_param(
            "~ransac_distance", 0.03
        )
        self.ransac_n = rospy.get_param(
            "~ransac_n", 4
        )
        self.ransac_edge_length = rospy.get_param(
            "~ransac_edge_length", 0.9
        )
        self.ransac_max_iter = rospy.get_param(
            "~ransac_max_iter", 50000
        )
        self.ransac_confidence = rospy.get_param(
            "~ransac_confidence", 0.999
        )
        self.ransac_mutual_filter = rospy.get_param(
            "~ransac_mutual_filter", True
        )
        self.accumulate = rospy.get_param(
            "~accumulate", True
        )
        self.accumulate_voxel = rospy.get_param(
            "~accumulate_voxel", 0.0
        )
        self.pastel_saturation = rospy.get_param(
            "~pastel_saturation", 0.35
        )
        self.pastel_value = rospy.get_param(
            "~pastel_value", 0.95
        )
        self.pastel_hue_step = rospy.get_param(
            "~pastel_hue_step", 0.61803398875
        )

        # Coverage parameters
        self.enable_coverage = rospy.get_param(
            "~enable_coverage", True
        )
        self.coverage_distance = rospy.get_param(
            "~coverage_distance", 0.002
        )
        self.coverage_use_downsampled_ref = rospy.get_param(
            "~coverage_use_downsampled_ref", False
        )
        self.coverage_use_accumulated = rospy.get_param(
            "~coverage_use_accumulated", True
        )
        self.coverage_log_dir = rospy.get_param(
            "~coverage_log_dir", "/home/user/workspace/data/csv"
        )
        self.coverage_log_flush = rospy.get_param(
            "~coverage_log_flush", True
        )
        self.path_sum_topic = rospy.get_param(
            "~path_sum_topic", "/policy/path_sum"
        )

        # Save aligned point clouds
        self.save_aligned_enable = rospy.get_param(
            "~save_aligned_enable", True
        )
        self.save_aligned_dir = rospy.get_param(
            "~save_aligned_dir", "/home/user/workspace/save_dir"
        )
        self.save_aligned_prefix = rospy.get_param(
            "~save_aligned_prefix", "points_aligned"
        )
        self.save_aligned_min = np.array(
            rospy.get_param("~save_aligned_min", [-0.2, -0.2, 0.0]),
            dtype=np.float32,
        )
        self.save_aligned_max = np.array(
            rospy.get_param("~save_aligned_max", [0.2, 0.2, 0.75]),
            dtype=np.float32,
        )

        # Internal state
        self._lock = threading.Lock()
        self._last_cloud_msg = None
        self._last_cloud_time = None
        self._last_ds102_time = None
        self._last_arm_time = None
        self._last_trigger_time = None
        self._pending_trigger_time = None
        self._processing = False
        self._capture_count = 0
        self._publish_count = 0
        self._cloud_count = 0
        self._ds102_count = 0
        self._arm_count = 0
        self._last_status_time = rospy.Time(0)
        self._log_times = {}

        self._reference_cloud = None
        self._reference_cloud_full = None
        self._reference_frame = None
        self._reference_stamp = None
        self._reference_center_point = self.reference_center
        self._accumulated_cloud = None
        self._accumulated_colors = None
        self._startup_pending = bool(self.startup_capture)
        self._reference_pub = None
        self._coverage_history = []
        self._coverage_log_error = False
        self._coverage_log_file = None
        self._path_sum_value = None
        self._save_seq = 0
        self._save_dir_checked = False

        # ROS I/O
        self._cloud_sub = rospy.Subscriber(
            self.input_cloud_topic, PointCloud2, self._cloud_cb, queue_size=1
        )
        self._ds102_sub = rospy.Subscriber(
            self.ds102_done_topic, Float32, self._ds102_done_cb, queue_size=1
        )
        self._arm_sub = rospy.Subscriber(
            self.arm_done_topic, Bool, self._arm_done_cb, queue_size=1
        )
        self._path_sum_sub = None
        if self.path_sum_topic:
            self._path_sum_sub = rospy.Subscriber(
                self.path_sum_topic, Float32, self._path_sum_cb, queue_size=1
            )
        self._aligned_pub = rospy.Publisher(
            self.output_cloud_topic, PointCloud2, queue_size=1
        )
        self._reference_pub = rospy.Publisher(
            self.reference_cloud_pub_topic, PointCloud2, queue_size=1, latch=True
        )

        self._timer = rospy.Timer(rospy.Duration(0.05), self._timer_cb)

        rospy.loginfo("PointCloudAlignNode ready")
        rospy.loginfo("input_cloud_topic: %s", self.input_cloud_topic)
        rospy.loginfo("output_cloud_topic: %s", self.output_cloud_topic)
        rospy.loginfo("ds102_done_topic: %s", self.ds102_done_topic)
        rospy.loginfo("arm_done_topic: %s", self.arm_done_topic)
        rospy.loginfo("reference_mode: %s", self.reference_mode)
        if self.reference_mode == "external":
            rospy.loginfo("reference_cloud_topic: %s",
                          self.reference_cloud_topic)
        if self.reference_mode == "pcd":
            rospy.loginfo("reference_pcd_path: %s",
                          self.reference_pcd_path)
            if self.reference_frame_id:
                rospy.loginfo("reference_frame_id: %s",
                              self.reference_frame_id)
            if self.reference_center is not None:
                rospy.loginfo(
                    "reference_center: [%.6f, %.6f, %.6f]",
                    float(self.reference_center[0]),
                    float(self.reference_center[1]),
                    float(self.reference_center[2]),
                )
        rospy.loginfo("startup_capture: %s", str(self.startup_capture))
        rospy.loginfo("accumulate: %s", str(self.accumulate))
        rospy.loginfo("accumulate_voxel: %s", str(self.accumulate_voxel))
        rospy.loginfo("status_period: %.3f sec", self.status_period)
        rospy.loginfo("reference_cloud_pub_topic: %s",
                      self.reference_cloud_pub_topic)
        rospy.loginfo("use_ransac: %s", str(self.use_ransac))
        rospy.loginfo("colorize_captures: %s", str(self.colorize_captures))
        rospy.loginfo("enable_coverage: %s", str(self.enable_coverage))
        if self.enable_coverage:
            rospy.loginfo("coverage_distance: %.6f",
                          float(self.coverage_distance))
            rospy.loginfo(
                "coverage_use_downsampled_ref: %s",
                str(self.coverage_use_downsampled_ref),
            )
            rospy.loginfo(
                "coverage_use_accumulated: %s",
                str(self.coverage_use_accumulated),
            )
            rospy.loginfo("coverage_log_dir: %s", self.coverage_log_dir)
            if not _HAS_CKDTREE:
                rospy.logwarn(
                    "scipy not available, coverage computation disabled")
                self.enable_coverage = False
        if self.path_sum_topic:
            rospy.loginfo("path_sum_topic: %s", self.path_sum_topic)
        if self.save_aligned_enable:
            rospy.loginfo("save_aligned_dir: %s", self.save_aligned_dir)
            rospy.loginfo("save_aligned_prefix: %s", self.save_aligned_prefix)
            rospy.loginfo(
                "save_aligned_min: [%.3f, %.3f, %.3f]",
                float(self.save_aligned_min[0]),
                float(self.save_aligned_min[1]),
                float(self.save_aligned_min[2]),
            )
            rospy.loginfo(
                "save_aligned_max: [%.3f, %.3f, %.3f]",
                float(self.save_aligned_max[0]),
                float(self.save_aligned_max[1]),
                float(self.save_aligned_max[2]),
            )

        if self.reference_mode == "pcd":
            self._load_pcd_reference()

    def _log_info(self, msg, *args):
        rospy.loginfo(msg, *args)

    def _log_info_throttle(self, key, period_sec, msg, *args):
        now = rospy.Time.now().to_sec()
        last = self._log_times.get(key, 0.0)
        if (now - last) >= period_sec:
            self._log_times[key] = now
            rospy.loginfo(msg, *args)

    def _silence_logging(self):
        def _noop(*_args, **_kwargs):
            return None

        rospy.logdebug = _noop
        rospy.loginfo = _noop
        rospy.logwarn = _noop
        rospy.logerr = _noop
        rospy.logfatal = _noop
        rospy.logdebug_throttle = _noop
        rospy.loginfo_throttle = _noop
        rospy.logwarn_throttle = _noop
        rospy.logerr_throttle = _noop
        rospy.logfatal_throttle = _noop

    def _maybe_log_status(self):
        if not self.enable_status_log:
            return
        if self.status_period <= 0:
            return
        now = rospy.Time.now()
        if (now - self._last_status_time).to_sec() < self.status_period:
            return
        self._last_status_time = now
        cloud_age = None
        if self._last_cloud_time is not None:
            cloud_age = (now - self._last_cloud_time).to_sec()
        ds102_age = None
        if self._last_ds102_time is not None:
            ds102_age = (now - self._last_ds102_time).to_sec()
        arm_age = None
        if self._last_arm_time is not None:
            arm_age = (now - self._last_arm_time).to_sec()
        pending = None
        if self._pending_trigger_time is not None:
            pending = self._pending_trigger_time.to_sec()
        last_trigger = None
        if self._last_trigger_time is not None:
            last_trigger = self._last_trigger_time.to_sec()
        self._log_info(
            "status: clouds=%d ds102=%d arm=%d captures=%d publishes=%d processing=%s",
            self._cloud_count,
            self._ds102_count,
            self._arm_count,
            self._capture_count,
            self._publish_count,
            str(self._processing),
        )
        self._log_info(
            "status: cloud_age=%s ds102_age=%s arm_age=%s pending_trigger=%s last_trigger=%s",
            f"{cloud_age:.3f}" if cloud_age is not None else "None",
            f"{ds102_age:.3f}" if ds102_age is not None else "None",
            f"{arm_age:.3f}" if arm_age is not None else "None",
            f"{pending:.3f}" if pending is not None else "None",
            f"{last_trigger:.3f}" if last_trigger is not None else "None",
        )

    def _cloud_cb(self, msg):
        with self._lock:
            self._last_cloud_msg = msg
            self._last_cloud_time = rospy.Time.now()
            self._cloud_count += 1

    def _ds102_done_cb(self, _msg):
        with self._lock:
            self._last_ds102_time = rospy.Time.now()
            self._ds102_count += 1
            self._update_trigger_locked()

    def _arm_done_cb(self, msg):
        if not bool(msg.data):
            return
        with self._lock:
            self._last_arm_time = rospy.Time.now()
            self._arm_count += 1
            self._update_trigger_locked()

    def _path_sum_cb(self, msg):
        self._path_sum_value = float(msg.data)

    def _update_trigger_locked(self):
        if self._pending_trigger_time is not None:
            self._log_info_throttle(
                "pending_trigger", 1.0, "trigger skipped: pending trigger exists"
            )
            return
        if self._last_ds102_time is None or self._last_arm_time is None:
            self._log_info_throttle(
                "missing_done", 1.0, "trigger skipped: waiting for both done signals"
            )
            return
        dt = abs((self._last_ds102_time - self._last_arm_time).to_sec())
        if dt > self.trigger_sync_window:
            self._log_info_throttle(
                "sync_window",
                1.0,
                "trigger skipped: sync window exceeded (dt=%.3f > %.3f)",
                dt,
                self.trigger_sync_window,
            )
            return
        if self._last_trigger_time is not None:
            if self._last_ds102_time <= self._last_trigger_time:
                self._log_info_throttle(
                    "ds102_old", 1.0, "trigger skipped: ds102 timestamp not newer"
                )
                return
            if self._last_arm_time <= self._last_trigger_time:
                self._log_info_throttle(
                    "arm_old", 1.0, "trigger skipped: arm timestamp not newer"
                )
                return
            if self.capture_interval > 0:
                if (rospy.Time.now() - self._last_trigger_time).to_sec() < self.capture_interval:
                    self._log_info_throttle(
                        "capture_interval",
                        1.0,
                        "trigger skipped: capture interval not elapsed (%.3f < %.3f)",
                        (rospy.Time.now() - self._last_trigger_time).to_sec(),
                        self.capture_interval,
                    )
                    return
        self._pending_trigger_time = max(
            self._last_ds102_time, self._last_arm_time)
        self._log_info("trigger queued at %.3f",
                       self._pending_trigger_time.to_sec())

    def _timer_cb(self, _event):
        if self._processing:
            return
        if self._startup_pending:
            self._startup_pending = False
            self._processing = True
            try:
                ok = self._handle_trigger(None, reason="startup")
                if not ok:
                    self._startup_pending = True
            finally:
                self._processing = False
                self._maybe_log_status()
            return
        if self._pending_trigger_time is None:
            self._maybe_log_status()
            return
        with self._lock:
            trigger_time = self._pending_trigger_time
            self._pending_trigger_time = None
            self._last_trigger_time = trigger_time
        if self.max_captures > 0 and self._capture_count >= self.max_captures:
            self._log_info_throttle(
                "max_captures",
                1.0,
                "trigger ignored: max_captures reached (%d)",
                self.max_captures,
            )
            return
        self._processing = True
        try:
            self._handle_trigger(trigger_time, reason="trigger")
        finally:
            self._processing = False
            self._maybe_log_status()

    def _handle_trigger(self, trigger_time, reason="trigger"):
        cloud_msg = self._get_cloud_for_trigger()
        if cloud_msg is None:
            if reason == "startup":
                rospy.logwarn_throttle(
                    2.0, "Startup capture waiting for point cloud"
                )
            else:
                rospy.logerr("Failed to get point cloud (%s)", reason)
            return False

        points_full = self._cloud_to_xyz(cloud_msg)
        if points_full.shape[0] < self.min_points:
            rospy.logwarn("Point cloud too small: %d points",
                          points_full.shape[0])
            return False

        capture_index = self._capture_count
        capture_colors = self._make_capture_colors(
            points_full.shape[0], capture_index
        )

        points_down = self._voxel_downsample(
            points_full, self.downsample_voxel)

        if self.reference_mode == "external" and self._reference_cloud is None:
            self._load_external_reference()
        elif self.reference_mode == "pcd" and self._reference_cloud is None:
            self._load_pcd_reference()

        if self._reference_cloud is None:
            if self.reference_mode == "first":
                self._reference_cloud = points_down
                self._reference_cloud_full = points_full
                self._reference_frame = cloud_msg.header.frame_id
                self._reference_stamp = cloud_msg.header.stamp
                rospy.loginfo("Reference cloud set (%d points)",
                              points_down.shape[0])
                self._publish_reference(
                    points_down, frame_id=self._reference_frame)
                if self.accumulate:
                    if capture_colors is not None:
                        self._accumulated_cloud, self._accumulated_colors = self._accumulate_colored(
                            self._accumulated_cloud,
                            self._accumulated_colors,
                            points_full,
                            capture_colors,
                        )
                        self._publish_aligned(
                            self._accumulated_cloud, cloud_msg.header, self._accumulated_colors
                        )
                    else:
                        self._accumulated_cloud = self._accumulate_points(
                            self._accumulated_cloud, points_full
                        )
                        self._accumulated_colors = None
                        self._publish_aligned(
                            self._accumulated_cloud, cloud_msg.header
                        )
                else:
                    self._publish_aligned(
                        points_full, cloud_msg.header, capture_colors)
                self._maybe_compute_and_record_coverage(
                    points_full,
                    trigger_time,
                    capture_index,
                    source_label="reference",
                )
                self._capture_count += 1
                return True
            rospy.logwarn_throttle(
                2.0, "Reference cloud not ready for mode '%s'", self.reference_mode
            )
            return False

        if self._reference_frame and self._reference_frame != cloud_msg.header.frame_id:
            rospy.logwarn(
                "Frame mismatch: reference=%s input=%s",
                self._reference_frame,
                cloud_msg.header.frame_id,
            )

        init_transform = self._compute_init_transform(points_down)
        result = self._run_icp(
            points_down, self._reference_cloud, init_transform)
        if result is None:
            rospy.logerr("ICP failed")
            return False

        transform, mean_error, fitness = result
        aligned_full = self._apply_transform(points_full, transform)

        rospy.loginfo("ICP done: mean_error=%.6f fitness=%.3f",
                      mean_error, fitness)
        rospy.loginfo("Transform:\n%s", self._format_matrix(transform))

        if self.accumulate:
            if capture_colors is not None:
                self._accumulated_cloud, self._accumulated_colors = self._accumulate_colored(
                    self._accumulated_cloud,
                    self._accumulated_colors,
                    aligned_full,
                    capture_colors,
                )
                self._publish_aligned(
                    self._accumulated_cloud, cloud_msg.header, self._accumulated_colors
                )
                coverage_points = self._accumulated_cloud
            else:
                self._accumulated_cloud = self._accumulate_points(
                    self._accumulated_cloud, aligned_full
                )
                self._accumulated_colors = None
                self._publish_aligned(
                    self._accumulated_cloud, cloud_msg.header
                )
                coverage_points = self._accumulated_cloud
        else:
            self._publish_aligned(
                aligned_full, cloud_msg.header, capture_colors)
            coverage_points = aligned_full
        if self.accumulate and not self.coverage_use_accumulated:
            coverage_points = aligned_full
        self._maybe_compute_and_record_coverage(
            coverage_points,
            trigger_time,
            capture_index,
            source_label="accumulated" if self.accumulate and self.coverage_use_accumulated else "aligned",
        )
        self._capture_count += 1
        if trigger_time is not None:
            rospy.loginfo(
                "Capture %d complete (%s) at %.3f sec",
                self._capture_count,
                reason,
                trigger_time.to_sec(),
            )
        else:
            rospy.loginfo(
                "Capture %d complete (%s)", self._capture_count, reason
            )
        return True

    def _get_cloud_for_trigger(self):
        with self._lock:
            msg = self._last_cloud_msg
            msg_time = self._last_cloud_time
        if msg is not None and msg_time is not None:
            age = (rospy.Time.now() - msg_time).to_sec()
            if age <= self.cloud_max_age:
                self._log_info("using cached cloud (age=%.3f sec)", age)
                return msg
            self._log_info_throttle(
                "cloud_stale",
                1.0,
                "cached cloud too old (age=%.3f > %.3f), waiting for new cloud",
                age,
                self.cloud_max_age,
            )
        if self.cloud_wait_timeout <= 0:
            self._log_info("cloud_wait_timeout <= 0, no blocking wait")
            return None
        try:
            self._log_info(
                "waiting for cloud (timeout=%.3f sec)", self.cloud_wait_timeout
            )
            return rospy.wait_for_message(
                self.input_cloud_topic, PointCloud2, timeout=self.cloud_wait_timeout
            )
        except rospy.ROSException:
            self._log_info_throttle(
                "cloud_wait_timeout",
                1.0,
                "wait_for_message timed out (%.3f sec)",
                self.cloud_wait_timeout,
            )
            return None

    def _load_external_reference(self):
        if not self.reference_cloud_topic:
            rospy.logerr("reference_cloud_topic is empty")
            return
        try:
            msg = rospy.wait_for_message(
                self.reference_cloud_topic, PointCloud2, timeout=self.reference_wait_timeout
            )
        except rospy.ROSException:
            rospy.logwarn("Reference cloud not available yet")
            return
        points = self._cloud_to_xyz(msg)
        if points.shape[0] < self.min_points:
            rospy.logwarn("Reference cloud too small: %d points",
                          points.shape[0])
            return
        points_down = self._voxel_downsample(points, self.downsample_voxel)
        self._reference_cloud = points_down
        self._reference_cloud_full = points
        self._reference_frame = msg.header.frame_id
        self._reference_stamp = msg.header.stamp
        rospy.loginfo("External reference cloud loaded (%d points)",
                      points_down.shape[0])
        self._publish_reference(points_down, frame_id=self._reference_frame)

    def _load_pcd_reference(self):
        if not self.reference_pcd_path:
            rospy.logerr("reference_pcd_path is empty")
            return
        if not _HAS_OPEN3D:
            rospy.logerr("open3d not available, cannot load PCD reference")
            return
        if not os.path.isfile(self.reference_pcd_path):
            rospy.logerr("PCD file not found: %s", self.reference_pcd_path)
            return
        try:
            pcd = o3d.io.read_point_cloud(self.reference_pcd_path)
        except Exception as exc:
            rospy.logerr("Failed to read PCD: %s", exc)
            return
        if pcd is None:
            rospy.logerr("Failed to read PCD: %s", self.reference_pcd_path)
            return
        points = np.asarray(pcd.points, dtype=np.float32)
        if points.size == 0 or points.shape[0] < self.min_points:
            rospy.logwarn("Reference PCD too small: %d points",
                          points.shape[0])
            return
        points_down = self._voxel_downsample(points, self.downsample_voxel)
        self._reference_cloud = points_down
        self._reference_cloud_full = points
        if self.reference_frame_id:
            self._reference_frame = self.reference_frame_id
        self._reference_stamp = rospy.Time.now()
        rospy.loginfo("PCD reference cloud loaded (%d points)",
                      points_down.shape[0])
        if self._reference_center_point is None and self.reference_center is not None:
            self._reference_center_point = self.reference_center
        self._publish_reference(points_down, frame_id=self._reference_frame)

    def _parse_center_param(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
            except Exception:
                rospy.logwarn("Invalid reference_center list: %s", str(value))
                return None
        if isinstance(value, str):
            parts = value.replace(",", " ").split()
            if len(parts) == 3:
                try:
                    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
                except Exception:
                    rospy.logwarn("Invalid reference_center string: %s", value)
                    return None
        rospy.logwarn("Invalid reference_center format: %s", str(value))
        return None

    def _cloud_to_xyz(self, msg):
        points = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        )
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return points[:, :3]

    def _voxel_downsample(self, points, voxel):
        if voxel is None or voxel <= 0:
            return points
        coords = np.floor(points / float(voxel)).astype(np.int32)
        _, idx = np.unique(coords, axis=0, return_index=True)
        return points[idx]

    def _voxel_downsample_with_colors(self, points, colors, voxel):
        if voxel is None or voxel <= 0:
            return points, colors
        coords = np.floor(points / float(voxel)).astype(np.int32)
        _, idx = np.unique(coords, axis=0, return_index=True)
        return points[idx], colors[idx]

    def _accumulate_points(self, existing, new_points):
        if existing is None or existing.size == 0:
            merged = new_points
        else:
            merged = np.concatenate([existing, new_points], axis=0)
        if self.accumulate_voxel and self.accumulate_voxel > 0:
            merged = self._voxel_downsample(merged, self.accumulate_voxel)
        return merged

    def _accumulate_colored(self, existing_points, existing_colors, new_points, new_colors):
        if existing_points is None or existing_points.size == 0:
            merged_points = new_points
            merged_colors = new_colors
        else:
            merged_points = np.concatenate(
                [existing_points, new_points], axis=0)
            if existing_colors is None:
                merged_colors = None
            else:
                merged_colors = np.concatenate(
                    [existing_colors, new_colors], axis=0)
        if self.accumulate_voxel and self.accumulate_voxel > 0:
            if merged_colors is None:
                merged_points = self._voxel_downsample(
                    merged_points, self.accumulate_voxel
                )
            else:
                merged_points, merged_colors = self._voxel_downsample_with_colors(
                    merged_points, merged_colors, self.accumulate_voxel
                )
        return merged_points, merged_colors

    def _make_capture_colors(self, num_points, capture_index):
        if not self.colorize_captures or num_points <= 0:
            return None
        hue = (float(capture_index) * float(self.pastel_hue_step)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(
            hue, float(self.pastel_saturation), float(self.pastel_value)
        )
        colors = np.empty((int(num_points), 3), dtype=np.float32)
        colors[:] = (r, g, b)
        return colors

    @staticmethod
    def _pack_rgb_float(r, g, b):
        rgb_uint32 = struct.unpack(
            "I",
            struct.pack("BBBB", int(b), int(g), int(r), 0),
        )[0]
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]

    def _run_icp(self, src_points, dst_points, init_transform=None):
        if self.use_open3d and _HAS_OPEN3D:
            return self._run_icp_open3d(src_points, dst_points, init_transform)
        if self.use_open3d and not _HAS_OPEN3D:
            self._log_info("open3d unavailable, falling back to numpy ICP")
        if self.use_ransac:
            rospy.logwarn_throttle(
                5.0,
                "use_ransac is true but open3d is unavailable; running ICP without RANSAC",
            )
        return self._run_icp_numpy(src_points, dst_points, init_transform)

    def _run_icp_open3d(self, src_points, dst_points, init_transform=None):
        if src_points.shape[0] == 0 or dst_points.shape[0] == 0:
            return None
        src = o3d.geometry.PointCloud()
        dst = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(src_points.astype(np.float64))
        dst.points = o3d.utility.Vector3dVector(dst_points.astype(np.float64))
        if init_transform is None:
            init_transform = np.eye(4)
        if self.use_ransac:
            ransac_transform = self._run_ransac_open3d(src_points, dst_points)
            if ransac_transform is not None:
                init_transform = ransac_transform
            else:
                self._log_info("RANSAC failed, using initial transform")
        result = o3d.pipelines.registration.registration_icp(
            src,
            dst,
            self.icp_max_corr_dist,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_max_iter
            ),
        )
        transform = result.transformation
        mean_error = float(result.inlier_rmse)
        fitness = float(result.fitness)
        return transform, mean_error, fitness

    def _run_icp_numpy(self, src_points, dst_points, init_transform=None):
        if src_points.shape[0] == 0 or dst_points.shape[0] == 0:
            return None
        if not _HAS_CKDTREE:
            rospy.logerr("scipy not available, cannot run ICP")
            return None

        src = src_points.copy()
        dst = dst_points
        if init_transform is None:
            init_transform = np.eye(4)
        transform = init_transform.copy()
        if not np.allclose(init_transform, np.eye(4)):
            src = self._apply_transform(src, init_transform)
        prev_error = None

        tree = cKDTree(dst)
        for _ in range(int(self.icp_max_iter)):
            distances, indices = tree.query(src, k=1)
            if self.icp_max_corr_dist > 0:
                mask = distances <= self.icp_max_corr_dist
            else:
                mask = np.ones_like(distances, dtype=bool)
            if np.count_nonzero(mask) < self.min_points:
                break

            src_corr = src[mask]
            dst_corr = dst[indices[mask]]
            t_step = self._best_fit_transform(src_corr, dst_corr)
            src = self._apply_transform(src, t_step)
            transform = t_step @ transform

            mean_error = float(np.mean(distances[mask])) if np.any(
                mask) else float("inf")
            if prev_error is not None:
                if abs(prev_error - mean_error) < self.icp_tolerance:
                    break
            prev_error = mean_error

        if prev_error is None:
            return None
        fitness = float(np.count_nonzero(mask)) / float(src_points.shape[0])
        return transform, prev_error, fitness

    def _run_ransac_open3d(self, src_points, dst_points):
        voxel = float(self.ransac_voxel) if self.ransac_voxel else 0.0
        if voxel <= 0:
            rospy.logwarn_throttle(
                5.0, "ransac_voxel <= 0; RANSAC skipped"
            )
            return None
        if src_points.shape[0] < self.min_points or dst_points.shape[0] < self.min_points:
            self._log_info("RANSAC skipped: not enough points")
            return None

        src = o3d.geometry.PointCloud()
        dst = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(src_points.astype(np.float64))
        dst.points = o3d.utility.Vector3dVector(dst_points.astype(np.float64))

        src_down = src.voxel_down_sample(voxel)
        dst_down = dst.voxel_down_sample(voxel)
        if len(src_down.points) < self.min_points or len(dst_down.points) < self.min_points:
            self._log_info("RANSAC skipped: downsampled clouds too small")
            return None

        radius_normal = voxel * 2.0
        radius_feature = voxel * 5.0

        src_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=30
            )
        )
        dst_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=30
            )
        )
        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=100
            ),
        )
        dst_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            dst_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=100
            ),
        )

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down,
            dst_down,
            src_fpfh,
            dst_fpfh,
            bool(self.ransac_mutual_filter),
            float(self.ransac_distance),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            int(self.ransac_n),
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    float(self.ransac_edge_length)
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    float(self.ransac_distance)
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=int(self.ransac_max_iter),
                confidence=float(self.ransac_confidence),
            ),
        )
        if result is None:
            return None
        self._log_info(
            "RANSAC done: fitness=%.3f inlier_rmse=%.6f",
            float(result.fitness),
            float(result.inlier_rmse),
        )
        if np.any(np.isnan(result.transformation)):
            rospy.logwarn("RANSAC returned NaN transform")
            return None
        return result.transformation

    def _compute_init_transform(self, src_points):
        if self._reference_center_point is None:
            return np.eye(4)
        if src_points is None or src_points.size == 0:
            return np.eye(4)
        src_center = np.mean(src_points, axis=0)
        t = self._reference_center_point - src_center
        transform = np.eye(4)
        transform[:3, 3] = t
        self._log_info(
            "initial translation: [%.6f, %.6f, %.6f]",
            float(t[0]),
            float(t[1]),
            float(t[2]),
        )
        return transform

    def _best_fit_transform(self, src, dst):
        src_centroid = np.mean(src, axis=0)
        dst_centroid = np.mean(dst, axis=0)
        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid

        h = src_centered.T @ dst_centered
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[2, :] *= -1
            r = vt.T @ u.T
        t = dst_centroid - r @ src_centroid
        transform = np.eye(4)
        transform[:3, :3] = r
        transform[:3, 3] = t
        return transform

    def _apply_transform(self, points, transform):
        rot = transform[:3, :3]
        trans = transform[:3, 3]
        return (rot @ points.T).T + trans

    def _publish_aligned(self, points, header, colors=None):
        out_header = header
        if self._reference_frame:
            out_header.frame_id = self._reference_frame
        cloud_msg = None
        if colors is None or points.size == 0:
            cloud_msg = pc2.create_cloud_xyz32(out_header, points.tolist())
        else:
            if colors.shape[0] != points.shape[0]:
                rospy.logwarn(
                    "Color count mismatch: points=%d colors=%d (publishing XYZ only)",
                    int(points.shape[0]),
                    int(colors.shape[0]),
                )
                cloud_msg = pc2.create_cloud_xyz32(out_header, points.tolist())
            else:
                colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
                fields = [
                    PointField("x", 0, PointField.FLOAT32, 1),
                    PointField("y", 4, PointField.FLOAT32, 1),
                    PointField("z", 8, PointField.FLOAT32, 1),
                    PointField("rgb", 12, PointField.FLOAT32, 1),
                ]
                cloud_data = []
                for point, color in zip(points, colors_u8):
                    rgb = self._pack_rgb_float(color[0], color[1], color[2])
                    cloud_data.append(
                        (float(point[0]), float(point[1]),
                         float(point[2]), rgb)
                    )
                cloud_msg = pc2.create_cloud(out_header, fields, cloud_data)
        self._aligned_pub.publish(cloud_msg)
        self._publish_count += 1
        self._log_info(
            "published aligned cloud (%d points) on %s",
            int(points.shape[0]),
            self.output_cloud_topic,
        )
        self._save_aligned_ply(points, out_header)

    def _publish_reference(self, points, frame_id=None):
        if self._reference_pub is None:
            return
        use_frame = frame_id or self._reference_frame or self.reference_frame_id or ""
        if not use_frame:
            rospy.logwarn_throttle(
                5.0,
                "reference frame_id is empty; set ~reference_frame_id for reference publish",
            )
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = use_frame
        cloud_msg = pc2.create_cloud_xyz32(header, points.tolist())
        self._reference_pub.publish(cloud_msg)
        self._log_info(
            "published reference cloud (%d points) on %s",
            int(points.shape[0]),
            self.reference_cloud_pub_topic,
        )

    def _maybe_compute_and_record_coverage(self, input_points, trigger_time, capture_index, source_label="aligned"):
        if not self.enable_coverage:
            return
        model_points = self._reference_cloud_full
        if self.coverage_use_downsampled_ref or model_points is None or model_points.size == 0:
            model_points = self._reference_cloud
        if model_points is None or model_points.size == 0:
            rospy.logwarn_throttle(
                5.0, "Coverage skipped: reference cloud empty")
            return
        if input_points is None or input_points.size == 0:
            rospy.logwarn_throttle(5.0, "Coverage skipped: input cloud empty")
            return
        if not _HAS_CKDTREE:
            rospy.logwarn_throttle(
                5.0, "Coverage skipped: scipy not available")
            return

        kdtree = cKDTree(input_points)
        distances, _ = kdtree.query(model_points)
        covered_mask = distances < float(self.coverage_distance)
        covered_count = int(np.count_nonzero(covered_mask))
        total_count = int(model_points.shape[0])
        coverage_ratio = float(covered_count) / float(total_count)

        stamp = trigger_time if trigger_time is not None else rospy.Time.now()
        stamp_sec = float(stamp.to_sec())
        path_sum_value = self._path_sum_value
        self._coverage_history.append(
            (stamp_sec, int(capture_index), coverage_ratio,
             source_label, path_sum_value)
        )
        rospy.loginfo(
            "Coverage: %.4f (%d/%d) source=%s input=%d",
            coverage_ratio,
            covered_count,
            total_count,
            source_label,
            int(input_points.shape[0]),
        )
        self._append_coverage_log(
            stamp_sec,
            int(capture_index),
            coverage_ratio,
            source_label,
            path_sum_value,
        )

    def _append_coverage_log(
        self,
        stamp_sec,
        capture_index,
        coverage_ratio,
        source_label,
        path_sum_value,
    ):
        log_file = self._resolve_coverage_log_file()
        if not log_file:
            return
        if self._coverage_log_error:
            return
        try:
            is_new = (not os.path.exists(log_file)
                      ) or os.path.getsize(log_file) == 0
            with open(log_file, "a", encoding="utf-8") as handle:
                if is_new:
                    handle.write(
                        "stamp_sec,capture_index,coverage_ratio,source,path_sum\n"
                    )
                if path_sum_value is None:
                    path_sum_value = float("nan")
                handle.write(
                    f"{stamp_sec:.6f},{capture_index},{coverage_ratio:.6f},{source_label},{path_sum_value}\n"
                )
                if self.coverage_log_flush:
                    handle.flush()
        except Exception as exc:
            rospy.logerr("Failed to append coverage log: %s", exc)
            self._coverage_log_error = True

    def _resolve_coverage_log_file(self):
        if self._coverage_log_file:
            return self._coverage_log_file
        if not self.coverage_log_dir:
            return ""
        log_dir = self.coverage_log_dir
        if log_dir and not os.path.isdir(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as exc:
                rospy.logerr("Failed to create coverage log dir: %s", exc)
                self._coverage_log_error = True
                return ""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coverage_{timestamp}.csv"
        self._coverage_log_file = os.path.join(log_dir, filename)
        return self._coverage_log_file

    def _ensure_save_dir(self):
        if self._save_dir_checked:
            return bool(self.save_aligned_dir)
        self._save_dir_checked = True
        if not self.save_aligned_dir:
            rospy.logerr("save_aligned_dir is empty; disabling save")
            self.save_aligned_enable = False
            return False
        try:
            os.makedirs(self.save_aligned_dir, exist_ok=True)
        except Exception as exc:
            rospy.logerr("Failed to create save_aligned_dir: %s", exc)
            self.save_aligned_enable = False
            return False
        return True

    def _save_aligned_ply(self, points, header):
        if not self.save_aligned_enable:
            return
        if points is None or points.size == 0:
            rospy.logwarn_throttle(
                5.0, "Aligned cloud empty; skipping PLY save"
            )
            return
        points = self._filter_save_points(points)
        if points.size == 0:
            rospy.logwarn_throttle(
                5.0, "Aligned cloud empty after bounds filter; skipping PLY save"
            )
            return
        if not self._ensure_save_dir():
            return
        stamp = None
        if header is not None and hasattr(header, "stamp"):
            try:
                if header.stamp.to_sec() > 0:
                    stamp = header.stamp
            except Exception:
                stamp = None
        if stamp is None:
            stamp = rospy.Time.now()
        stamp_ns = int(stamp.to_nsec())
        seq = int(self._save_seq)
        self._save_seq += 1
        filename = f"{seq:06d}.ply"
        path = os.path.join(self.save_aligned_dir, filename)
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                handle.write("ply\n")
                handle.write("format ascii 1.0\n")
                handle.write("comment point cloud from /points_aligned\n")
                handle.write(f"element vertex {int(points.shape[0])}\n")
                handle.write("property float x\n")
                handle.write("property float y\n")
                handle.write("property float z\n")
                handle.write("end_header\n")
                for x, y, z in points:
                    handle.write(
                        f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n"
                    )
            os.replace(tmp_path, path)
            self._log_info(
                "saved aligned PLY (%d points) to %s",
                int(points.shape[0]),
                path,
            )
        except Exception as exc:
            rospy.logerr("Failed to save PLY: %s", exc)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _filter_save_points(self, points):
        if points is None or points.size == 0:
            return points
        min_v = self.save_aligned_min
        max_v = self.save_aligned_max
        if min_v.shape[0] != 3 or max_v.shape[0] != 3:
            rospy.logwarn_throttle(
                5.0, "save_aligned_min/max invalid; skipping bounds filter"
            )
            return points
        mask = np.all((points >= min_v) & (points <= max_v), axis=1)
        return points[mask]

    @staticmethod
    def _format_matrix(mat):
        lines = []
        for row in mat:
            lines.append("[" + ", ".join(f"{v: .6f}" for v in row) + "]")
        return "\n".join(lines)


if __name__ == "__main__":
    try:
        node = PointCloudAlignNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
