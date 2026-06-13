#!/usr/bin/env python3

import math
import socket
import sys
import threading
import time

import geometry_msgs.msg
import rospy
import tf2_ros
from std_msgs.msg import Bool, Float32, Float32MultiArray
from tf.transformations import quaternion_from_euler


class DS102ROSDriver:
    def __init__(self):
        rospy.init_node('ds102_driver', anonymous=False)

        # パラメータ取得 (省略なし)
        self.server_ip = rospy.get_param('~server_ip', '192.168.0.2')
        self.server_port = rospy.get_param('~server_port', 5000)
        self.com_port = rospy.get_param('~com_port', 'COM3')
        self.baudrate = rospy.get_param('~baudrate', 38400)
        self.axis_id = rospy.get_param('~axis_id', 1)
        self.socket_timeout = rospy.get_param('~socket_timeout', 2.0)
        self.move_timeout = rospy.get_param('~move_timeout', 180.0)
        self.post_move_settle_time = rospy.get_param(
            '~post_move_settle_time', 0.3)
        self.position_tolerance_deg = rospy.get_param(
            '~position_tolerance_deg', 0.1)
        self.status_poll_hz = rospy.get_param('~status_poll_hz', 10.0)
        self.position_poll_hz = rospy.get_param(
            '~position_poll_hz', 30.0)
        self.parent_frame = rospy.get_param('~parent_frame', 'map')
        self.child_frame = rospy.get_param('~child_frame', 'ds102_link')
        self.origin_xyz = rospy.get_param('~origin_xyz', [0.0, 0.0, 0.0])
        self.l_speed = self._coerce_speed_param(
            rospy.get_param('~l_speed', 100), '~l_speed', 100)
        self.rate = self._coerce_speed_param(
            rospy.get_param('~rate', 100), '~rate', 100)
        self.s_rate = self._coerce_speed_param(
            rospy.get_param('~s_rate', 100), '~s_rate', 100)
        self.speed = self._coerce_speed_param(
            rospy.get_param('~speed', 1000), '~speed', 2000)

        self.deg_per_pulse = 0.004
        self.current_angle_deg = 0.0
        self.target_angle_deg = 0.0
        self.is_moving = False
        self.last_move_done_published = True  # move_doneの重複送信防止
        self.last_angle_deg = None
        self.last_angle_change_time = time.monotonic()

        self.sock = None
        self.sock_file = None
        self.socket_lock = threading.Lock()
        self.speed_lock = threading.Lock()

        self.connect_to_server()
        self.update_current_angle_from_device()

        self.br = tf2_ros.TransformBroadcaster()

        # Publisher
        self.current_angle_pub = rospy.Publisher(
            '/ds102/current_angle', Float32, queue_size=10)
        self.target_angle_pub = rospy.Publisher(
            '/ds102/target_angle', Float32, queue_size=10, latch=True)
        self.is_moving_pub = rospy.Publisher(
            '/ds102/is_moving', Bool, queue_size=10)
        self.move_done_pub = rospy.Publisher(
            '/ds102/move_done', Float32, queue_size=10)

        rospy.Subscriber('/ds102/command_angle', Float32, self.angle_callback)
        rospy.Subscriber('/ds102/command_speed', Float32MultiArray,
                         self.speed_callback)

        # Timer
        rospy.Timer(rospy.Duration(0.1), self.publish_tf)
        if self.position_poll_hz > 0:
            rospy.Timer(rospy.Duration(
                1.0 / self.position_poll_hz), self.poll_status)

        rospy.loginfo("DS102 Driver Ready.")

    def _coerce_speed_param(self, value, name, default):
        try:
            return self._sanitize_speed(float(value))
        except (TypeError, ValueError):
            rospy.logwarn(f"{name} invalid, fallback to {default}")
            return self._sanitize_speed(float(default))

    def _sanitize_speed(self, value):
        if math.isnan(value) or math.isinf(value):
            return 0
        return int(max(0.0, round(value)))

    def connect_to_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.socket_timeout)
            self.sock.connect((self.server_ip, self.server_port))
            self.sock_file = self.sock.makefile('r')
            self.receive_response()  # 初期メッセージ破棄
            self.send_command(f"CONNECT {self.com_port} {self.baudrate}")
        except Exception as e:
            rospy.logerr(f"Connection Failed: {e}")
            sys.exit(1)

    def send_command(self, cmd_str):
        if not self.sock:
            return ""
        try:
            with self.socket_lock:
                if not cmd_str.endswith("\r"):
                    cmd_str += "\r"
                self.sock.sendall(cmd_str.encode())
                return self.sock_file.readline().strip()
        except Exception as e:
            rospy.logerr(f"Socket error: {e}")
            return ""

    def receive_response(self):
        try:
            return self.sock_file.readline().strip()
        except:
            return ""

    def angle_callback(self, msg):
        target_angle = self.normalize_angle_deg(msg.data)
        diff_angle = self.shortest_angle_diff_deg(self.current_angle_deg, target_angle)

        if abs(diff_angle) < self.deg_per_pulse:
            rospy.loginfo("Target angle change too small, ignoring.")
            return

        pulses = int(abs(diff_angle) / self.deg_per_pulse)
        direction = "CW" if diff_angle > 0 else "CCW"

        # 移動開始フラグ
        self.target_angle_deg = target_angle
        self.target_angle_pub.publish(Float32(self.target_angle_deg))
        self.is_moving = True
        self.last_move_done_published = False
        self.is_moving_pub.publish(Bool(True))

        with self.speed_lock:
            l_speed = self.l_speed
            rate = self.rate
            s_rate = self.s_rate
            speed = self.speed
        cmd = (
            f"AXI{self.axis_id}:L0 {l_speed}:R0 {rate}:S0 {s_rate}:F0 {speed}"
            f":PULS {pulses}:GO {direction}"
        )
        self.send_command(cmd)

        # デバイス側で移動フラグが立つのを待つための微小待機
        time.sleep(0.1)

        # 動作完了待ち
        success = self.wait_for_motion_complete()

        # 最終位置の更新
        self.update_current_angle_from_device()

        # 位置安定待ち（微小揺れ対策）
        if success and self.post_move_settle_time > 0:
            self.wait_for_settle(self.post_move_settle_time)

        if success:
            rospy.loginfo(f"Move success: {self.current_angle_deg:.2f} deg")
        else:
            rospy.logwarn("Move timeout or interrupted")

        self.is_moving = False
        self.is_moving_pub.publish(Bool(False))
        self.move_done_pub.publish(Float32(self.current_angle_deg))
        self.last_move_done_published = True

    @staticmethod
    def normalize_angle_deg(angle_deg):
        """角度を [-180, 180] に正規化する"""
        normalized = (angle_deg + 180.0) % 360.0 - 180.0
        if normalized == -180.0:
            return 180.0
        return normalized

    @classmethod
    def shortest_angle_diff_deg(cls, current_deg, target_deg):
        """現在角度から目標角度への最短回転差分 ([-180, 180])"""
        current = cls.normalize_angle_deg(current_deg)
        target = cls.normalize_angle_deg(target_deg)
        return cls.normalize_angle_deg(target - current)

    def wait_for_motion_complete(self):
        """ブロッキングで動作完了を監視"""
        start_time = time.monotonic()
        while not rospy.is_shutdown():
            if time.monotonic() - start_time > self.move_timeout:
                return False

            # SB1のBusyフラグ(0x40)を確認
            is_busy = self.query_axis_busy()
            if not is_busy:
                # 念のため、現在位置が目標に十分近いか確認しても良い
                return True

            rospy.sleep(0.1)
        return False

    def wait_for_settle(self, settle_time):
        """位置変化が一定時間止まるまで待機"""
        end_time = time.monotonic() + settle_time
        last_angle = self.current_angle_deg
        last_change = time.monotonic()
        while not rospy.is_shutdown():
            self.update_current_angle_from_device()
            diff = abs(self.current_angle_deg - last_angle)
            if diff > self.position_tolerance_deg:
                last_angle = self.current_angle_deg
                last_change = time.monotonic()

            if time.monotonic() - last_change >= settle_time:
                return True
            if time.monotonic() >= end_time + settle_time:
                return False
            time.sleep(0.05)
        return False

    def query_axis_busy(self):
        resp = self.send_command(f"WRR AXI{self.axis_id}:SB1?")
        try:
            status = int(resp)
            return bool(status & 0x40)
        except:
            return False

    def update_current_angle_from_device(self):
        resp = self.send_command(f"WRR AXI{self.axis_id}:POS?")
        try:
            pulses = int(resp)
            self.current_angle_deg = pulses * self.deg_per_pulse
            return True
        except:
            return False

    def poll_status(self, event):
        """定期的な位置更新。移動中も含めて更新する"""
        self.update_current_angle_from_device()
        if self.last_angle_deg is None:
            self.last_angle_deg = self.current_angle_deg
            self.last_angle_change_time = time.monotonic()
        elif abs(self.current_angle_deg - self.last_angle_deg) > self.position_tolerance_deg:
            self.last_angle_deg = self.current_angle_deg
            self.last_angle_change_time = time.monotonic()

    def speed_callback(self, msg):
        data = list(msg.data)
        if not data:
            return

        with self.speed_lock:
            if len(data) == 1:
                self.speed = self._sanitize_speed(data[0])
            elif len(data) == 4:
                self.l_speed = self._sanitize_speed(data[0])
                self.rate = self._sanitize_speed(data[1])
                self.s_rate = self._sanitize_speed(data[2])
                self.speed = self._sanitize_speed(data[3])
            else:
                rospy.logwarn(
                    "command_speed expects 1 or 4 values: "
                    "[speed] or [l_speed, rate, s_rate, speed]"
                )

    def publish_tf(self, event):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        t.transform.translation.x = self.origin_xyz[0]
        t.transform.translation.y = self.origin_xyz[1]
        t.transform.translation.z = self.origin_xyz[2]

        rad = -math.radians(self.current_angle_deg)
        q = quaternion_from_euler(0, 0, rad)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q

        self.br.sendTransform(t)
        self.current_angle_pub.publish(Float32(self.current_angle_deg))
        # 非移動時かつ未送信の場合のみ、再度is_moving=Falseを流して安定させる
        if not self.is_moving:
            self.is_moving_pub.publish(Bool(False))

    def shutdown(self):
        if self.sock:
            self.send_command("STOP 0")
            self.sock.close()


if __name__ == '__main__':
    try:
        driver = DS102ROSDriver()
        rospy.on_shutdown(driver.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
