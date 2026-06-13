#!/usr/bin/env python3
import math
import xmlrpc.client

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class I611HardwareInterface:
    def __init__(self):
        rospy.init_node('i611_hardware_interface')

        # パラメータ設定
        self.host = rospy.get_param('~robot_ip', '192.168.0.23')
        self.port = rospy.get_param('~robot_port', 4416)
        self.joint_names = ['Joint1', 'Joint2',
                            'Joint3', 'Joint4', 'Joint5', 'Joint6']

        # XML-RPC クライアント設定
        self.proxy = xmlrpc.client.ServerProxy(
            f"http://{self.host}:{self.port}/", allow_none=True)

        # ROS 側の通信設定
        self.joint_state_pub = rospy.Publisher(
            '/joint_states', JointState, queue_size=10)
        # Controllerからの指令をサブスクライブ
        self.sub = rospy.Subscriber(
            '/i611/arm_controller/command', JointTrajectory, self.command_cb)

        rospy.loginfo("Connecting to i611 XML-RPC Server...")

        # 制御ループ
        self.rate = rospy.Rate(20)  # 20Hz (通信負荷を考慮)
        self.main_loop()

    def rad_to_deg(self, rad_list):
        return [math.degrees(x) for x in rad_list]

    def deg_to_rad(self, deg_list):
        return [math.radians(x) for x in deg_list]

    def command_cb(self, msg):
        """
        MoveIt (JointTrajectoryController) からの指令を受け取る
        """
        if msg.points:
            # 最新の目標値を取得
            target_rad = msg.points[-1].positions
            target_deg = self.rad_to_deg(target_rad)

            try:
                # ロボットへ指令 (ノンブロッキングのmove_joint_noresを推奨)
                self.proxy.move_joint_nores(target_deg)
            except Exception as e:
                rospy.logerr(f"Failed to send command: {e}")

    def main_loop(self):
        while not rospy.is_shutdown():
            try:
                # 1. ロボットから現在値を取得
                curr_jnt_deg = self.proxy.getjnt()

                if curr_jnt_deg:
                    # 2. JointState メッセージの作成
                    msg = JointState()
                    msg.header.stamp = rospy.Time.now()
                    msg.name = self.joint_names
                    msg.position = self.deg_to_rad(curr_jnt_deg)

                    # 3. 公開
                    self.joint_state_pub.publish(msg)

            except Exception as e:
                rospy.logerr(f"Communication error: {e}")

            self.rate.sleep()


if __name__ == '__main__':
    try:
        I611HardwareInterface()
    except rospy.ROSInterruptException:
        pass
