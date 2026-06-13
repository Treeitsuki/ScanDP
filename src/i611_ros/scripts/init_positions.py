#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xmlrpc.client

import rospy
from std_msgs.msg import Float32


def _wait_for_subscriber(pub, topic_name, timeout_sec):
    if timeout_sec <= 0:
        return
    start = rospy.Time.now()
    rate = rospy.Rate(20)
    while pub.get_num_connections() == 0:
        if (rospy.Time.now() - start).to_sec() >= timeout_sec:
            break
        rate.sleep()
    if pub.get_num_connections() == 0:
        rospy.logwarn("No subscribers on %s (timeout %.1f sec)",
                      topic_name, timeout_sec)


def main():
    rospy.init_node("init_positions", anonymous=False)

    arm_xmlrpc_url = rospy.get_param(
        "~arm_xmlrpc_url", "http://192.168.0.23:4416/")
    arm_init_joint = rospy.get_param(
        "~arm_init_joint", [90, 50, 115, -180, 166, 180])
    turntable_command_topic = rospy.get_param(
        "~turntable_command_topic", "/ds102/command_angle")
    turntable_angle = rospy.get_param("~turntable_angle", 0.0)
    turntable_sub_timeout = rospy.get_param("~turntable_sub_timeout", 5.0)
    turntable_post_delay = rospy.get_param("~turntable_post_delay", 0.2)

    try:
        server = xmlrpc.client.ServerProxy(arm_xmlrpc_url)
        server.move_joint(arm_init_joint)
        rospy.loginfo("Arm init requested: %s", str(arm_init_joint))
    except Exception as exc:
        rospy.logerr("Arm init failed: %s", exc)

    pub = rospy.Publisher(turntable_command_topic, Float32, queue_size=1)
    _wait_for_subscriber(pub, turntable_command_topic,
                         float(turntable_sub_timeout))
    pub.publish(Float32(float(turntable_angle)))
    rospy.loginfo("Turntable init requested: %.3f", float(turntable_angle))

    if turntable_post_delay > 0:
        rospy.sleep(float(turntable_post_delay))


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
