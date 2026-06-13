#include <ros/ros.h>
#include <controller_manager/controller_manager.h>
#include "i611_ros_control/i611_hw_interface.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "i611_hw_node");
    ros::NodeHandle nh;

    i611_ros_control::I611HWInterface robot(nh);
    controller_manager::ControllerManager cm(&robot, nh);

    ros::AsyncSpinner spinner(1);
    spinner.start();

    ros::Rate rate(50); // 50Hz

    ros::Time last = ros::Time::now();

    while (ros::ok())
    {
        ros::Time now = ros::Time::now();
        ros::Duration period = now - last;
        last = now;

        robot.read();
        cm.update(now, period);
        robot.write();

        rate.sleep();
    }
    return 0;
}
