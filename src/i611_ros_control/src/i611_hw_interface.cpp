#include "i611_ros_control/i611_hw_interface.h"
#include <cmath>

namespace i611_ros_control
{

    I611HWInterface::I611HWInterface(ros::NodeHandle &nh)
        : nh_(nh),
          pnh_("~"),
          last_sent_cmd_(NUM_JOINTS, 0.0)
    {
        std::string robot_ip;
        int robot_port = 0;
        pnh_.param<std::string>("robot_ip", robot_ip, std::string("192.168.0.23"));
        pnh_.param<int>("robot_port", robot_port, 4416);
        pnh_.param<std::string>("command_method", command_method_, std::string("move_joint_nores"));
        rpc_ = std::make_unique<XmlRpc::XmlRpcClient>(robot_ip.c_str(), robot_port);

        joint_names_[0] = "Joint1";
        joint_names_[1] = "Joint2";
        joint_names_[2] = "Joint3";
        joint_names_[3] = "Joint4";
        joint_names_[4] = "Joint5";
        joint_names_[5] = "Joint6";

        for (int i = 0; i < NUM_JOINTS; i++)
        {
            joint_position_[i] = 0.0;
            joint_velocity_[i] = 0.0;
            joint_effort_[i] = 0.0;
            joint_command_[i] = 0.0;

            hardware_interface::JointStateHandle state_handle(
                joint_names_[i],
                &joint_position_[i],
                &joint_velocity_[i],
                &joint_effort_[i]);
            jnt_state_interface_.registerHandle(state_handle);

            hardware_interface::JointHandle pos_handle(
                jnt_state_interface_.getHandle(joint_names_[i]),
                &joint_command_[i]);
            pos_jnt_interface_.registerHandle(pos_handle);
        }

        registerInterface(&jnt_state_interface_);
        registerInterface(&pos_jnt_interface_);

        ROS_INFO_STREAM("I611HWInterface initialized (ip=" << robot_ip
                                                           << ", port=" << robot_port
                                                           << ", command_method=" << command_method_ << ")");
    }

    void I611HWInterface::read()
    {
        XmlRpc::XmlRpcValue result;
        if (!rpc_->execute("getjnt", XmlRpc::XmlRpcValue(), result))
        {
            ROS_WARN_THROTTLE(1.0, "Failed to call getjnt");
            return;
        }

        ROS_DEBUG_STREAM_THROTTLE(1.0, "getjnt result: " << result);

        for (int i = 0; i < NUM_JOINTS; i++)
        {
            joint_position_[i] = deg2rad(static_cast<double>(result[i]));
        }
    }

    void I611HWInterface::write()
    {
        bool changed = false;
        XmlRpc::XmlRpcValue cmd;

        for (int i = 0; i < NUM_JOINTS; i++)
        {
            double deg = rad2deg(joint_command_[i]);
            cmd[i] = deg;
            if (std::fabs(deg - last_sent_cmd_[i]) > 0.01)
                changed = true;
        }

        if (!changed)
            return;

        ROS_DEBUG_STREAM_THROTTLE(0.5, "Sending " << command_method_ << ": " << cmd);

        XmlRpc::XmlRpcValue args;
        args[0] = cmd;

        XmlRpc::XmlRpcValue result;
        if (!rpc_->execute(command_method_.c_str(), args, result))
        {
            ROS_ERROR_THROTTLE(1.0, "Failed to send command method");
            return;
        }

        for (int i = 0; i < NUM_JOINTS; i++)
            last_sent_cmd_[i] = cmd[i];
    }

}
