#pragma once

#include <ros/ros.h>
#include <hardware_interface/robot_hw.h>
#include <hardware_interface/joint_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <controller_manager/controller_manager.h>
#include <xmlrpcpp/XmlRpc.h>
#include <memory>

namespace i611_ros_control
{

    class I611HWInterface : public hardware_interface::RobotHW
    {
    public:
        I611HWInterface(ros::NodeHandle &nh);
        void read();
        void write();

    private:
        static constexpr int NUM_JOINTS = 6;

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;

        std::string joint_names_[NUM_JOINTS];

        double joint_position_[NUM_JOINTS];
        double joint_velocity_[NUM_JOINTS];
        double joint_effort_[NUM_JOINTS];
        double joint_command_[NUM_JOINTS];

        hardware_interface::JointStateInterface jnt_state_interface_;
        hardware_interface::PositionJointInterface pos_jnt_interface_;

        std::unique_ptr<XmlRpc::XmlRpcClient> rpc_;
        std::string command_method_;

        double rad2deg(double r) { return r * 180.0 / M_PI; }
        double deg2rad(double d) { return d * M_PI / 180.0; }

        std::vector<double> last_sent_cmd_;
    };

}
