# i611_ros_control Requirements

**Overview**
This package provides a ROS Control hardware interface for the i611 robot, exposing joint state feedback and accepting joint trajectory commands for controller execution.

**Scope**
In scope:
- Hardware interface for 6-DoF i611 robot
- Joint state publishing and command execution
- Controller manager integration

Out of scope:
- ROS2 support
- Non-catkin build systems

**Functional Requirements**
- FR-01 Expose a `hardware_interface::RobotHW` implementation for the i611 arm.
- FR-02 Read joint positions from the robot controller and publish `/joint_states`.
- FR-03 Accept trajectory or position commands via ROS control controllers.
- FR-04 Provide a launch entry to start the hardware interface and controller manager.

**Interfaces**
ROS Control interfaces:
- `hardware_interface::JointStateInterface`
- `hardware_interface::PositionJointInterface`

ROS topics and services:
- `/joint_states`
- Controller manager services under `/controller_manager`

**Configuration**
- `config/controllers.yaml` defines joint controllers and parameters.
- Joint names must match the robot description and controller configuration.

**Launch Files**
- `launch/i611_ros_control.launch` Start hardware interface and controllers.

**Dependencies**
- ROS Noetic: `roscpp`, `hardware_interface`, `controller_manager`, `position_controllers`, `trajectory_msgs`, `control_msgs`, `xmlrpcpp`

**Operational Notes**
- Ensure the i611 XML-RPC server is reachable before starting the node.
- Controller parameters must align with the robot URDF and MoveIt configuration.
