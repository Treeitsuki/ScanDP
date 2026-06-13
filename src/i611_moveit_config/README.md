# i611_moveit_config Requirements

**Overview**
This package provides the MoveIt configuration for the i611 (fsrobo_r) robot, including URDF/SRDF, planning pipelines, controller settings, and launch files for simulation and real execution.

**Scope**
In scope:
- MoveIt planning configuration and pipelines
- Robot description and semantics
- Controller and execution settings
- RViz visualization and demo launches

Out of scope:
- ROS2 support
- Non-catkin build systems

**Functional Requirements**
- FR-01 Provide a complete robot description and semantics for planning.
- FR-02 Provide motion planning pipelines (OMPL, CHOMP, STOMP, Pilz).
- FR-03 Provide controller configurations for fake and real controllers.
- FR-04 Provide launch files for MoveIt bring-up, RViz visualization, and real execution.
- FR-05 Integrate with `i611_ros` for real hardware execution.

**Configuration Files**
- `config/fsrobo_r.srdf` semantic robot description
- `config/joint_limits.yaml` joint limit definitions
- `config/kinematics.yaml` kinematics solver settings
- `config/ros_controllers.yaml` controller mapping
- `config/fake_controllers.yaml` fake controller mapping
- `config/ompl_planning.yaml` OMPL planning settings
- `config/chomp_planning.yaml` CHOMP planning settings
- `config/stomp_planning.yaml` STOMP planning settings
- `config/sensors_3d.yaml` 3D sensor config

**Launch Files**
- `launch/real.launch` Real robot execution pipeline
- `launch/move_group.launch` Core MoveIt node
- `launch/moveit_rviz.launch` RViz visualization
- `launch/demo.launch` Demo with fake controllers
- `launch/gazebo.launch` Simulation pipeline
- Planning pipeline launch files under `launch/*planning_pipeline*`

**Dependencies**
- MoveIt core packages and planners
- `tf2_ros`, `joint_state_publisher`, `robot_state_publisher`, `rviz`
- `i611_ros` for real hardware integration

**Operational Notes**
- Ensure the robot description matches the actual i611 URDF used by hardware.
- Use `real.launch` when connected to the robot, and `demo.launch` for offline testing.
