# i611_ros

ROS Noetic nodes for operating the i611 robot arm and DS102 turntable, integrating
RealSense + RTAB-Map sensing, gridmap generation, and MoveIt coordination for ScanDP experiments.

## Scope

In scope:
- Robot and turntable control nodes
- Sensor ingestion and synchronization
- Gridmap construction from RealSense point clouds
- MoveIt goal generation from policy outputs

Out of scope:
- ROS 2 support
- Non-catkin build systems

## Functional Requirements

- FR-01: Provide an XML-RPC based control path for the i611 arm and publish `/joint_states`.
- FR-02: Provide a FollowJointTrajectory action server interface for MoveIt execution.
- FR-03: Provide a DS102 driver that accepts target angle commands and publishes status topics.
- FR-04: Provide a gridmap node that converts point clouds into a 3D occupancy map with optional features.
- FR-05: Provide a policy-pose bridge that transforms `/policy/poses` into MoveIt goals and coordinates turntable motion.
- FR-06: Provide helper tools for initial robot pose setup and synthetic pose publishing for testing.

## Key Nodes

| Node | Description |
|---|---|
| `i611_interface.py` | XML-RPC bridge for command and joint-state feedback |
| `i611_driver.py` | FollowJointTrajectory action server for MoveIt execution |
| `ds102_driver.py` | TCP driver for DS102 turntable with status outputs |
| `gridmap_node.py` | Gridmap update from point clouds using TF transforms |
| `pc_align_node.py` | Point-cloud capture and ICP alignment |
| `trans_node.py` | Policy pose intake, MoveIt goal creation, and DS102 coordination |
| `ee_state_node.py` | End-effector state publisher (FK from joint states) |
| `i611_init_pose.py` | One-shot initial joint movement |
| `publish_policy_poses.py` | Test publisher for PoseArray |

## Topics

### Published

| Topic | Type | Description |
|---|---|---|
| `/joint_states` | `JointState` | Joint positions |
| `/ds102/current_angle` | `Float32` | Turntable current angle |
| `/ds102/is_moving` | `Bool` | Turntable motion flag |
| `/ds102/move_done` | `Bool` | Turntable completion flag |
| `/policy/pose` | `PoseStamped` | Policy target pose |
| `/policy/path` | `Path` | Planned path |
| `/policy/poses` | `PoseArray` | Full policy pose array |

### Subscribed

| Topic | Type | Description |
|---|---|---|
| `/follow_joint_trajectory` | Action | MoveIt trajectory execution |
| `/ds102/command_angle` | `Float32` | Turntable target angle |
| `/camera/color/image_raw` | `Image` | RGB from RealSense |
| `/camera/aligned_depth_to_color/image_raw` | `Image` | Depth from RealSense |
| `/camera/color/camera_info` | `CameraInfo` | Camera intrinsics |
| `/camera/depth/color/points` | `PointCloud2` | Raw RealSense point cloud |

## TF Frames

| Frame | Description |
|---|---|
| `map` | World frame (RTAB-Map origin) |
| `base_link` | i611 base |
| `Link6` | i611 end-effector |
| `camera_link` | RealSense camera frame |
| `ds102_link` | DS102 turntable centre |

## Parameters

- **i611 interface**: `~robot_ip`, `~robot_port`
- **DS102 driver**: `~server_ip`, `~server_port`, `~com_port`, `~axis_id`, `~move_timeout`, `~parent_frame`, `~child_frame`
- **trans node**: `~policy_pose_topic`, `~map_frame`, `~base_link_frame`, `~move_group`, `~move_group_action`, `~command_angle_topic`

## Launch Files

- `launch/main.launch` — Full real-robot bring-up (RealSense, RTAB-Map, MoveIt, gridmap, etc.)
- `launch/dataset_rosbag.launch` — Dataset playback and gridmap processing

## Dependencies

- ROS Noetic: `rospy`, `tf2_ros`, `geometry_msgs`, `sensor_msgs`, `std_msgs`, `moveit_commander`
- Python: `opencv-python`, `torch`, `open3d`, `numpy`

## Operational Notes

- RealSense and RTAB-Map must be running for synchronized capture.
- GPU is optional but recommended for gridmap feature extraction.
- Use safe defaults to avoid unintended robot motion.
