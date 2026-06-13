# ScanDP Usage Guide

This document provides step-by-step instructions for the full ScanDP pipeline:
environment setup, dataset creation, training, simulation evaluation, and real-robot deployment.

**日本語版は [USAGE_ja.md](USAGE_ja.md) を参照してください。**

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Pipeline](#2-dataset-pipeline)
   - [2.1 Record demonstrations (real system)](#21-record-demonstrations-real-system)
   - [2.2 Convert HDF5 to demo chunks](#22-convert-hdf5-to-demo-chunks)
   - [2.3 Convert demo chunks to zarr](#23-convert-demo-chunks-to-zarr)
   - [2.4 Visualize the dataset](#24-visualize-the-dataset)
3. [Training](#3-training)
   - [3.1 Configuration overview](#31-configuration-overview)
   - [3.2 Run training](#32-run-training)
   - [3.3 Monitor training](#33-monitor-training)
4. [Simulation Evaluation (Genesis)](#4-simulation-evaluation-genesis)
   - [4.1 Evaluate a trained checkpoint](#41-evaluate-a-trained-checkpoint)
   - [4.2 Evaluate from recorded trajectories](#42-evaluate-from-recorded-trajectories)
5. [Real-World Deployment](#5-real-world-deployment)
   - [5.1 Hardware launch order](#51-hardware-launch-order)
   - [5.2 Start ROS components](#52-start-ros-components)
   - [5.3 Run policy inference](#53-run-policy-inference)
6. [ROS Node Reference](#6-ros-node-reference)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Environment Setup

### Prerequisites

| Requirement | Version |
|---|---|
| OS | Ubuntu 20.04 |
| ROS | Noetic |
| Python | ≥ 3.11 |
| CUDA | 11.8 (tested) |
| Package manager | [`uv`](https://github.com/astral-sh/uv) |

### Install Python environment

```bash
# From the repository root
uv sync
source .venv/bin/activate

# Add ROS Python bindings to the virtual environment
uv pip install --extra-index-url https://rospypi.github.io/simple rospy-all
uv pip install setuptools==81.0.0
```

### Install Genesis (simulation only)

Follow the [Genesis installation guide](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html).

### Build the ROS workspace

```bash
source /opt/ros/noetic/setup.bash
catkin build
source devel/setup.bash
```

> **Note:** The workspace mixes a ROS Noetic installation with a Python 3 virtual environment.
> Always activate the venv *after* sourcing the ROS setup files.

### Recommended shell initialization

Add to your `~/.bashrc` (or run manually before each session):

```bash
source /opt/ros/noetic/setup.bash
source /path/to/ScanDP/devel/setup.bash
source /path/to/ScanDP/.venv/bin/activate
```

---

## 2. Dataset Pipeline

The data pipeline has three stages:

```
Real system / rosbag
        │
        ▼
  HDF5 recording (.h5)         ← create_dataset_real.py
        │
        ▼
  Demo chunks (per-episode)    ← create_dataset.py
        │
        ▼
  zarr dataset                 ← convert_demos.py
```

### 2.1 Record demonstrations (real system)

`create_dataset_real.py` is a ROS node that subscribes to the sensor topics listed below
and writes synchronized frames to an HDF5 file when triggered.

**Required ROS topics:**

| Topic | Type | Description |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB image |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Aligned depth |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |
| `/camera/depth/color/points` | `sensor_msgs/PointCloud2` | RealSense point cloud |
| TF: `map → camera_link` | — | Camera pose in world frame |

**Run:**

```bash
source devel/setup.bash
python scandp/create_dataset_real.py \
  _hdf5_out_path:=data/record_001.h5 \
  _seq_length:=150
```

| Parameter | Default | Description |
|---|---|---|
| `_hdf5_out_path` | `data/record.h5` | Output file path |
| `_seq_length` | `150` | Number of frames per episode |

> The node records one episode and exits. Repeat for each demonstration.

### 2.2 Convert HDF5 to demo chunks

```bash
python scandp/create_dataset.py \
  data/record_001.h5 \
  --output-dir save_dir/
```

Run once per HDF5 file. Each episode is saved as a separate subdirectory under `save_dir/`.

**Batch processing:**

```bash
for f in data/hdf5/*.h5; do
  python scandp/create_dataset.py "$f" --output-dir save_dir/
done
```

### 2.3 Convert demo chunks to zarr

```bash
python scandp/convert_demos.py \
  --demo_dir save_dir/ \
  --save_dir dataset/gridmap_real \
  --save_img 1 \
  --save_depth 1
```

| Argument | Description |
|---|---|
| `--demo_dir` | Directory containing per-episode demo chunks |
| `--save_dir` | Output zarr dataset directory |
| `--save_img` | Save RGB images into zarr (`1` = yes) |
| `--save_depth` | Save depth images into zarr (`1` = yes) |

The resulting `dataset/gridmap_real/` is the zarr path you pass to training and evaluation.

### 2.4 Visualize the dataset

```bash
cd scandp
python vis_dataset.py \
  --dataset_path /path/to/dataset/gridmap_real \
  --use_img 1 \
  --vis_cloud 0 \
  --use_pc_color 0 \
  --downsample 1
```

---

## 3. Training

### 3.1 Configuration overview

Training is configured through [Hydra](https://hydra.cc/). The main configuration files are:

```
scandp/diffusion_policy_3d/config/
├── scandp_spconv.yaml      ← main algorithm config (ScanDP + SpConv)
├── scandp_conv3d.yaml      ← dense 3D conv variant
├── scandp_img_r3m.yaml     ← image-only baseline
├── dp3.yaml                ← DP3 baseline
├── idp3.yaml               ← IDP3 baseline
└── task/
    └── cam_gridmap.yaml    ← task / observation space config
```

**Key parameters in `cam_gridmap.yaml`:**

| Parameter | Description |
|---|---|
| `shape_meta` | Observation and action dimensions |
| `dataset.zarr_path` | Path to the zarr dataset |
| `dataset.max_train_episodes` | Number of episodes to use for training |

**Key parameters in the algorithm config (e.g., `scandp_spconv.yaml`):**

| Parameter | Description |
|---|---|
| `horizon` | Prediction horizon |
| `n_action_steps` | Number of action steps to execute |
| `n_obs_steps` | Number of observation steps |
| `policy.noise_scheduler` | DDIM scheduler parameters |

Override any parameter at the command line without editing config files:

```bash
python train.py --config-name=scandp_spconv.yaml \
  task.dataset.max_train_episodes=50
```

### 3.2 Run training

```bash
cd scandp

python train.py \
  --config-name=scandp_spconv.yaml \
  task=cam_gridmap \
  hydra.run.dir=data/outputs/cam_gridmap-scandp_spconv-run1_seed0 \
  training.seed=0 \
  training.device=cuda:0 \
  exp_name=cam_gridmap-scandp_spconv-run1 \
  logging.mode=offline \
  checkpoint.save_ckpt=true \
  task.dataset.zarr_path=/path/to/dataset/gridmap_real
```

**Important parameters:**

| Parameter | Description |
|---|---|
| `--config-name` | Algorithm config (`scandp_spconv`, `dp3`, …) |
| `task` | Task config (`cam_gridmap`) |
| `hydra.run.dir` | Output directory for logs and checkpoints |
| `training.seed` | Random seed |
| `training.device` | PyTorch device (`cuda:0`, `cuda:1`, …) |
| `exp_name` | Experiment name (used by WandB) |
| `logging.mode` | WandB mode: `online` or `offline` |
| `checkpoint.save_ckpt` | Whether to save checkpoints |
| `task.dataset.zarr_path` | Path to zarr dataset |

Checkpoints are saved to `hydra.run.dir/checkpoints/`.

### 3.3 Monitor training

If `logging.mode=online`, metrics appear on [WandB](https://wandb.ai/).
For offline mode, logs are written to `hydra.run.dir/logs/`.

---

## 4. Simulation Evaluation (Genesis)

Genesis simulation uses the same 3D mesh assets stored in `scandp/diffusion_policy_3d/assets/`.

### 4.1 Evaluate a trained checkpoint

```bash
cd scandp

python deploy_gridmap.py \
  --config-name=scandp_spconv.yaml \
  task=cam_gridmap \
  hydra.run.dir=data/outputs/cam_gridmap-scandp_spconv-run1_seed0 \
  training.seed=0 \
  training.device=cuda:0 \
  exp_name=cam_gridmap-scandp_spconv-run1 \
  logging.mode=offline \
  task.dataset.zarr_path=/path/to/dataset/gridmap_real
```

The script automatically loads the latest checkpoint from `hydra.run.dir/checkpoints/`.
Results (coverage metrics and point-cloud outputs) are saved to `hydra.run.dir/`.

**Available target objects** (set `TARGET` inside the script or pass as a config override):

`bunny`, `armadillo`, `spot`, `dragon`, `bust`, `bike`, `happy`, `teapot`

### 4.2 Evaluate from recorded trajectories

`deploy_demo.py` replays a set of pre-recorded trajectories (`.pth` files) and measures coverage:

```bash
cd scandp

python deploy_demo.py \
  --target bunny \
  --scale 1 \
  --pth /path/to/demo.pth \
  --csv results.csv
```

| Argument | Description |
|---|---|
| `--target` | Object name (see list above) |
| `--scale` | Object scale factor (e.g., `1` or `1.5`) |
| `--pth` | Path to a recorded trajectory `.pth` file |
| `--csv` | Output CSV file for results |

---

## 5. Real-World Deployment

> **Hardware required:** i611 arm, DS102 turntable, Intel RealSense, and the proprietary server programs
> for both the i611 and DS102. These server programs are not included in this repository.

### 5.1 Hardware launch order

Start components in the following order to avoid initialization failures:

```
1. i611 XML-RPC server (on the robot controller PC)
2. DS102 TCP server (on the turntable controller PC)
3. Intel RealSense → realsense2_camera
4. RTAB-Map
5. MoveIt (i611_moveit_config)
6. ROS nodes in i611_ros
7. Policy deployment node
```

### 5.2 Start ROS components

#### RealSense

```bash
roslaunch realsense2_camera rs_aligned_depth.launch
```

#### RTAB-Map

```bash
roslaunch rtabmap_ros rtabmap.launch \
  rtabmap_args:="--delete_db_on_start" \
  depth_topic:=/camera/aligned_depth_to_color/image_raw \
  rgb_topic:=/camera/color/image_raw \
  camera_info_topic:=/camera/color/camera_info
```

#### MoveIt

```bash
roslaunch i611_moveit_config real.launch robot_ip:=<ROBOT_IP>
```

Replace `<ROBOT_IP>` with the actual IP address of your i611 controller.

#### i611_ros nodes

```bash
# DS102 turntable driver
rosrun i611_ros ds102_driver.py _server_ip:=<DS102_IP>

# Grid-map generator (uses /camera/depth/color/points from RealSense)
rosrun i611_ros gridmap_node.py
```

Replace `<DS102_IP>` with the actual IP address of your DS102 TCP server.

### 5.3 Run policy inference

```bash
cd scandp

python deploy_real.py \
  --config-name=scandp_spconv.yaml \
  task=cam_gridmap \
  hydra.run.dir=data/outputs/cam_gridmap-scandp_spconv-run1_seed0 \
  training.seed=0 \
  training.device=cuda:0 \
  exp_name=cam_gridmap-scandp_spconv-run1 \
  logging.mode=offline \
  task.dataset.zarr_path=/path/to/dataset/gridmap_real
```

The node loads the trained checkpoint and publishes joint targets to MoveIt via
`/follow_joint_trajectory`. Policy outputs (Cartesian waypoints) are also published on
`/policy/pose`, `/policy/path`, and `/policy/poses`.

---

## 6. ROS Node Reference

### Nodes in `src/i611_ros/scripts/`

| Node | Description | Key topics / services |
|---|---|---|
| `i611_driver.py` | XML-RPC bridge to the i611 robot controller | Subscribes: joint commands; Publishes: `/joint_states` |
| `i611_interface.py` | High-level i611 command interface | Used by `i611_driver.py` |
| `ds102_driver.py` | DS102 turntable driver (TCP client) | Pub: `/ds102/current_angle`, `/ds102/is_moving`, `/ds102/move_done`; Sub: `/ds102/command_angle` |
| `gridmap_node.py` | 3D occupancy grid-map generator from point cloud | Sub: `/camera/depth/color/points`, TF `map→camera_link`; Pub: grid-map topic |
| `pc_align_node.py` | Point-cloud capture and ICP alignment | Sub: `/camera/depth/color/points` + completion topics; Pub: aligned point cloud |
| `ee_state_node.py` | End-effector state publisher (FK from joint states) | Pub: `/ee_state` |
| `trans_node.py` | TF broadcaster for the turntable frame | Pub: `ds102_link` TF |
| `moveit_done_bridge.py` | Bridges MoveIt action completion to `/ds102`-style topics | — |
| `policy_path_node.py` | Converts policy output to a ROS Path message | Pub: `/policy/path` |
| `publish_policy_poses.py` | Publishes policy poses as a PoseArray | Pub: `/policy/poses` |
| `init_positions.py` | Moves the arm to a predefined initial pose | — |

### Key TF frames

| Frame | Description |
|---|---|
| `map` | World frame (RTAB-Map origin) |
| `base_link` | i611 base |
| `Link6` | i611 end-effector |
| `camera_link` | RealSense camera optical frame |
| `ds102_link` | DS102 turntable centre |

### Main ROS topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/camera/color/image_raw` | `Image` | Input | RGB |
| `/camera/aligned_depth_to_color/image_raw` | `Image` | Input | Depth |
| `/camera/color/camera_info` | `CameraInfo` | Input | Intrinsics |
| `/camera/depth/color/points` | `PointCloud2` | Input | Raw RealSense point cloud |
| `/ds102/command_angle` | `Float32` | Input | Target angle (degrees) |
| `/ds102/current_angle` | `Float32` | Output | Current angle |
| `/ds102/move_done` | `Bool` | Output | Motion completion flag |
| `/joint_states` | `JointState` | Output | Joint positions |
| `/policy/pose` | `PoseStamped` | Output | Policy target pose |
| `/policy/path` | `Path` | Output | Planned path |

---

## 7. Troubleshooting

See [ERROR_CATCH.md](ERROR_CATCH.md) for common issues.

### Dataset issues

**Grid-map is empty or all zeros**  
Confirm that `/camera/depth/color/points` is being published and the TF `map→camera_link` is available.

**zarr conversion fails with shape mismatch**  
All HDF5 files in a batch must have the same `seq_length`. Re-record any that were cut short.

### Training issues

**CUDA out of memory**  
Reduce `training.batch_size` or switch to a smaller model variant (`scandp_conv3d`).

**WandB authentication error**  
Use `logging.mode=offline` to skip WandB, or run `wandb login` before training.

### Simulation issues

**Genesis segmentation fault on startup**  
Ensure the Genesis version matches the one tested with this codebase.
Check that `CUDA_VISIBLE_DEVICES` is set to a valid GPU index.

**Coverage metric is 0**  
Verify that the target object name (`TARGET`) matches one of the asset files in
`scandp/diffusion_policy_3d/assets/ply/`.

### Real-robot issues

**MoveIt reports "no motion plan found"**  
Check that the robot is near the configured initial pose before launching deployment.
Run `scandp/create_dataset_real.py` with `_seq_length:=1` to verify sensor connectivity first.

**DS102 does not respond**  
Confirm the TCP server is running and `<DS102_IP>` in the launch command matches the actual address.
Check connectivity with `nc -zv <DS102_IP> 5000`.
