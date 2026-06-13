# ScanDP: Generalizable 3D Scanning with Diffusion Policy
<!-- <a href='https://hogehoge'><img src='https://img.shields.io/badge/IROS 2026-blue' alt='Project'></a> -->
<a href='https://treeitsuki.github.io/ScanDP/'><img src='https://img.shields.io/badge/Project-ScanDP-green' alt='Project'></a>
<a href='https://arxiv.org/abs/2603.10390'><img src='https://img.shields.io/badge/Paper-Arxiv-red' alt='Arxiv'></a>


This is the official code release for the paper:

> **ScanDP: Generalizable 3D Scanning with Diffusion Policy**  
> Itsuki Hirako, Ryo Hakoda, Yubin Liu, Matthew Hwang, Yoshihiro Sato, Takeshi Oishi  
> arXiv:2603.10390

[日本語版 README はこちら](README_ja.md)

![ScanDP overview](docs/imgs/teaser_real.png)

## Overview

ScanDP is a robot learning system for 3D object scanning using imitation learning with a diffusion policy.
Given an initial viewpoint, the robot arm moves around an object to collect 3D point clouds while a turntable rotates the object, achieving generalizable coverage across unseen objects.

The system runs on a **ROS Noetic** catkin workspace and integrates:

- **i611** 6-DoF robot arm
- **DS102** motorized turntable
- **Intel RealSense** RGB-D camera
- **RTAB-Map** for real-time mapping and localization
- **Genesis** physics simulator for training-time evaluation

## What is included

| Component | Path | Description |
|---|---|---|
| Training code | `scandp/train.py` | ScanDP / DP3 / IDP3 training via Hydra |
| Real-world deployment | `scandp/deploy_real.py` | Policy inference on the real robot |
| Simulation evaluation | `scandp/deploy_gridmap.py`, `deploy_demo.py` | Genesis-based evaluation |
| Dataset pipeline | `scandp/create_dataset_real.py`, `create_dataset.py`, `convert_demos.py` | HDF5 recording → zarr conversion |
| ROS nodes | `src/i611_ros/scripts/` | i611 control, DS102, grid-map, MoveIt bridge |
| Hydra configs | `scandp/diffusion_policy_3d/config/` | All training and task configurations |
| Ground-truth tools | `scandp/ground_truth/` | Coverage evaluation utilities |
| Helper scripts | `scripts/` | Convenience wrappers for common workflows |

## What is NOT included

- The full dataset used in the paper
- Pretrained model checkpoints
- The i611 robot-side XML-RPC server (proprietary, hardware-specific)
- The DS102 TCP server (proprietary, hardware-specific)
- Lab-specific calibration files and network configuration
- Object segmentation module (unpublished; the pipeline uses the raw RealSense point cloud `/camera/depth/color/points`)

Reproducing the full real-world setup requires the above external components.

## Repository structure

```
.
├── scandp/                          Training, evaluation, and data pipeline
│   ├── train.py                     Entry point for training
│   ├── deploy_real.py               Real-world policy deployment
│   ├── deploy_gridmap.py            Genesis simulation evaluation (main)
│   ├── deploy_demo.py               Genesis demo evaluation from recorded trajectories
│   ├── create_dataset_real.py       Record demos from real system (ROS subscriber)
│   ├── create_dataset.py            Convert a single HDF5 recording to demo chunks
│   ├── convert_demos.py             Convert demo chunks to zarr dataset
│   ├── vis_dataset.py               Dataset visualization
│   └── diffusion_policy_3d/
│       ├── config/                  Hydra configuration files
│       ├── model/                   Network architectures
│       ├── policy/                  Policy wrappers
│       ├── workspace/               Training and deployment workspace classes
│       ├── dataset/                 zarr dataset loaders
│       ├── module/                  Grid-map, feature extractor, utilities
│       ├── common/                  Shared utilities (replay buffer, samplers, etc.)
│       ├── tools/                   Visualization and analysis tools
│       └── assets/                  3D mesh assets for Genesis simulation
├── src/
│   ├── i611_ros/                    ROS package: i611 arm + DS102 + grid-map nodes
│   ├── i611_ros_control/            ROS Control hardware interface
│   └── i611_moveit_config/          MoveIt configuration for the i611
├── scripts/                         Shell script wrappers for training / evaluation
├── test/                            Development and diagnostic scripts
└── docs/                            Supplementary notes and figures
```

## Requirements

### Software
- Ubuntu 20.04 with **ROS Noetic**
- Python ≥ 3.11
- CUDA-capable GPU (tested with CUDA 11.8)
- [`uv`](https://github.com/astral-sh/uv) package manager

### Python dependencies (managed via `uv`)
PyTorch, Open3D, Hydra, spconv, WandB, zarr, h5py, diffusers, and others (see `pyproject.toml`).

### For simulation evaluation
- [Genesis](https://genesis-world.readthedocs.io/) physics simulator

### For real-robot deployment
- Intel RealSense SDK and `realsense2_camera` ROS package
- RTAB-Map (`rtabmap_ros`)
- i611 XML-RPC server (hardware-side, not included)
- DS102 TCP server (hardware-side, not included)

> **Note:** The object segmentation module used in the paper is not yet released.
> The point-cloud source in the released code uses the raw RealSense topic `/camera/depth/color/points`.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Treeitsuki/ScanDP.git
cd ScanDP
```

### 2. Install Python dependencies

```bash
uv sync
source .venv/bin/activate
uv pip install --extra-index-url https://rospypi.github.io/simple rospy-all
uv pip install setuptools==81.0.0
```

### 3. Build the ROS workspace

```bash
source /opt/ros/noetic/setup.bash
catkin build
source devel/setup.bash
```

See [docs/SETUP.md](docs/SETUP.md) for additional environment notes.

## Quickstart: Simulation evaluation

The simplest way to run ScanDP without physical hardware is the Genesis-based simulator.

### 1. Prepare a dataset

If you do not have real-robot data, you can create a synthetic dataset using `create_dataset.py`
from recorded HDF5 files or rosbag replays (see [docs/USAGE.md](docs/USAGE.md) for details).

### 2. Train

```bash
cd scandp

python train.py \
  --config-name=scandp_spconv.yaml \
  task=cam_gridmap \
  hydra.run.dir=data/outputs/scandp_example_seed0 \
  training.seed=0 \
  training.device=cuda:0 \
  exp_name=scandp_example \
  logging.mode=offline \
  checkpoint.save_ckpt=true \
  task.dataset.zarr_path=/path/to/your/dataset
```

### 3. Evaluate in simulation

```bash
python deploy_gridmap.py \
  --config-name=scandp_spconv.yaml \
  task=cam_gridmap \
  hydra.run.dir=data/outputs/scandp_example_seed0 \
  training.seed=0 \
  training.device=cuda:0 \
  exp_name=scandp_example \
  logging.mode=offline \
  task.dataset.zarr_path=/path/to/your/dataset
```

## Available configurations

| Config file | Description |
|---|---|
| `scandp_spconv.yaml` | ScanDP with sparse 3D convolution (SpConv) — main method |
| `scandp_conv3d.yaml` | ScanDP with dense 3D convolution |
| `scandp_img_r3m.yaml` | Image-only baseline (R3M features) |
| `dp3.yaml` | DP3 (3D Diffusion Policy) baseline |
| `idp3.yaml` | IDP3 baseline |

Task: use `task=cam_gridmap` for the standard grid-map observation space.

## Dataset creation pipeline

See [docs/USAGE.md](docs/USAGE.md) — **Dataset Pipeline** section for step-by-step instructions covering:

1. Recording demonstrations from the real system or a rosbag
2. Converting HDF5 recordings to demo chunks
3. Converting demo chunks to a zarr dataset

## Real-world deployment

See [docs/USAGE.md](docs/USAGE.md) — **Real-world Deployment** section for the full launch sequence.

## Known limitations

- Some scripts retain legacy comments with absolute paths. Pass dataset paths via Hydra overrides or
  environment variables rather than editing scripts directly.
- The real-world pipeline is tightly coupled to the specific hardware setup described in the paper.
- Bundled 3D assets in `scandp/diffusion_policy_3d/assets/` follow their respective license terms.
- This repository alone is not sufficient to reproduce every configuration used in the paper.

## Citation

```bibtex
@article{hirako2026scandp,
  title   = {ScanDP: Generalizable 3D Scanning with Diffusion Policy},
  author  = {Itsuki Hirako and Ryo Hakoda and Yubin Liu and Matthew Hwang and Yoshihiro Sato and Takeshi Oishi},
  year    = {2026},
  eprint  = {2603.10390},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url     = {https://arxiv.org/abs/2603.10390}
}
```

## License

See [LICENSE](LICENSE) for the main codebase.
