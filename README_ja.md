# ScanDP: 拡散政策による汎化3Dスキャン

[![arXiv](https://img.shields.io/badge/arXiv-2603.10390-b31b1b.svg)](https://arxiv.org/abs/2603.10390)

本リポジトリは以下の論文の公式コードです：

> **ScanDP: Generalizable 3D Scanning with Diffusion Policy**  
> Hirako Itsuki, Hakoda Ryo, Liu Yubin, Matthew Hwang, Sato Yoshihiro, Oishi Takeshi  
> arXiv:2603.10390

[English README](README.md)

![ScanDP overview](docs/imgs/art.png)

## 概要

ScanDP は、拡散政策（Diffusion Policy）による模倣学習を用いた 3D 物体スキャンのためのロボット学習システムです。
初期視点から出発し、ターンテーブルで物体を回転させながらロボットアームが物体の周囲を移動して 3D 点群を取得することで、未見物体にも汎化したカバレッジを実現します。

システムは **ROS Noetic** の catkin ワークスペース上で動作し、以下を統合しています：

- **i611** 6 自由度ロボットアーム
- **DS102** 電動ターンテーブル
- **Intel RealSense** RGB-D カメラ
- **RTAB-Map** リアルタイムマッピング・位置推定
- **Genesis** 物理シミュレータ（訓練時評価）

## 含まれるもの

| コンポーネント | パス | 説明 |
|---|---|---|
| 学習コード | `scandp/train.py` | Hydra を使った ScanDP / DP3 / IDP3 の訓練 |
| 実機デプロイ | `scandp/deploy_real.py` | 実ロボットでのポリシー推論 |
| シミュレーション評価 | `scandp/deploy_gridmap.py`, `deploy_demo.py` | Genesis ベースの評価 |
| データセットパイプライン | `scandp/create_dataset_real.py`, `create_dataset.py`, `convert_demos.py` | HDF5 収録 → zarr 変換 |
| ROS ノード | `src/i611_ros/scripts/` | i611 制御・DS102・グリッドマップ・MoveIt ブリッジ |
| Hydra 設定 | `scandp/diffusion_policy_3d/config/` | 全訓練・タスク設定 |
| 評価ツール | `scandp/ground_truth/` | カバレッジ評価ユーティリティ |
| ヘルパースクリプト | `scripts/` | よく使うワークフローのラッパー |

## 含まれないもの

- 論文で使用した完全なデータセット
- 事前学習済みモデルの重み
- i611 ロボット側 XML-RPC サーバー（独自実装、ハードウェア依存）
- DS102 TCP サーバー（独自実装、ハードウェア依存）
- 研究室固有のキャリブレーションファイルおよびネットワーク設定

実世界の完全再現にはこれら外部コンポーネントが必要です。

## リポジトリ構成

```
.
├── scandp/                          訓練・評価・データパイプライン
│   ├── train.py                     訓練エントリポイント
│   ├── deploy_real.py               実世界ポリシーデプロイ
│   ├── deploy_gridmap.py            Genesis シミュレーション評価（主要）
│   ├── deploy_demo.py               記録済み軌跡からの Genesis デモ評価
│   ├── create_dataset_real.py       実システムからのデモ収録（ROS サブスクライバ）
│   ├── create_dataset.py            HDF5 収録を デモチャンクに変換
│   ├── convert_demos.py             デモチャンクを zarr データセットに変換
│   ├── vis_dataset.py               データセット可視化
│   └── diffusion_policy_3d/
│       ├── config/                  Hydra 設定ファイル
│       ├── model/                   ネットワークアーキテクチャ
│       ├── policy/                  ポリシーラッパー
│       ├── workspace/               訓練・デプロイワークスペースクラス
│       ├── dataset/                 zarr データセットローダ
│       ├── module/                  グリッドマップ・特徴抽出器・ユーティリティ
│       ├── common/                  共通ユーティリティ（リプレイバッファ等）
│       ├── tools/                   可視化・分析ツール
│       └── assets/                  Genesis シミュレーション用 3D メッシュアセット
├── src/
│   ├── i611_ros/                    ROS パッケージ：i611・DS102・グリッドマップノード
│   ├── i611_ros_control/            ROS Control ハードウェアインタフェース
│   └── i611_moveit_config/          i611 の MoveIt 設定
├── scripts/                         訓練・評価用シェルスクリプトラッパー
├── test/                            開発・診断スクリプト
└── docs/                            補足ノートと図
```

## 動作要件

### ソフトウェア
- Ubuntu 20.04 + **ROS Noetic**
- Python ≥ 3.11
- CUDA 対応 GPU（CUDA 11.8 で動作確認済み）
- [`uv`](https://github.com/astral-sh/uv) パッケージマネージャ

### Python 依存パッケージ（`uv` で管理）
PyTorch, Open3D, Hydra, spconv, WandB, zarr, h5py, diffusers など（詳細は `pyproject.toml` を参照）。

### シミュレーション評価に必要なもの
- [Genesis](https://genesis-world.readthedocs.io/) 物理シミュレータ

### 実ロボットデプロイに必要なもの
- Intel RealSense SDK および `realsense2_camera` ROS パッケージ
- RTAB-Map（`rtabmap_ros`）
- i611 XML-RPC サーバー（ハードウェア側、本リポジトリには含まれない）
- DS102 TCP サーバー（ハードウェア側、本リポジトリには含まれない）
- 物体セグメンテーションモジュール（未発表のため非公開。本リポジトリでは RealSense の生点群トピック `/camera/depth/color/points` を使用）

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/Treeitsuki/ScanDP.git
cd ScanDP
```

### 2. Python 依存パッケージのインストール

```bash
uv sync
source .venv/bin/activate
uv pip install --extra-index-url https://rospypi.github.io/simple rospy-all
uv pip install setuptools==81.0.0
```

### 3. ROS ワークスペースのビルド

```bash
source /opt/ros/noetic/setup.bash
catkin build
source devel/setup.bash
```

追加の環境設定については [docs/SETUP.md](docs/SETUP.md) を参照してください。

## クイックスタート：シミュレーション評価

物理ハードウェアなしで ScanDP を試す最も簡単な方法は Genesis ベースのシミュレータです。

### 1. データセットの準備

実ロボットデータがない場合は、収録済み HDF5 ファイルや rosbag リプレイから `create_dataset.py` を使って合成データセットを作成できます（詳細は [docs/USAGE_ja.md](docs/USAGE_ja.md) を参照）。

### 2. 訓練

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

### 3. シミュレーション評価

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

## 利用可能な設定

| 設定ファイル | 説明 |
|---|---|
| `scandp_spconv.yaml` | ScanDP with Sparse 3D Convolution（SpConv）— 主要手法 |
| `scandp_conv3d.yaml` | ScanDP with Dense 3D Convolution |
| `scandp_img_r3m.yaml` | 画像のみベースライン（R3M 特徴量） |
| `dp3.yaml` | DP3（3D Diffusion Policy）ベースライン |
| `idp3.yaml` | IDP3 ベースライン |

タスク：標準グリッドマップ観測空間には `task=cam_gridmap` を使用してください。

## データセット作成パイプライン

[docs/USAGE_ja.md](docs/USAGE_ja.md) の **データセットパイプライン** セクションを参照してください。以下の手順をカバーしています：

1. 実システムまたは rosbag からのデモ収録
2. HDF5 収録からデモチャンクへの変換
3. デモチャンクから zarr データセットへの変換

## 実ロボットデプロイ

[docs/USAGE_ja.md](docs/USAGE_ja.md) の **実ロボットデプロイ** セクションを参照してください。

## 既知の制限事項

- 一部スクリプトの旧コメントに絶対パスが残っています。スクリプトを直接編集せず、Hydra のオーバーライドや環境変数でパスを指定してください。
- 実世界パイプラインは論文に記載の特定ハードウェア構成と密に結合しています。
- `scandp/diffusion_policy_3d/assets/` 内の 3D アセットはそれぞれのライセンスに従います。
- 本リポジトリ単体では論文で使用した全設定の再現には不十分です。

## 引用

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

## ライセンス

本コードベースのライセンスは [LICENSE](LICENSE) を参照してください。
