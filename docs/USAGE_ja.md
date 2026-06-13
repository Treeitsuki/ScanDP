# ScanDP 使用ガイド

本ドキュメントでは、ScanDP の全パイプライン（環境セットアップ、データセット作成、訓練、シミュレーション評価、実ロボットデプロイ）について手順を説明します。

**English version: [USAGE.md](USAGE.md)**

---

## 目次

1. [環境セットアップ](#1-環境セットアップ)
2. [データセットパイプライン](#2-データセットパイプライン)
   - [2.1 デモの収録（実システム）](#21-デモの収録実システム)
   - [2.2 HDF5 からデモチャンクへの変換](#22-hdf5-からデモチャンクへの変換)
   - [2.3 デモチャンクから zarr への変換](#23-デモチャンクから-zarr-への変換)
   - [2.4 データセットの可視化](#24-データセットの可視化)
3. [訓練](#3-訓練)
   - [3.1 設定ファイルの概要](#31-設定ファイルの概要)
   - [3.2 訓練の実行](#32-訓練の実行)
   - [3.3 訓練のモニタリング](#33-訓練のモニタリング)
4. [シミュレーション評価（Genesis）](#4-シミュレーション評価genesis)
   - [4.1 訓練済みチェックポイントの評価](#41-訓練済みチェックポイントの評価)
   - [4.2 記録済み軌跡からの評価](#42-記録済み軌跡からの評価)
5. [実ロボットデプロイ](#5-実ロボットデプロイ)
   - [5.1 ハードウェアの起動順序](#51-ハードウェアの起動順序)
   - [5.2 ROS コンポーネントの起動](#52-ros-コンポーネントの起動)
   - [5.3 ポリシー推論の実行](#53-ポリシー推論の実行)
6. [ROS ノードリファレンス](#6-ros-ノードリファレンス)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. 環境セットアップ

### 前提条件

| 項目 | バージョン |
|---|---|
| OS | Ubuntu 20.04 |
| ROS | Noetic |
| Python | ≥ 3.11 |
| CUDA | 11.8（動作確認済み） |
| パッケージマネージャ | [`uv`](https://github.com/astral-sh/uv) |

### Python 環境のインストール

```bash
# リポジトリルートで実行
uv sync
source .venv/bin/activate

# 仮想環境に ROS Python バインディングを追加
uv pip install --extra-index-url https://rospypi.github.io/simple rospy-all
uv pip install setuptools==81.0.0
```

### Genesis のインストール（シミュレーションのみ）

[Genesis インストールガイド](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html) に従ってインストールしてください。

### ROS ワークスペースのビルド

```bash
source /opt/ros/noetic/setup.bash
catkin build
source devel/setup.bash
```

> **注意:** このワークスペースは ROS Noetic と Python 3 仮想環境を組み合わせて使用します。
> 必ず ROS セットアップファイルを source した *後に* 仮想環境を有効化してください。

### 推奨シェル初期化

`~/.bashrc` に追記するか、セッション開始時に手動で実行：

```bash
source /opt/ros/noetic/setup.bash
source /path/to/ScanDP/devel/setup.bash
source /path/to/ScanDP/.venv/bin/activate
```

---

## 2. データセットパイプライン

データパイプラインは 3 つのステージで構成されます：

```
実システム / rosbag
        │
        ▼
  HDF5 収録ファイル (.h5)       ← create_dataset_real.py
        │
        ▼
  デモチャンク（エピソード単位）  ← create_dataset.py
        │
        ▼
  zarr データセット              ← convert_demos.py
```

### 2.1 デモの収録（実システム）

`create_dataset_real.py` は以下のセンサートピックを購読し、センサーフレームを同期して HDF5 ファイルに書き込む ROS ノードです。

**必要な ROS トピック：**

| トピック | 型 | 説明 |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB 画像 |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | アライン済み深度画像 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | カメラ内部パラメータ |
| `/camera/depth/color/points` | `sensor_msgs/PointCloud2` | RealSense 点群 |
| TF: `map → camera_link` | — | ワールド座標系でのカメラ姿勢 |

**実行：**

```bash
source devel/setup.bash
python scandp/create_dataset_real.py \
  _hdf5_out_path:=data/record_001.h5 \
  _seq_length:=150
```

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `_hdf5_out_path` | `data/record.h5` | 出力ファイルパス |
| `_seq_length` | `150` | エピソードあたりのフレーム数 |

> ノードは 1 エピソードを収録して終了します。デモごとに繰り返し実行してください。

### 2.2 HDF5 からデモチャンクへの変換

```bash
python scandp/create_dataset.py \
  data/record_001.h5 \
  --output-dir save_dir/
```

HDF5 ファイル 1 つに対して 1 回実行します。各エピソードは `save_dir/` 以下の個別サブディレクトリに保存されます。

**バッチ処理：**

```bash
for f in data/hdf5/*.h5; do
  python scandp/create_dataset.py "$f" --output-dir save_dir/
done
```

### 2.3 デモチャンクから zarr への変換

```bash
python scandp/convert_demos.py \
  --demo_dir save_dir/ \
  --save_dir dataset/gridmap_real \
  --save_img 1 \
  --save_depth 1
```

| 引数 | 説明 |
|---|---|
| `--demo_dir` | エピソード単位のデモチャンクが入ったディレクトリ |
| `--save_dir` | 出力 zarr データセットのディレクトリ |
| `--save_img` | RGB 画像を zarr に保存する（`1` = する） |
| `--save_depth` | 深度画像を zarr に保存する（`1` = する） |

生成された `dataset/gridmap_real/` が訓練・評価時に指定する zarr パスです。

### 2.4 データセットの可視化

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

## 3. 訓練

### 3.1 設定ファイルの概要

訓練は [Hydra](https://hydra.cc/) で設定します。主な設定ファイルは以下の通りです：

```
scandp/diffusion_policy_3d/config/
├── scandp_spconv.yaml      ← アルゴリズム設定（ScanDP + SpConv、主要手法）
├── scandp_conv3d.yaml      ← Dense 3D Conv バリアント
├── scandp_img_r3m.yaml     ← 画像のみベースライン
├── dp3.yaml                ← DP3 ベースライン
├── idp3.yaml               ← IDP3 ベースライン
└── task/
    └── cam_gridmap.yaml    ← タスク・観測空間設定
```

**`cam_gridmap.yaml` の主要パラメータ：**

| パラメータ | 説明 |
|---|---|
| `shape_meta` | 観測・行動の次元 |
| `dataset.zarr_path` | zarr データセットのパス |
| `dataset.max_train_episodes` | 訓練に使用するエピソード数 |

**アルゴリズム設定（例：`scandp_spconv.yaml`）の主要パラメータ：**

| パラメータ | 説明 |
|---|---|
| `horizon` | 予測ホライゾン |
| `n_action_steps` | 実行する行動ステップ数 |
| `n_obs_steps` | 観測ステップ数 |
| `policy.noise_scheduler` | DDIM スケジューラのパラメータ |

設定ファイルを直接編集せず、コマンドラインからパラメータをオーバーライドできます：

```bash
python train.py --config-name=scandp_spconv.yaml \
  task.dataset.max_train_episodes=50
```

### 3.2 訓練の実行

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

**重要なパラメータ：**

| パラメータ | 説明 |
|---|---|
| `--config-name` | アルゴリズム設定（`scandp_spconv`, `dp3` など） |
| `task` | タスク設定（`cam_gridmap`） |
| `hydra.run.dir` | ログとチェックポイントの出力ディレクトリ |
| `training.seed` | 乱数シード |
| `training.device` | PyTorch デバイス（`cuda:0`, `cuda:1` など） |
| `exp_name` | 実験名（WandB で使用） |
| `logging.mode` | WandB モード：`online` または `offline` |
| `checkpoint.save_ckpt` | チェックポイントを保存するか否か |
| `task.dataset.zarr_path` | zarr データセットのパス |

チェックポイントは `hydra.run.dir/checkpoints/` に保存されます。

### 3.3 訓練のモニタリング

`logging.mode=online` の場合、メトリクスは [WandB](https://wandb.ai/) に表示されます。
オフラインモードの場合、ログは `hydra.run.dir/logs/` に書き込まれます。

---

## 4. シミュレーション評価（Genesis）

Genesis シミュレーションは `scandp/diffusion_policy_3d/assets/` に格納された 3D メッシュアセットを使用します。

### 4.1 訓練済みチェックポイントの評価

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

スクリプトは `hydra.run.dir/checkpoints/` から最新のチェックポイントを自動的に読み込みます。
結果（カバレッジメトリクスと点群出力）は `hydra.run.dir/` に保存されます。

**利用可能なターゲット物体**（スクリプト内の `TARGET` 変数または設定オーバーライドで指定）：

`bunny`, `armadillo`, `spot`, `dragon`, `bust`, `bike`, `happy`, `teapot`

### 4.2 記録済み軌跡からの評価

`deploy_demo.py` は記録済みの軌跡（`.pth` ファイル）を再生してカバレッジを測定します：

```bash
cd scandp

python deploy_demo.py \
  --target bunny \
  --scale 1 \
  --pth /path/to/demo.pth \
  --csv results.csv
```

| 引数 | 説明 |
|---|---|
| `--target` | 物体名（上記リスト参照） |
| `--scale` | 物体スケール係数（例：`1` または `1.5`） |
| `--pth` | 記録済み軌跡 `.pth` ファイルのパス |
| `--csv` | 結果を書き込む CSV ファイル |

---

## 5. 実ロボットデプロイ

> **必要なハードウェア：** i611 アーム、DS102 ターンテーブル、Intel RealSense、
> および i611・DS102 それぞれの独自サーバープログラム。これらのサーバープログラムは本リポジトリに含まれていません。

### 5.1 ハードウェアの起動順序

初期化失敗を防ぐため、以下の順序でコンポーネントを起動します：

```
1. i611 XML-RPC サーバー（ロボットコントローラ PC 側）
2. DS102 TCP サーバー（ターンテーブルコントローラ PC 側）
3. Intel RealSense → realsense2_camera
4. RTAB-Map
5. MoveIt（i611_moveit_config）
6. i611_ros の ROS ノード群
7. ポリシーデプロイノード
```

### 5.2 ROS コンポーネントの起動

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
roslaunch i611_moveit_config real.launch robot_ip:=<ロボットIPアドレス>
```

`<ロボットIPアドレス>` を実際の i611 コントローラの IP に置き換えてください。

#### i611_ros ノード群

```bash
# DS102 ターンテーブルドライバ
rosrun i611_ros ds102_driver.py _server_ip:=<DS102のIPアドレス>

# グリッドマップ生成ノード（RealSense の /camera/depth/color/points を使用）
rosrun i611_ros gridmap_node.py
```

`<DS102のIPアドレス>` を実際の DS102 TCP サーバーの IP に置き換えてください。

### 5.3 ポリシー推論の実行

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

ノードは訓練済みチェックポイントを読み込み、`/follow_joint_trajectory` 経由で MoveIt にジョイント目標を送信します。
ポリシー出力（デカルト座標ウェイポイント）は `/policy/pose`、`/policy/path`、`/policy/poses` にも配信されます。

---

## 6. ROS ノードリファレンス

### `src/i611_ros/scripts/` のノード一覧

| ノード | 説明 | 主なトピック・サービス |
|---|---|---|
| `i611_driver.py` | i611 ロボットコントローラへの XML-RPC ブリッジ | 購読：関節コマンド；配信：`/joint_states` |
| `i611_interface.py` | 高レベル i611 コマンドインタフェース | `i611_driver.py` が内部で使用 |
| `ds102_driver.py` | DS102 ターンテーブルドライバ（TCP クライアント） | 配信：`/ds102/current_angle`, `/ds102/is_moving`, `/ds102/move_done`；購読：`/ds102/command_angle` |
| `gridmap_node.py` | 点群から 3D 占有グリッドマップを生成 | 購読：`/camera/depth/color/points`, TF `map→camera_link`；配信：グリッドマップトピック |
| `pc_align_node.py` | 点群取得と ICP アライメント | 購読：`/camera/depth/color/points` + 完了通知トピック；配信：アライン済み点群 |
| `ee_state_node.py` | エンドエフェクタ状態配信（関節状態からの FK） | 配信：`/ee_state` |
| `trans_node.py` | ターンテーブルフレームの TF ブロードキャスタ | 配信：`ds102_link` TF |
| `moveit_done_bridge.py` | MoveIt アクション完了を `/ds102` スタイルのトピックへ変換 | — |
| `policy_path_node.py` | ポリシー出力を ROS Path メッセージに変換 | 配信：`/policy/path` |
| `publish_policy_poses.py` | ポリシー姿勢を PoseArray として配信 | 配信：`/policy/poses` |
| `init_positions.py` | アームを定義済みの初期姿勢に移動 | — |

### 主な TF フレーム

| フレーム | 説明 |
|---|---|
| `map` | ワールドフレーム（RTAB-Map 原点） |
| `base_link` | i611 ベース |
| `Link6` | i611 エンドエフェクタ |
| `camera_link` | RealSense カメラ光学フレーム |
| `ds102_link` | DS102 ターンテーブル中心 |

### 主な ROS トピック

| トピック | 型 | 方向 | 説明 |
|---|---|---|---|
| `/camera/color/image_raw` | `Image` | 入力 | RGB 画像 |
| `/camera/aligned_depth_to_color/image_raw` | `Image` | 入力 | 深度画像 |
| `/camera/color/camera_info` | `CameraInfo` | 入力 | カメラ内部パラメータ |
| `/camera/depth/color/points` | `PointCloud2` | 入力 | RealSense 生点群 |
| `/ds102/command_angle` | `Float32` | 入力 | 目標角度（度） |
| `/ds102/current_angle` | `Float32` | 出力 | 現在角度 |
| `/ds102/move_done` | `Bool` | 出力 | 動作完了フラグ |
| `/joint_states` | `JointState` | 出力 | 関節位置 |
| `/policy/pose` | `PoseStamped` | 出力 | ポリシー目標姿勢 |
| `/policy/path` | `Path` | 出力 | 計画経路 |

---

## 7. トラブルシューティング

よくあるエラーは [ERROR_CATCH.md](ERROR_CATCH.md) も参照してください。

### データセット関連

**グリッドマップが空またはゼロのまま**  
`/camera/depth/color/points` が配信されているか、TF `map→camera_link` が利用可能かを確認してください。

**zarr 変換でシェイプの不一致エラー**  
バッチ内の全 HDF5 ファイルの `seq_length` が同じである必要があります。途中で切れたファイルは再収録してください。

### 訓練関連

**CUDA メモリ不足**  
`training.batch_size` を小さくするか、より小さいモデルバリアント（`scandp_conv3d`）に変更してください。

**WandB 認証エラー**  
`logging.mode=offline` を指定して WandB をスキップするか、訓練前に `wandb login` を実行してください。

### シミュレーション関連

**Genesis 起動時にセグメンテーションフォルト**  
Genesis のバージョンが本コードベースでテスト済みのものと一致しているか確認してください。
`CUDA_VISIBLE_DEVICES` が有効な GPU インデックスに設定されているかも確認してください。

**カバレッジメトリクスが 0**  
ターゲット物体名（`TARGET`）が `scandp/diffusion_policy_3d/assets/ply/` 内のアセットファイルと一致しているか確認してください。

### 実ロボット関連

**MoveIt が "no motion plan found" を報告**  
デプロイ前にロボットが設定済みの初期姿勢付近にあることを確認してください。
まず `scandp/create_dataset_real.py` を `_seq_length:=1` で実行してセンサーの接続性を確認してください。

**DS102 が応答しない**  
TCP サーバーが起動しているか、起動コマンドの `<DS102のIPアドレス>` が実際のアドレスと一致しているか確認してください。
`nc -zv <DS102のIPアドレス> 5000` で接続を確認できます。

