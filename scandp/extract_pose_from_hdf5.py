# python scandp/extract_pose_from_hdf5.py --hdf5 /home/cvl/cvl/ScanDP/demo_dir/5.h5 --output /home/cvl/cvl/ScanDP/demo_dir/demo_pth/5.pth

import h5py
import torch
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pose from hdf5 and save as demo_pose.pth")
    parser.add_argument('--hdf5', type=str, required=True, help='Path to input hdf5 file')
    parser.add_argument('--output', type=str, default='demo_pose.pth', help='Output .pth file path')
    args = parser.parse_args()

    # hdf5ファイルからactionデータセットを読み込む
    with h5py.File(args.hdf5, 'r') as f:
        if 'action' in f:
            action = f['action'][:]
        elif 'env_qpos_proprioception' in f:
            action = f['env_qpos_proprioception'][:]
        else:
            raise KeyError('No action or env_qpos_proprioception dataset found in hdf5 file')

    # torch.Tensorに変換
    action_tensor = torch.tensor(action)

    # 保存
    torch.save(action_tensor, args.output)
    print(f"Saved pose tensor of shape {action_tensor.shape} to {args.output}") 