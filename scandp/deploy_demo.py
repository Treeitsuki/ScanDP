import numpy as np
import genesis as gs
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat
import torch
import open3d as o3d
import os
import time
from skimage import filters
import pandas as pd
import glob
import csv
import argparse

# 追加: occupancy grid mapやデータ記録用のクラス
import diffusion_policy_3d.module.gridmap as gridmap
from ground_truth.coverage_gt import eval_inference

DEPLOY_LENGTH = 500
INIT_POS = [0.0, 1.0, 0.7]
SCALE = 1
TARGET = "spot"  # "bunny", "spot", "armadillo", "teapot", "dragon", "bust", "bike", "happy"
NAME = "demo"

class CameraInference:
    def __init__(self, image_size=512, target=TARGET, scale=SCALE):
        self.IMAGE_SIZE = image_size
        self.median_filter = True
        self.add_noise = False
        self.pcd_total = o3d.geometry.PointCloud()
        self.csv_list = []
        self.path_sum = 0
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.gridmap_array = []
        self.action_array = []
        self.target_name = target
        self.scale = scale

        # ターゲット設定
        self.target_map = {
            "bunny": dict(
                file="stanford-bunny.obj", pos=(0.05, 0, 0.5), scale=1.0, quat=(1, 1, 0, 0)
            ),
            "spot": dict(
                file="spot.obj", pos=(0, 0.05, 0.8), scale=0.28, quat=(1, 1, 0, 0)
            ),
            "armadillo": dict(
                file="armadillo.obj", pos=(0, 0, 0.8), scale=0.0025, quat=(1, 1, 0, 0)
            ),
            "teapot": dict(
                file="teapot.obj", pos=(0, 0, 0.55), scale=0.07, quat=(1, 1, 0, 0)
            ),
            "dragon": dict(
                file="dragon.obj", pos=(0, 0, 0.8), scale=0.5, quat=(1, 1, 0, 0)
            ),
            "bust": dict(
                file="bust.obj", pos=(0, 0, 0.8), scale=0.002, quat=(1, 0, 0, 0)
            ),
            "bike": dict(
                file="bike.obj", pos=(0.0, 0.11, 0.45), scale=0.00023, quat=(1, 1, 0, 0)
            ),
            "happy": dict(
                file="happy.obj", pos=(0, 0, 0.4), scale=3, quat=(1, 1, 0, 0)
            ),
        }
        self.asset_dir = "/home/cvl/cvl/ScanDP/scandp/diffusion_policy_3d/assets/"
        self.mesh_cfg = self.target_map[self.target_name]

        # occupancy grid map
        self.mapper = gridmap.LocalMap3D(
            X_lim=np.array([-0.4, 0.4]),
            Y_lim=np.array([-0.4, 0.4]),
            Z_lim=np.array([0.4, 1.2]),
            resolution=0.02,
            p=0.5,
            device="cuda"
        )

        ##### initialize genesis #####
        gs.init(
            backend=gs.gpu,
            logging_level='warning',
            )

        self.scene = gs.Scene(
            vis_options = gs.options.VisOptions(
                show_world_frame = False,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = True,
                plane_reflection = True,
                ambient_light    = (0.1, 0.1, 0.1),
            ),
        )

        self.cam = self.scene.add_camera(
            res    = (self.IMAGE_SIZE, self.IMAGE_SIZE),
            pos    = (0.0, 1.0, 0.7),
            lookat = (0, 0, 0.7),
            fov    = 45,
            GUI    = False,
        )

        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.target = self.scene.add_entity(
            morph = gs.morphs.Mesh(
                file = os.path.join(self.asset_dir, self.mesh_cfg["file"]),
                pos = self.mesh_cfg["pos"],
                quat = self.mesh_cfg["quat"],
                scale = self.mesh_cfg["scale"] * self.scale,
                fixed = True,
            ),
            surface = gs.surfaces.Default(
                color = (0.82, 0.82, 0.82),
            ),
        )
        self.drone = self.scene.add_entity(
            morph = gs.morphs.Drone(
                file = os.path.join(self.asset_dir, "drones/cf2x.urdf"),
                pos = tuple(INIT_POS),
                visualization = False,
            ),
        )
        self.scene.build(n_envs = 1, env_spacing = (2.0, 2.0))

    def reset(self, demo_pose):
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.gridmap_array = []
        self.action_array = []
        self.pcd_total = o3d.geometry.PointCloud()
        self.path_sum = 0
        # 初期位置
        self.drone.set_pos(torch.tensor([INIT_POS]))
        self.drone.set_quat(xyz_to_quat(torch.tensor([[90, 0, 0]])))
        cam_transform = trans_quat_to_T(np.array([0, 0, 0]), xyz_to_quat(np.array([90, 0, 0])))
        self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu().numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform,)
        img, depth, segmentation, _ = self.cam.render(depth=True, segmentation=True)
        self.color_array.append(img)
        self.depth_array.append(depth)
        self.gridmap_array.append(self.update_occupancy_gridmap(depth, segmentation).cpu())
        env_qpos = np.concatenate([self.drone.get_pos().cpu().view(-1), self.drone.get_quat().cpu().view(-1)])
        self.env_qpos_array.append(env_qpos)
        print("Drone ready!")

    def update_occupancy_gridmap(self, depth, segmentation):
        segmentation_mask = segmentation == self.target.idx
        depth_filtered = np.where(segmentation_mask, depth, 0)
        if self.median_filter:
            depth_filtered = filters.median(depth_filtered, behavior='ndimage')
        depth_image = o3d.geometry.Image(depth_filtered)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width = self.IMAGE_SIZE,
            height = self.IMAGE_SIZE,
            fx = self.cam.intrinsics[0, 0],
            fy = self.cam.intrinsics[1, 1],
            cx = self.cam.intrinsics[0, 2],
            cy = self.cam.intrinsics[1, 2],
        )
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsic,
            self.cam.extrinsics,
        )
        self.pcd_total += pcd
        points = torch.tensor(np.array(pcd.points), device="cuda")
        self.mapper.update(
            torch.tensor(self.cam.pos[0], device="cuda"),
            torch.tensor(self.cam.pos[1], device="cuda"),
            torch.tensor(self.cam.pos[2], device="cuda"),
            points, 
            p_free=0.4, 
            p_occ=0.7
        )
        occ_grid_map = self.mapper.to_prob_occ_map()
        return occ_grid_map

    def step(self, pos, quat):
        self.scene.step()
        self.drone.set_pos(pos)
        self.drone.set_quat(quat)
        cam_transform = trans_quat_to_T(np.array([0, 0, 0]), xyz_to_quat(np.array([90, 0, 0])))
        self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu().numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform,)
        img, depth, segmentation, _ = self.cam.render(depth=True, segmentation=True)
        self.color_array.append(img)
        self.depth_array.append(depth)
        self.gridmap_array.append(self.update_occupancy_gridmap(depth, segmentation).cpu())
        env_qpos = np.concatenate([self.drone.get_pos().cpu().view(-1), self.drone.get_quat().cpu().view(-1)])
        self.env_qpos_array.append(env_qpos)
        # 距離計算
        if len(self.env_qpos_array) > 1:
            prev_pos = torch.tensor(self.env_qpos_array[-2][:3]).cuda()
            distance = torch.linalg.norm(self.drone.get_pos() - prev_pos)
            self.path_sum += distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--pth", required=True)
    parser.add_argument("--csv", default="results_summary.csv")
    args = parser.parse_args()

    demo_pose = torch.load(args.pth)
    env = CameraInference(image_size=512, target=args.target, scale=args.scale)
    env.reset(demo_pose)
    step = 0
    env.csv_list.append((0, 0, 0))
    for i in range(DEPLOY_LENGTH):
        pos = demo_pose[step, :3].unsqueeze(0)
        quat = demo_pose[step, 3:].unsqueeze(0)
        env.step(pos, quat)
        step = (step + 1) % demo_pose.shape[0]
    final_cover_ratio, pcd_final = eval_inference(
        env.target_name,
        env.pcd_total,
        scale=args.scale,
        threshold_icp=1,
        threshold_dis=0.002,
        visualize=False,
        noise_remove=False,
    )
    print(f"[DONE] {os.path.basename(args.pth)} | TARGET={args.target} | SCALE={args.scale} | Final Coverage ratio: {final_cover_ratio:.4f}")

    # CSV追記
    write_header = not os.path.exists(args.csv)
    with open(args.csv, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "target", "scale", "final_coverage"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "file": os.path.basename(args.pth),
            "target": args.target,
            "scale": args.scale,
            "final_coverage": final_cover_ratio
        })