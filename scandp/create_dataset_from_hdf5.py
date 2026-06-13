import asyncio
import os
import sys
import time

import cv2
import diffusion_policy_3d.module.gridmap as gridmap
import genesis as gs
import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat
from scipy.spatial.transform import Rotation as R
from skimage import filters
from skimage.util import random_noise
from termcolor import cprint
from tqdm import tqdm


class Teleoperator:
    def __init__(self, fps=50, hdf5_file_path=None):

        self.fps = fps
        self.IMAGE_SIZE = 224

        self.pos = np.zeros(3)
        self.quat = np.zeros(4)

        self.img_array = []
        self.depth_array = []
        self.cloud_array = []
        self.gridmap_array = []
        self.lock = asyncio.Lock()

        self.median_filter = True
        self.mapper = gridmap.LocalMap3D(
            X_lim=np.array([-0.4, 0.4]),
            Y_lim=np.array([-0.4, 0.4]),
            Z_lim=np.array([0.4, 1.2]),
            resolution=0.02,
            p=0.5,
            device="cuda"
        )
        self.pcd_total = o3d.geometry.PointCloud()

        self.hdf5_file = hdf5_file_path
        with h5py.File(self.hdf5_file, "r") as f:
            self.action_array = f["action"][:]
            print(f"load data from {self.hdf5_file}")
        print(f"action", self.action_array.shape)

        ##### initialize genesis #####
        gs.init(backend=gs.gpu, logging_level='warning')

        self.scene = gs.Scene(
            vis_options=gs.options.VisOptions(
                show_world_frame=False,   # Show xyz axes
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=True,
                ambient_light=(0.1, 0.1, 0.1),
            ),
        )

        self.cam = self.scene.add_camera(
            res=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            pos=(0.0, 1.0, 0.7),
            lookat=(0, 0, 0.7),
            fov=45,
            GUI=False,
        )

        ########################### entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane()
        )

        # WARNING: You must add bunny before drone
        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="/home/cvl/cvl/ScanDP/scandp/diffusion_policy_3d/assets/stanfordbunny.stl",
                pos=(0, 0, 0.7),
                scale=3.0,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.72, 0.72, 0.72),
            ),
        )

        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file="/home/cvl/cvl/ScanDP/scandp/diffusion_policy_3d/assets/drones/cf2x.urdf",
                pos=(0.0, -1.0, 0.02),
                visualization=False,
            ),
        )

        ########################## build ##########################
        n_envs = 1
        self.scene.build(n_envs=n_envs, env_spacing=(2.0, 2.0))

    def main(self):

        if True:
            #### some parameters ####
            self.save_img = True
            self.save_depth = True
            # length of demo
            length = 500
            demo_dir = "save_dir"
            demo_name = os.path.basename(self.hdf5_file).replace(".h5", "")

            #### initialize demo saving ####
            os.makedirs(demo_dir, exist_ok=True)
            record_file_name = os.path.join(demo_dir, demo_name + ".h5")
            env_qpos_array = []
            action_array = []
            pos_array = []
            quat_array = []
            # self.prev_pos = self.drone.get_pos().cpu().view(-1)
            # self.prev_quat = self.drone.get_quat().cpu().view(-1)

        for i in tqdm(range(length), desc="Processing frames"):
            start = time.time()

            self.scene.step()
            x, y, z = self.action_array[i, :3]
            quaternion = self.action_array[i, 3:]

            self.drone.set_pos(
                pos=torch.tensor([[x, y, z]]),
            )
            self.drone.set_quat(torch.tensor([quaternion]))

            cam_transform = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu(
            ).numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform)
            img, depth, segmentation, _ = self.cam.render(
                depth=True, segmentation=True)

            # create point cloud
            # segmentation_mask = segmentation == self.bunny.idx
            # depth_masked = np.where(depth, segmentation_mask, 0)
            depth_filtered = filters.median(depth, behavior="ndimage")
            depth_filtered = o3d.geometry.Image(
                depth_filtered.astype(np.float32))
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=self.IMAGE_SIZE,
                height=self.IMAGE_SIZE,
                fx=self.cam.intrinsics[0, 0],
                fy=self.cam.intrinsics[1, 1],
                cx=self.cam.intrinsics[0, 2],
                cy=self.cam.intrinsics[1, 2],
            )

            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_filtered,
                intrinsic,
                self.cam.extrinsics,
            )
            if len(pcd.points) < 4096:
                points = np.asarray(pcd.points)
                padding = np.zeros((4096 - len(points), 3))
                points_padded = np.vstack((points, padding))
                pcd.points = o3d.utility.Vector3dVector(points_padded)
            else:
                pcd = pcd.farthest_point_down_sample(4096)
            pcd_np = np.asarray(pcd.points)

            occ_grid_map = self.update_occupancy_gridmap(depth, segmentation)
            # print(occ_grid_map.shape) # torch.Size([1, 200, 200, 200])

            self.write_data(img, depth, pcd_np, occ_grid_map.cpu())
            # await self.write_data(img, depth, pcd_np)

            drone_pos = self.drone.get_pos().cpu().view(-1)
            drone_quat = self.drone.get_quat().cpu().view(-1)

            # TODO: Change to delta pos and quat
            # delta_pos = drone_pos - self.prev_pos
            # self.prev_pos = drone_pos # update previous position

            # drone_quat_R = R.from_quat(drone_quat)
            # self.prev_quat_R = R.from_quat(self.prev_quat)
            # delta_euler = drone_quat_R.as_euler('xyz', degrees=True) - self.prev_quat_R.as_euler('xyz', degrees=True)
            # self.prev_quat = drone_quat # update previous quaternion
            # action = torch.cat((delta_pos, delta_euler), dim=0).numpy()

            action = torch.cat((drone_pos, drone_quat), dim=0).numpy()
            action_array.append(action)
            env_qpos_array.append(action)

        # save the data
        with h5py.File(record_file_name, "w") as f:
            seq_length = len(action_array)
            img_array = np.array(self.img_array)[:seq_length]
            depth_array = np.array(self.depth_array)[:seq_length]
            cloud_array = np.array(self.cloud_array)[:seq_length]
            env_qpos_array = np.array(env_qpos_array)[:seq_length]
            action_array = np.array(action_array)[:seq_length]
            gridmap_array = np.array(self.gridmap_array)[:seq_length]

            f.create_dataset("color", data=img_array[:])
            f.create_dataset("depth", data=depth_array[:])
            f.create_dataset("cloud", data=cloud_array[:])
            f.create_dataset("env_qpos_proprioception", data=env_qpos_array[:])
            f.create_dataset("action", data=action_array[:])
            f.create_dataset(
                "gridmap", data=gridmap_array[:], compression='gzip')

            cprint(f"color shape: {img_array.shape}", "yellow")
            cprint(f"depth shape: {depth_array.shape}", "yellow")
            cprint(f"cloud shape: {cloud_array.shape}", "yellow")
            cprint(f"action shape: {action_array.shape}", "yellow")
            cprint(f"env_qpos shape: {env_qpos_array.shape}", "yellow")
            cprint(f"gridmap shape: {gridmap_array.shape}", "yellow")
            cprint(
                f"save data at step: {seq_length} in {record_file_name}", "yellow")

    def write_data(self, img, depth, point_cloud, occ_grid_map):
        self.img_array.append(img)
        self.depth_array.append(depth)
        self.cloud_array.append(point_cloud)
        self.gridmap_array.append(occ_grid_map)

    def update_occupancy_gridmap(self, depth, segmentation):

        segmentation_mask = segmentation == self.target.idx
        depth_filtered = np.where(segmentation_mask, depth, 0)

        # Apply a median filter to the depth image
        if self.median_filter:
            depth_filtered = filters.median(depth_filtered, behavior='ndimage')

        depth_image = o3d.geometry.Image(depth_filtered.astype(np.float32))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.IMAGE_SIZE,
            height=self.IMAGE_SIZE,
            fx=self.cam.intrinsics[0, 0],
            fy=self.cam.intrinsics[1, 1],
            cx=self.cam.intrinsics[0, 2],
            cy=self.cam.intrinsics[1, 2],
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
        # torch.Size([1, 200, 200, 200])
        occ_grid_map = self.mapper.to_prob_occ_map()
        return occ_grid_map


def process_hdf5_file(input_dir, output_dir):
    """Processes all HDF5 files in the given directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.h5'):
            file_path = os.path.join(input_dir, file_name)
            teleoperator = Teleoperator(fps=50, hdf5_file_path=file_path)
            asyncio.run(teleoperator.main())


if __name__ == "__main__":
    teleoperator = Teleoperator(
        fps=50, hdf5_file_path="/home/cvl/cvl/ScanDP/demo_dir/4.h5")
    asyncio.run(teleoperator.main())
    # print(os.getcwd())
    # input_directory = "/home/cvl/cvl/ScanDP/demo_dir"
    # output_directory = "./saved_data"
    # process_hdf5_file(input_directory, output_directory)
