import os 
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import genesis as gs
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat
import hydra
import numpy as np
import open3d as o3d
import pathlib
import time
import torch
import tqdm
from omegaconf import OmegaConf
from termcolor import cprint
from skimage import filters, util
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pandas as pd


from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import diffusion_policy_3d.module.gridmap as gridmap
import diffusion_policy_3d.module.vis_trajectory as vis_trajectory
from diffusion_policy_3d.module.waypoint_extraction.extract_waypoints import dp_waypoint_selection
from ground_truth.coverage_gt import eval_inference
from diffusion_policy_3d.module.bubble import compute_bubble

os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

DEPLOY_LENGTH = 500
INIT_POS = [0, -1, 0.7]
TARGET = "bunny"  # "bunny", "spot", "armadillo", "teapot", "dragon", "bust", "bike", "happy"
NAME = "spconv" # "dp", "spconv"
SCALE = 1
use_waypoints = True

class CameraInference:
    def __init__(self, obs_horizon=2, 
                 action_horizon=8, 
                 device="gpu",
                 use_point_cloud=False, num_points=4096,
                 use_image=True, img_size=224,
                 use_gridmap=True,
                 use_waist=False):
        
        # obs/action
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        self.use_gridmap = use_gridmap
        
        self.use_waist = use_waist
        
        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        
        # inference device
        if device == "gpu":
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")
        
        # self.mapper = gridmap.LocalMap3D(
        #     X_lim=np.array([-0.25, 0.25]),
        #     Y_lim=np.array([-0.25, 0.25]),
        #     Z_lim=np.array([0.5, 1.0]),
        #     resolution=0.02,
        #     p=0.5,
        #     device="cuda"
        # )
        self.mapper = gridmap.LocalMap3D(
            X_lim=np.array([-0.4, 0.4]),
            Y_lim=np.array([-0.4, 0.4]),
            Z_lim=np.array([0.4, 1.2]),
            resolution=0.02,
            p=0.5,
            device="cuda"
        )
        self.IMAGE_SIZE_W = 224
        self.IMAGE_SIZE_H = 224
        self.median_filter = True
        self.add_noise = False
        self.pcd_total = o3d.geometry.PointCloud()
        self.csv_list = []
        self.path_sum = 0
        
        ##### initialize genesis #####
        gs.init(backend=gs.gpu)

        self.scene = gs.Scene(
            vis_options = gs.options.VisOptions(
                show_world_frame = False,   # Show xyz axes
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = False,
                plane_reflection = True,
                ambient_light    = (0.1, 0.1, 0.1),
            ),
        )

        self.cam = self.scene.add_camera(
            res    = (self.IMAGE_SIZE_W, self.IMAGE_SIZE_H),
            pos    = (0.0, 1.0, 0.7),
            lookat = (0, 0, 0.7),
            fov    = 45,
            GUI    = False,
        )

        # MARK: target
        ########################### entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane()
        )

        # WARNING: You must add bunny before drone 
        self.target_map = {
            "bunny": dict(
                file="stanfordbunny.stl", pos=(0, 0, 0.8), scale=3.0, quat=(1, 0, 0, 0)
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
        # mesh_cfg = target_map[self.cfg.target_mesh.name]
        self.mesh_cfg = self.target_map[TARGET]
        self.asset_dir = "/home/cvl/cvl/ScanDP/scandp/diffusion_policy_3d/assets/"

        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file=os.path.join(self.asset_dir, self.mesh_cfg["file"]),
                pos=self.mesh_cfg["pos"],
                scale=self.mesh_cfg["scale"]*SCALE,
                fixed=True,
                quat=self.mesh_cfg.get("quat", None)
            ),
            surface=gs.surfaces.Default(color=(0.82, 0.82, 0.82))
        )

        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file="/home/cvl/cvl/ScanDP/scandp/diffusion_policy_3d/assets/drones/cf2x.urdf",
                pos=(0.0, -1.0, 0.02),
                visualization=False,
            ),
        )

        # self.pedestal = self.scene.add_entity(
        #     gs.morphs.Box(
        #         pos = (0, 0, 0.25),
        #         # size = (0.6, 0.6, 0.56),    #  x1
        #         size = (0.6, 0.6, 0.35),
        #         collision = True,
        #         fixed = True,
        #         visualization = True,
        #     )
        # )

        ########################## build ##########################
        n_envs = 1
        self.scene.build(n_envs = n_envs, env_spacing = (2.0, 2.0))
    
    
    # MARK: 
    def step(self, action_list):
        
        for action_id in range(self.action_horizon):
            self.scene.step()
            act = action_list[action_id]
            self.action_array.append(act)
            
            # #### Changed ####
            # drone_pos = self.prev_pos + torch.tensor(np.array(act[:3]))
            # self.drone.set_pos(drone_pos.unsqueeze(0))  # Reshape to 2D tensor
            # # delta_euler = R.from_euler('xyz', act[3:], degrees=True).as_euler('xyz', degrees=True)
            # # prev_euler = R.from_quat(self.prev_quat).as_euler('xyz', degrees=True)
            # # drone_euler = prev_euler + delta_euler
            # # drone_quat = R.from_euler('xyz', drone_euler, degrees=True).as_quat()
            # # drone_quat = torch.tensor(drone_quat).unsqueeze(0)

            # # self.drone.set_quat(drone_quat)
            
            # self.prev_pos = self.drone.get_pos().cpu().view(-1)
            # # self.prev_quat = self.drone.get_quat().cpu().view(-1)
            
            # #### Original ####
            # # self.drone.set_pos(torch.tensor([act[:3]]))
            
            self.drone.set_pos(torch.tensor([act[:3]]))
            self.drone.set_quat(torch.tensor([act[3:]]))
            
            cam_transform = trans_quat_to_T(np.array([0, 0, 0]), xyz_to_quat(np.array([90, 0, 0])))
            self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu().numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform,)
            img, depth, segmentation, _ = self.cam.render(depth=True, segmentation=True)
            # Set all values in depth that are exactly 100 to 0
            depth = np.where(depth > 10, 0, depth)
            # if depth is not None:
            #     plt.imshow(depth)
            #     plt.pause(0.001)  # 0.001秒だけ描画して進む
            #     plt.clf()         # 毎回クリアして重ならないように
            
            if self.add_noise:
                depth_orig = depth.copy()
                segmentation_mask_orig = segmentation == self.target.idx
                depth_filtered = np.where(segmentation_mask_orig, depth, 0)
                # Apply a median filter to the depth image
                if self.median_filter:
                    depth_filtered = filters.median(depth_filtered, behavior='ndimage')
                depth_image = o3d.geometry.Image(depth_filtered)
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width = self.IMAGE_SIZE_W,
                    height = self.IMAGE_SIZE_H,
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
                
                # papper_mask = np.random.rand(*depth.shape) >0.25
                # depth = depth * papper_mask
                depth += np.random.normal(loc=0.0, scale=0.01, size=depth.shape).astype(np.float32)  # loc=mean, scale=std
            
            occ_grid_map = self.update_occupancy_gridmap(depth, segmentation)
            
            prev_pos = torch.tensor(self.env_qpos_array[-1][:3]).cuda()
            distance = torch.linalg.norm(self.drone.get_pos() - prev_pos)
            self.path_sum += distance
            
            self.color_array.append(img)
            self.depth_array.append(depth)
            
            self.env_qpos_array.append(np.concatenate([self.drone.get_pos().cpu().view(-1), self.drone.get_quat().cpu().view(-1)]))
            self.gridmap_array.append(occ_grid_map.cpu())
        
        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
        
        obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)
        obs_gridmap = np.stack(self.gridmap_array[-self.obs_horizon:], axis=0)

        
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
        if self.use_gridmap:
            obs_dict['gridmap'] = torch.from_numpy(obs_gridmap).unsqueeze(0) #  torch.Size([2, 25, 25, 25])
        return obs_dict
    
    # MARK:
    def reset(self, first_init=True):
        # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
        self.gridmap_array = []
        
        if first_init:
            # self.drone.set_pos(torch.tensor([[0, -1, 0.7]]))
            self.drone.set_pos(torch.tensor([INIT_POS]))
            self.drone.set_quat(xyz_to_quat(torch.tensor([[90, 0, 0]])))
            cam_transform = trans_quat_to_T(np.array([0, 0, 0]), xyz_to_quat(np.array([90, 0, 0])))
            self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu().numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform,)
            img, depth, segmentation, _ = self.cam.render(depth=True, segmentation=True)
            # Set all values in depth that are exactly 100 to 0
            depth = np.where(depth == 100, 0, depth)
        
        self.color_array.append(img)
        self.gridmap_array.append(self.update_occupancy_gridmap(depth, segmentation).cpu())
        action = np.zeros(7)
        # action = np.zeros(6)
        print("Drone ready!")
        
        env_qpos = np.concatenate([self.drone.get_pos().cpu().view(-1), self.drone.get_quat().cpu().view(-1)])
        
        self.env_qpos_array.append(env_qpos)
        agent_pos = np.stack([self.env_qpos_array[-1]]*self.obs_horizon, axis=0)
        
        obs_img = np.stack([self.color_array[-1]]*self.obs_horizon, axis=0)
        obs_gridmap = np.stack([self.gridmap_array[-1]]*self.obs_horizon, axis=0)
        
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
        if self.use_gridmap:
            obs_dict['gridmap'] = torch.from_numpy(obs_gridmap).unsqueeze(0) #  torch.Size([2, 25, 25, 25])
        return obs_dict

    def update_occupancy_gridmap(self, depth, segmentation):
        
        segmentation_mask = segmentation == self.target.idx
        depth_filtered = np.where(segmentation_mask, depth, 0)
        # Apply a median filter to the depth image
        if self.median_filter:
            depth_filtered = filters.median(depth_filtered, behavior='ndimage')
        
        depth_image = o3d.geometry.Image(depth_filtered)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width = self.IMAGE_SIZE_W,
            height = self.IMAGE_SIZE_H,
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
        if self.add_noise == False:
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
        occ_grid_map = self.mapper.to_prob_occ_map()  # torch.Size([1, 200, 200, 200])
        return occ_grid_map

@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config')),
    version_base=None
)
# MARK:
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    
    if workspace.__class__.__name__ == 'DPWorkspace':
        use_image = True
        use_point_cloud = False
    else:
        use_image = False
        use_point_cloud = True

    # fetch policy model
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1
    
    roll_out_length = DEPLOY_LENGTH
    
    img_size = 224
    num_points = 4096
    use_waist = True
    first_init = True
    record_data = True
    
    env = CameraInference(
        obs_horizon=2, 
        action_horizon=action_horizon, 
        device="gpu",
        use_point_cloud=use_point_cloud,
        use_image=use_image,
        img_size=img_size,
        num_points=num_points,
        use_waist=use_waist
    )
    
    obs_dict = env.reset(first_init=first_init)
    env.csv_list.append((0, 0, 0))
    
    step_count = 0
    while step_count < roll_out_length:
        with torch.no_grad():
            action = policy(obs_dict)[-1]   # torch.Size([15, 7])
            if use_waypoints:
                action_torch = torch.stack([torch.tensor(act, dtype=torch.float32, device="cuda") for act in action])
                action_list_pos = [act[:3] for act in action] 
                action_torch_pos = torch.stack([torch.tensor(act, dtype=torch.float32, device="cuda") for act in action_list_pos])
                
                # action_torch_pos = compute_bubble(
                #     occupancy_grid=env.mapper.to_prob_occ_map(),
                #     positions=action_torch_pos,
                #     dis_threshold=0.05,
                # )
                
                waypoints = dp_waypoint_selection(
                    action_torch_pos,
                    gt_states=action_torch_pos,
                    err_threshold=0.01,
                    pos_only=True,
                )
                selected_points = action_torch[waypoints]
                action_list = [point.cpu().numpy() for point in selected_points]
                if len(action_list) < 15:
                    while len(action_list) < 15:
                        action_list.append(action_list[-1])
            else:
                action_list = [act.numpy() for act in action]
            
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"step_count: {step_count}")

        path_length = env.path_sum.item()
        cprint(f"Path length: {env.path_sum:.2f} meters", "green")
        cover_ratio, _ = eval_inference(
            TARGET,
            env.pcd_total,
            scale=SCALE,
            threshold_icp=1,
            threshold_dis=0.002,
            visualize=False,
            noise_remove=False,
        )
        coverage = cover_ratio * 100
        cprint(f"Coverage ratio: {coverage:.4f}", "green")
        env.csv_list.append((step_count, coverage, path_length))

    # For the final result
    final_cover_ratio, pcd_final = eval_inference(
            TARGET,
            env.pcd_total,
            scale=SCALE,
            threshold_icp=1,
            threshold_dis=0.002,
            visualize=True, 
            noise_remove=False,
        )
    cprint(f"Final Coverage ratio: {final_cover_ratio:.4f}", "yellow")

    # Save coverage data to a CSV file using pandas
    csv_save_path = os.path.join(os.getcwd(), f"data/logs/csv_{INIT_POS}_{SCALE}", f"{TARGET}_{NAME}_{DEPLOY_LENGTH}.csv")
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    coverage_df = pd.DataFrame(env.csv_list, columns=["step", "coverage", "path_length"])
    coverage_df.to_csv(csv_save_path, index=False)
    print(f"Coverage data saved to {csv_save_path}")
    
    # env.mapper.visualize_occupancy_grid(threshold_p_occ=0.9)
    # prob_map = env.mapper.to_prob_occ_map()
    # torch.save(prob_map, "prob_map.pth")

    # def plot_histogram_of_tensor_values(env):
    #     prob_map = env.mapper.to_prob_occ_map().cpu().numpy()
    #     map_flatten = prob_map.flatten()
    #     plt.figure(figsize=(10, 8))
    #     plt.hist(map_flatten, bins=50, range=(0, 1), edgecolor='black', alpha=0.75)
    #     plt.xlabel("Value", fontsize=14)
    #     plt.ylabel("Frequency", fontsize=14)
    #     plt.title("Histogram of Tensor Values", fontsize=16)
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
    #     plt.tight_layout()
    #     plt.show()
    # # plot_histogram_of_tensor_values(env)
    
    # o3d.visualization.draw_geometries([env.pcd_total])
    
    # Save the accumulated point cloud
    pcd_save_path = os.path.join(os.getcwd(), "data", "pointcloud.ply")
    o3d.io.write_point_cloud(pcd_save_path, pcd_final)

    choice = input("whether to rename of ply: y/n")
    if choice == "y":
        renamed = input("file rename of ply:")
        os.rename(src=pcd_save_path, dst=pcd_save_path.replace("pointcloud.ply", renamed+'.ply'))
        new_name = pcd_save_path.replace("pointcloud.ply", renamed+'.ply')
        cprint(f"save data at step: {roll_out_length} in {new_name}", "yellow")
    else:
        cprint(f"save data at step: {roll_out_length} in {pcd_save_path}", "yellow")
    
    if record_data:
        import h5py
        root_dir = os.getcwd()
        save_dir = os.path.join(root_dir, "deploy_dir")
        os.makedirs(save_dir, exist_ok=True)
        
        record_file_name = f"{save_dir}/demo.h5"
        color_array = np.array(env.color_array)
        depth_array = np.array(env.depth_array)
        cloud_array = np.array(env.cloud_array)
        qpos_array = np.array(env.env_qpos_array)
        with h5py.File(record_file_name, 'w') as f:
            f.create_dataset("color_array", data=color_array)
            f.create_dataset("depth_array", data=depth_array)
            f.create_dataset("cloud_array", data=cloud_array)
            f.create_dataset("qpos_array", data=qpos_array)
        
        choice = input("whether to rename: y/n")
        if choice == "y":
            renamed = input("file rename:")
            os.rename(src=record_file_name, dst=record_file_name.replace("demo.h5", renamed+'.h5'))
            record_file_name = record_file_name.replace("demo.h5", renamed+'.h5')
            cprint(f"save data at step: {roll_out_length} in {record_file_name}", "yellow")
        else:
            cprint(f"save data at step: {roll_out_length} in {record_file_name}", "yellow")

    # plot trajectory
    vis_trajectory.plot_trajectory_and_obj(
        hdf5_file=record_file_name, 
        obj_path=os.path.join(env.asset_dir, env.mesh_cfg["file"]), 
        scale=env.mesh_cfg["scale"]*SCALE,
        pos=env.mesh_cfg["pos"],
        quat=env.mesh_cfg.get("quat", None),
        show_background=False,
        show_axes=False,
        )

if __name__ == "__main__":
    main()