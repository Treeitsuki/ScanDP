import os 
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import genesis as gs
import hydra
import numpy as np
import open3d as o3d
import pathlib
import time
import torch
import tqdm
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat
from omegaconf import OmegaConf
from termcolor import cprint
from skimage import filters, util
import matplotlib.pyplot as plt

from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace

import diffusion_policy_3d.module.gridmap as gridmap
import diffusion_policy_3d.module.vis_trajectory as vis_trajectory
from ground_truth.coverage_gt import eval_inference

os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

DEPLOY_LENGTH = 500
INIT_POS = [0, 0, 1]
TARGET = "bunny"  # "bunny", "spot", "armadillo", "teapot", "dragon", "bust", "bike"
POLICY = "hemi_random" # "random", "hemi_random", "hemi_uni"
SCALE = 1

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
        
        self.mapper = gridmap.LocalMap3D(
            X_lim=np.array([-0.25, 0.25]),
            Y_lim=np.array([-0.25, 0.25]),
            Z_lim=np.array([0.5, 1.0]),
            resolution=0.02,
            p=0.5,
            device="cuda"
        )
        # self.mapper = gridmap.LocalMap3D(
        #     X_lim=np.array([-1, 1]),
        #     Y_lim=np.array([-1, 1]),
        #     Z_lim=np.array([0, 2]),
        #     resolution=0.02,
        #     p=0.5,
        #     device="cuda"
        # )
        self.IMAGE_SIZE = 224
        self.median_filter = True
        self.add_noise = False
        self.pcd_total = o3d.geometry.PointCloud()
        
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
            res    = (self.IMAGE_SIZE, self.IMAGE_SIZE),
            pos    = (0.0, 1.0, 0.7),
            lookat = (0, 0, 0.7),
            fov    = 45,
            GUI    = True,
        )

        # MARK: target
        ########################### entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane()
        )

        # # WARNING: You must add bunny before drone 
        self.target_map = {
            "bunny": dict(
                file="stanfordbunny.stl", pos=(0, 0, 0.7), scale=3.0, quat=(1, 0, 0, 0)
            ),
            "bunny2": dict(
                file="stanford-bunny.obj", pos=(0, 0, 0.7), scale=3.0, quat=(1, 1, 0, 0)
            ),
            "spot": dict(
                file="spot.obj", pos=(0, 0.05, 0.72), scale=0.28, quat=(1, 1, 0, 0)
            ),
            "armadillo": dict(
                file="armadillo.obj", pos=(0, 0, 0.7), scale=0.0025, quat=(1, 1, 0, 0)
            ),
            "teapot": dict(
                file="teapot.obj", pos=(0, 0, 0.55), scale=0.07, quat=(1, 1, 0, 0)
            ),
            "dragon": dict(
                file="dragon.obj", pos=(0, 0, 0.7), scale=0.5, quat=(1, 1, 0, 0)
            ),
            "bust": dict(
                file="bust.obj", pos=(0, 0, 0.8), scale=0.002, quat=(1, 0, 0, 0)
            ),
            "bike": dict(
                file="bike.obj", pos=(0.0, 0.11, 0.45), scale=0.00023, quat=(1, 1, 0, 0)
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
        #         size = (0.6, 0.6, 0.56),
        #         collision = True,
        #         fixed = True,
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

            self.drone.set_pos(torch.tensor([act[:3]]))
            self.drone.set_quat(torch.tensor([act[3:]]))
            
            cam_transform = trans_quat_to_T(np.array([0, 0, 0]), xyz_to_quat(np.array([90, 0, 0])))
            self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu().numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform,)
            
            # # For random hemisphere
            if POLICY == "hemi_random" or POLICY == "hemi_uni":
                self.cam.set_pose(pos=act[:3], )
                self.cam.set_pose(lookat=(0, 0, 0.7))

            img, depth, segmentation, _ = self.cam.render(depth=True, segmentation=True)
            
            if self.add_noise:
                depth = util.random_noise(depth, mode='s&p', amount=0.95)
            
            occ_grid_map = self.update_occupancy_gridmap(depth, segmentation)
            # self.cloud_array.append()
            self.color_array.append(img)
            self.depth_array.append(depth)
            self.env_qpos_array.append(np.concatenate([self.drone.get_pos().cpu().view(-1), self.drone.get_quat().cpu().view(-1)]))
            self.gridmap_array.append(occ_grid_map.cpu())
        
        
        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
        
        # obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
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
            self.drone.set_pos(torch.tensor([INIT_POS]))
            self.drone.set_quat(xyz_to_quat(torch.tensor([[90, 0, 0]])))
            cam_transform = trans_quat_to_T(np.array([0, 0, 0]), xyz_to_quat(np.array([90, 0, 0])))
            self.cam.set_pose(transform=trans_quat_to_T(self.drone.get_pos()[0].cpu().numpy(), self.drone.get_quat()[0].cpu().numpy()) @ cam_transform,)
            img, depth, segmentation, _ = self.cam.render(depth=True, segmentation=True)
        
        self.color_array.append(img)
        self.gridmap_array.append(self.update_occupancy_gridmap(depth, segmentation).cpu())
        action = np.zeros(7)
        print("Drone ready!")
        
        env_qpos = np.concatenate([self.drone.get_pos().cpu().view(-1), self.drone.get_quat().cpu().view(-1)])
        self.env_qpos_array.append(env_qpos)
        agent_pos = np.stack([self.env_qpos_array[-1]]*self.obs_horizon, axis=0)
        
        # obs_cloud = np.stack([self.cloud_array[-1]]*self.obs_horizon, axis=0)
        obs_img = np.stack([self.color_array[-1]]*self.obs_horizon, axis=0)
        obs_gridmap = np.stack([self.gridmap_array[-1]]*self.obs_horizon, axis=0)
        
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        # if self.use_point_cloud:
        #     obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
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
        occ_grid_map = self.mapper.to_prob_occ_map()  # torch.Size([1, 200, 200, 200])
        return occ_grid_map


from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def random_cube(n, lower_bound=(-1, -1, -1), upper_bound=(1, 1, 1)):
    points = np.random.uniform(low=lower_bound, high=upper_bound, size=(n, 3))
    return points

# Sample points evenly on a hemisphere (Fibonacci lattice)
def sample_points_on_hemisphere(n_points, radius=1.0, center=(0.0, 0.0, 0.0)):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    mask = z >= 0
    x, y, z = x[mask], y[mask], z[mask]

    points = np.stack((x, y, z), axis=1)
    points *= radius
    points += np.array(center)

    return points

# Randomly sample points on a hemisphere
def sample_points_on_hemisphere_random(n_points, radius=1.0, center=(0.0, 0.0, 0.0)):
    points = []
    while len(points) < n_points:
        # Random point on a unit sphere
        vec = np.random.normal(size=3)
        vec /= np.linalg.norm(vec)
        if vec[2] >= 0:  # On the hemisphere (Z >= 0)
            point = radius * vec + np.array(center)
            points.append(point)
    return np.array(points)

# Create a distance matrix
def create_distance_matrix(points):
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(points[i] - points[j])
    return dist_matrix

# Solve TSP using OR-Tools
def solve_tsp(distance_matrix):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(distance_matrix[from_node][to_node] * 1e6)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route
    else:
        return None

@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config')),
    version_base=None
)
# MARK:
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
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
    
    
    num_points = DEPLOY_LENGTH + 10
    radius = 1.0
    center = (0, 0, 0.7)
    # shortest_path, total_distance = tsp_shortest_path(points)

    print(f"num_points: {num_points}")
    
    if POLICY == "hemi_uni":
        points = sample_points_on_hemisphere(num_points, radius=radius, center=center)
    elif POLICY == "hemi_random":
        points = sample_points_on_hemisphere_random(num_points, radius=radius, center=center)
    elif POLICY == "random":
        points = random_cube(num_points, lower_bound=(-0.5, -0.5, 0.5), upper_bound=(0.5, 0.5, 1))

    distance_matrix = create_distance_matrix(points)
    path  = solve_tsp(distance_matrix)
    ordered_points = points[path]
    shortest_path = [torch.tensor(p, dtype=torch.float32) for p in ordered_points]
    
    step_count = 0
    while step_count < roll_out_length:
        action_list = []
        for i in range(action_horizon):
            if step_count + i < len(shortest_path):
                random_quat = np.random.randn(4)
                random_quat /= np.linalg.norm(random_quat)
                
                action = np.concatenate([shortest_path[step_count + i].numpy(), random_quat])  # Add a default quaternion
                action_list.append(action)
            else:
                break
        
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"step_count: {step_count}")
    
    # For the final result
    final_cover_ratio, _ = eval_inference(
            TARGET,
            env.pcd_total,
            scale=SCALE,
            threshold_icp=1,
            threshold_dis=0.002,
            visualize=True,
            noise_remove=True,
        )
    coverage = final_cover_ratio * 100
    cprint(f"Coverage ratio: {coverage:.4f}", "green")
    
    # env.mapper.visualize_occupancy_grid(threshold_p_occ=0.9)
    
    # o3d.visualization.draw_geometries([env.pcd_total])
    
    # Save the accumulated point cloud
    pcd_save_path = os.path.join(os.getcwd(), "data", "pointcloud.ply")
    o3d.io.write_point_cloud(pcd_save_path, env.pcd_total)

    # Remove noise from the point cloud
    cl, ind = env.pcd_total.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    env.pcd_total = env.pcd_total.select_by_index(ind)
    o3d.visualization.draw_geometries([env.pcd_total])

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
    # vis_trajectory.plot_trajectory_with_plotly(record_file_name)
    vis_trajectory.plot_trajectory_and_obj(
        hdf5_file=record_file_name, 
        obj_path=os.path.join(env.asset_dir, env.mesh_cfg["file"]), 
        scale=env.mesh_cfg["scale"]*SCALE,
        pos=env.mesh_cfg["pos"],
        quat=env.mesh_cfg.get("quat", None),
        show_axes=False,
        show_background=False,
    )

if __name__ == "__main__":
    main()