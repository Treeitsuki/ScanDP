import os
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import torch

def plot_ply_as_point_cloud(ply_file_path, voxel_size=0.005):
    # Load the PLY file
    pcd = o3d.io.read_point_cloud(ply_file_path)
    o3d.visualization.draw_geometries([pcd])
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd])
    
    # Assign colors to the point cloud
    # Normalize the Z values to range between 0 and 1
    z_values = np.array(pcd.points)[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_normalized = (z_values - z_min) / (z_max - z_min)

    # Create a gradient from light blue to blue based on the normalized Z values
    colors = np.zeros((len(pcd.points), 3))
    colors[:, 0] = 0.253  # Light blue (R value)
    colors[:, 1] = 0.558  # Light blue (G value)
    colors[:, 2] = z_normalized  # Blue gradient based on Z values
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    o3d.visualization.draw_geometries([pcd])

    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_sampled, voxel_size)
    o3d.visualization.draw_geometries([voxel])
    # Display the number of voxels
    num_voxels = len(voxel.get_voxels())
    print(f"Number of voxels: {num_voxels}")

    # # Get the coordinates of the point cloud
    # pcd_sampled = pcd.voxel_down_sample(voxel_size=0.005)
    # points = np.array(pcd_sampled.points)
    # print(pcd_sampled.points)

    # # Plot using Plotly
    # fig = go.Figure(data=[go.Scatter3d(
    #     x=points[:, 0],
    #     y=points[:, 1],
    #     z=points[:, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=2,
    #         color=points[:, 2],  # Set color based on Z value
    #         colorscale='Viridis',
    #         opacity=0.8
    #     )
    # )])

    # fig.update_layout(scene=dict(
    #     xaxis_title='X',
    #     yaxis_title='Y',
    #     zaxis_title='Z'
    # ))

    # fig.show()

def evaluation_point_cloud(stl_file_path, ply_file_path, threshold=0.02, voxel_size=0.01):
    # Load the STL file and convert to point cloud
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    translation = (0, 0, 0.7)
    mesh.translate(translation)
    mesh.scale(3, center=mesh.get_center())
    pcd_model = mesh.sample_points_uniformly(number_of_points=100_000)
    # pcd_model = pcd_model.voxel_down_sample(voxel_size=0.001)
    voxel_model = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_model, 0.01)

    # Load the PLY file
    pcd_input = o3d.io.read_point_cloud(ply_file_path)
    pcd_input = pcd_input.voxel_down_sample(voxel_size=0.001)

    # Perform ICP (Iterative Closest Point) alignment
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_input, pcd_model, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation_icp = reg_p2p.transformation
    pcd_input.transform(transformation_icp)

    # Convert aligned scanned point cloud to voxel grid
    voxel_input = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_input, voxel_size)
    coverage_ratio = compute_voxel_coverage(voxel_model, voxel_input)
    print(f"Voxel Coverage Ratio: {coverage_ratio * 100:.2f}%")
    return coverage_ratio

    # # Visualize the aligned point clouds
    # pcd_model.paint_uniform_color([0.945, 0.660, 0])  # Yellow
    # pcd_input.paint_uniform_color([0.253, 0.558, 0.867])  # Blue
    # o3d.visualization.draw_geometries([pcd_model, pcd_input])

def compute_voxel_coverage(voxel_model, voxel_input):
    """
    Compute the coverage ratio between the ground-truth voxel model and the scanned point cloud voxel model.
    """
    model_voxels = set([tuple(v.grid_index) for v in voxel_model.get_voxels()])
    input_voxels = set([tuple(v.grid_index) for v in voxel_input.get_voxels()])
    
    intersecting_voxels = model_voxels.intersection(input_voxels)
    coverage_ratio = len(intersecting_voxels) / len(model_voxels) if len(model_voxels) > 0 else 0
    
    return coverage_ratio

# if __name__ == '__main__':
#     stl_file_path = os.path.join(os.getcwd(), "scandp/diffusion_policy_3d/assets", "stanfordbunny.stl")
#     ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/outputs/cam_gridmap-scandp_conv3d-0219_100_seed0/bunny_wobatch.ply"
    
#     proportion = evaluation_point_cloud(stl_file_path, ply_file_path)


if __name__ == '__main__':
    # obj_name = "stanfordbunny.stl"
    obj_name = "spot.obj"
    stl_file_path = os.path.join(os.getcwd(), "scandp/diffusion_policy_3d/assets", obj_name)
    # plot_stl_as_point_cloud(stl_file_path)
    
    # ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/bunny_img.ply"
    ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/bunny_wobatch.ply"
    plot_ply_as_point_cloud(ply_file_path)
