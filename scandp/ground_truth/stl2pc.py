import os
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import torch

def plot_stl_as_point_cloud(stl_file_path, number_of_points=100_000, translation=(0, 0, 0.7), scale=1.0, quat=(1, 0, 0, 0)):
    # Load the STL file
    mesh = o3d.io.read_triangle_mesh(stl_file_path)

    # Apply translation and scaling
    mesh.translate(translation)
    mesh.scale(scale, center=mesh.get_center())

    # Apply rotation using quaternion
    rotation = R.from_quat(quat)
    rotation_matrix = rotation.as_matrix()
    mesh.rotate(rotation_matrix, center=mesh.get_center())

    # Convert the mesh to a point cloud
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    print(pcd.points)

    # Get the coordinates of the point cloud
    points = torch.tensor(pcd.points)

    # Plot using Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],  # Set color based on Z value
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    fig.show()

def plot_ply_as_point_cloud(ply_file_path, number_of_points=100_000):
    # Load the PLY file
    pcd = o3d.io.read_point_cloud(ply_file_path)
    # o3d.visualization.draw_geometries([pcd])

    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.01)
    o3d.visualization.draw_geometries([voxel])

    # Get the coordinates of the point cloud
    pcd_sampled = pcd.voxel_down_sample(voxel_size=0.005)
    points = np.array(pcd_sampled.points)
    print(pcd_sampled.points)

    # Plot using Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],  # Set color based on Z value
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    fig.show()

def evaluation_point_cloud(stl_file_path, ply_file_path, threshold=0.02):
    # Load the STL file and convert to point cloud
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    translation = (0, 0, 0.7)
    mesh.translate(translation)
    mesh.scale(3, center=mesh.get_center())
    pcd_model = mesh.sample_points_uniformly(number_of_points=100_000)

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

    # Transform the input point cloud
    pcd_input.transform(transformation_icp)

    chamfer_dist = Chamfer_Distance(pcd_model, pcd_input)
    print(f"Chamfer Distance: {chamfer_dist}")

    # Visualize the aligned point clouds
    pcd_model.paint_uniform_color([0.945, 0.660, 0])  # Yellow
    pcd_input.paint_uniform_color([0.253, 0.558, 0.867])  # Blue
    o3d.visualization.draw_geometries([pcd_model, pcd_input])

    # # Perform ICP (Iterative Closest Point) alignment
    # trans_init = np.eye(4)
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     pcd_input, pcd_model, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # )
    # transformation_icp = reg_p2p.transformation

    # # Transform the input point cloud
    # pcd_input.transform(transformation_icp)

    # # Visualize the aligned point clouds
    # pcd_model.paint_uniform_color([0.945, 0.660, 0])  # Yellow
    # pcd_input.paint_uniform_color([0.253, 0.558, 0.867])  # Blue
    # o3d.visualization.draw_geometries([pcd_model, pcd_input])

    # # # Voxelize the point clouds
    # # voxel_size = 0.01
    # # voxel_model = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_model, voxel_size)
    # # voxel_input = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_input, voxel_size)

    # o3d.visualization.draw_geometries([voxel_model, voxel_input])

def Chamfer_Distance(pcd_cad, pcd_real):
    source_pcd = np.asarray(pcd_cad.points)
    target_pcd = np.asarray(pcd_real.points)
    tree1 = cKDTree(source_pcd)
    tree2 = cKDTree(target_pcd)
    dist1, _ = tree1.query(target_pcd)
    dist2, _ = tree2.query(source_pcd)
    chamfer_dist = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer_dist * 1e3 


if __name__ == '__main__':
    stl_file_path = os.path.join(os.getcwd(), "scandp/diffusion_policy_3d/assets", "stanfordbunny.stl")
    ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/bunny_intrinsic.ply"
    
    proportion = evaluation_point_cloud(stl_file_path, ply_file_path)


# if __name__ == '__main__':
#     # obj_name = "stanfordbunny.stl"
#     obj_name = "spot.obj"
#     stl_file_path = os.path.join(os.getcwd(), "scandp/diffusion_policy_3d/assets", obj_name)
#     # plot_stl_as_point_cloud(stl_file_path)
    
#     ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/bunny_img.ply"
#     plot_ply_as_point_cloud(ply_file_path)
