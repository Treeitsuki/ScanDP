import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import torch

def plot_point_cloud(path_pc):

    pcd = o3d.io.read_point_cloud(path_pc)
    o3d.visualization.draw_geometries([pcd])
    
    # Remove noise from the point cloud
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd])

    pcd_sampled = pcd.voxel_down_sample(voxel_size=0.01)
    o3d.visualization.draw_geometries([pcd_sampled])

    # mesh = o3d.io.read_triangle_mesh(path_pc)
    # mesh.translate(-mesh.get_center())
    # translation = (0, 0, 0.7)
    # mesh.translate(translation)
    # mesh.scale(0.003, center=mesh.get_center())
    # # Apply quaternion rotation to the mesh
    # quat = np.array([1, 1, 0, 0])
    # R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
    # mesh.rotate(R, center=mesh.get_center())
    # pcd_model = mesh.sample_points_poisson_disk(number_of_points=100_000)
    # pcd_sampled = pcd_model.voxel_down_sample(voxel_size=0.005)


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

if __name__ == "__main__":
    path_pc = "/home/cvl/cvl/ScanDP/scandp/data/armadillo_img.ply"
    # path_pc = "/home/cvl/cvl/ScanDP/scandp/diffusion_policy_3d/assets/armadillo.obj"
    plot_point_cloud(path_pc)