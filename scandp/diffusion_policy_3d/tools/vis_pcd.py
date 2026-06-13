import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def load_and_sample_pcd(ply_path, voxel_size=0.005):
    # Load the point cloud from the specified PLY file
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Perform Poisson disk sampling
    sampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # Compute the signed distance function (SDF) from the point cloud
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # Compute the SDF
    sdf = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]
    return sampled_pcd, sdf

# Example usage
if __name__ == "__main__":
    ply_path = "/home/cvl/cvl/ScanDP/scandp/data/outputs/cam_gridmap-scandp_conv3d-0219_100_seed0/bunny_wobatch.ply"
    sampled_pcd, sdf = load_and_sample_pcd(ply_path)
    o3d.visualization.draw_geometries([sampled_pcd])
    o3d.visualization.draw_geometries([sdf])
    # Convert Open3D point cloud to numpy array
    points = np.asarray(sampled_pcd.points)
    
    # Create a Plotly scatter plot
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],  # Color by Z axis values
            colorscale='Viridis',
            opacity=0.8
        )
    )

    # Create the layout for the plot
    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(r=0, l=0, b=0, t=0)
    )

    # Create the figure and plot it
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()