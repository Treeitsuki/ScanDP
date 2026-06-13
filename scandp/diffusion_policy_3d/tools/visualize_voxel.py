import open3d as o3d
import numpy as np
import plotly.graph_objs as go

# Resolution for the voxel size
voxel_size = 1

# Load and process the 3D model
bunny = o3d.data.KnotMesh()
mesh = o3d.io.read_triangle_mesh(bunny.path)
mesh.compute_vertex_normals()

# Convert the mesh to a point cloud
pcd = mesh.sample_points_uniformly(number_of_points=50000)
down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

# Extract voxel centers and ensure no overlap by rounding to voxel grid
voxel_centers = np.asarray(down_pcd.points)
voxel_centers = np.round(voxel_centers / voxel_size) * voxel_size

# Remove duplicates to avoid overlapping voxels
voxel_centers = np.unique(voxel_centers, axis=0)
print(f"Number of voxels: {len(voxel_centers)}")

def cube_points(center, size):
    """Generate the 8 vertices of a cube given its center and size."""
    half_size = size / 2.0
    offsets = np.array([
        [-half_size, -half_size, -half_size],
        [ half_size, -half_size, -half_size],
        [ half_size,  half_size, -half_size],
        [-half_size,  half_size, -half_size],
        [-half_size, -half_size,  half_size],
        [ half_size, -half_size,  half_size],
        [ half_size,  half_size,  half_size],
        [-half_size,  half_size,  half_size]
    ])
    return center + offsets

def triangulate_cube_faces(centers, size):
    """Create vertices and faces for a collection of cubes."""
    all_cubes = [cube_points(center, size) for center in centers]
    vertices, indices = np.unique(np.vstack(all_cubes), axis=0, return_inverse=True)

    faces = []
    for i in range(len(centers)):
        base_idx = indices[i * 8:(i + 1) * 8]
        faces.extend([
            [base_idx[0], base_idx[1], base_idx[2]],
            [base_idx[0], base_idx[2], base_idx[3]],
            [base_idx[4], base_idx[5], base_idx[6]],
            [base_idx[4], base_idx[6], base_idx[7]],
            [base_idx[0], base_idx[1], base_idx[5]],
            [base_idx[0], base_idx[5], base_idx[4]],
            [base_idx[2], base_idx[3], base_idx[7]],
            [base_idx[2], base_idx[7], base_idx[6]],
            [base_idx[1], base_idx[2], base_idx[6]],
            [base_idx[1], base_idx[6], base_idx[5]],
            [base_idx[3], base_idx[0], base_idx[4]],
            [base_idx[3], base_idx[4], base_idx[7]]
        ])
    faces = np.array(faces)
    return vertices, faces[:, 0], faces[:, 1], faces[:, 2]

# Generate vertices and triangular faces for the voxels
vertices, I, J, K = triangulate_cube_faces(voxel_centers, voxel_size)
X, Y, Z = vertices.T

# Create the 3D mesh for visualization
mesh3d = go.Mesh3d(
    x=X, y=Y, z=Z, i=I, j=J, k=K,
    flatshading=True,
    color="#ce6a6b",
    opacity=1
)

# Layout for the plot
layout = go.Layout(
    width=700, height=700,
    title_text='Voxel Data Visualization',
    title_x=0.5,
    scene=dict(
        xaxis=dict(),
        yaxis=dict(),
        zaxis=dict(),
        camera=dict(eye=dict(x=1.4, y=-2.5, z=1.0))
    )
)

# Create the figure and display it
fig = go.Figure(data=[mesh3d], layout=layout)
fig.show()