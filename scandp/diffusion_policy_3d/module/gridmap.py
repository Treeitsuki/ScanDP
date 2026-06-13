import logging
import numpy as np
import torch
import open3d as o3d
import plotly.graph_objs as go
from diffusion_policy_3d.module.bresenham_torch import bresenham3D


class LocalMap3D:
    def __init__(self, X_lim, Y_lim, Z_lim, resolution, p, device=None):
        self.ratio = 100    # For conversion to integer
        self.X_lim = X_lim * self.ratio
        self.Y_lim = Y_lim * self.ratio
        self.Z_lim = Z_lim * self.ratio
        self.resolution = resolution * self.ratio
        self.p = p
        self.device = device if device is not None else torch.device("cpu")
        self.base = (self.X_lim[1] - self.X_lim[0]) / self.resolution

        x = torch.arange(start=self.X_lim[0], end=self.X_lim[1], step=self.resolution, device=self.device)
        y = torch.arange(start=self.Y_lim[0], end=self.Y_lim[1], step=self.resolution, device=self.device)
        z = torch.arange(start=self.Z_lim[0], end=self.Z_lim[1], step=self.resolution, device=self.device)

        self.x_max = len(x)
        self.y_max = len(y)
        self.z_max = len(z)

        self.occ_map_int = torch.full(
            (self.x_max, self.y_max, self.z_max),
            fill_value=self.log_odds(p),
            device=self.device
        )        
        
        x_orig = torch.arange(start=X_lim[0], end=X_lim[1], step=resolution, device=self.device)
        y_orig = torch.arange(start=Y_lim[0], end=Y_lim[1], step=resolution, device=self.device)
        z_orig = torch.arange(start=Z_lim[0], end=Z_lim[1], step=resolution, device=self.device)

        self.x_max_orig = len(x_orig)
        self.y_max_orig = len(y_orig)
        self.z_max_orig = len(z_orig)
        
        self.occ_map = torch.full(
            (self.x_max_orig, self.y_max_orig, self.z_max_orig),
            fill_value=self.log_odds(self.p),
            device=self.device
        )

    def log_odds(self, p):
        p = torch.tensor(p, device=self.device)
        return torch.log(p / (1 - p))

    def retrieve_p(self, log_map):
        prob_map = 1 - 1 / (1 + torch.exp(log_map))
        return prob_map

    def is_valid(self, x_idx, y_idx, z_idx):
        """Check if voxel indices are within valid range"""
        flag_valid = (x_idx < self.x_max) & (y_idx < self.y_max) & (z_idx < self.z_max) & \
                        (x_idx >= 0) & (y_idx >= 0) & (z_idx >= 0)
        return flag_valid

    def discretize(self, points):
        """
        Convert continuous point coordinates to discrete voxel indices

        Args:
            points: Point cloud tensor of shape (num_points, 3)
        Returns:
            binary_map: Binary occupancy grid of shape (size[0], size[1], x_max, y_max, z_max)
        """
        points = points * self.ratio
        points = points.to(self.device)

        # Translate the physical position to grid indicies
        x_idx = torch.floor((points[:, 0] - self.X_lim[0]) / self.resolution).to(torch.int64)
        y_idx = torch.floor((points[:, 1] - self.Y_lim[0]) / self.resolution).to(torch.int64)
        z_idx = torch.floor((points[:, 2] - self.Z_lim[0]) / self.resolution).to(torch.int64)

        # Get valid batch indices
        valid_mask = self.is_valid(x_idx, y_idx, z_idx)
        binary_map = torch.zeros(self.x_max, self.y_max, self.z_max, device=self.device)

        binary_map[x_idx[valid_mask], y_idx[valid_mask], z_idx[valid_mask]] = 1

        return binary_map

    def update(self, x0, y0, z0, points, p_free, p_occ):
        """
        Efficiently update a 3D occupancy grid map using a point cloud.

        Args:
            x0, y0, z0: Floats. Sensor origin (start point) in 3D space.
            points: Tensor of shape (N, 3). Point cloud of observed obstacles.
            p_free: Float. Probability of free space.
            p_occ: Float. Probability of occupied space.
        """
        x0, y0, z0 = (x0 * self.ratio).to(self.device), (y0 * self.ratio).to(self.device), (z0 * self.ratio).to(self.device)
        points = (points * self.ratio).to(self.device)

        x0_idx = torch.floor((x0 - self.X_lim[0]) / self.resolution).to(torch.int64)
        y0_idx = torch.floor((y0 - self.Y_lim[0]) / self.resolution).to(torch.int64)
        z0_idx = torch.floor((z0 - self.Z_lim[0]) / self.resolution).to(torch.int64)
        start = torch.tensor([x0_idx, y0_idx, z0_idx], device=self.device)

        x_idx = torch.floor((points[:, 0] - self.X_lim[0]) / self.resolution).to(torch.int64)
        y_idx = torch.floor((points[:, 1] - self.Y_lim[0]) / self.resolution).to(torch.int64)
        z_idx = torch.floor((points[:, 2] - self.Z_lim[0]) / self.resolution).to(torch.int64)
        end = torch.stack([x_idx, y_idx, z_idx], dim=1)

        valid_mask = self.is_valid(x_idx, y_idx, z_idx)
        end = end[valid_mask]
        if end.shape[0] == 0:
            return
        free_points = []
        # free_points.append(bresenhamline(start, end, max_iter=-1).to(self.device))
        free_points.append(bresenham3D(start, end, map_size=25).to(self.device))

        # Remove points that are the same as the start point and end points
        free_points = torch.cat(free_points, dim=0)
        base = self.base
        free_points_encoded = free_points[:, 0] * base * base + free_points[:, 1] * base + free_points[:, 2]
        end_encoded = end[:, 0] * base * base + end[:, 1] * base + end[:, 2]
        result = ~torch.isin(free_points_encoded, end_encoded)
        free_points = free_points[result]

        free_x, free_y, free_z = free_points[:, 0], free_points[:, 1], free_points[:, 2]
        valid_free_mask = self.is_valid(free_x, free_y, free_z)
        free_x, free_y, free_z = free_x[valid_free_mask], free_y[valid_free_mask], free_z[valid_free_mask]

        self.occ_map_int[free_x, free_y, free_z] += self.log_odds(p_free)
        self.occ_map_int[end[:, 0], end[:, 1], end[:, 2]] += self.log_odds(p_occ)

        # Back to original resolution
        self.occ_map = self.occ_map_int[:self.x_max, :self.y_max, :self.z_max]

    def to_prob_occ_map(self):
        log_map = self.occ_map
        prob_map = self.retrieve_p(log_map)
        return prob_map


    ### Visualization methods ###
    def visualize_occupancy_grid(self, threshold_p_occ=0.5):
        """
        Visualize the occupancy grid using cube meshes for voxels, ensuring all faces are rendered.
        
        Args:
            threshold_p_occ: Float. Probability threshold for visualization (default: 0.5)
        """
        # Get probability map and convert to CPU numpy array
        prob_map = self.to_prob_occ_map().cpu().numpy()

        # Back to original resolution
        self.resolution = self.resolution / self.ratio
        self.X_lim = self.X_lim / self.ratio
        self.Y_lim = self.Y_lim / self.ratio
        self.Z_lim = self.Z_lim / self.ratio
        
        # Create meshgrid for all coordinates
        x_coords = np.arange(prob_map.shape[0]) * self.resolution + self.X_lim[0]
        y_coords = np.arange(prob_map.shape[1]) * self.resolution + self.Y_lim[0]
        z_coords = np.arange(prob_map.shape[2]) * self.resolution + self.Z_lim[0]
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Find occupied voxels above threshold
        occupied = prob_map > threshold_p_occ
        
        # Extract coordinates and probabilities of occupied voxels
        voxels = np.stack([
            X[occupied],
            Y[occupied],
            Z[occupied]
        ], axis=1)
        
        # Skip visualization if no voxels are occupied
        if len(voxels) == 0:
            logging.warning("No voxels above threshold found for visualization")
            return

        
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
        vertices, I, J, K = triangulate_cube_faces(voxels, self.resolution)
        X, Y, Z = vertices.T

        # Create the 3D mesh for visualization
        mesh3d = go.Mesh3d(
            x=X, y=Y, z=Z, i=I, j=J, k=K,
            flatshading=True,
            color="rgba(65, 143, 222, 1)",
            # intensity=prob_map[occupied],
            # intensitymode='cell',
            # colorscale='RdBu',
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

    def reset(self):
        self.occ_map = torch.full(
            (self.x_max, self.y_max, self.z_max),
            fill_value=self.log_odds(self.p),
            device=self.device
        )

if __name__ == "__main__":

    resolution = 0.1
    points = torch.randint(0, 1, (200, 3), dtype=torch.float32, device="cuda")

    mapper = LocalMap3D(
        X_lim=np.array([-0.25, 0.25]),
        Y_lim=np.array([-0.25, 0.25]),
        Z_lim=np.array([0.5, 1.0]),
        resolution=resolution,
        p=0.5,
        device="cuda"
    )

    print(mapper.occ_map.shape) # torch.Size([1, 1, 200, 200, 200])

    logging.info("Updating the occupancy grid map.")
    mapper.update(torch.tensor(0.0, device="cuda"), 
                  torch.tensor(0.0, device="cuda"), 
                  torch.tensor(0.0, device="cuda"), 
                  points, p_free=0.3, p_occ=0.7)

    occ_grid_map = mapper.to_prob_occ_map()  # torch.Size([1, 200, 200, 200])
    print(occ_grid_map.shape)
    
    logging.info("Visualizing the occupancy grid map.")
    mapper.visualize_occupancy_grid(threshold_p_occ=0.5)