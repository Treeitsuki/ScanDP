import os
import numpy as np
import open3d as o3d

def convert_ground_truth(obj_file_path, scale, voxel_size=0.002):
    # Load the STL file and convert to point cloud
    mesh = o3d.io.read_triangle_mesh(obj_file_path)
    mesh.translate(-mesh.get_center())
    # translation = (0, 0, 0.7)
    # translation = (0.0, 0.11, 0.45)  # bike
    translation = (0, 0, 0.4)  # happy
    mesh.translate(translation)
    scale = scale * 1
    mesh.scale(scale, center=mesh.get_center())

    quat = np.array([1, 1, 0, 0])
    R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
    mesh.rotate(R, center=mesh.get_center())
    o3d.visualization.draw_geometries([mesh])
    pcd_model = mesh.sample_points_poisson_disk(number_of_points=100_000)
    pcd_model = pcd_model.voxel_down_sample(voxel_size=voxel_size)
    o3d.visualization.draw_geometries([pcd_model])

    # Save the point cloud to a PLY file
    output_file_path = os.path.splitext(obj_file_path)[0] + "_1_5x" + ".ply"
    o3d.io.write_point_cloud(output_file_path, pcd_model)


if __name__ == "__main__":
    obj_file_path = "/home/cvl/cvl/ScanDP/scandp/diffusion_policy_3d/assets/happy.obj"
    scale = 3
    convert_ground_truth(obj_file_path, scale)