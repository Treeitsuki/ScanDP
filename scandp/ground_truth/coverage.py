import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def evaluation_point_cloud(stl_file_path, ply_file_path, threshold_icp=1, threshold_dis=0.002):
    # Load the STL file and convert to point cloud
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    mesh.translate(-mesh.get_center())
    translation = (0, 0, 0.7)
    # translation = (0.0, 0.11, 0.45) # bike
    mesh.translate(translation)

    # mesh.scale(3.0, center=mesh.get_center())  # Bunny
    # mesh.scale(0.0025, center=mesh.get_center())   # Armadillo
    # mesh.scale(0.28, center=mesh.get_center()) # Spot
    # mesh.scale(0.07, center=mesh.get_center())  # teapot
    # mesh.scale(0.002, center=mesh.get_center())  # bust
    # mesh.scale(0.00023, center=mesh.get_center())  # bike

    quat = np.array([1, 1, 0, 0])
    R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
    mesh.rotate(R, center=mesh.get_center())
    pcd_model = mesh.sample_points_poisson_disk(number_of_points=100_000)
    pcd_model = pcd_model.voxel_down_sample(voxel_size=0.002)

    # Load the PLY file
    pcd_input = o3d.io.read_point_cloud(ply_file_path)
    cl, ind = pcd_input.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_input = pcd_input.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd_input])
    pcd_input = pcd_input.voxel_down_sample(voxel_size=0.002)
    o3d.visualization.draw_geometries([pcd_model, pcd_input])

    # Perform ICP (Iterative Closest Point) alignment
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_input, pcd_model, threshold_icp, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation_icp = reg_p2p.transformation
    pcd_input.transform(transformation_icp)
    o3d.visualization.draw_geometries([pcd_model, pcd_input])

    # Compute coverage ratio
    model_points = np.asarray(pcd_model.points)
    # model_points = model_points[model_points[:, 2] > 0.56]
    input_points = np.asarray(pcd_input.points)
    # print(len(np.nonzero(model_points[:, 2] < 0.55)[0]) / model_points.shape[0])
    print(f"Model Points: {model_points.shape}, Input Points: {input_points.shape}")

    # Use KD-Tree for fast nearest neighbor search
    kdtree = cKDTree(input_points)
    distances, _ = kdtree.query(model_points)
    covered_points = model_points[distances < threshold_dis]
    uncovered_points = model_points[distances > threshold_dis]

    coverage_ratio = len(covered_points) / len(model_points)

    # Visualize the aligned point clouds
    pcd_covered = o3d.geometry.PointCloud()
    pcd_covered.points = o3d.utility.Vector3dVector(covered_points)
    pcd_uncovered = o3d.geometry.PointCloud()
    pcd_uncovered.points = o3d.utility.Vector3dVector(uncovered_points)
    
    pcd_covered.paint_uniform_color([0.253, 0.558, 0.867])  # Blue
    pcd_uncovered.paint_uniform_color([0.945, 0.660, 0])  # Yellow
    o3d.visualization.draw_geometries([pcd_covered, pcd_uncovered])

    return coverage_ratio


if __name__ == '__main__':
    # stl_file_path = os.path.join(os.getcwd(), "scandp/diffusion_policy_3d/assets", "stanfordbunny.stl")
    stl_file_path = os.path.join(os.getcwd(), "scandp/diffusion_policy_3d/assets", "armadillo.obj")
    ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/armadillo_spconv_08m_x15.ply"
    # ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/result/idp/spot_idp.ply"
    # ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/result/spconv/bunny_spconv_50.ply"
    
    proportion = evaluation_point_cloud(stl_file_path, ply_file_path)
    coverage = proportion * 100
    print(f"Final Coverage Proportion: {coverage:.4f}")
