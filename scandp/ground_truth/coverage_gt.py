import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def evaluation_point_cloud(stl_file_path, ply_file_path, threshold_icp=1, threshold_dis=0.002):
    # Load ground truth ply file
    pcd_model = o3d.io.read_point_cloud(stl_file_path)

    # Load the PLY file
    pcd_input = o3d.io.read_point_cloud(ply_file_path)
    cl, ind = pcd_input.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
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
    print(
        f"Model Points: {model_points.shape}, Input Points: {input_points.shape}")

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


def eval_inference(taeget_name, pcd, scale=1, threshold_icp=1, threshold_dis=0.002, visualize=False, noise_remove=False):
    # Load ground truth ply file
    if scale == 1:
        asset_folder = "/home/user/workspace/scandp/diffusion_policy_3d/assets/ply"
        model_path = os.path.join(asset_folder, f"{taeget_name}.ply")
    elif scale == 1.5:
        asset_folder = "/home/user/workspace/scandp/diffusion_policy_3d/assets/ply_1_5x"
        model_path = os.path.join(asset_folder, f"{taeget_name}_1_5x.ply")
    else:
        raise ValueError("Scale must be either 1 or 1.5")
    pcd_model = o3d.io.read_point_cloud(model_path)

    # Load the PLY file
    pcd_input = pcd
    if noise_remove:
        cl, ind = pcd_input.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        pcd_input = pcd_input.select_by_index(ind)
    # o3d.visualization.draw_geometries([pcd_input])
    pcd_input = pcd_input.voxel_down_sample(voxel_size=0.002)
    # o3d.visualization.draw_geometries([pcd_model, pcd_input])

    # Perform ICP (Iterative Closest Point) alignment
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_input, pcd_model, threshold_icp, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation_icp = reg_p2p.transformation
    pcd_input.transform(transformation_icp)
    # o3d.visualization.draw_geometries([pcd_model, pcd_input])

    # Compute coverage ratio
    model_points = np.asarray(pcd_model.points)
    input_points = np.asarray(pcd_input.points)

    # Use KD-Tree for fast nearest neighbor search
    kdtree = cKDTree(input_points)
    distances, _ = kdtree.query(model_points)
    covered_points = model_points[distances < threshold_dis]
    uncovered_points = model_points[distances > threshold_dis]
    coverage_ratio = len(covered_points) / len(model_points)

    # Visualize the aligned point clouds
    if visualize:
        pcd_covered = o3d.geometry.PointCloud()
        pcd_covered.points = o3d.utility.Vector3dVector(covered_points)
        pcd_uncovered = o3d.geometry.PointCloud()
        pcd_uncovered.points = o3d.utility.Vector3dVector(uncovered_points)
        pcd_covered.paint_uniform_color([0.253, 0.558, 0.867])  # Blue
        pcd_uncovered.paint_uniform_color([0.945, 0.660, 0])  # Yellow
        pcd_result = pcd_covered + pcd_uncovered
        o3d.visualization.draw_geometries([pcd_covered, pcd_uncovered])
    else:
        pcd_result = None

    return coverage_ratio, pcd_result


if __name__ == '__main__':
    # stl_file_path = os.path.join(os.getcwd(), "scandp/diffusion_policy_3d/assets", "stanfordbunny.stl")
    stl_file_path = os.path.join(
        os.getcwd(), "scandp/diffusion_policy_3d/assets/ply", "teapot.ply")
    ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/teapot_hemi_uni.ply"
    # ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/result/idp/spot_idp.ply"
    # ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/result/spconv/bunny_spconv_50.ply"

    proportion = evaluation_point_cloud(stl_file_path, ply_file_path)
    coverage = proportion * 100
    print(f"Final Coverage Proportion: {coverage:.4f}")
