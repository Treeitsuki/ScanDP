import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def compute_fpfh_feature(pcd, voxel_size=0.005):
    """Compute FPFH feature for a given point cloud."""
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return fpfh


def ransac_registration(source, target, source_feature, target_feature, threshold=0.05):
    """Perform RANSAC-based alignment using FPFH features."""
    ransac_n = 4
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold),
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feature, target_feature,
        mutual_filter=False,
        max_correspondence_distance=threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=ransac_n,
        checkers=checkers,
        criteria=criteria
    )
    return result.transformation


def evaluation_point_cloud(stl_file_path, ply_file_path, threshold_ransac=0.1, threshold_dis=0.002):
    # Load and preprocess the STL file
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    mesh.translate(-mesh.get_center())
    translation = (0, 0, 0.7)
    mesh.translate(translation)
    mesh.scale(0.0025, center=mesh.get_center())

    quat = np.array([1, 1, 0, 0])
    R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
    mesh.rotate(R, center=mesh.get_center())

    pcd_model = mesh.sample_points_poisson_disk(number_of_points=100_000)
    pcd_model = pcd_model.voxel_down_sample(voxel_size=0.002)

    # Load and preprocess the PLY file
    pcd_input = o3d.io.read_point_cloud(ply_file_path)
    cl, ind = pcd_input.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
    pcd_input = pcd_input.select_by_index(ind)
    pcd_input = pcd_input.voxel_down_sample(voxel_size=0.002)

    o3d.visualization.draw_geometries([pcd_model, pcd_input])

    # Compute FPFH features
    model_fpfh = compute_fpfh_feature(pcd_model)
    input_fpfh = compute_fpfh_feature(pcd_input)

    # Perform RANSAC-based alignment
    transformation_ransac = ransac_registration(pcd_input, pcd_model, input_fpfh, model_fpfh, threshold_ransac)
    pcd_input.transform(transformation_ransac)
    o3d.visualization.draw_geometries([pcd_model, pcd_input])

    # Compute coverage ratio
    model_points = np.asarray(pcd_model.points)
    # model_points = model_points[model_points[:, 2] > 0.56]
    input_points = np.asarray(pcd_input.points)

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
    stl_file_path = os.path.join(os.getcwd(), "scandp/diffusion_policy_3d/assets", "armadillo.obj")
    ply_file_path = "/home/cvl/cvl/ScanDP/scandp/data/armadillo_wobatch.ply"

    proportion = evaluation_point_cloud(stl_file_path, ply_file_path)
    coverage = proportion * 100
    print(f"Final Coverage Proportion: {coverage:.4f}")
