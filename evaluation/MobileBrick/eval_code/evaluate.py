# =============================================================================
#  Imports
# =============================================================================

import os
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import open3d as o3d
import numpy as np

import utils.o3d_helper as o3d_helper

# =============================================================================
#  Helper Functions
# =============================================================================

def compute_curvature(points, radius=0.005):
    tree = KDTree(points)
    curvature = [0] * points.shape[0]
    for index, point in enumerate(points):
        indices = tree.query_ball_point(point, radius)
        if len(indices) < 3:
            print("invalid points")
            continue
        M = np.array([points[i] for i in indices]).T
        M = np.cov(M)
        V, E = np.linalg.eig(M)
        h1, h2, h3 = V
        curvature[index] = h3 / (h1 + h2 + h3)
    return np.asarray(curvature)

def visibility_test(volume, min_pts, resolution, voxel_size, mesh, device):
    points = np.asarray(mesh.vertices)
    volume = torch.from_numpy(volume).float().to(device)
    voxels = (points - min_pts) / voxel_size
    voxels = voxels / (resolution - 1) * 2 - 1
    voxels = torch.from_numpy(voxels)[..., [2, 1, 0]].float().to(device)
    mask = F.grid_sample(volume.unsqueeze(0).unsqueeze(0), voxels.unsqueeze(0).unsqueeze(0).unsqueeze(0), mode="nearest", padding_mode="zeros", align_corners=True)
    mask = mask[0, 0, 0, 0].cpu().numpy() > 0
    mesh.remove_vertices_by_mask(mask == False)
    mesh.compute_vertex_normals()
    return mesh

def evaluate(pred_points, gt_points, threshold, verbose=False):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(gt_points)
    distances, indices = nbrs.kneighbors(pred_points)
    pred_gt_dist = np.mean(distances)
    precision = np.sum(distances < threshold) / len(distances)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pred_points)
    distances, indices = nbrs.kneighbors(gt_points)
    gt_pred_dist = np.mean(distances)
    recall = np.sum(distances < threshold) / len(distances)
    F1 = 2 * precision * recall / (precision + recall)
    chamfer = pred_gt_dist + gt_pred_dist
    if verbose:
        print("precision @ {}: {:.6f}".format(threshold, precision))
        print("recall @ {}: {:.6f}".format(threshold, recall))
        print("F1: {:.6f}".format(F1))
        print("Chamfer: {:.6f}".format(chamfer))
    out = {'pred_gt': pred_gt_dist, 'accuracy': precision, 'gt_pred': gt_pred_dist, 'recall': recall, 'chamfer': chamfer, 'F1': F1}
    return out

def sample_surface_points(mesh, n_samples):
    return np.asarray(mesh.sample_points_poisson_disk(n_samples).points)

# =============================================================================
#  Evaluation Function
# =============================================================================

def evaluate_single(gt_dir, pred_path, exp_path, scan_name, device='cuda'):
    visibility_mask = np.load(os.path.join(gt_dir, "visibility_mask.npy"), allow_pickle=True).item()
    resolution = visibility_mask['resolutions']
    volume = visibility_mask['mask'].reshape(resolution)
    voxel_size = visibility_mask['voxel_size']
    min_pts = visibility_mask['min_pts']
    gt_mesh = o3d.io.read_triangle_mesh(os.path.join(gt_dir, "mesh", "gt_mesh.ply"))
    gt_points = sample_surface_points(gt_mesh, 100000)
    pred_mesh = o3d.io.read_triangle_mesh(pred_path)

    # Perform ICP registration
    gt_pts = o3d_helper.np2pc(gt_mesh.vertices)
    pred_pts = o3d_helper.np2pc(pred_mesh.vertices)
    threshold = 0.02
    trans_init = np.eye(4)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        gt_pts, pred_pts, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10))
    if reg_p2l.fitness > 0.99:
        new_pose = reg_p2l.transformation
        pred_mesh.transform(np.linalg.inv(new_pose))

    pred_mesh = visibility_test(volume, min_pts, resolution, voxel_size, pred_mesh, device)
    if len(np.asarray(pred_mesh.triangles)) > 0:
        pred_points = sample_surface_points(pred_mesh, 100000)
    else:
        pred_points = np.random.permutation(np.asarray(pred_mesh.vertices))[:100000]
        
    cropped_mesh_path = os.path.join(exp_path, f"{scan_name}_cropped.ply")
    o3d.io.write_triangle_mesh(cropped_mesh_path, pred_mesh)
    
    out_2_5mm = evaluate(pred_points, gt_points, threshold=0.0025, verbose=False)        
    out_5mm = evaluate(pred_points, gt_points, threshold=0.005, verbose=False)
    out = [out_2_5mm['chamfer'], out_2_5mm['accuracy'], out_2_5mm['recall'], out_2_5mm['F1'], out_5mm['accuracy'], out_5mm['recall'], out_5mm['F1']]
    
    return out