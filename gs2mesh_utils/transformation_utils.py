# =============================================================================
#  Imports
# =============================================================================

import numpy as np
import os
import cv2
import math

# =============================================================================
#  Small functions for correcting numerical errors
# =============================================================================

ZERO = 1e-7
fix_zero = lambda x: np.where(np.abs(x) < ZERO, 0, x)
round_float = lambda x: round(x, -int(math.floor(math.log10(abs(x)))) + 1) if x != 0 else 0

# =============================================================================
#  Helper functions to deal with various 2D/3D transformations
# =============================================================================

def RT_from_rot_pos(rot, pos):
    """
    Create a rotation-translation matrix from rotation (Euler angles) and position.

    Parameters:
    rot (tuple): Rotation in Euler angles.
    pos (tuple): Position vector.

    Returns:
    np.ndarray: 4x4 rotation-translation matrix.
    """
    R = eul2rotm(rot)
    R[:, 1:] *= -1
    T = np.array(pos)
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = T
    return RT

def convert_R_T_to_GS(R, T):
    """
    Convert R and T to a format compatible with Gaussian Splatting code.

    Parameters:
    blender_R (tuple): Original rotation.
    blender_T (tuple): Original translation.

    Returns:
    tuple: Rotation matrix and translation vector in Gaussian Splatting format.
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = eul2rotm(R)
    Rt[:3, 3] = np.asarray(T, dtype=np.float32)
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    GS_T = W2C[:3, 3]
    GS_T[1:] *= -1
    GS_R = W2C[:3, :3].transpose()
    GS_R[:, 1:] *= -1
    return GS_R, GS_T

def intrinsic_from_camera_params(camera_params):
    """
    Create an intrinsic matrix from camera parameters (fx,fy,cx,cy).

    Parameters:
    camera_params (dict): Dictionary containing camera parameters.

    Returns:
    np.ndarray: Intrinsic matrix.
    """
    return np.array([[camera_params['fx'], 0, camera_params['cx']],
                     [0, camera_params['fy'], camera_params['cy']],
                     [0, 0, 1]])

def eul2rotm(R):
    """
    Convert Euler angles to a rotation matrix.

    Parameters:
    R (tuple): Euler angles.

    Returns:
    np.ndarray: Rotation matrix.
    """
    rotx, roty, rotz = np.radians(R)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rotx), -np.sin(rotx)],
        [0, np.sin(rotx), np.cos(rotx)]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(roty), 0, np.sin(roty)],
        [0, 1, 0],
        [-np.sin(roty), 0, np.cos(roty)]
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(rotz), -np.sin(rotz), 0],
        [np.sin(rotz), np.cos(rotz), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R = Rz @ Ry @ Rx
    return fix_zero(R)

def rotm2eul(R):
    """
    Convert a rotation matrix to Euler angles.

    Parameters:
    R (np.ndarray): Rotation matrix.

    Returns:
    np.ndarray: Euler angles.
    """
    R = np.asarray(R, dtype=np.float32)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    out = np.degrees([x, y, z])
    return fix_zero(out)

def depth_image_to_point_cloud(depth, K):
    """
    Project 2D points into 3D space using the intrinsic matrix K.

    Parameters:
    depth (np.ndarray): Depth image.
    K (np.ndarray): Intrinsic matrix.

    Returns:
    np.ndarray: Point cloud.
    """
    h, w = depth.shape
    i, j = np.meshgrid(range(w), range(h), indexing='xy')
    points = np.stack([i, j, np.ones_like(i)], axis=-1).reshape(-1, 3)
    points = np.dot(np.linalg.inv(K), points.T) * depth.flatten()
    return points.T

def project_points_to_image(points, K):
    """
    Project 3D points onto a 2D image plane using the intrinsic matrix K.

    Parameters:
    points (np.ndarray): 3D points.
    K (np.ndarray): Intrinsic matrix.

    Returns:
    np.ndarray: 2D points in image coordinates.
    """
    points = np.dot(K, points.T).T
    points[:, :2] /= points[:, 2:3]
    return points[:, :2]

def transform_points(points, R, T):
    """
    Transform points given a rotation R and translation T.

    Parameters:
    points (np.ndarray): Points to transform.
    R (np.ndarray): Rotation matrix.
    T (np.ndarray): Translation vector.

    Returns:
    np.ndarray: Transformed points.
    """
    return points @ R.T + T
    
def calculate_right_camera_pose(R_left, T_left, baseline):
    """
    Calculate the pose of a stereo-calibrated right camera given the left camera pose and the horizontal baseline.

    Parameters:
    R_left (tuple): Euler angles for the left camera rotation.
    T_left (tuple): Translation vector for the left camera.
    baseline (float): Horizontal baseline for the stereo pair.

    Returns:
    tuple: Rotation and translation for the right camera.
    """
    R_mat = eul2rotm(R_left)
    baseline_vector = np.array([baseline, 0, 0], dtype=np.float32)
    rotated_baseline = R_mat @ baseline_vector
    T_left_tensor = np.array(T_left, dtype=np.float32)
    T_right = T_left_tensor + rotated_baseline
    return tuple(R_left), tuple(fix_zero(T_right).tolist())

def get_shading(img, shading_eps):
    """
    Compute the shading magnitude image - depth gradient. Used as an indication for stereo matching quality.

    Parameters:
    img (np.ndarray): Input image.
    shading_eps (float): Small value to avoid division by zero.

    Returns:
    np.ndarray: Shading magnitude image.
    """
    gX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    magnitude = 1 / (np.sqrt((gX ** 2) + (gY ** 2) + shading_eps))
    return magnitude

def project_depth_image(depth, mask, K1, K2, R1, T1, R2, T2):
    """
    Project a depth image onto another camera view. Used for the masking process.

    Parameters:
    depth (np.ndarray): Depth image.
    mask (np.ndarray): The mask being projected.
    K1 (np.ndarray): Intrinsic matrix of the first camera.
    K2 (np.ndarray): Intrinsic matrix of the second camera.
    R1 (np.ndarray): Rotation matrix of the first camera.
    T1 (np.ndarray): Translation vector of the first camera.
    R2 (np.ndarray): Rotation matrix of the second camera.
    T2 (np.ndarray): Translation vector of the second camera.

    Returns:
    np.ndarray: Projected depth image.
    """
    points = depth_image_to_point_cloud(depth * mask, K2)
    points_world = transform_points(points, R2, T2)
    points_camera1 = transform_points(points_world, np.linalg.inv(R1), -np.dot(np.linalg.inv(R1), T1))
    projected_points = project_points_to_image(points_camera1, K1)
    
    depth_image_rendered = np.full_like(depth, np.nan, dtype=np.float32)
    
    x_indices = np.round(projected_points[:, 0]).astype(np.int32)
    y_indices = np.round(projected_points[:, 1]).astype(np.int32)
    
    valid_mask = (x_indices >= 0) & (x_indices < depth_image_rendered.shape[1]) & (y_indices >= 0) & (y_indices < depth_image_rendered.shape[0])
    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    z_values = points_camera1[valid_mask, 2]
    
    sorted_indices = np.argsort(z_values)
    x_indices = x_indices[sorted_indices]
    y_indices = y_indices[sorted_indices]
    z_values = z_values[sorted_indices]
    
    depth_image_rendered[y_indices, x_indices] = z_values
    depth_image_rendered[np.isnan(depth_image_rendered)] = 0
    return depth_image_rendered