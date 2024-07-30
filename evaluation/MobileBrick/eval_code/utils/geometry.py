import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple, Sequence
from scipy.spatial.transform import Rotation
from typing import Optional


class Simulator:
    def __init__(self):
        fname = "/home/kejie/repository/fast_sdf/dist-model.txt"
        data = np.loadtxt(fname, comments='%', skiprows=5)
        dist = np.empty([80, 80, 5])

        for y in range(0, 80):
            for x in range(0, 80):
                idx = (y * 80 + x) * 23 + 3
                if (data[idx:idx + 5] < 8000).all():
                    dist[y, x, :] = 0
                else:
                    dist[y, x, :] = data[idx + 15: idx + 20]

        self.model = dist

    def undistort(self, x, y, z):

        i2 = int((z + 1) / 2)
        i1 = i2 - 1
        a = (z - (i1 * 2 + 1)) / 2
        x = int(x / 8)
        y = int(y / 6)
        f = (1 - a) * self.model[y, x, min(max(i1, 0), 4)
                                 ] + a * self.model[y, x, min(i2, 4)]

        if f == 0:
            return 0
        else:
            return z / f

    def simulate(self, clean_depth):

        # a = np.array(Image.open(inputpng)).astype(np.float32) / 1000.0
        img_h, img_w = clean_depth.shape
        b = np.copy(clean_depth)
        it = np.nditer(clean_depth, flags=[
                       'multi_index'], op_flags=['writeonly'])

        while not it.finished:

            # pixel shuffle
            x = min(
                max(round(it.multi_index[1] + np.random.normal(0, 0.25)), 0), img_w-1)
            y = min(
                max(round(it.multi_index[0] + np.random.normal(0, 0.25)), 0), img_h-1)

            # downsample
            d = b[y - y % 2, x - x % 2]

            # distortion
            d = self.undistort(x, y, d)

            # quantization and high freq noise
            if d == 0:
                it[0] = 0
            else:
                it[0] = 35.130 * 8 / \
                    round((35.130 / d + np.random.normal(0, 0.027778)) * 8)

            it.iternext()
        return clean_depth


def get_homogeneous(
    pts: Union['np.ndarray', 'torch.tensor']
    ) -> Union['np.ndarray', 'torch.tensor']:
    """ convert [(b), N, 3] pts to homogeneous coordinate

    Args:
        pts ([(b), N, 3] Union['np.ndarray', 'torch.tensor']): input point cloud

    Returns:
        homo_pts ([(b), N, 4] Union['np.ndarray', 'torch.tensor']): output point
            cloud

    Raises:
        ValueError: if the input tensor/array is not with the shape of [b, N, 3]
            or [N, 3]
        TypeError: if input is not either tensor or array
    """

    batch = False
    if len(pts.shape) == 3:
        batch = True
    elif len(pts.shape) == 2:
        pts = pts
    else:
        raise ValueError("only accept [b, n_pts, 3] or [n_pts, 3]")

    if isinstance(pts, torch.Tensor):
        ones = torch.ones_like(pts[..., :1])
        homo_pts = torch.cat([pts, ones], dim=-1)
    elif isinstance(pts, np.ndarray):
        ones = np.ones_like(pts[..., :1])
        homo_pts = np.concatenate([pts, ones], axis=-1)
    else:
        raise TypeError("wrong data type")
    return homo_pts


def get_aabb(pc: 'np.ndarray') -> 'np.ndarray':
    """ get aabb of a point cloud

    Args:
        pc ([N, 3] np.ndarray): input point cloud

    Returns:
        aabb ([2, 3] np.ndarray): a 3D bbox represent by
            [[x_min, y_min, z_min], [x_max, y_max, z_max]]    
    """

    x_min, y_min, z_min = np.min(pc, axis=0)
    x_max, y_max, z_max = np.max(pc, axis=0)
    aabb = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])
    return aabb


# def get_aabb(pc: 'np.ndarray', img_w: int, img_h: int) -> 'np.ndarray':
#     """ get aabb of a point cloud

#     Args:
#         pc ([N, 2] np.ndarray): input point cloud

#     Returns:
#         aabb ([2, 2] np.ndarray): a 2D bbox represent by
#             [[x_min, y_min], [x_max, y_max]]    
#     """

#     x_min, y_min = np.min(pc, axis=0)
#     x_max, y_max = np.max(pc, axis=0)
#     x_min = max(0, x_min)
#     y_min = max(0, y_min)
#     x_max = min(img_w, x_max)
#     y_max = min(img_h, y_max)
#     aabb = np.array([[x_min, y_min], [x_max, y_max]])
#     return aabb


def transform_point_cloud(point_cloud, transform_matrix):

    """
    Transforms a point cloud using a 4x4 transformation matrix.
    
    Parameters
    ----------
    point_cloud : numpy.ndarray, shape (N, 3)
        The input point cloud, represented as a Nx3 numpy array.
    transform_matrix : numpy.ndarray, shape (4, 4)
        The transformation matrix, represented as a 4x4 numpy array.
    
    Returns
    -------
    numpy.ndarray, shape (N, 3)
        The transformed point cloud, represented as a Nx3 numpy array.
    """

    transformed_point_cloud = np.dot(point_cloud, transform_matrix[:3, :3].T) + transform_matrix[:3, 3][np.newaxis, :]
    return transformed_point_cloud


def depth2xyz(depth, intr_mat):
    """ convert depth map to xyz map

    Args:
        depth ([H, W] np.ndarray): depth map
    
    Returns:
        xyz ([H, W, 3] np.ndarray): xyz map
    """

    height, width = depth.shape
    fx, fy, cx, cy = intr_mat[0, 0], intr_mat[1, 1], intr_mat[0, 2], intr_mat[1, 2]

    urange = (
        np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) - cx
    ) / fx
    vrange = (
        np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) - cy
    ) / fy
    xyz = np.stack([urange, vrange, np.ones(urange.shape)], axis=-1)
    xyz = xyz * depth.reshape(height, width, 1)
    return xyz


def angle2class(angles, num_classes=30):
    """ convert angles between [0, 180] to class index for classification
    
    Args:
        angles (np.ndarray): angle in radian
    
    Returns:
        out_class (np.ndarray): angle is converted to class, the number of which
            is defined in num_classes
    """
    y = torch.sin(angles)
    x = torch.cos(angles)
    angles = torch.atan2(y, x) / np.pi * 180.
    angles = torch.where(angles<0, angles + 180, angles)
    out_class = angles // (180 / num_classes)
    assert (out_class >= 0).all()
    assert (out_class <= num_classes).all()
    out_class = np.clip(out_class, a_min=0, a_max=num_classes-1)
    return out_class


def iou_2d(bboxA: 'np.ndarray', bboxB: 'np.ndarray') -> float:
    """ calculate IoU between two 2D bboxes
    
    Args:
        bboxA ([2, 2] np.ndarray): input bbox A in AABB format
        bboxB ([2, 2] np.ndarray): input bbox B in AABB format
        
    Returns:
        IoU (float): output IoU
    """

    x_min = max(bboxA[0, 0], bboxB[0, 0])
    y_min = max(bboxA[0, 1], bboxB[0, 1])
    x_max = min(bboxA[1, 0], bboxB[1, 0])
    y_max = min(bboxA[1, 1], bboxB[1, 1])
    
    inter_area = max(0, (x_max - x_min)) * max(0, (y_max - y_min))
    area_A = np.prod(bboxA[1] - bboxA[0])
    area_B = np.prod(bboxB[1] - bboxB[0])
    IoU = inter_area / (area_A + area_B - inter_area)
    assert IoU <= 1 and IoU >= 0, "invalid IoU value"
    return IoU


def iou_3d(bboxA: 'np.ndarray', bboxB: 'np.ndarray') -> float:
    """ calculate 3D IoU between two 3D bboxes

    Args:
        bboxA ([2, 3] np.ndarray): input bbox A in AABB format
        bboxB ([2, 3] np.ndarray): input bbox B in AABB format
        
    Returns:
        IoU (float): 3D IoU
    """

    x_min = max(bboxA[0, 0], bboxB[0, 0])
    y_min = max(bboxA[0, 1], bboxB[0, 1])
    z_min = max(bboxA[0, 2], bboxB[0, 2])
    x_max = min(bboxA[1, 0], bboxB[1, 0])
    y_max = min(bboxA[1, 1], bboxB[1, 1])
    z_max = min(bboxA[1, 2], bboxB[1, 2])
    
    inter_volume = max(0, (x_max - x_min)) * max(0, (y_max - y_min)) * max(0, (z_max - z_min))
    volume_A = np.prod(bboxA[1] - bboxA[0])
    volume_B = np.prod(bboxB[1] - bboxB[0])
    IoU = inter_volume / (volume_A + volume_B - inter_volume)
    assert IoU <= 1 and IoU >= 0, "invalid IoU value"
    return IoU


def giou_3d(bboxA: 'np.ndarray', bboxB: 'np.ndarray') -> float:
    """ calculate generalized 3D IoU between two 3D bboxes

    Args:
        bboxA ([2, 3] np.ndarray): input bbox A in AABB format
        bboxB ([2, 3] np.ndarray): input bbox B in AABB format
        
    Returns:
        IoU (float): 3D Generalized IoU
    """

    x_min = max(bboxA[0, 0], bboxB[0, 0])
    y_min = max(bboxA[0, 1], bboxB[0, 1])
    z_min = max(bboxA[0, 2], bboxB[0, 2])
    x_max = min(bboxA[1, 0], bboxB[1, 0])
    y_max = min(bboxA[1, 1], bboxB[1, 1])
    z_max = min(bboxA[1, 2], bboxB[1, 2])
    
    inter_volume = max(0, (x_max - x_min)) * max(0, (y_max - y_min)) * max(0, (z_max - z_min))
    volume_A = np.prod(bboxA[1] - bboxA[0])
    volume_B = np.prod(bboxB[1] - bboxB[0])
    volume_union = (volume_A + volume_B - inter_volume)

    iou = iou_3d(bboxA, bboxB)

    x_min = min(bboxA[0, 0], bboxB[0, 0])
    y_min = min(bboxA[0, 1], bboxB[0, 1])
    z_min = min(bboxA[0, 2], bboxB[0, 2])
    x_max = max(bboxA[1, 0], bboxB[1, 0])
    y_max = max(bboxA[1, 1], bboxB[1, 1])
    z_max = max(bboxA[1, 2], bboxB[1, 2])
    
    volume_complete = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    giou = iou - (volume_complete - volume_union) / volume_complete
    return giou


def get_corner_by_dims(dimensions) -> np.ndarray:
    """get 8 corner points of 3D bbox defined by self.dimensions

    Returns:
        a np.ndarray with shape [8,3] to represent 8 corner points'
        position of the 3D bounding box.
    """

    w, h, l = dimensions[0], dimensions[1], dimensions[2]
    x_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    corner_pts = np.array([x_corners, y_corners, z_corners], dtype=np.float32).T
    return corner_pts


def scale_vertices_to_bbox(
    vertices: 'np.ndarray', bbox_dims: 'np.ndarray'
    ) -> 'np.ndarray':
    """scale the vertices such that they are tightly bounded by the 3D bbox
    
    Args:
        vertices ([N, 3] np.ndarray): input vertices
        bbox_dims ([3] np.ndarray): bbox dimension in x, y, z axis
    
    Returns:
        vertices: the scaled vertices
    """

    vertices[:, 0] *= (bbox_dims[0] / (np.max(vertices[:, 0]) - np.min(vertices[:, 0])))
    vertices[:, 1] *= (bbox_dims[1] / (np.max(vertices[:, 1]) - np.min(vertices[:, 1])))
    vertices[:, 2] *= (bbox_dims[2] / (np.max(vertices[:, 2]) - np.min(vertices[:, 2])))

    return vertices


def unproject(pixel, depth, intr_mat):
    """ unproject from pixels and depths to 3D

    Args:
        pixel: [n, 2]
        depth: [n]
    """
    fx = intr_mat[0, 0]
    fy = intr_mat[1, 1]
    cx = intr_mat[0, 2]
    cy = intr_mat[1, 2]
    pts = np.concatenate([pixel, np.ones_like(pixel)[:, :1]], axis=1)
    pts[:, 0] = (pts[:, 0] - cx) / fx
    pts[:, 1] = (pts[:, 1] - cy) / fy
    pts = pts * depth[:, None]
    return pts


def projection(pts, intr_mat, keep_z=False):
    """perspective projection
    
    Args:
        pts ([(b), N, 3] or [(b), N, 4] np.ndarray or torch.tensor): 3D points
        intr_mat ([(b), 3, 3] or [(b), 3, 4] np.ndarray or torch.tensor): intrinsic
            matrix
    
    Returns:
        pts ([(b), N, 3], np.ndarray or torch.tensor): projected points
    """

    batch = False
    if len(pts.shape) == 3:
        assert len(intr_mat.shape) == 3, "intr_mat shape needs to match pts"
        batch = True
    elif len(pts.shape) == 2:
        assert len(intr_mat.shape) == 2, "intr_mat shape needs to match pts"
    else:
        ValueError("only accept [b, n_pts, 3] or [n_pts, 3]")
    if batch:
        if isinstance(pts, torch.Tensor):
            intr_mat = intr_mat.transpose(1, 2)
        else:
            intr_mat = intr_mat.transpose(0, 2, 1)
    else:
        intr_mat = intr_mat.T
    pts = pts @ intr_mat
    if isinstance(pts, torch.Tensor):
        z = torch.ones_like(pts[..., -1])
    else:
        z = np.ones_like(pts[..., -1])
    if batch:
        if keep_z:
            z = pts[:, :, -1]
        pts = pts / pts[:, :, -1:]
        pts[:, :, -1] *= z
    else:
        if keep_z:
            z = pts[:, -1]
        pts = pts / pts[:, -1:]
        pts[:, -1] *= z
    return pts


def pad_transform_matrix(mat: 'np.ndarray') -> 'np.ndarray':
    """ pad a [3, 4] transform matrix to a [4, 4] matrix

    Args:
        mat ([3, 4] np.ndarray): the input [3, 4] matrix
    Returns:
        mat ([4, 4] np.ndarray): the output [4, 4] matrix
    """

    if mat.shape[0] < 4:
        pad = np.zeros((1, 4), dtype=np.float32)
        pad[0,-1] = 1
        return np.concatenate([mat, pad], axis=0)
    else:
        return mat


def rgbd_to_colored_pc(
    rgb: 'np.ndarray',
    depth: 'np.ndarray',
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cap: float = 200) -> Tuple['np.ndarray', 'np.ndarray']:
    """ convert a pair of rgb and depth iamge to a 3D colored point cloud

    Args:
        rgb ([H, W, 3] np.ndarray): rgb image
        depth ([H, W] np.ndarray): depth image
        fx, fy, cx, cy (float, float, float, float): camera intrinsic matrix
        cap (float): depth capping value
    
    Returns:
        a tuple containing:
            points ([N, 3] np.ndarray): 3D point positions
            colors ([N, 3] np.ndarray): color for each point
    """

    rgb_height, rgb_width, _ = rgb.shape
    X, Y = np.meshgrid(np.arange(rgb_width), np.arange(rgb_height))
    xyz_rgb = np.concatenate(
        [X[:, :, None], Y[:, :, None], depth[:, :, None], rgb],
        axis=2
    )
    xyz_rgb[:, :, 0] = (xyz_rgb[:, :, 0] - cx) * xyz_rgb[:, :, 2] / fx
    xyz_rgb[:, :, 1] = (xyz_rgb[:, :, 1] - cy) * xyz_rgb[:, :, 2] / fy
    points = xyz_rgb[:, :, :3].reshape(-1, 3)
    colors = xyz_rgb[:, :, 3:].reshape(-1, 3) / 255.
    cap_ind = np.logical_and((points[:, 2] < cap), (points[:, 2] > 0))
    points = points[cap_ind]
    colors = colors[cap_ind]
    return points, colors


def geodesic_distance(R1: 'np.ndarray', R2: 'np.ndarray') -> float:
    '''Returns the geodesic distance between two rotation matrices.

    Args:
        R1 ([3, 3] np.ndarray): input rotation matrix
        R2 ([3, 3] np.ndarray): input rotation matrix
    
    Returns:
        delta_theta (float): geodesic distance between the input rotation
            matrices
    '''

    delta_R = np.dot(R1, R2.T)
    rotvec = Rotation.from_dcm(delta_R).as_rotvec()
    delta_theta = np.linalg.norm(rotvec)
    return delta_theta


def pts_in_box(pts: 'np.ndarray', img_shape: 'np.ndarray') -> 'np.ndarray':
    """ check projected points are within image frame

    Args:
        pts ([N, 2] np.ndarray): a set of 2D points on image plane
        img_shape (aabb): bbox_size [x_min, y_min, x_max, y_max]
    Return:
        a boolean array of shape [N] indicating whether a point is within
        image frame
    """

    img_shape = img_shape.reshape(2, 2)
    larger_x_min = pts[:, 0] > img_shape[0, 0]
    smaller_x_max = pts[:, 0] < img_shape[1, 0]
    larger_y_min = pts[:, 1] > img_shape[0, 1]
    smaller_y_max = pts[:, 1] < img_shape[1, 1]
    return (larger_x_min * smaller_x_max * \
        larger_y_min * smaller_y_max)


def get_normalize_factor(pts):
    norm = np.max(np.linalg.norm(pts, axis=1))
    return norm


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.
        device (torch.device): the device on which the grid will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated gird. Default: float32.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def calculate_normals_finite_difference(pts, func):
    """
    compute the normal of an implicit reprentation using finite difference

    pts: (B, N, 3)

    """

    batch_size, n_pts = pts.shape[:2]
    epsilon = np.abs(np.random.normal(scale=0.01, size=(batch_size, n_pts)))
    epsilon = torch.from_numpy(epsilon).float().to(pts.device)
    epsilon = torch.clamp(epsilon, min=1e-3, max=1)
    epsilon_x = torch.stack(
        [epsilon, torch.zeros_like(epsilon), torch.zeros_like(epsilon)],
        dim=-1
    ).detach()
    epsilon_y = torch.stack(
        [torch.zeros_like(epsilon), epsilon, torch.zeros_like(epsilon)],
        dim=-1
    ).detach()
    epsilon_z = torch.stack(
        [torch.zeros_like(epsilon), torch.zeros_like(epsilon), epsilon],
        dim=-1
    ).detach()

    pts_x0 = pts - epsilon_x
    pts_x1 = pts + epsilon_x

    pts_y0 = pts - epsilon_y
    pts_y1 = pts + epsilon_y

    pts_z0 = pts - epsilon_z
    pts_z1 = pts + epsilon_z
    f_out = func(
        torch.cat([pts_x0, pts_x1, pts_y0, pts_y1, pts_z0, pts_z1], dim=0)
    )
    f_x0, f_x1, f_y0, f_y1, f_z0, f_z1 = torch.split(f_out, batch_size, dim=0)

    g_x = (f_x1 - f_x0) / (2 * epsilon)
    g_y = (f_y1 - f_y0) / (2 * epsilon)
    g_z = (f_z1 - f_z0) / (2 * epsilon)

    normals = torch.stack([g_x, g_y, g_z], dim=-1)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-5)
    return normals


def recenter(vertices):
    min_ = np.min(vertices, axis=0)
    max_ = np.max(vertices, axis=0)
    center = (max_ + min_) / 2.
    vertices = vertices - center[None, :]
    return vertices, center