# =============================================================================
#  Imports
# =============================================================================

import torch
import numpy as np
import open3d as o3d

# =============================================================================
#  Functions for dealing with input/output
# =============================================================================

def read_ply(filename):
    """
    Read a PLY file and return points and colors as tensors.

    Parameters:
    filename (str): Path to the PLY file.

    Returns:
    tuple: Points and colors as tensors.
    """
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return torch.tensor(points, dtype=torch.float32), torch.tensor(colors, dtype=torch.float32)