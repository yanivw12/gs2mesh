# =============================================================================
#  Imports
# =============================================================================

import torch
import numpy as np
import os
import json
import cv2
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import copy

from gs2mesh_utils.colmap_utils import poses_from_file
from gs2mesh_utils.transformation_utils import rotm2eul, eul2rotm, intrinsic_from_camera_params, RT_from_rot_pos, convert_R_T_to_GS, calculate_right_camera_pose
from gs2mesh_utils.io_utils import read_ply
import gs2mesh_utils.third_party.visualization.visualize as visualize
from gs2mesh_utils.third_party.colmap_runner.utils.read_write_model import read_cameras_text

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'third_party', 'gaussian-splatting')))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import Scene, GaussianModel, cameras

# =============================================================================
#  Helper functions for camera sorting
# =============================================================================

def find_nearest_neighbors(current_index, coordinates, visited):
    """
    Find the nearest neighbors to the current camera.

    Parameters:
    current_index (int): Index of the current camera.
    coordinates (np.ndarray): Array of camera coordinates.
    visited (np.ndarray): Array indicating whether each camera has been visited.

    Returns:
    np.ndarray: Indices of the nearest neighbors.
    """
    distances = np.linalg.norm(coordinates - coordinates[current_index], axis=1)
    distances[visited] = np.inf
    distances[current_index] = np.inf
    nearest_indices = np.argsort(distances)[:2]
    return nearest_indices

def choose_by_close_z(current_index, candidates, coordinates):
    """
    Choose the next camera based on the closest z-coordinate.

    Parameters:
    current_index (int): Index of the current camera.
    candidates (np.ndarray): Indices of candidate next cameras.
    coordinates (np.ndarray): Array of camera coordinates.

    Returns:
    int: Index of the chosen camera.
    """
    z_diff = np.abs(coordinates[candidates][:, 2] - coordinates[current_index][2])
    return candidates[np.argmin(z_diff)]

# =============================================================================
#  Camera sorting
# =============================================================================

def sort_camera_coordinates(coordinates):
    """
    Sort camera coordinates to create a sequence of neighboring cameras.

    Parameters:
    coordinates (np.ndarray): Array of unsorted camera coordinates.

    Returns:
    list: Sorted order of camera indices.
    """
    visited = np.zeros(len(coordinates), dtype=bool)
    order = []
    
    # Start with the camera with the lowest z-coordinate
    current_index = np.argmin(coordinates[:, 2])
    
    while not np.all(visited):
        visited[current_index] = True
        order.append(current_index)
        
        if np.all(visited):
            break
        
        nearest_neighbors = find_nearest_neighbors(current_index, coordinates, visited)
        if len(nearest_neighbors) == 0:
            break
        
        # Choose the next camera based on closer z-coordinate
        current_index = choose_by_close_z(current_index, nearest_neighbors, coordinates)
    
    return order

# =============================================================================
#  Class for Gaussian Splatting renderer
# =============================================================================

class Renderer:
    def __init__(self, base_dir, colmap_dir, output_dir_root, render_name, dataset='custom', splatting='custom', experiment_name=None, splatting_iteration=30000, white_background=False, baseline_absolute=None, baseline_percentage=7, folder_name=None, save_json=True, sort_cameras=False, device='cuda'):
        """
        Initialize the Renderer class.

        Parameters:
        base_dir (str): Base directory of the repository.
        colmap_dir (str): Directory containing the COLMAP sparse model.
        output_dir_root (str): Root directory of the output.
        render_name (str): Name of the specific scan name.
        dataset (str): Name of the dataset.
        splatting (str): name of the splatting output folder.
        experiment_name (str): Name of the experiment. Use if you want to override the default name.
        splatting_iteration (int): The Gaussian Splatting iteration from which the point cloud will be taken. Default 30000.
        white_background (bool): Flag for Gaussian Splatting to use/not use an artificial white background.
        baseline_absolute (float): The horizontal stereo baseline as an absolute value. defualt is None, which forces usage of baseline_percentage.
        baseline_percentage (float): The horizontal stereo baseline as percentage of the scene radius. defualt is using 7% of the scene radius.
        folder_name (str): Name of the folder for saving the renders.
        save_json (bool): Flag to save camera data as JSON.
        device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.render_name = render_name
        self.white_background = white_background
        self.base_dir = base_dir
        self.colmap_dir = colmap_dir
        self.output_dir_root = output_dir_root
        self.device = device
        self.splatting_iteration = splatting_iteration
        self.splatting_dir = os.path.join(base_dir, 'splatting_output', splatting, render_name)
        splatting_ply_file_path = os.path.join(self.splatting_dir, 'point_cloud', f"iteration_{self.splatting_iteration}", 'point_cloud.ply')
        
        self.poses = poses_from_file(os.path.join(self.colmap_dir,'sparse','0','images.txt'))
        
        poses_inv = [np.linalg.inv(np.vstack((pose, np.array([0, 0, 0, 1])))) for pose in self.poses]
        camera_rotations = [rotm2eul(pose[:3,:3]) for pose in poses_inv]
        for i in range(len(camera_rotations)):
            rotation = eul2rotm(camera_rotations[i])
            rotation[:,1:]*=-1
            camera_rotations[i] = rotm2eul(rotation)

        camera_locations = [pose[:3,3].tolist() for pose in poses_inv]
        
        camera_params = read_cameras_text(os.path.join(self.colmap_dir,'sparse','0','cameras.txt'))
        camera_params = [{'width':camera_params[i].width, 
                          'height':camera_params[i].height, 
                          'fx':camera_params[i].params[0], 
                          'fy':camera_params[i].params[0 if camera_params[i].model=='SIMPLE_RADIAL' else 1], 
                          'cx':camera_params[i].params[1 if camera_params[i].model=='SIMPLE_RADIAL' else 2], 
                          'cy':camera_params[i].params[2 if camera_params[i].model=='SIMPLE_RADIAL' else 3]} for i in range(1,1+len(camera_params))] 
        if len(camera_params)!=len(camera_locations):
            camera_params = [camera_params[0]]*len(camera_locations)
            
        if baseline_absolute is not None:
            self.baseline = baseline_absolute
        else:
            ts = np.array(camera_locations)
            self.baseline = np.median(np.linalg.norm(ts - ts.mean(axis=0),axis=1)) * (baseline_percentage/100)

        if sort_cameras:
            self.sorted_camera_indices = sort_camera_coordinates(np.array(camera_locations))
            self.poses = self.poses[torch.tensor(self.sorted_camera_indices)]
        else:
            self.sorted_camera_indices = range(len(camera_locations))

        self.cameras = []
        for i in range(len(camera_locations)):
            camera_index = self.sorted_camera_indices[i]
            R_right, T_right = calculate_right_camera_pose(camera_rotations[camera_index], camera_locations[camera_index], self.baseline)
            self.cameras.append({'left':
                               {'rot':tuple(camera_rotations[camera_index].tolist()),
                                'pos':tuple(camera_locations[camera_index]), 
                                'width':camera_params[camera_index]['width'], 
                                'height':camera_params[camera_index]['height'], 
                                'fx':camera_params[camera_index]['fx'].item(), 
                                'fy':camera_params[camera_index]['fy'].item(), 
                                'cx':camera_params[camera_index]['cx'].item(), 
                                'cy':camera_params[camera_index]['cy'].item(),
                                'intrinsic': intrinsic_from_camera_params(camera_params[camera_index]),
                                'extrinsic': RT_from_rot_pos(tuple(camera_rotations[camera_index]),tuple(camera_locations[camera_index])),
                                'baseline': self.baseline
                               }, 'right':
                               {'rot':R_right, 
                                'pos':T_right, 
                                'width':camera_params[camera_index]['width'], 
                                'height':camera_params[camera_index]['height'], 
                                'fx':camera_params[camera_index]['fx'].item(), 
                                'fy':camera_params[camera_index]['fy'].item(), 
                                'cx':camera_params[camera_index]['cx'].item(), 
                                'cy':camera_params[camera_index]['cy'].item(),
                                'intrinsic': intrinsic_from_camera_params(camera_params[camera_index]),
                                'extrinsic': RT_from_rot_pos(tuple(camera_rotations[camera_index]),tuple(camera_locations[camera_index]))
                               }
                              })    
        
        print(f"num views: {len(self.cameras)}")
        print(f"baseline: {self.baseline}")
        
        self.left_cameras = [camera['left'] for camera in self.cameras]

        if save_json:
            self.save_camera_data()
        
        self.GS_ply_points, _ = read_ply(splatting_ply_file_path)
    
    def __len__(self):
        """
        Get the number of cameras.

        Returns:
        int: Number of cameras.
        """
        return len(self.cameras)
    
    def visualize_poses(self, depth_scale=1, subsample=100):
        """
        Visualize camera poses using Plotly.

        Parameters:
        depth_scale (float): Adjust according to scale of the scene, in order to see the cameras.
        subsample (int): Subsampling factor to reduce the number of visualized points. Use a larger value if your mesh is large.

        Returns:
        None
        """
        vis_depth = 0.02 * depth_scale
        xyzs = self.GS_ply_points[::subsample]
        x, y, z = *xyzs.T,
        
        traces_poses = visualize.plotly_visualize_pose(self.poses, vis_depth=vis_depth, xyz_length=0.02, center_size=0.01, xyz_width=0.005, mesh_opacity=0.05)
        trace_points = go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=1, opacity=1))
        traces_all = traces_poses+[trace_points]
        layout = go.Layout(scene=dict(xaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                                      yaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                                      zaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                                      xaxis_title="X", yaxis_title="Y", zaxis_title="Z", dragmode="orbit",
                                      aspectratio=dict(x=1, y=1, z=1), aspectmode="data"), height=800)
        fig = go.Figure(data=traces_all, layout=layout)
        
        fig.show()

    def render_folder_name(self, render_number):
        """
        Generate the folder name for a specific render number.

        Parameters:
        render_number (int): Render number.

        Returns:
        str: Output folder path corresponding to the render number.
        """
        return os.path.join(self.output_dir_root, f"{render_number:03}")

    def save_camera_data(self):
        """
        Save camera data to a JSON file.

        Returns:
        None
        """
        os.makedirs(self.output_dir_root, exist_ok=True)
        camera_data_path = os.path.join(self.output_dir_root,'camera_data.json')
        cameras_for_save = copy.deepcopy(self.cameras)
        for i in range(len(cameras_for_save)):
            cameras_for_save[i]['left']['intrinsic'] = cameras_for_save[i]['left']['intrinsic'].tolist()
            cameras_for_save[i]['left']['extrinsic'] = cameras_for_save[i]['left']['extrinsic'].tolist()
            cameras_for_save[i]['right']['intrinsic'] = cameras_for_save[i]['right']['intrinsic'].tolist()
            cameras_for_save[i]['right']['extrinsic'] = cameras_for_save[i]['right']['extrinsic'].tolist()
        with open(camera_data_path, 'w') as f:
            json.dump(cameras_for_save, f, indent=4)
            
    def prepare_renderer(self):
        """
        Prepare the renderer for generating images.

        Returns:
        None
        """
        parser = ArgumentParser(description="Testing script parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        args = Namespace(compute_cov3D_python=False, 
                         convert_SHs_python=False, 
                         data_device=self.device, 
                         debug=False, 
                         eval=False, 
                         feature_dim=32, 
                         feature_model_path='', 
                         idx=0, 
                         images='images', 
                         init_from_3dgs_pcd=False, 
                         iteration=self.splatting_iteration, 
                         model_path=self.splatting_dir, 
                         need_features=False, 
                         need_masks=False, 
                         precomputed_mask=None, 
                         quiet=False, 
                         resolution=1, 
                         segment=False, 
                         sh_degree=3, 
                         skip_test=False, 
                         skip_train=False, 
                         source_path=self.colmap_dir, 
                         target='scene', 
                         white_background=self.white_background)
        
        dataset = model.extract(args)
        self.pipeline = pipeline.extract(args)
        dataset.need_features = dataset.need_masks = False
        gaussians = None
        
        with torch.no_grad():
            self.gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, self.gaussians, load_iteration=self.splatting_iteration, shuffle=False)
            bg_color = [1,1,1] if self.white_background else [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
            self.render_func = render

    def render_image_pair(self, camera_number, visualize=False):
        """
        Render a stereo-aligned pair of images corresponding to a specific view number.

        Parameters:
        camera_number (int): View number for which to render the stereo-aligned pair.
        visualize (bool): Flag to visualize the rendered images for debugging.

        Returns:
        None
        """
        with torch.no_grad():
            camera_pair = self.cameras[camera_number]
            output_dir = self.render_folder_name(camera_number)
            l_r = []
            for camera_name, camera in camera_pair.items():
                rot = tuple(camera['rot'])
                pos = tuple(camera['pos'])
                R, T = convert_R_T_to_GS(rot, pos)
                w, h = camera['width'], camera['height']
                fx, fy = camera['fx'], camera['fy']
                FoVx = 2 * np.arctan2(w, 2 * fx)
                FoVy = 2 * np.arctan2(h, 2 * fy)
                view = cameras.Camera(0, R, T, FoVx, FoVy, torch.rand(3,h,w), None, "abcd", 0)
                render_path = os.path.join(os.path.join(output_dir))
                os.makedirs(render_path, exist_ok=True)
                rendering = (self.render_func(view, self.gaussians, self.pipeline, self.background)["render"].permute(1, 2, 0) * 255).cpu().numpy()
                cv2.imwrite(os.path.join(render_path, f'{camera_name}.png'), cv2.cvtColor(rendering, cv2.COLOR_BGR2RGB))
                l_r.append(rendering / 255)
            if visualize:
                plt.imshow(l_r[0])
                plt.imshow(l_r[1], alpha=0.5)
                plt.show()
