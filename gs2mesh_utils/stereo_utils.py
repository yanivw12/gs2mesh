# =============================================================================
#  Imports
# =============================================================================

import torch
import numpy as np
import os
from tqdm import tqdm
import cv2
from PIL import Image
from argparse import Namespace
import matplotlib.pyplot as plt

from gs2mesh_utils.transformation_utils import get_shading

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'third_party', 'DLNR')))
from core.dlnr import DLNR
from core.utils.utils import InputPadder as DLNR_InputPadder

# =============================================================================
#  Class for stereo matching model
# =============================================================================

class Stereo:
    def __init__(self, base_dir, renderer, args, device='cuda'):
        """
        Initialize the Stereo class.

        Parameters:
        base_dir (str): Base directory of the repository.
        renderer (Renderer): Renderer class object.
        args (ArgParser): Program arguments.
        device (str): Device to run the model on.
        """
        self.base_dir = base_dir
        self.renderer = renderer
        self.args = args
        self.model_name = self.args.stereo_model
        self.device = device
        self.disparity_signs = {'DLNR_Middlebury': -1, 'DLNR_SceneFlow': -1}

        if "DLNR" in self.model_name:
            DLNR_args = Namespace(corr_implementation='reg', 
                                  corr_levels=4, 
                                  corr_radius=4, 
                                  dataset='things', 
                                  hidden_dims=[128, 128, 128], 
                                  mixed_precision=True, 
                                  n_downsample=2, 
                                  n_gru_layers=3, 
                                  restore_ckpt=os.path.join(self.base_dir, 'third_party', 'DLNR', 'pretrained', f'{self.model_name}.pth'),
                                  shared_backbone=False, 
                                  slow_fast_gru=False, 
                                  valid_iters=10)
            DLNR_model = torch.nn.DataParallel(DLNR(DLNR_args), device_ids=[0])
            DLNR_model.load_state_dict(torch.load(DLNR_args.restore_ckpt))
            DLNR_model = DLNR_model.module
            DLNR_model.to(self.device)
            DLNR_model.eval()
            self.model = DLNR_model
            self.input_padder = DLNR_InputPadder
            self.model_args = DLNR_args
        else:
            print("MODEL NOT SUPPORTED")
            return

    def load_image(self, imfile):
        """
        Load an image and prepare it for the stereo model.

        Parameters:
        imfile (str): Path to the image file.

        Returns:
        torch.Tensor: Image tensor prepared for the model.
        """
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.device)
    
    def run(self, start=0, visualize=False):
        """
        Run the stereo model: render a pair of images and compute the disparity and depth using the stereo model.

        Parameters:
        start (int): The view number from which to start. Default is 0.
        visualize (bool): Flag to visualize the results for debugging.

        Returns:
        None
        """
        with torch.no_grad():
            prev_flows = {'LR': None, 'RL': None}
            for camera_number, left_camera in enumerate(tqdm(self.renderer.left_cameras)):
                if camera_number < start:
                    continue
                baseline = self.renderer.baseline
                output_dir = self.renderer.render_folder_name(camera_number)
                self.renderer.render_image_pair(camera_number, visualize=False)
                
                image1 = self.load_image(os.path.join(output_dir, 'left.png'))
                image2 = self.load_image(os.path.join(output_dir, 'right.png'))

                disparities = {'LR': None, 'RL': None}
                for direction in ['LR', 'RL']:
                    padder = self.input_padder(image1.shape, divis_by=32)
                    image1, image2 = padder.pad(image1, image2)
                    if direction == 'LR':
                        image1_to_model = image1
                        image2_to_model = image2
                    elif direction == 'RL':
                        image1_to_model = torch.flip(image2, dims=[3])
                        image2_to_model = torch.flip(image1, dims=[3])
                    
                    prev_flow, flow_up = self.model(image1_to_model, image2_to_model, iters=self.model_args.valid_iters, test_mode=True, flow_init=prev_flows[direction] if self.args.stereo_warm else None)
                    if direction == 'RL':
                        prev_flow = torch.flip(prev_flow, dims=[3])
                        flow_up = torch.flip(flow_up, dims=[3])
                    flow_up = padder.unpad(flow_up).squeeze()
                    
                    prev_flows[direction] = prev_flow
                    
                    disparities[direction] = self.disparity_signs[self.model_name] * flow_up.detach().cpu().numpy().squeeze()
                    
                    output_directory = os.path.join(output_dir, f'out_{self.model_name}')
                    os.makedirs(output_directory, exist_ok=True)
                    
                    np.save(os.path.join(output_directory, f"disparity_{direction}.npy"), disparities[direction])
                    plt.imsave(os.path.join(output_directory, f"disparity_{direction}.png"), disparities[direction], cmap='jet')
                        
                occlusion_mask = self.get_occlusion_mask(disparities['LR'], disparities['RL'], self.args.stereo_occlusion_threshold)
                depth = (left_camera['fx'] * baseline) / (disparities['LR'])
                
                np.save(os.path.join(output_directory, "occlusion_mask.npy"), occlusion_mask)
                plt.imsave(os.path.join(output_directory, "occlusion_mask.png"), occlusion_mask)
                np.save(os.path.join(output_directory, "depth.npy"), depth)
                cv2.imwrite(os.path.join(output_directory, 'depth.png'), depth)            
                shading = get_shading(depth, self.args.stereo_shading_eps)
                cv2.imwrite(os.path.join(output_directory, 'shading.png'), shading)
                
                if visualize:
                    print(f"baseline: {baseline}")
                    print(f"minimal depth: {depth.min()}, maximal depth: {depth.max()}")
                    self.view_results_single(camera_number)

                torch.cuda.empty_cache()
                
    def get_occlusion_mask(self, L2R_disparity, R2L_disparity, occlusion_threshold):
        """
        Calculate the occlusion mask given a pair of disparities.

        Parameters:
        L2R_disparity (np.ndarray): Left-to-right disparity map.
        R2L_disparity (np.ndarray): Right-to-left disparity map.
        occlusion_threshold (int): Threshold on the reprojection error.

        Returns:
        np.ndarray: Binary occlusion mask where 0 indicates occluded pixels and 1 indicates visible pixels.
        """
        height, width = L2R_disparity.shape
    
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        
        x_projected = (x_grid - L2R_disparity).astype(np.int32)
        x_projected_clipped = np.clip(x_projected, 0, width - 1)
        
        x_reprojected = x_projected_clipped + R2L_disparity[y_grid, x_projected_clipped]
        x_reprojected_clipped = np.clip(x_reprojected, 0, width - 1)
        
        disparity_difference = np.abs(x_grid - x_reprojected_clipped)
    
        occlusion_mask = (disparity_difference > occlusion_threshold).astype(np.uint8)
        
        occlusion_mask[(x_projected < 0) | (x_projected >= width)] = 1

        occlusion_mask = occlusion_mask > 0.5
        
        return ~occlusion_mask

    def view_results_single(self, camera_number):
        """
        Visualize the rendering and stereo results for a single view.

        Parameters:
        camera_number (int): View number to visualize.

        Returns:
        None
        """
        output_dir = self.renderer.render_folder_name(camera_number)

        paths = {'left_img': 'left.png',
                 'right_img': 'right.png',
                 'object_mask': 'left_mask.png',
                 'occlusion_mask': f'out_{self.model_name}/occlusion_mask.png',
                 'disparity': f'out_{self.model_name}/disparity_LR.png',
                 'shading': f'out_{self.model_name}/shading.png'
                }

        images = {path_name: None for path_name in paths.keys()}

        for path_name, filename in paths.items():
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                images[path_name] = Image.open(path)
            else:
                images[path_name] = Image.fromarray(np.random.randint(0, 255, (self.renderer.left_cameras[0]['height'], self.renderer.left_cameras[0]['width'], 3), dtype=np.uint8))

        images['lr_img'] = Image.blend(images['left_img'], images['right_img'], alpha=0.5)
        
        images_list = [images[key] for key in ['lr_img', 'object_mask', 'disparity', 'occlusion_mask', 'shading']]
        
        widths, heights = zip(*(i.size for i in images_list))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        disparity_min, disparity_max = None, None
        for i, im in enumerate(images_list):
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
            if i == 2:
                disparity_min = np.array(im).min()
                disparity_max = np.array(im).max()
        concatenated_image = new_im
        
        factor = 0.25
        smaller_image = concatenated_image.resize((int(total_width * factor), int(max_height * factor)))

        plt.figure(figsize=(20, 10))
        plt.imshow(smaller_image)
        plt.title(f'{camera_number:03}')
        plt.axis('off')
        plt.show()
        print(f"minimal disparity: {disparity_min}, maximal disparity: {disparity_max}")

    def view_results(self):
        """
        Visualize the rendering and stereo results for all views.

        Returns:
        None
        """
        for camera_number in range(len(self.renderer)):
            self.view_results_single(camera_number)
