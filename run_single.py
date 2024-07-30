# =============================================================================
#  Imports
# =============================================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.colmap_utils import extract_frames, run_colmap
from gs2mesh_utils.eval_utils import create_strings
from gs2mesh_utils.renderer_utils import Renderer
from gs2mesh_utils.stereo_utils import Stereo
from gs2mesh_utils.tsdf_utils import TSDF

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = os.path.abspath(os.getcwd())

# =============================================================================
#  Run
# =============================================================================

def run_single(args):

    TSDF_voxel_length=args.TSDF_voxel/512
    colmap_dir = os.path.abspath(os.path.join(base_dir,'data',args.dataset_name,args.colmap_name))
    
    strings = create_strings(args)
    
    # =============================================================================
    #  Extract frames from a video
    # =============================================================================
    
    if not args.skip_video_extraction:
        video_name = f'{args.colmap_name}.{args.video_extension}'
        extract_frames(os.path.join(colmap_dir, video_name), os.path.join(colmap_dir, 'images') , interval=args.video_interval)

    # =============================================================================
    #  Run COLMAP with unknown poses
    # =============================================================================
    
    if not args.skip_colmap:
        run_colmap(colmap_dir, use_gpu=True) # If there's an error regarding SiftGPU not being supported, set use_gpu to False

    # =============================================================================
    #  Run Gaussian Splatting
    # =============================================================================
    
    if not args.skip_GS:
        try:
            os.chdir(os.path.join(base_dir, 'third_party', 'gaussian-splatting'))
            iterations_str = ' '.join([str(iteration) for iteration in args.GS_save_test_iterations])
            os.system(f"python train.py -s {colmap_dir} --port {args.GS_port} --model_path {os.path.join(base_dir, 'splatting_output', strings['splatting'], args.colmap_name)} --iterations {args.GS_iterations} --test_iterations {iterations_str} --save_iterations {iterations_str}{' --white_background' if args.GS_white_background else ''}")
            os.chdir(base_dir)
        except:
            os.chdir(base_dir)
            print("ERROR")

    # =============================================================================
    #  Initialize renderer
    # =============================================================================
    
    renderer = Renderer(base_dir, 
                    colmap_dir,
                    strings['output_dir_root'],
                    args.colmap_name, 
                    dataset = strings['dataset'], 
                    splatting = strings['splatting'],
                    experiment_name = strings['experiment_name'],
                    splatting_iteration = args.GS_iterations, 
                    white_background = args.GS_white_background, 
                    baseline_absolute = args.renderer_baseline_absolute, 
                    baseline_percentage = args.renderer_baseline_percentage * (2 if args.dataset_name=="DTU" else 1), 
                    folder_name = args.renderer_folder_name,
                    save_json = args.renderer_save_json,
                    sort_cameras = args.renderer_sort_cameras,
                    device = device)

    # =============================================================================
    #  Prepare renderer
    # =============================================================================
    
    if not args.skip_rendering:
        renderer.prepare_renderer()

    # =============================================================================
    #  Initialize stereo
    # =============================================================================
    
    stereo = Stereo(base_dir, renderer, model_name=args.stereo_model, device=device)

    # =============================================================================
    #  Run stereo
    # =============================================================================
    
    if not args.skip_rendering:
        stereo.run(shading_eps=1e-4, occlusion_threshold=args.stereo_occlusion_threshold, start=0, warm=args.stereo_warm, visualize=False)

    # =============================================================================
    #  Copy masks for DTU / MobileBrick
    # =============================================================================
    
    if not args.skip_masking:
        if args.dataset_name == 'DTU':
            masks_dir = os.path.join(colmap_dir,'mask')
            masks_files = sorted([f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))], key=lambda x: x) 
            masks_files = [mask for mask in masks_files if mask[0]!='.']
            for i, mask_file in enumerate(tqdm(masks_files)):
                stereo_output_dir = renderer.render_folder_name(i)
                mask = plt.imread(os.path.join(masks_dir,mask_file))[:,:,0]
                cx, cy = 823.204, 619.071
                H, W = mask.shape
                W2 = min((W - cx), cx)
                H2 = min((H - cy), cy)
                crop_box = (
                    int(cx-W2),     # left
                    int(cx+W2),     # upper
                    int(cy-H2),     # right
                    int(cy+H2)      # lower
                )
                mask = mask[crop_box[2]:crop_box[3], crop_box[0]:crop_box[1]]
                plt.imsave(os.path.join(stereo_output_dir,'left_mask.png'), mask)
                np.save(os.path.join(stereo_output_dir,'left_mask.npy'), mask)

        elif args.dataset_name == 'MobileBrick':
            masks_dir = os.path.join(colmap_dir,'mask')
            masks_files = sorted([f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))], key=lambda x: x)  
            for i, mask_file in enumerate(tqdm(masks_files)):
                stereo_output_dir = renderer.render_folder_name(i)
                mask = plt.imread(os.path.join(masks_dir,mask_file))[:,:,0]
                plt.imsave(os.path.join(stereo_output_dir,'left_mask.png'), mask)
                np.save(os.path.join(stereo_output_dir,'left_mask.npy'), mask)
                
    # =============================================================================
    #  Initialize TSDF
    # =============================================================================
    
    tsdf = TSDF(renderer, stereo, strings['TSDF'])

    if not args.skip_TSDF:
        # ================================================================================
        #  Run TSDF. the TSDF class will have an attribute "mesh" with the resulting mesh
        # ================================================================================
        
        tsdf.run(scale = args.TSDF_scale,
                 dilate = args.TSDF_dilate,
                 valid = args.TSDF_valid if args.TSDF_valid is not None else list(range(len(renderer))),
                 skip = args.TSDF_skip if args.TSDF_skip is not None else [],
                 use_occlusion_mask = args.TSDF_use_occlusion_mask, 
                 use_mask = args.TSDF_use_mask, 
                 invert_mask = args.TSDF_invert_mask,
                 erode_mask = args.TSDF_erode_mask, 
                 erosion_kernel_size = args.TSDF_erosion_kernel_size, 
                 closing_kernel_size = args.TSDF_closing_kernel_size, 
                 voxel_length = TSDF_voxel_length, 
                 sdf_trunc = args.TSDF_sdf_trunc, 
                 min_depth_baselines = args.TSDF_min_depth_baselines,
                 max_depth_baselines = args.TSDF_max_depth_baselines, 
                 visualize = False)

        # =============================================================================
        #  Save the original mesh before cleaning
        # =============================================================================
        
        tsdf.save_mesh()

        # =============================================================================
        #  Clean the mesh using clustering and save the cleaned mesh.
        # =============================================================================
        
        # original mesh is still available under tsdf.mesh (the cleaned is tsdf.clean_mesh)
        tsdf.clean_mesh(thres=args.TSDF_cleaning_threshold/args.TSDF_scale)

    # =============================================================================
    #  Return the path of the cleaned mesh for dataset evaluations
    # =============================================================================
    
    return os.path.join(renderer.output_dir_root, f'{tsdf.out_name}_cleaned_mesh.ply')

# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('custom')
    args = parser.parse_args()
    run_single(args)