# =============================================================================
#  Imports
# =============================================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.colmap_utils import extract_frames, create_downsampled_colmap_dir, run_colmap
from gs2mesh_utils.eval_utils import create_strings
from gs2mesh_utils.renderer_utils import Renderer
from gs2mesh_utils.stereo_utils import Stereo
from gs2mesh_utils.masker_utils import init_predictor, Masker
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
    #  Create downsampled COLMAP directory
    # =============================================================================
    
    if args.downsample > 1.0:
        create_downsampled_colmap_dir(colmap_dir, args.downsample)
        args.colmap_name = f"{args.colmap_name}_downsample{args.downsample}"
        TSDF_voxel_length=args.TSDF_voxel/512
        colmap_dir = os.path.abspath(os.path.join(base_dir,'data',args.dataset_name,args.colmap_name))
        strings = create_strings(args)
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
                        args,
                        dataset = strings['dataset'], 
                        splatting = strings['splatting'],
                        experiment_name = strings['experiment_name'],
                        device=device)

    # =============================================================================
    #  Prepare renderer
    # =============================================================================
    
    if not args.skip_rendering:
        renderer.prepare_renderer()

    # =============================================================================
    #  Initialize stereo
    # =============================================================================
    
    stereo = Stereo(base_dir, renderer, args, device=device)

    # =============================================================================
    #  Run stereo
    # =============================================================================
    
    if not args.skip_rendering:
        stereo.run(start=0, visualize=False)

    # =============================================================================
    #  Perform automatic masking form custom, or copy masks for DTU / MobileBrick
    # =============================================================================
    
    if not args.skip_masking:
        if args.dataset_name == 'custom':
            if args.masker_automask:
                GD_model, predictor, inference_state, images_dir = init_predictor(base_dir, renderer, args, device=device) 
                masker = Masker(GD_model, predictor, inference_state, images_dir, renderer, stereo, args, image_number=0, visualize=False)
                masker.segment()
                args.TSDF_use_mask = True
            else:
                print("Automask must be enabled for masking in script mode. Skipping.")
                
        elif args.dataset_name == 'DTU':
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
    
    tsdf = TSDF(renderer, stereo, args, strings['TSDF'])

    if not args.skip_TSDF:
        # ================================================================================
        #  Run TSDF. the TSDF class will have an attribute "mesh" with the resulting mesh
        # ================================================================================
        
        tsdf.run(visualize = False)

        # =============================================================================
        #  Save the original mesh before cleaning
        # =============================================================================
        
        tsdf.save_mesh()

        # =============================================================================
        #  Clean the mesh using clustering and save the cleaned mesh.
        # =============================================================================
        
        # original mesh is still available under tsdf.mesh (the cleaned is tsdf.clean_mesh)
        tsdf.clean_mesh()

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
    print(args)
    run_single(args)