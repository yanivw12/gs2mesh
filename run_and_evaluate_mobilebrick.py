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

from run_single import run_single
from gs2mesh_utils.argument_utils import ArgParser, encode_string
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', 'evaluation', 'MobileBrick', 'eval_code')))
from evaluate import evaluate_single

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
#  Run
# =============================================================================

def run_MobileBrick(args):
    
    # =============================================================================
    #  Prepare evaluation
    # =============================================================================
      
    _, exp_path, csv_file = prepare_eval(args)
    
    # =============================================================================
    #  Create meshes and evaluate
    # =============================================================================
    
    for scan_name in args.scans:

        # =============================================================================
        #  Create mesh
        # =============================================================================

        args.colmap_name = scan_name
        args.GS_port = GS_port_orig + encode_string(scan_name)
        print(args.colmap_name)
        print(args)
        ply_file = run_single(args)

        # =============================================================================
        #  Evaluate
        # =============================================================================
        
        gt_dir = os.path.join(os.getcwd(), 'data', 'MobileBrick', scan_name)        
        out = evaluate_single(gt_dir, ply_file, exp_path, scan_name, device=device)
        
        # =============================================================================
        #  Write evaluation results to CSV
        # =============================================================================
        
        write_to_csv(args.dataset_name, csv_file, [scan_name] + out)

# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('MobileBrick')
    args = parser.parse_args()  
    GS_port_orig = args.GS_port
    run_MobileBrick(args)
