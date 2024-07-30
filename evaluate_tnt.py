# =============================================================================
#  Imports
# =============================================================================

import os

from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import create_strings, prepare_eval, write_to_csv

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', 'evaluation', 'TNT', 'eval_code','python_toolbox','evaluation')))
from run import run_evaluation

base_dir = os.path.abspath(os.getcwd())
    
# =============================================================================
#  Run evaluation
# =============================================================================

def evaluate_TNT(args):
    
    # =============================================================================
    #  Create output for evaluation
    # =============================================================================

    dataset_string, exp_path, csv_file = prepare_eval(args)
    
    # =============================================================================
    #  Evaluate
    # =============================================================================
    
    for scan_name in args.scans:
        
        # =============================================================================
        #  Get ply filename since we can't run it together with the mesh creation
        # =============================================================================  
                
        args.colmap_name = scan_name
        strings = create_strings(args)

        # =============================================================================
        #  Run evaluation on specific scan and save output
        # =============================================================================
        scan_output_path = os.path.join(exp_path, scan_name)
        metrics = run_evaluation(dataset_dir=os.path.join(base_dir,'data','TNT',scan_name), traj_path=os.path.join(base_dir,'data','TNT',scan_name, f'{scan_name}_COLMAP_SfM.log'), ply_path=strings['ply_path'], out_dir=scan_output_path)

        write_to_csv(args.dataset_name, csv_file, [scan_name] + metrics)
            
# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('TNT')
    args = parser.parse_args()   
    evaluate_TNT(args)