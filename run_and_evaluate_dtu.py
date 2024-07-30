# =============================================================================
#  Imports
# =============================================================================

import os
from pathlib import Path
import subprocess

from run_single import run_single
from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', 'evaluation', 'DTU', 'eval_code')))
from evaluate_single_scene import cull_scan

# =============================================================================
#  Run
# =============================================================================

def run_DTU(args):
    
    # =============================================================================
    #  Create output for evaluation
    # =============================================================================
    
    Offical_DTU_Dataset = os.path.join(os.getcwd(), 'data', 'DTU', 'SampleSet', 'MVS_Data')
    dataset_string, exp_path, csv_file = prepare_eval(args)

    # =============================================================================
    #  Create meshes and evaluate
    # =============================================================================
    
    for scan_num in args.scans:

        # =============================================================================
        #  Create mesh
        # =============================================================================

        args.colmap_name = f'scan{scan_num}'
        args.GS_port = GS_port_orig + scan_num
        print(args.colmap_name)
        print(args)
        ply_file = run_single(args)
        
        # =============================================================================
        #  Evaluate
        # =============================================================================
        
        out_dir = os.path.join(exp_path, str(scan_num))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        vis_out_dir = os.path.join(exp_path, str(scan_num))
        Path(vis_out_dir).mkdir(parents=True, exist_ok=True)
        result_mesh_file = os.path.join(out_dir, f"{dataset_string}_scan{scan_num}.ply")
        cull_scan(scan_num, ply_file, result_mesh_file, Offical_DTU_Dataset)
        cmd = f"python {os.path.join(os.getcwd(), 'evaluation', 'DTU', 'eval_code', 'eval.py')} --data {result_mesh_file} --scan {scan_num} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {vis_out_dir}"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        output = output.replace(" ", ",").split(",")
        output[-1] = output[-1].strip()
        output = [scan_num] + output
       
        write_to_csv(args.dataset_name, csv_file, output)
        
# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('DTU')
    args = parser.parse_args()
    GS_port_orig = args.GS_port
    run_DTU(args)
