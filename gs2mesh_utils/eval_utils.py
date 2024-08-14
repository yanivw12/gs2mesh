# =============================================================================
#  Imports
# =============================================================================

import os
from pathlib import Path
import csv

base_dir = os.path.abspath(os.path.join(__file__, '..', '..'))

csv_headers = {
    'DTU': ['Scan Number', 'd2s', 's2d', 'f1'],
    'TNT': ['Scan Name', 'Precision', 'Recall', 'F1 Score'],
    'MobileBrick': ['Scan Name', 'Chamfer Distance', 'Accuracy (2.5mm)', 'Recall (2.5mm)', 'F1 Score (2.5mm)', 'Accuracy (5mm)', 'Recall (5mm)', 'F1 Score (5mm)']
}

# =============================================================================
#  Functions
# =============================================================================

float2str = lambda x: str(x).replace('.', '_')

def create_strings(args):
    """
    Creates and returns a dictionary of formatted strings used in the rendering and evaluation process.

    Parameters:
    args (Namespace): The arguments from the command line given to the reconstruction/evaluation function.

    Returns:
    dict: A dictionary containing the formatted strings.
    """
    splatting_string = f"{args.dataset_name}{'_nw' if args.GS_white_background == False else ''}_iterations{args.GS_iterations}"
    baseline_string = f"{args.renderer_baseline_absolute}a" if args.renderer_baseline_absolute is not None else f"{float2str(args.renderer_baseline_percentage)}p"
    dataset_string = f"{splatting_string}_{args.stereo_model}_baseline{baseline_string}"
    TSDF_string = f"{args.colmap_name}_{dataset_string}_mask{'1' if args.TSDF_use_mask else '0'}_occ{'1' if args.TSDF_use_occlusion_mask else '0'}_scale{float2str(float(args.TSDF_scale))}_voxel{str(args.TSDF_voxel)}_512_trunc{args.TSDF_min_depth_baselines}_{args.TSDF_max_depth_baselines}"
    experiment_name_string = args.experiment_folder_name if args.experiment_folder_name is not None else dataset_string
    output_dir_root_string = os.path.join(base_dir, 'output', experiment_name_string if experiment_name_string is not None else dataset_string, args.renderer_folder_name if args.renderer_folder_name is not None else args.colmap_name)
    ply_path_string = os.path.join(output_dir_root_string, f'{TSDF_string}_cleaned_mesh.ply')
    
    string_dict = {
        'splatting': splatting_string,
        'baseline': baseline_string,
        'dataset': dataset_string,
        'TSDF': TSDF_string,
        'experiment_name': experiment_name_string,
        'output_dir_root': output_dir_root_string,
        'ply_path': ply_path_string
    }
    
    return string_dict
    
def prepare_eval(args):
    """
    Prepares the evaluation output directory and CSV file for storing evaluation results.

    Parameters:
    args (Namespace): The arguments from the command line given to the reconstruction/evaluation function.

    Returns:
    tuple: A tuple containing the dataset string, evaluation path, and CSV file path.
    """
    strings = create_strings(args)
    out_dir_prefix = os.path.join(os.getcwd(), 'evaluation', args.dataset_name, 'eval_output')
    Path(out_dir_prefix).mkdir(parents=True, exist_ok=True)
    exp_path = os.path.join(out_dir_prefix, strings['dataset'])
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    
    csv_file = os.path.join(exp_path, 'evaluation_results.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_headers[args.dataset_name])
    
    return strings['dataset'], exp_path, csv_file

def write_to_csv(dataset, csv_file, line):
    """
    Writes a line of data to the specified CSV file.

    Parameters:
    dataset (str): A string indicating the dataset name, used to fetch the correct CSV headers for printing.
    csv_file (str): The path to the CSV file where the data will be written.
    line (list): A list of values to write as a new row in the CSV file.

    Returns:
    None
    """
    print(list(zip(csv_headers[dataset], line)))
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(line)
