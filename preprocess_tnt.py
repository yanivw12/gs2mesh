# =============================================================================
#  Imports
# =============================================================================

import os
import shutil

base_dir = os.path.abspath(os.getcwd())

# =============================================================================
#  Helper functions
# =============================================================================

def clean_directory(dir_path):
    files_to_delete = [
        'images_raw',
        'stereo',
        'pinhole_dict.json',
        'run-colmap-geometric.sh',
        'run-colmap-photometric.sh',
        'scene.json'
    ]
    for item in files_to_delete:
        item_path = os.path.join(dir_path, item)
        if os.path.exists(item_path):
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

def move_files_to_sparse_zero(dir_path):
    sparse_dir = os.path.join(dir_path, 'sparse')
    sparse_zero_dir = os.path.join(sparse_dir, '0')
    os.makedirs(sparse_zero_dir, exist_ok=True)

    for file_name in os.listdir(sparse_dir):
        file_path = os.path.join(sparse_dir, file_name)
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join(sparse_zero_dir, file_name))
            
# =============================================================================
#  Main code to run COLMAP and clean output directories
# =============================================================================
    
if __name__ == "__main__":
            
    # =============================================================================
    #  Run COLMAP
    # =============================================================================

    os.chdir(os.path.join(base_dir,'gs2mesh_utils','third_party','colmap_runner'))
    os.system(f"python convert_tnt_to_json.py --tnt_path {os.path.join(base_dir,'data','TNT')}")

    # =============================================================================
    #  Convert model to txt
    # =============================================================================

    scans = ['Barn', 'Caterpillar', 'Truck', 'Ignatius']
    
    for scan_name in scans:
        scan_path = os.path.join(base_dir,'data','TNT',scan_name)
        os.system(f"colmap model_converter --input_path {os.path.join(scan_path,'sparse')} --output_path {os.path.join(scan_path,'sparse')} --output_type TXT")
        clean_directory(scan_path)
        move_files_to_sparse_zero(scan_path)
