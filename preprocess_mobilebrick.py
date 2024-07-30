# =============================================================================
#  Imports
# =============================================================================

import os

from gs2mesh_utils.colmap_utils import create_mobile_brick_colmap_files, run_colmap_known_poses

base_dir = os.path.abspath(os.getcwd())

# =============================================================================
#  Preprocess MobileBrick
# =============================================================================

if __name__ == "__main__":
    
    scans = ['aston', 'audi', 'beetles', 'big_ben', 'boat', 'bridge', 'cabin', 'camera', 'castle', 'colosseum', 'convertible', 'ferrari', 'jeep', 'london_bus', 'motorcycle', 'porsche', 'satellite', 'space_shuttle']

    # =============================================================================
    #  Preprocess scans
    # =============================================================================

    for scan_name in scans:

        colmap_name = scan_name
        print(colmap_name)
        colmap_dir = os.path.abspath(os.path.join(base_dir,'data','MobileBrick',colmap_name))
        if os.path.exists(os.path.join(colmap_dir, 'image')):
            os.rename(os.path.join(colmap_dir, 'image'), os.path.join(colmap_dir, 'images'))
        create_mobile_brick_colmap_files(colmap_dir, colmap_name)
        run_colmap_known_poses(colmap_dir, use_gpu=True)