# =============================================================================
#  Imports
# =============================================================================

import os

from gs2mesh_utils.colmap_utils import convert_to_txt

base_dir = os.path.abspath(os.getcwd())

# =============================================================================
#  Preprocess MobileBrick
# =============================================================================

if __name__ == "__main__":
    
    scans = ['counter', 'garden', 'bicycle', 'bonsai', 'kitchen']

    # =============================================================================
    #  Preprocess scans
    # =============================================================================

    for scan_name in scans:

        colmap_name = scan_name
        print(colmap_name)
        colmap_dir = os.path.abspath(os.path.join(base_dir,'data','MipNerf360',colmap_name))
        convert_to_txt(colmap_dir)