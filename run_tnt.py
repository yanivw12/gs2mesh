# =============================================================================
#  Imports
# =============================================================================

from run_single import run_single
from gs2mesh_utils.argument_utils import ArgParser, encode_string

# =============================================================================
#  Run
# =============================================================================

def run_TNT(args):
    
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
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('TNT')
    args = parser.parse_args()  
    GS_port_orig = args.GS_port
    run_TNT(args)
