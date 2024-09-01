# =============================================================================
#  Imports
# =============================================================================

import argparse

# =============================================================================
#  Hash a string to a 2 digit number
# =============================================================================

encode_string = lambda s: sum(s.encode()) % 100

# =============================================================================
#  ArgParser class
# =============================================================================

class ArgParser:
    def __init__(self, dataset):
        """
        Initialize the argument parser.

        Parameters:
        dataset (str): Name of the dataset.
        """
        self.dataset = dataset
        
        self.parser = argparse.ArgumentParser(description="GS2Mesh arguments.")

        self.default_values = {
            'colmap_name':{'custom':'sculpture', 'DTU':'scan24', 'TNT':'Ignatius', 'MobileBrick':'aston', 'MipNerf360':'garden'},
            'dataset_name':{'custom':'custom', 'DTU':'DTU', 'TNT':'TNT', 'MobileBrick':'MobileBrick', 'MipNerf360':'MipNerf360'},
            'downsample':{'custom':1, 'DTU':1, 'TNT':1, 'MobileBrick':1, 'MipNerf360':3},
            'renderer_baseline_percentage':{'custom':7.0, 'DTU':7.0, 'TNT':7.0, 'MobileBrick':14.0, 'MipNerf360':7.0},
            'stereo_warm':{'custom':False, 'DTU':True, 'TNT':True, 'MobileBrick':True, 'MipNerf360':False},
            'TSDF_scale':{'custom':1.0, 'DTU':1.0, 'TNT':1.0, 'MobileBrick':0.1, 'MipNerf360':1.0},
            'TSDF_use_mask':{'custom':False, 'DTU':True, 'TNT':False, 'MobileBrick':True, 'MipNerf360':False},
            'TSDF_min_depth_baselines':{'custom':4, 'DTU':4, 'TNT':2, 'MobileBrick':4, 'MipNerf360':4},
            'TSDF_max_depth_baselines':{'custom':20, 'DTU':20, 'TNT':10, 'MobileBrick':20, 'MipNerf360':15},
            'TSDF_cleaning_threshold':{'custom':100000, 'DTU':100000, 'TNT':100000, 'MobileBrick':10000, 'MipNerf360':100000},
            'skip_video_extraction':{'custom':False, 'DTU':True, 'TNT':True, 'MobileBrick':True, 'MipNerf360':True},
            'skip_colmap':{'custom':False, 'DTU':True, 'TNT':True, 'MobileBrick':True, 'MipNerf360':True}
        }
        
        # General params
        self.parser.add_argument('--colmap_name', type=str, default=self.default_value('colmap_name'), help='Name of the directory with the COLMAP sparse model')
        self.parser.add_argument('--dataset_name', type=str, default=self.default_value('dataset_name'), help='Name of the dataset. Options: custom, DTU, TNT, MobileBrick, MipNerf360')
        self.parser.add_argument('--experiment_folder_name', type=str, default=None, help='Name of the experiment folder')

        # Preprocessing params
        self.parser.add_argument('--downsample', type=int, default=self.default_value('downsample'), help='Downsampling factor')
        
        # GS params
        self.parser.add_argument('--GS_iterations', type=int, default=30000, help='Number of Gaussian Splatting iterations')
        self.parser.add_argument('--GS_save_test_iterations', type=int, nargs='+', default=[7000, 30000], help='Gaussian Splatting test iterations to save')
        self.parser.add_argument('--GS_white_background', action='store_true', help='Use white background in Gaussian Splatting')

        # Renderer params
        self.parser.add_argument('--renderer_baseline_absolute', type=float, default=None, help='Absolute value of the renderer baseline (None uses 7 percent of scene radius)')
        self.parser.add_argument('--renderer_baseline_percentage', type=float, default=self.default_value('renderer_baseline_percentage'), help='Percentage value of the renderer baseline')
        self.parser.add_argument('--renderer_scene_360', action='store_true', default=True, help='Scene is a 360 scene')
        self.parser.add_argument('--no-renderer_scene_360', action='store_false', dest='renderer_scene_360', help="Scene is not a 360 scene")
        self.parser.add_argument('--renderer_folder_name', type=str, default=None, help='Name of the renderer folder (None uses the colmap name)')
        self.parser.add_argument('--renderer_save_json', action='store_true', default=True, help='Save renderer data to JSON (default True - disable with no-renderer_save_json)')
        self.parser.add_argument('--no-renderer_save_json', action='store_false', dest='renderer_save_json', help="Disable renderer_save_json")
        self.parser.add_argument('--renderer_sort_cameras', action='store_true', help='Sort cameras in the renderer (True if using unordered set of views)')
        
        # Stereo params
        self.parser.add_argument('--stereo_model', type=str, default='DLNR_Middlebury', help='Stereo model to use')
        self.parser.add_argument('--stereo_occlusion_threshold', type=int, default=3, help='Occlusion threshold for stereo model (Lower value masks out more areas)')
        self.parser.add_argument('--stereo_warm', action='store_true', default=self.default_value('stereo_warm'), help='Use the previous disparity as initial disparity for current view (False if views are not sorted) (disable with no-stereo_warm)')
        self.parser.add_argument('--stereo_shading_eps', type=float, default=1e-4, help='Small value used for visualization of the depth gradient. Adjusted according to the scale of the scene.')

        # TSDF params
        self.parser.add_argument('--TSDF_scale', type=float, default=self.default_value('TSDF_scale'), help='Fix depth scale')
        self.parser.add_argument('--TSDF_dilate', type=int, default=1, help='Take every n-th image (1 to take all images)')
        self.parser.add_argument('--TSDF_valid', type=str, default=None, help='Choose valid images as a list of indices (None to ignore)')
        self.parser.add_argument('--TSDF_skip', type=str, default=None, help='Choose non-valid images as a list of indices (None to ignore)')
        self.parser.add_argument('--TSDF_use_occlusion_mask', action='store_true', default=True, help='Ignore occluded regions in stereo pairs for better geometric consistency (default True - disable with no-TSDF_use_occlusion_mask)')
        self.parser.add_argument('--no-TSDF_use_occlusion_mask', action='store_false', dest='TSDF_use_occlusion_mask', help="Disable TSDF_use_occlusion_mask")
        self.parser.add_argument('--TSDF_use_mask', action='store_true', default=self.default_value('TSDF_use_mask'), help='Use object masks (optional - disable with no-TSDF_use_mask)')
        self.parser.add_argument('--TSDF_invert_mask', action='store_true', help='Invert the background mask for TSDF')
        self.parser.add_argument('--TSDF_erode_mask', action='store_true', default=True, help='Erode masks in TSDF (default True - disable with no-TSDF_erode_mask)')
        self.parser.add_argument('--no-TSDF_erode_mask', action='store_false', dest='TSDF_erode_mask', help="Disable TSDF_erode_mask")
        self.parser.add_argument('--TSDF_erosion_kernel_size', type=int, default=10, help='Erosion kernel size in TSDF')
        self.parser.add_argument('--TSDF_closing_kernel_size', type=int, default=10, help='Closing kernel size in TSDF')
        self.parser.add_argument('--TSDF_voxel', type=int, default=2, help='Voxel size (voxel length is TSDF_voxel/512)')
        self.parser.add_argument('--TSDF_sdf_trunc', type=float, default=0.04, help='SDF truncation in TSDF')
        self.parser.add_argument('--TSDF_min_depth_baselines', type=int, default=self.default_value('TSDF_min_depth_baselines'), help='Minimum depth baselines in TSDF')
        self.parser.add_argument('--TSDF_max_depth_baselines', type=int, default=self.default_value('TSDF_max_depth_baselines'), help='Maximum depth baselines in TSDF')
        self.parser.add_argument('--TSDF_cleaning_threshold', type=int, default=self.default_value('TSDF_cleaning_threshold'), help='Minimal cluster size for clean mesh')
        
        # Running params
        self.parser.add_argument('--GS_port', type=int, default=8080, help='GS port number (relevant if running several instances at the same time).')
        self.parser.add_argument('--skip_video_extraction', action='store_true', default=self.default_value('skip_video_extraction'), help='Skip the video extraction stage.')
        self.parser.add_argument('--skip_colmap', action='store_true', default=self.default_value('skip_colmap'), help='Skip the COLMAP stage.')
        self.parser.add_argument('--skip_GS', action='store_true', help='Skip the GS stage.')
        self.parser.add_argument('--skip_rendering', action='store_true', help='Skip the rendering stage.')
        self.parser.add_argument('--skip_masking', action='store_true', help='Skip the masking stage.')
        self.parser.add_argument('--skip_TSDF', action='store_true', help='Skip the TSDF stage.')
        
        # Add params based on dataset
        if self.dataset == 'custom':
            self.parser.add_argument('--video_extension', type=str, default='mp4', help='Video file extension.')
            self.parser.add_argument('--video_interval', type=int, default=10, help='Extract every n-th frame - aim for 3fps.')
            # Masker params
            self.parser.add_argument('--masker_automask', action='store_true', help='Use GroundingDINO for automatic object detection for masking with SAM2')
            self.parser.add_argument('--masker_prompt', type=str, default='main_object', help='Prompt for GroundingDINO')
            self.parser.add_argument('--masker_SAM2_local', action='store_true', help='Use local SAM2 weights')
        if self.dataset == 'DTU':
            self.parser.add_argument('--scans', type=int, nargs='+', default=[24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122], help='Scan numbers')
            self.parser.add_argument('--no-stereo_warm', action='store_false', dest='stereo_warm', help="Disable stereo_warm")
            self.parser.add_argument('--no-TSDF_use_mask', action='store_false', dest='TSDF_use_mask', help="Disable TSDF_use_mask")
        if self.dataset == 'TNT':
            self.parser.add_argument('--scans', type=str, nargs='+', default=['Barn', 'Caterpillar', 'Ignatius', 'Truck'], help='Scan names')
            self.parser.add_argument('--no-stereo_warm', action='store_false', dest='stereo_warm', help="Disable stereo_warm")
        if self.dataset == 'MobileBrick':
            self.parser.add_argument('--scans', type=str, nargs='+', default=['aston', 'audi', 'beetles', 'big_ben', 'boat', 'bridge', 'cabin', 'camera', 'castle', 'colosseum', 'convertible', 'ferrari', 'jeep', 'london_bus', 'motorcycle', 'porsche', 'satellite', 'space_shuttle'], help='Scan names')
            self.parser.add_argument('--no-stereo_warm', action='store_false', dest='stereo_warm', help="Disable stereo_warm")
            self.parser.add_argument('--no-TSDF_use_mask', action='store_false', dest='TSDF_use_mask', help="Disable TSDF_use_mask")
        if self.dataset == 'MipNerf360':
            self.parser.add_argument('--scans', type=str, nargs='+', default=['counter', 'garden'], help='Scan names')

    def default_value(self, param):
        """
        Retrieve the default value for a given parameter based on the dataset.

        Parameters:
        param (str): Parameter name.

        Returns:
        Any: Default value for the parameter.
        """
        return self.default_values[param][self.dataset]
        
    def parse_args(self):
        """
        Parse the command-line arguments.

        Returns:
        Namespace: Parsed arguments.
        """
        return self.parser.parse_args()
