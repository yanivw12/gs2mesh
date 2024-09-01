<p align="center">
  <h1 align="center">GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views</h1>
  <p align="center"> Yaniv Wolf · Amit Bracha · Ron Kimmel</p>
  <h3 align="center">ECCV 2024</h3>
  <h3 align="center"> <a href="https://arxiv.org/pdf/2404.01810">Paper (arXiv)</a> | <a href="https://gs2mesh.github.io/">Project Page</a>  </h3>
  <div align="center"></div>
</p>

![pipeline](assets/pipeline.jpeg)

We introduce a novel pipeline for multi-view 3D reconstruction from uncontrolled videos, capable of extracting smooth meshes from noisy Gaussian Splatting clouds. While other geometry extraction methods rely on the location of the Gaussian elements, and attempt to regularize them using geometric constraints, we simply utilize a pre-trained stereo model as a real-world geometric prior, that can extract accurate depths from every view, and fuse them together to create a smooth mesh, without any further optimization needed. Our method achieves state-of-the art reconstruction results, and only requires a short overhead on top of the GS optimization.

### Please read "Common Issues and Tips" before opening a new issue!

## Updates

- **2024/09/01:**
    - Added support for automatic masking using GroundingDINO! Now you can remove background automatically or reconstruct a specific object in the scene using a text prompt as input to the script. See "Custom Data" section for more details.
    - Improved visualizations in *custom_data.ipynb* - You can now see which areas of the scene will be taken into account in the TSDF process when visualizing the renderer poses.
    - Cleaned function calls.
    - Improved handling for non-360 scenes , which is initiated by a parameter *no-renderer_scene_360* for scenes which aren't 360 (for example, DTU scenes). If your scene is taken from a partial sphere, set this parameter to get a better estimation of the scene radius and the required horizontal baseline.
- **2024/08/22:**
    - Added option for local SAM2 weights.
- **2024/08/21:**
    - Small bug fix in renderer initialization.
- **2024/08/14:** 
	- Added support for SAM2! You can now extract meshes of specific objects with ease using the interactive notebook *custom_data.ipynb*.
	- Changed default CUDA to 11.8 and default python to 3.8 for better compatibility. If you have already installed a previous version of the environment, please remove it using *"conda remove -n gs2mesh --all"* and re-install it using the instructions below.
	- Improved COLMAP on custom data to fix distortion.
	- Added support for downsampling - relevant especially for MipNeRF360 dataset.
- **2024/07/30:** Code is released! Includes preprocessing, reconstruction and evaluation code for DTU, TNT and MobileBrick, as well as code for reconstruction of custom data and MipNerf360.

## Future Work

- [ ] Add support for additional state-of-the-art GS models, and the gsplat framework
- [x] Add support for automatic background removal with SAM2 (without the need for an interactive prompt)
- [x] Add support for SAM2 in the interactive notebook
- [ ] Release Google Colab demo
- [x] Release notebook for interactive reconstruction of custom data
- [x] Release reconstruction/evaluation code for DTU, TNT, MobileBrick, MipNerf360

## Installation

**Environment**

We tested the environment on Ubuntu 20.04, with Python 3.8.18 and CUDA 11.8, with an Nvidia A40/L40 GPU.

First, clone the repository:
```bash
git clone https://github.com/yanivw12/gs2mesh.git
```
Then, create the environment, activate it and install dependencies:
```bash
# create the environment
conda create --name gs2mesh python=3.8
# activate the environment
conda activate gs2mesh
# install conda dependencies
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 cudatoolkit=11.8 colmap -c pytorch -c nvidia -c conda-forge
# install additional dependencies
cd gs2mesh
pip install -r requirements.txt
```

**Download DLNR Stereo weights**

```bash
# create the folder for the pretrained models
cd third_party/DLNR
mkdir pretrained
cd pretrained
# download Middlebury model
wget https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network/releases/download/v1.0.0/DLNR_Middlebury.pth
# download Sceneflow model
wget https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network/releases/download/v1.0.0/DLNR_SceneFlow.pth
```

**Download SAM2 weights**

SAM2 local weights is optional, as it automatically downloads weights using Huggingface. If you are still interested in local weights, download them and set the parameter *masker_SAM2_local* to True if using local weights.

```bash
# navigate to the folder for the pretrained model
cd third_party/segment-anything-2/checkpoints
# download pretrained model
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```


**Download GroundingDINO weights**

```bash
# create the folder for the pretrained model
cd third_party/GroundingDINO
mkdir weights
cd weights
# download pretrained model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Datasets

**DTU**

For the DTU dataset, download the preprocessed dataset from [2DGS](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9), the DTU sample set with observation masks from [the DTU website](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) and the ground truth point clouds from [the DTU website](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) as well.

The dataset should look as follows, under *gs2mesh/data/DTU/*:
```
DTU
|---SampleSet
|	|---...
|	|---MVS_Data
|	|	|---...
|	|	|---Points
|---scan24
|	|---...
|	|---depths
|	|---images
|---...
```

Notice that the ground truth points are placed in *gs2mesh/SampleSet/MVS_Data/*.

To run the reconstruction and evaluation, run the command
```bash
python run_and_evaluate_dtu.py
```
For additional parameters, see the 'parameters' table below.

The output of the evaluation should be in *gs2mesh/evaluation/DTU/eval_output/*.

**Tanks and Temples**

For the TNT dataset, download the training dataset from the [TNT](https://www.tanksandtemples.org/download/) website. You will need the image set, Camera Poses, Alignment, Cropfiles and ground truth. The dataset should be organized as follows, under *gs2mesh/data/TNT/*:
```
TNT
|---Barn
|	|---images_raw
|	|	|---000001.jpg
|	|	|---...
|	|---Barn_COLMAP_SfM.log
|	|---Barn_trans.txt
|	|---Barn.json
|	|---Barn.ply
|---...
```

Run the preprocessing code:
```bash
python preprocess_tnt.py
```

Afterwards, the dataset should look as follows:
```
TNT
|---Barn
|	|---images
|	|	|---000001.jpg
|	|	|---...
|	|---sparse
|	|	|---0
|	|	|	|---cameras.bin
|	|	|	|---cameras.txt
|	|	|	|---images.bin
|	|	|	|---images.txt
|	|	|	|---points3D.bin
|	|	|	|---points3D.txt
|	|---Barn_COLMAP_SfM.log
|	|---Barn_trans.txt
|	|---Barn.json
|	|---Barn.ply
|	|---database.db
|---...
```

To run the reconstruction, run the command
```bash
python run_tnt.py
```
For additional parameters, see the 'parameters' table below.

The TNT evaluation code requires an older version of Open3D (9.0). create a separate Conda environment with the following libraries:
```bash
matplotlib>=1.3
open3d==0.9
```
Activate it, and run the command
```bash
python evaluate_tnt.py
```
Use the same parameters that you used for the reconstruction, as it constructs a ply filename based on these parameters.

The output of the evaluation should be in *gs2mesh/evaluation/TNT/eval_output/*.

**Note: The results reported in the paper are from an older version where the RADIAL camera model was used as input to GS (an older GS version as well). The current results using the evaluation code with the updated GS model and a PINHOLE camera model are slightly better:
<details>
<summary>TNT Results</summary>

|    | Barn   | Caterpillar | Ignatius | Truck  | Mean **↑**  | 
|--------|--------|-------------|----------|--------|------------|
| Paper (RADIAL)| 0.21  | 0.17      | 0.64   | 0.46 | 0.37 |
| Code (PINHOLE)| 0.33  | 0.20      | 0.68   | 0.47 | 0.42 |
</details>
<br>

**MobileBrick**

For the MobileBrick dataset, download the dataset from [the official site](https://www.robots.ox.ac.uk/~victor/data/MobileBrick/MobileBrick_Mar23.zip) and extract it into *gs2mesh/data/MobileBrick/*.

Run the preprocessing code:
```bash
python preprocess_mobilebrick.py
```
To run the reconstruction and evaluation, run the command
```bash
python run_and_evaluate_mobilebrick.py
```
For additional parameters, see the 'parameters' table below.

The output of the evaluation should be in *gs2mesh/evaluation/MobileBrick/eval_output/*.

**Note: In the results reported in the paper, we used a horizontal baseline of 14% of the scene radius instead of 7%. The results for the 7% baseline are slightly worse (most likely due to the camera views being above the object, making the scene radius significantly different from the camera location radius which is used to calculate the baseline):

<details>
<summary>MobileBrick Results</summary>

| | Accuracy (2.5mm) **↑** | Recall (2.5mm) **↑** | F1 (2.5mm) **↑**  | Accuracy (5mm) **↑**  | Recall (5mm) **↑**  | F1 (5mm) **↑**  | Chamfer Distance **↓**|
|--------|------------------|----------------|------------|----------------|--------------|----------|-----------------------|
| Paper (14%)| 68.77            | 69.27          | 68.94      | 89.46          | 87.37        | 88.28    | 4.94                  |
| Code (7%)| 67.63           | 69.00          | 68.20      | 88.57          | 87.36        | 87.85    | 5.05                  |

</details>
<br>

**MipNerf360**

For the MipNerf360 dataset, download the dataset from [the official site](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) and extract it into *gs2mesh/data/MipNerf360/*.

Run the preprocessing code:
```bash
python preprocess_mipnerf360.py
```
To run the reconstruction, run the command
```bash
python run_mipnerf360.py
```
Results are saved in *gs2mesh/output/MipNerf360_\<params>*.

**Note: Since the original resolution of the MipNeRF360 scenes is very large, we added a downsample argument that can reduce the size of the images before processing.

## Custom Data

Place your data under *gs2mesh/data/custom/\<data_name>*.

If your input is a video:
```
custom
|---<data_name>
|	|---<data_name>.<extension>
```

If your input is a set of images:
```
custom
|---<data_name>
|	|---images
|	|	|---000001.png
|	|	|---...
```

If your input is a COLMAP dataset:
```
custom
|---<data_name>
|	|---images
|	|	|---000001.png
|	|	|---...
|	|---sparse
|	|	|---0
|	|	|	|---cameras.bin
|	|	|	|---cameras.txt
|	|	|	|---images.bin
|	|	|	|---images.txt
|	|	|	|---points3D.bin
|	|	|	|---points3D.txt
```

You can either run an interactive notebook *custom_data.ipynb*, which also has visualizations and allows masking of objects interactively, or run a script *run_single.py* and use the automatic masking parameters:
```bash
# if your input is a video
python run_single.py --colmap_name <data_name> --video_extension <video_extension without '.'>
# if your input is a set of images
python run_single.py --colmap_name <data_name> --skip_video_extraction
# if your input is a COLMAP dataset
python run_single.py --colmap_name <data_name> --skip_video_extraction --skip_colmap

# For automatic masking add params:
--masker_automask --masker_prompt <object_name>
# (default object name is main_object which works on most object-centered scenes)
```
Results are saved in *gs2mesh/output/custom_\<params>*.

<details>
<summary>Parameters</summary>

| Parameter                        | Description                                                                     | Custom       | DTU          | TNT        | MobileBrick  | MipNerf360   |
|----------------------------------|---------------------------------------------------------------------------------|--------------|--------------|------------|--------------|--------------|
| `colmap_name`                    | Name of the directory with the COLMAP sparse model                              | sculpture     | scan24       | Ignatius   | aston        | counter       |
| `dataset_name`                   | Name of the dataset. Options: custom, DTU, TNT, MobileBrick, MipNerf360         | custom       | DTU          | TNT        | MobileBrick  | MipNerf360   |
| `experiment_folder_name`         | Name of the experiment folder                                                   | None         | None         | None       | None         | None         |
| `downsample`                  | Downsampling factor                                         | 1        | 1        | 1      | 1        | 3   
| `GS_iterations`                  | Number of Gaussian Splatting iterations                                         | 30000        | 30000        | 30000      | 30000        | 30000        |
| `GS_save_test_iterations`        | Gaussian Splatting test iterations to save                                      | [7000, 30000]| [7000, 30000]| [7000, 30000]| [7000, 30000]| [7000, 30000]|
| `GS_white_background`            | Use white background in Gaussian Splatting                                      | False        | False        | False      | False        | False        |
| `renderer_max_images`            | Maximum number of images to render                                              | 1000         | 1000         | 1000       | 1000         | 1000         |
| `renderer_baseline_absolute`     | Absolute value of the renderer baseline (None uses 7 percent of scene radius)   | None         | None         | None       | None         | None         |
| `renderer_baseline_percentage`   | Percentage value of the renderer baseline                                       | 7            | 7            | 7          | 14            | 7            |
| `renderer_scene_360`   | Scene is a 360 scene                                       | True            | True            | True          | True| True            |
| `no-renderer_scene_360`   | Scene is not a 360 scene                                      |False            | False            | False          | False            | False            |
| `renderer_folder_name`           | Name of the renderer folder (None uses the colmap name)                         | None         | None         | None       | None         | None         |
| `renderer_save_json`             | Save renderer data to JSON (default True - disable with no-renderer_save_json)  | True         | True         | True       | True         | True         |
| `no-renderer_save_json`          | Disable renderer_save_json                                                      | False        | False        | False      | False        | False        |
| `renderer_sort_cameras`          | Sort cameras in the renderer (True if using unordered set of views)             | False        | False        | False      | False        | False        |
| `stereo_model`                   | Stereo model to use                                                             | DLNR_Middlebury | DLNR_Middlebury | DLNR_Middlebury | DLNR_Middlebury | DLNR_Middlebury |
| `stereo_occlusion_threshold`     | Occlusion threshold for stereo model (Lower value masks out more areas)         | 3            | 3            | 3          | 3            | 3            |
| `stereo_warm`                    | Use the previous disparity as initial disparity for current view (disable with no-stereo_warm) | False| True         | True       | True         | False|
| `no-stereo_warm`                 | Disable stereo_warm                                                             | ----| False        | False      | False        | ----|
| `stereo_shading_eps`                 | Small value used for visualization of the depth gradient. Adjusted according to the scale of the scene.                                                             | 1e-4| 1e-4| 1e-4| 1e-4|1e-4|
| `masker_automask`                    | Use GroundingDINO for automatic object detection for masking with SAM2 | False| ----| ----| ----| ----|
| `masker_prompt`                    | Prompt for GroundingDINO | main_object| ----| ----| ----| ----|
| `masker_SAM2_local`                    | Use local SAM2 weights | False| ----| ----| ----| ----|
| `TSDF_scale`                     | Fix depth scale                                                                 | 1.0          | 1.0          | 1.0        | 0.1          | 1.0          |
| `TSDF_dilate`                    | Take every n-th image (1 to take all images)                                    | 1            | 1            | 1          | 1            | 1            |
| `TSDF_valid`                     | Choose valid images as a list of indices (None to ignore)                       | None         | None         | None       | None         | None         |
| `TSDF_skip`                      | Choose non-valid images as a list of indices (None to ignore)                   | None         | None         | None       | None         | None         |
| `TSDF_use_occlusion_mask`        | Ignore occluded regions in stereo pairs for better geometric consistency (default True - disable with no-TSDF_use_occlusion_mask) | True         | True         | True       | True         | True         |
| `no-TSDF_use_occlusion_mask`     | Disable TSDF_use_occlusion_mask                                                 | False        | False        | False      | False        | False        |
| `TSDF_use_mask`                  | Use object masks (optional) (default True - disable with no-TSDF_use_mask)      | False         | True         | False      | True         | False        |
| `no-TSDF_use_mask`               | Disable TSDF_use_mask                                                           | ----| False        | ----| False        | ----|
| `TSDF_invert_mask`               | Invert the background mask for TSDF                                             | False        | False        | False      | False        | False        |
| `TSDF_erode_mask`                | Erode masks in TSDF (default True - disable with no-TSDF_erode_mask)            | True         | True         | True       | True         | True         |
| `no-TSDF_erode_mask`             | Disable TSDF_erode_mask                                                         | False        | False        | False      | False        | False        |
| `TSDF_erosion_kernel_size`       | Erosion kernel size in TSDF                                                     | 10           | 10           | 10         | 10           | 10           |
| `TSDF_closing_kernel_size`       | Closing kernel size in TSDF                                                     | 10           | 10           | 10         | 10           | 10           |
| `TSDF_voxel`                     | Voxel size (voxel length is TSDF_voxel/512)                                     | 2            | 2            | 2          | 2            | 2            |
| `TSDF_sdf_trunc`                 | SDF truncation in TSDF                                                          | 0.04         | 0.04         | 0.04       | 0.04         | 0.04         |
| `TSDF_min_depth_baselines`       | Minimum depth baselines in TSDF                                                 | 4            | 4            | 2          | 4            | 4            |
| `TSDF_max_depth_baselines`       | Maximum depth baselines in TSDF                                                 | 20           | 20           | 10         | 20           | 15           |
| `TSDF_cleaning_threshold`        | Minimal cluster size for clean mesh                                             | 100000       | 100000       | 100000     | 10000        | 100000       |
| `video_extension`                | Video file extension                                                            | mp4          | ----| ----| ----| ----|
| `video_interval`                 | Extract every n-th frame - aim for 3fps                                         | 10           | ----|----| ----           | ----           |
| `GS_port`                        | GS port number (relevant if running several instances at the same time)         | 8080         | 8080         | 8080       | 8080         | 8080         |
| `skip_video_extraction`          | Skip the video extraction stage                                                 | False        | True         | True       | True         | True         |
| `skip_colmap`                    | Skip the COLMAP stage                                                           | False        | True         | True       | True         | True         |
| `skip_GS`                        | Skip the GS stage                                                               | False        | False        | False      | False        | False        |
| `skip_rendering`                 | Skip the rendering stage                                                        | False        | False        | False      | False        | False        |
| `skip_masking`                   | Skip the masking stage                                                          | False        | False        | False      | False        | False        |
| `skip_TSDF`                      | Skip the TSDF stage                                                             | False        | False        | False      | False        | False        |
| `scans`                          | Scan names/numbers                                                              | ----| [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122] | [Barn, Caterpillar, Ignatius, Truck] | [aston, audi, beetles, big_ben, boat, bridge, cabin, camera, castle, colosseum, convertible, ferrari, jeep, london_bus, motorcycle, porsche, satellite, space_shuttle] | [counter, garden] |

</details>


## Common Issues and Tips

Below are some common issues that have risen by users. Please note that we are constantly improving the code  and adding new features according to user feedback.

- **TSDF integration gets killed/segfault.** This is a bug with Open3D ScalableTSDFVolume. It seems to happen in older versions of Ubuntu such as 18.04, and in newer versions of Python (from our experiments, it happened on Python 3.9 and 3.10, and worked on Python 3.7 and 3.8). To avoid this bug, we recommend using Ubuntu 20.04 with Python 3.8 and Open3D 0.17.0.
- **Mesh not visible.** The main culprit is usually TSDF_min/max_depth_baselines. It's a parameter that truncates the depth maps before TSDF fusion, expressed as a multiple of the horizontal baseline. The default for a custom 360 scene with 7% baseline is between 4 and 20 baselines. If the object is not visible, it usually means that your baseline is too small. If your scene is not a 360 scene (for example, the DTU dataset), use a larger baseline. To avoid dealing with the min/max depth baselines, you can segment the object using SAM2 in the interactive notebook *custom_data.py* or using the parameters *--masker_automask* and optionally *--masker_prompt \<object\>* as input to the script *run_single.py*, and set TSDF_max_depth_baselines to a high value that won't affect truncation.
- **Poor mesh quality.** There are many factors that contribute to the quality of the mesh:
	- The video should cover the reconstructed object completely. We recommend the following practices for a good video:
		-  Filming in landscape. 
		- If the object is larger, do 2 cycles around the object from 2 different heights and angles, while keeping the object centered and close to the camera. It's ok if the object is not fully in the frame at all times. 
		- When extracting the video, make sure you maintain around 3 FPS. Lower frame rates can cause COLMAP/GS to fail or produce low quality reconstructions. 
		- We recommend a resolution of 1920x1080, since larger resolutions can take much longer to process, and lower resolutions reduce the resulting reconstruction's quality.
	- Naturally, stereo matching models tend to struggle with transparent/textureless surfaces, and the resulting mesh might be noisy/missing in these places. We hope to improve robustness in an upcoming release. Setting the no-stereo_warm flag appears to help in this case, since it doesn't propagate errors between depth maps.
	- Sometimes, changing the horizontal baseline to a larger baseline for larger objects can improve stereo matching. Make sure to adjust TSDF_min/max_depth_baselines proportionally to how much you changed from the default baseline percentage value (for example, if you double the percentage from 7 to 14, then you need to divide the values of TSDF_min/max_depth_baselines by 2).
	- With custom datasets which come pre-computed poses (other than the ones officially supported), the TSDF_scale parameter might need to be adjusted (usually changed from 1 to 0.1), otherwise the resolution of the resulting mesh will be very low due to inconsistency with the scale in which the TSDF algorithm expects the data.
- **Code runs slow.** There are several factors that can slow down the code:
	- Image size - Images larger than 1920x1080 take much longer to process in the stereo model and TSDF. set the downsample argument to reduce the size of the image at the beginning of the process.
	- Depth truncation - If the depth truncation for TSDF is too loose, the TSDF will struggle to integrate the background and take much longer. Make sure that you are either truncating depth to ignore the background, or using a segmentation mask on the object you want to reconstruct using the automatic masking tool or the interactive masking tool.

## Acknowledgements

We thank the following works and repositories:

 - [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) for their pioneering work which made all this possible.
 - [DLNR](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network) for their stereo model.
 - [Open3D](https://github.com/isl-org/Open3D) for the TSDF fusion algorithm.
 - [SAM2](https://github.com/facebookresearch/segment-anything-2) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/tree/main) for the masking and object detection models.
 - [2DGS](https://github.com/hbb1/2d-gaussian-splatting/tree/main) and [GOF](https://github.com/autonomousvision/gaussian-opacity-fields/tree/main) for their preprocessing code for TNT and the preprocessed DTU dataset and evaluation code.
 - [Tanks and Temples](https://github.com/isl-org/TanksAndTemples) and [MobileBrick](https://github.com/ActiveVisionLab/MobileBrick) for their evaluation codes.

## Citation
If you use GS2Mesh or parts of our code in your work, please cite the following paper:
```bibtex
@article{wolf2024surface,
	title={GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views},
	author={Wolf, Yaniv and Bracha, Amit and Kimmel, Ron},
	journal={arXiv preprint arXiv:2404.01810},
	year={2024}
}
```
## License
The Gaussian Splatting library is under the [Gaussian-Splatting License](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).