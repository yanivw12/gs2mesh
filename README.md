


<p align="center">
  <h1 align="center">GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views</h1>
  <p align="center"> Yaniv Wolf · Amit Bracha · Ron Kimmel</p>
  <h3 align="center">ECCV 2024</h3>
  <h3 align="center"> <a href="https://arxiv.org/pdf/2404.01810">Paper (arXiv)</a> | <a href="https://gs2mesh.github.io/">Project Page</a>  </h3>
  <div align="center"></div>
</p>

![pipeline](assets/pipeline.jpeg)

We introduce a novel pipeline for multi-view 3D reconstruction from uncontrolled videos, capable of extracting smooth meshes from noisy Gaussian Splatting clouds. While other geometry extraction methods rely on the location of the Gaussian elements, and attempt to regularize them using geometric constraints, we simply utilize a pre-trained stereo model as a real-world geometric prior, that can extract accurate depths from every view, and fuse them together to create a smooth mesh, without any further optimization needed. Our method achieves state-of-the art reconstruction results, and only requires a short overhead on top of the GS optimization.

## Updates

- **2024/07/30:** Code is released! Includes preprocessing, reconstruction and evaluation code for DTU, TNT and MobileBrick, as well as code for reconstruction of custom data and MipNerf360.

## Future Work

- [ ] Add support for additional state-of-the-art GS and Stereo models
- [ ] Add support for SAM2
- [ ] Release Google Colab demo
- [x] Release notebook for interactive reconstruction of custom data
- [x] Release reconstruction/evaluation code for DTU, TNT, MobileBrick, MipNerf360

## Installation

First, clone the repository:
```bash
git clone https://github.com/yanivw12/gs2mesh.git
```
Then, install the environment and activate:
```bash
conda env create --file environment.yml
conda activate gs2mesh
```
For the stereo model, download the pre-trained Middlebury and Sceneflow models from [DLNR](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network), and place them in *gs2mesh/third_party/DLNR/pretrained/*.

For the Segment Anything Model masker, download the pretrained weights from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), and place them in *gs2mesh/third_party/SAM/*.

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

You can either run an interactive notebook *custom_data.ipynb*, which also has visualizations and allows masking of objects interactively, or run a script *run_single.py*:
```bash
# if your input is a video
python run_single.py --colmap_name <data_name> --video_extension <video_extension without '.'>
# if your input is a set of images
python run_single.py --colmap_name <data_name> --skip_video_extraction
# if your input is a COLMAP dataset
python run_single.py --colmap_name <data_name> --skip_video_extraction --skip_colmap
```
Results are saved in *gs2mesh/output/custom_\<params>*.

<details>
<summary>Parameters</summary>

| Parameter                        | Description                                                                     | Custom       | DTU          | TNT        | MobileBrick  | MipNerf360   |
|----------------------------------|---------------------------------------------------------------------------------|--------------|--------------|------------|--------------|--------------|
| `colmap_name`                    | Name of the directory with the COLMAP sparse model                              | sculpture     | scan24       | Ignatius   | aston        | garden       |
| `dataset_name`                   | Name of the dataset. Options: custom, DTU, TNT, MobileBrick, MipNerf360         | custom       | DTU          | TNT        | MobileBrick  | MipNerf360   |
| `experiment_folder_name`         | Name of the experiment folder                                                   | None         | None         | None       | None         | None         |
| `GS_iterations`                  | Number of Gaussian Splatting iterations                                         | 30000        | 30000        | 30000      | 30000        | 30000        |
| `GS_save_test_iterations`        | Gaussian Splatting test iterations to save                                      | [7000, 30000]| [7000, 30000]| [7000, 30000]| [7000, 30000]| [7000, 30000]|
| `GS_white_background`            | Use white background in Gaussian Splatting                                      | False        | False        | False      | False        | False        |
| `renderer_max_images`            | Maximum number of images to render                                              | 1000         | 1000         | 1000       | 1000         | 1000         |
| `renderer_baseline_absolute`     | Absolute value of the renderer baseline (None uses 7 percent of scene radius)   | None         | None         | None       | None         | None         |
| `renderer_baseline_percentage`   | Percentage value of the renderer baseline                                       | 7            | 7            | 7          | 14            | 7            |
| `renderer_folder_name`           | Name of the renderer folder (None uses the colmap name)                         | None         | None         | None       | None         | None         |
| `renderer_save_json`             | Save renderer data to JSON (default True - disable with no-renderer_save_json)  | True         | True         | True       | True         | True         |
| `no-renderer_save_json`          | Disable renderer_save_json                                                      | False        | False        | False      | False        | False        |
| `renderer_sort_cameras`          | Sort cameras in the renderer (True if using unordered set of views)             | False        | False        | False      | False        | False        |
| `stereo_model`                   | Stereo model to use                                                             | DLNR_Middlebury | DLNR_Middlebury | DLNR_Middlebury | DLNR_Middlebury | DLNR_Middlebury |
| `stereo_occlusion_threshold`     | Occlusion threshold for stereo model (Lower value masks out more areas)         | 3            | 3            | 3          | 3            | 3            |
| `stereo_warm`                    | Use the previous disparity as initial disparity for current view (default True - disable with no-stereo_warm) | True         | True         | True       | True         | True         |
| `no-stereo_warm`                 | Disable stereo_warm                                                             | False        | False        | False      | False        | False        |
| `TSDF_scale`                     | Fix depth scale                                                                 | 1.0          | 1.0          | 1.0        | 0.1          | 1.0          |
| `TSDF_dilate`                    | Take every n-th image (1 to take all images)                                    | 1            | 1            | 1          | 1            | 1            |
| `TSDF_valid`                     | Choose valid images as a list of indices (None to ignore)                       | None         | None         | None       | None         | None         |
| `TSDF_skip`                      | Choose non-valid images as a list of indices (None to ignore)                   | None         | None         | None       | None         | None         |
| `TSDF_use_occlusion_mask`        | Ignore occluded regions in stereo pairs for better geometric consistency (default True - disable with no-TSDF_use_occlusion_mask) | True         | True         | True       | True         | True         |
| `no-TSDF_use_occlusion_mask`     | Disable TSDF_use_occlusion_mask                                                 | False        | False        | False      | False        | False        |
| `TSDF_use_mask`                  | Use object masks (optional) (default True - disable with no-TSDF_use_mask)      | True         | True         | False      | True         | False        |
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
| `video_extension`                | Video file extension                                                            | MP4          | ----| ----| ----| ----|
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


## FAQ

This section will be updated with common issues.

## Acknowledgements

We thank the following works and repositories:

 - [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) for their pioneering work which made all this possible.
 - [DLNR](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network) for their stereo model.
 - [Open3D](https://github.com/isl-org/Open3D) for the TSDF fusion algorithm.
 - [SAM](https://github.com/facebookresearch/segment-anything) for the masking model.
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
The Gaussian Splatting library is under the [Gaussian-Splatting License](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md). All original code in this repository is under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.