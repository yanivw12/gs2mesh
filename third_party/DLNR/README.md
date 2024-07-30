# High-frequency Stereo Matching Network

[High-Frequency Stereo Matching Network](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network/releases/download/v1.0.0/High-frequency.Stereo.Matching.Network.pdf)<br/>
CVPR 2023, Highlight<br/>
Haoliang Zhao, Huizhou Zhou, Yongjun Zhang, Jie Chen, Yitong Yang and Yong Zhao<br/>

```bibtex
@inproceedings{zhao2023high,
  title={High-Frequency Stereo Matching Network},
  author={Zhao, Haoliang and Zhou, Huizhou and Zhang, Yongjun and Chen, Jie and Yang, Yitong and Zhao, Yong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1327--1336},
  year={2023}
}
```

## Software Requirements (Recommended)
PyTorch 1.12.0 <br/>
CUDA 11.7

![avatar](./DLNR.png)

```Shell
pip install scipy
pip install tqdm
pip install tensorboard
pip install opt_einsum
pip install imageio
pip install opencv-python
pip install scikit-image
pip install einops
```
The program runs in a variety of environments, but the results may vary slightly.

## Required Data
To evaluate/train High-Frequency Stereo Matching Network, you will need to download the required datasets. 
* [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#:~:text=on%20Academic%20Torrents-,FlyingThings3D,-Driving) 
* [Middlebury](https://vision.middlebury.edu/stereo/data/)
* [KITTI-2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

By default `stereo_datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── Monkaa
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── Driving
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── Middlebury
        ├── MiddEval3
```


## Build Sampler (Optional)
```Shell
cd sampler
rm -r build corr_sampler.egg-info dist
python setup.py install && cd ..
```

## Train
```Shell
bash ./train.sh
```

## Evaluate
Set the arguments in evaluate_stereo.py and execute
```Shell
python evaluate_stereo.py
```

## Pretrained Weights
* [Sceneflow](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network/releases/download/v1.0.0/DLNR_SceneFlow.pth) 
* [Middlebury](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network/releases/download/v1.0.0/DLNR_Middlebury.pth)
* [KITTI-2015](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network/releases/download/v1.0.0/DLNR_KITTI2015.pth)

## Acknowledgement
Special thanks to RAFT-Stereo for providing the code base for this work.

<details>
<summary>
<a href="https://github.com/princeton-vl/RAFT-Stereo">RAFT-Stereo</a> [<b>BibTeX</b>]
</summary>

```bibtex
@inproceedings{lipson2021raft,
  title={RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching},
  author={Lipson, Lahav and Teed, Zachary and Deng, Jia},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```
</details>
