---
title: "Deep Learning based Real-time Virtual YouTuber Face Projection System"
collection: teaching
type: "Personal project"
permalink: /teaching/2015-spring-teaching-2
venue: "Face Detection, Face Alignment, Pose Estimation, Iris Localization"
date: 2024-06-02
location: "Jetson Nano, ARM64"
---

An real-time face alignment toolkit for vitural youtuber on embedded device, which based on [RetinaFace](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html) (CVPR, 2020).

You can find a short wideo of the project on [Bilibili](https://www.bilibili.com/video/BV1MT421v7sQ).

![a](https://img.shields.io/badge/Python-3.6-green?style=flat-square) ![c](https://img.shields.io/badge/JetPack-4.6.1-orange?style=flat-square) ![b](https://img.shields.io/badge/Code%20Version-1.2-blue?style=flat-square) ![d](https://img.shields.io/badge/CVPR-2020-red?style=flat-square) 

## Introduction

**A Real-time Face Alignment Toolkit for Vitural Youtuber on Embedded Device**

In this project, we use the algorithm from [RetinaFace](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html), design a general toolkit on embedded device, towards algin face in real-time for vitural youtuber. Compared to the classic multilayer feature pyramid scaling, the optimized one performs better on the detection of the face capture system's  speed.
### Pipeline
![](https://md.cra.moe/uploads/0f55f6f5fb2d42fb15ae9eb04.png)

<!-- With the rising popularity of anime and manga culture, there has been a growing interest in technologies that bridge the gap between the real and virtual worlds. One such fascinating endeavor is real-time facial transformation into 2D anime characters coupled with motion recognition. This project aims to explore the intersection of computer vision and animation by developing a system capable of seamlessly converting facial expressions into corresponding anime avatars while simultaneously detecting and recognizing various facial movements and gestures.

The motivation behind this project stems from the desire to create immersive experiences for users, allowing them to express themselves through the lens of beloved anime characters. Additionally, such technology holds potential applications in entertainment, gaming, and virtual communication platforms.

In this paper, we present our approach to real-time facial expression recognition and conversion to 2D anime avatars. We discuss the methodologies employed, the challenges encountered, and the results achieved. Moreover, we explore potential avenues for future research and development in this exciting field at the intersection of computer graphics and artificial intelligence. -->
## Methodology

### Classic Face Detection Algorithm

We use the alogorithm from [RetinaFace](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html) (CVPR, 2020). The basic ider is that, design a simple one-stage tiny objects detecting algorithm, view the critical points as different classes. Then do face **Reconstruction** (a.) and **Localisation** (b.).
![framework](https://md.cra.moe/uploads/ea4e161632db500fc76a84501.png)

- (a.) It is clear that, the algorithm first design a mutii-layer (5) feature pyramid, by iterating convlotion. For each scale of the feature maps, there is a deformable context module.
- (b.) Following the context modules, we calculate a joint loss (face classification, face box regression, five facial landmarks regression and 1k 3D vertices regression) for each positive anchor. To minimise the residual of localisation, we employ cascade regression.

### Optimized Real-time Virtual YouTuber Face 
The total framework is an upgradation of the [Deep Pictorial Gaze Estimation](https://openaccess.thecvf.com/content_ECCV_2018/html/Seonwook_Park_Deep_Pictorial_Gaze_ECCV_2018_paper.html) (ECCV 2018).
Since the detection target of the face capture system is in the middle-close range, there is no need for complex pyramid scaling. Compared to Feature Pyramid Network showd in Network Structure, our model use less layer of feature pyramid. For middle-close range detection, it's enough to get precise detection with less feature pyramids, so we reduces the feature pyramids and it  performs as accuracy as [RetinaFace](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html) but with higer speed.

### Face Alignment
Apply the facial landmarks for calculating head pose and slicing the eye regions for gaze estimation. Moreover, the mouth and eys status can be inferenced via these key points.
<!-- ![](https://md.cra.moe/uploads/ea4e161632db500fc76a84509.jpeg)
![](https://md.cra.moe/uploads/ea4e161632db500fc76a8450a.jpeg)
![](https://md.cra.moe/uploads/ea4e161632db500fc76a8450c.jpeg) -->
![](https://md.cra.moe/uploads/ea4e161632db500fc76a84517.png)



### Pose Estimation
The Perspective-n-Point (PnP) is the problem of determining the 3D position and orientation (pose) of a camera from observations of known point features.
The PnP is typically formulated and solved linearly by employing [lifting](https://ieeexplore.ieee.org/document/1195992), or [algebraically](https://openaccess.thecvf.com/content_cvpr_2017/html/Ke_An_Efficient_Algebraic_CVPR_2017_paper.html) or [directly](https://ieeexplore.ieee.org/document/6126266).

Briefily, for head pose estimation, a set of pre-defined 3D facial landmarks and the corresponding 2D image projections need to be given. In this project, we employed the eyebrow, eye, nose, mouth and jaw landmarks in the [AIFI Anthropometric Model](https://aifi.isr.uc.pt/Downloads.html) as origin 3D feature points. The pre-defined vectors and mapping proctol can be found at [here](PythonClient/pretrained/head_pose_object_points.npy).

### Iris Localization

Estimating human gaze from a single RGB face image is a challenging task.
Theoretically speaking, the gaze direction can be defined by pupil and eyeball center, however, the latter is unobservable in 2D images. Previous work of [Swook, et al.](https://openaccess.thecvf.com/content_ECCV_2018/html/Seonwook_Park_Deep_Pictorial_Gaze_ECCV_2018_paper.html) presents a method to extract the semantic information of iris and eyeball into the intermediate representation, which so called gazemaps, and then decode the gazemaps into euler angle through regression network.

Inspired by this, we propose a 3D semantic information based gaze estimation method. Instead of employing gazemaps as the intermediate representation, we estimate the center of the eyeball directly from the average geometric information of human gaze.

![](https://s3.ax1x.com/2020/12/15/rKWPK0.jpg)

### Fast Face Detection (Ours)

For middle-close range face detection, appropriately removing FPN layers and reducing the density of anchors could count-down the overall computational complexity. In addition, low-level APIs are used at preprocessing stage to bypass unnecessary format checks. While inferencing, runtime anchors are cached to avoid repeat calculations. More over, considerable speeding up can be obtained through vector acceleration and NMS algorithm improvement at post-processing stage.

![](https://md.cra.moe/uploads/ea4e161632db500fc76a84519.png)

## Hardware Platform
### Jetson Nano: ARM64
<!-- ![](https://md.cra.moe/uploads/ea4e161632db500fc76a84507.png)
 -->

| CPU        | GPU                            | MEM | Storage       | JetPack |
|:----------:|:------------------------------:|:---:|:-------------:|:-------:|
| Cortex-A57 | NVIDIA Maxwell (128 cuda core) | 4GB | 16GB Emmc 5.1 | 4.6.1   |

### Training Platform: x86-Debian-cluster

| CPU                        | GPU            | MEM   | Cuda Version | IP           |
|:--------------------------:|:--------------:|:-----:|:------------:|:------------:|
| Intel(R) Xeon(R) Gold 6240 | Tesla V100 x 4 | 128GB | 11.7         | 172.18.34.23 |

## Setup
<!-- ### Requirements
**·** Python 3.6+
**·** **pip3 install -r requirements.txt**
**·** node.js and npm or yarn
**·** **cd NodeServer && yarn** # install node modules
### Socket-IO Server
**·** **cd NodeServer**
**·** **yarn start**
### Python Client
**·** **cd PythonClient**
**·** **python3 vtuber_link_start.py <your-video-path>** -->

### Requirements
``` shell
conda create -n deepVTB python=3.6
conda activate deepVTB

git clone https://github.com/Kazawaryu/DeepVTB.git
cd DeepVTB
pip install --upgrade pip
pip install -r requirements.txt
-----------------------------------------------------------------------------
# To use advanced version, build mxnet from source
git clone --recursive https://github.com/apache/incubator-mxnet mxnet
cd mxnet
echo "USE_NCCL=1" >> make/config.mk
echo "USE_NCCP_PATH=path-to-nccl-installation-folder" >> make/config.mk
cp make/config.mk .
make -j"$(nproc)"

pip install mxnet
```

``` shell
sudo apt install nvm
nvm install 18.13.0
npm install yarn
cd DeepVTB/NodeServer
yarn install
```
### Usage
``` shell
cd DeepVTB/NodeServer
yarn start
-----------------------------------------------------------------------------
cd DeepVTB/PythonClient
python vtuber_link_start.py
```
    

## Performance

### Model Training
We use tensorflow as training toolkit. Use WiderFace (in COCO format) dataset as training set and validating set. The details of dataset are as follow.
| Dataset   | Frames | Sample-train | Sample-val | mAP-val   |
| --------- | ------ | ------------ | ---------- | --------- |
| WiderFace | 32203  | 158,989      | 39,496     | ***0.865*** | 

We set `epoch=80, batch_size=16, lr=0.0001-0.001 (auto set)`, this is the loss cruve and learning rate cruve.
![](https://md.cra.moe/uploads/ea4e161632db500fc76a84518.png)


### Real-time Performance
We test the real time performance on Jetson Nano.
| Face Detection | Face Alignment | Pose Estimate | Iris Localization | Sum     | FPS    |
| -------------- | -------------- | ------------- | ----------------- | ------- | ------ |
| 45.5ms         | 24.9ms         | 48.7ms        | 22.3ms            | 141.4ms | 7.07±1 | 

After using fast face detection optimization, the performance will be:
| Face Detection | Face Alignment | Pose Estimate | Iris Localization | Sum     | FPS    |
| -------------- | -------------- | ------------- | ----------------- | ------- | ------ |
| 12.3ms         | 18.1ms         | 49.2ms        | 22.6ms            | 104.2ms | 9.59±1 | 

### Our optimization (general test)
| Scale | RetinaFace           | Faster RetinaFace  | Speed Up |
| ----- | -------------------- | ------------------ | -------- |
| 0.1   | 2.854ms              | **2.155ms (Ours)** | 32%      |
| 0.4   | 3.481ms              | 2.916ms            | 19%      |
| 1.0   | **5.743ms (origin)** | 5.413ms            | 6.1%     |
| 2.0   | 22.351ms             | 20.599ms           | 8.5%     |

## Contributors

<a href="https://github.com/Kazawaryu/DeepVTB/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Kazawaryu/DeepVTB" />
</a>

## License

[MIT](LICENSE) © Kazawaryu


## Citation

``` bibtex
@InProceedings{Deng_2020_CVPR,
      author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
      booktitle = {2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
      title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild}, 
      year = {2020},
      pages = {5202-5211},
      keywords = {Face;Three-dimensional displays;Face detection;Two dimensional displays;Task analysis;Image reconstruction;Training},
      doi = {10.1109/CVPR42600.2020.00525}
}

@InProceedings{Park_2018_ECCV,
      author = {Park, Seonwook and Spurr, Adrian and Hilliges, Otmar},
      title = {Deep Pictorial Gaze Estimation},
      booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
      month = {September},
      year = {2018}
}

@inproceedings{Liu_2018_ECCV,
      author = {Liu, Songtao and Huang, Di and Wang, Yunhong},
      title = {Receptive Field Block Net for Accurate and Fast Object Detection},
      booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
      month = {September},
      year = {2018}
}
```