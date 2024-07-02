<div align="right">English | <a href="./README_CN.md">简体中文</a></div>

<div align="center">
    <h2><strong>4D Contrastive Superflows are Dense 3D Representation Learners</strong></h2>
</div>

<div align="center">
    <a href="https://xiangxu-0103.github.io/" target='_blank'>Xiang Xu</a><sup>1,*</sup>,&nbsp;&nbsp;&nbsp;
    <a href="https://ldkong.com/" target='_blank'>Lingdong Kong</a><sup>2,3,*</sup>,&nbsp;&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=zG3rgUcAAAAJ" target='_blank'>Hui Shuai</a><sup>4</sup>,&nbsp;&nbsp;&nbsp;
    <a href="http://zhangwenwei.cn/" target='_blank'>Wenwei Zhang</a><sup>2</sup>,&nbsp;&nbsp;&nbsp;
    </br>
    <a href="https://scholar.google.com/citations?user=lSDISOcAAAAJ" target='_blank'>Liang Pan</a><sup>2</sup>,&nbsp;&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=eGD0b7IAAAAJ" target='_blank'>Kai Chen</a><sup>2</sup>,&nbsp;&nbsp;&nbsp;
    <a href="https://liuziwei7.github.io/" target='_blank'>Ziwei Liu</a><sup>5</sup>,&nbsp;&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=2Pyf20IAAAAJ" target='_blank'>Qingshan Liu</a><sup>4</sup>
    </br>
    <sup>1</sup>Nanjing University of Aeronautics and Astronautics&nbsp;&nbsp;&nbsp;
    <sup>2</sup>Shanghai AI Laboratory&nbsp;&nbsp;&nbsp;
    <sup>3</sup>National University of Singapore&nbsp;&nbsp;&nbsp;
    <sup>4</sup>Nanjing University of Posts and Telecommunications&nbsp;&nbsp;&nbsp;
    <sup>5</sup>S-Lab, Nanyang Technological University&nbsp;&nbsp;&nbsp;
</div>

<div align="center">
    <a href="" target='_blank'>
        <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-blue">
    </a>
    <a href="https://xiangxu-0103.github.io/SuperFlow" target='_blank'>
        <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-yellow">
    </a>
    <a href="" target='_blank'>
        <img src="https://img.shields.io/badge/Demo-%F0%9F%8E%AC-violet">
    </a>
    <a href="" target='_blank'>
        <img src="https://img.shields.io/badge/%E4%B8%AD%E8%AF%91%E7%89%88-%F0%9F%90%BC-red">
    </a>
    <a href="" target='_blank'>
        <img src="https://visitor-badge.laobi.icu/badge?page_id=Xiangxu-0103.SuperFlow&left_color=gray&right_color=lightgreen">
    </a>
</div>

## About

SuperFlow is introduced to harness consecutive LiDAR-camera pairs for establishing spatiotemporal pretraining objectives. It stands out by integrating two key designs: 1) a dense-to-sparse consistency regularization, which promotes insensitivity to point cloud density variations during feature learning, and 2) a flow-based contrastive learning module, carefully crafted to extract meaningful temporal cues from readily available sensor calibrations.

<img src="docs/figs/superflow.png" align="center" width="100%">

## Updates

- \[2024.07\] - Our paper is accepted by [ECCV](https://eccv2024.ecva.net/).

## Outline

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Getting Started](#getting-started)
- [Main Results](#main-results)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## :gear: Installation

For details related to installation and environment setups, kindly refer to [INSTALL.md](./docs/INSTALL.md).

## :hotsprings: Data Preparation

Kindly refer to [DATA_PREPAER.md](./docs/DATA_PREPAER.md) for the details to prepare the datasets.

## :rocket: Getting Started

To learn more usage about this codebase, kindly refer to [GET_STARTED.md](./docs/GET_STARTED.md).

## :bar_chart: Main Results

### Comparisons of state-of-the-art pretraining methods

### Domain generalization study

### Out-of-distribution 3D robustness study

## License

This work is under the [Apache 2.0 license](LICENSE).

## Citation

If you find this work helpful for your research, please kindly consider citing our paper:

```latex
@inproceedings{xu2024superflow,
    title = {4D Contrastive Superflows are Dense 3D Representation Learners},
    author = {Xu, Xiang and Kong, Lingdong and Shuai, Hui and Zhang, Wenwei and Pan, Liang and Chen, Kai and Liu, Ziwei and Liu, Qingshan},
    booktitle = {European Conference on Computer Vision},
    year = {2024}
}
```

## Acknowledgements

This work is developed based on the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) codebase.

> <img src="https://github.com/open-mmlab/mmdetection3d/blob/main/resources/mmdet3d-logo.png" width="30%"/><br>
> MMDetection3D is an open-source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D perception. It is a part of the OpenMMLab project developed by MMLab.

We acknowledge the use of the following public resources during the couuse of this work: <sup>1</sup>[nuScenes](https://www.nuscenes.org/nuscenes), <sup>2</sup>[nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit), <sup>3</sup>[SemanticKITTI](http://www.semantic-kitti.org), <sup>4</sup>[SemanticKITTI-API](https://github.com/PRBonn/semantic-kitti-api), , <sup>5</sup>[WaymoOpenDataset](https://waymo.com/open), <sup>6</sup>[Synth4D](https://github.com/saltoricristiano/gipso-sfouda), <sup>7</sup>[ScribbleKITTI](https://github.com/ouenal/scribblekitti), <sup>8</sup>[RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D), <sup>9</sup>[SemanticPOSS](http://www.poss.pku.edu.cn/semanticposs.html), <sup>10</sup>[SemanticSTF](https://github.com/xiaoaoran/SemanticSTF), <sup>11</sup>[SynthLiDAR](https://github.com/xiaoaoran/SynLiDAR), <sup>12</sup>[DAPS-3D](https://github.com/subake/DAPS3D), <sup>13</sup>[Robo3D](https://github.com/ldkong1205/Robo3D), <sup>14</sup>[SLidR](https://github.com/valeoai/SLidR), <sup>15</sup>[DINOv2](https://github.com/facebookresearch/dinov2), <sup>16</sup>[Segment-Any-Point-Cloud](https://github.com/youquanl/Segment-Any-Point-Cloud), <sup>17</sup>[OpenSeeD](https://github.com/IDEA-Research/OpenSeeD), <sup>18</sup>[torchsparse](https://github.com/mit-han-lab/torchsparse). :heart_decoration:
