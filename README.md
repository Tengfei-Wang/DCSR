# Dual Camera Super-resolution
Implementation for our ICCV 2021 paper: Dual-Camera Super-Resolution with Aligned Attention Modules

[paper]( ) | [project website](https://tengfei-wang.github.io/Dual-Camera-SR/index.html) | [dataset]( ) | [demo video]( )

<img src="pics/demo.png" width="720px"/> 

## Introduction
We present a novel approach to reference-based super-resolution (RefSR) with the focus on real-world dual-camera super-resolution (DCSR).

## Setup
### Installation
```
git clone https://github.com/Tengfei-Wang/DualCameraSR.git
cd DualCameraSR
```

### Environment
This code is based on PyTorch.

The environment can be simply set up by Anaconda:
```
conda create -n DCSR python=3.7
conda activate DCSR
pip install -r requirements.txt
```


## Quick Start
```
python test.py
```


## Training
```
python test.py
```

## Citation
If you find this work useful for your research, please cite:
``` 
@InProceedings{wang2021DCSR,
author = {Wang, Tengfei and Xie, Jiaxin and Sun, Wenxiu and Yan, Qiong and Chen, Qifeng},
title = {Dual-Camera Super-Resolution with Aligned Attention Modules},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2021}
}
```
