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
conda create -n DCSR python=3.7R
conda activate DCSR
pip install -r requirements.txt
```

## Dataset
Download our CameraFusion dataset from [here](https://drive.google.com/file/d/1SxU6D1yYTTnZnCyytTObsZxZQigWLciT/view?usp=sharing).
```
cd DualCameraSR
mkdir data
mv CameraFusion.zip ./data
cd ./data
unzip CameraFusion.zip
```
For CUFED5 dataset, can be download from [SRNTT repo](https://github.com/ZZUTK/SRNTT).

## Quick Start
```
For 4K test(with ground-truth High-Resolution images):
sh test.py

For 8K test(without ground-truth High-Resolution images):
sh test_8k.sh
```


## Training
```
Regular training:
sh train.sh

After regular training, we can use Self-supervised Real-image Adaptation (SRA) to finetune the model for better 8K visual performance:
sh train_finetune.sh

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
