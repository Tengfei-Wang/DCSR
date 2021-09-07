# DCSR: Dual Camera Super-Resolution
Implementation for our ICCV 2021 oral paper: Dual-Camera Super-Resolution with Aligned Attention Modules

[paper](https://arxiv.org/abs/2109.01349) | [project website](https://tengfei-wang.github.io/Dual-Camera-SR/index.html) | [dataset](https://drive.google.com/file/d/1SxU6D1yYTTnZnCyytTObsZxZQigWLciT/view?usp=sharing) | [demo video]( )

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
The environment can be simply set up by Anaconda:
```
conda create -n DCSR python=3.7
conda activate DCSR
pip install -r requirements.txt
```

## Dataset
Download our CameraFusion dataset from [this link](https://drive.google.com/file/d/1SxU6D1yYTTnZnCyytTObsZxZQigWLciT/view?usp=sharing).
This dataset currently consists of 143 pairs of telephoto and wide-angle images in 4K resolution captured by smartphone dual-cameras.
```
mkdir data
cd ./data
unzip CameraFusion.zip
```


## Quick Start
The pretrained models have been put in `./experiments/pretrain`. For quick test, run the scipts: 

```
# For 4K test (with ground-truth High-Resolution images):
sh test.py

# For 8K test (without SRA):
sh test_8k.sh

# For 8K test (with SRA):
sh test_8k_SRA.sh
```


## Training
To train the DCSR model on CameraFusion, run:
```
sh train.sh
```
The trained model should perform well on 4K test, but may suffer performance degradation on 8K test.

After the regular training, we can use Self-supervised Real-image Adaptation (SRA) to finetune the trained model for real-world 8K image applications:
```
sh train_SRA.sh
```

## Results
4X SR results on CUFED5 testset can be found in [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfwang_connect_ust_hk/EidZ5B1jPC9PmTlSUtrMbN0B4a2VY1hXrteYZevijllhJg?e=hQwva7).

More 2X SR results on CameraFusion dataset can be found in our webpage.

<img src="pics/result-cufed.jpg" width="435px"/>  <img src="pics/result-CF.jpg" width="420px"/> 

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

##  Acknowledgement
We thank the authors of [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch), [CSNLN](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention), [TTSR](https://github.com/researchmm/TTSR) and [style-swap](https://github.com/rtqichen/style-swap) for sharing their codes.
