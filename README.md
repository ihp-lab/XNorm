# X-Norm: Exchanging Normalization Parameters for Bimodal Fusion.
Yufeng Yin*, Jiashu Xu*, Tianxin Zu, and Mohammad Soleymani

Correspondence to: 
  - Yufeng Yin (yin@ict.usc.edu)

## Introduction
This is the official Pytorch implementation for **X-Norm: Exchanging Normalization Parameters for Bimodal Fusion.**

This repo contains the following methods for multimodal fusion:
 - Late fusion
 - Early fusion
 - [Misa](https://github.com/declare-lab/MISA) [1]
 - [MulT](https://github.com/yaohungt/Multimodal-Transformer) [2]
 - Gradient-Blending [3]
 - X-Norm (our method)

## Overview
We present **X-Norm**, a novel, simple and efficient method for bimodal fusion that generates and exchanges limited but meaningful normalization parameters between the modalities implicitly aligning the feature spaces.

### Overview for X-Norm
![img](/figures/X-Norm.png)

### Architecture for NormExchange layer
![img](/figures/NormExchange.png)

## Usage
### Requirements
 - Python 3.9
 - PyTorch 1.11
 - CUDA 10.1

### Datasets and Pretrained Weights
Step 1: Download the RGB and Optical flow frames of kitchens P01, P08, and P22 from [EPIC_KITCHENS-100](https://github.com/epic-kitchens/epic-kitchens-100-annotations) and put them into the ```data/epic_kitchens``` fold.

Step 2: Download the pretrained weights [rgb_imagenet.pt](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt) and [flow_imagenet.pt](https://github.com/piergiaj/pytorch-i3d/blob/master/models/flow_imagenet.pt) and put them into the ```checkpoints``` fold.

### Run the Code
Unimodal methods (RGB or Optical flow)
```
python main.py --fusion rgb/flow
```
Multimodal methods
```
python main.py --fusion early/late/misa/mult/gb/xnorm
```
## Citation
If you find this work or code is helpful in your research, please cite:
```
Coming soon
```

## Reference
[1] Hazarika, Devamanyu, Roger Zimmermann, and Soujanya Poria. "Misa: Modality-invariant and-specific representations for multimodal sentiment analysis." Proceedings of the 28th ACM international conference on multimedia. 2020.

[2] Tsai, Yao-Hung Hubert, et al. "Multimodal transformer for unaligned multimodal language sequences." Proceedings of the conference. Association for Computational Linguistics. Meeting. Vol. 2019. NIH Public Access, 2019.

[3] Wang, Weiyao, Du Tran, and Matt Feiszli. "What makes training multi-modal classification networks hard?." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
