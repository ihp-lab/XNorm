# X-Norm: Exchanging Normalization Parameters for Bimodal Fusion.
Yufeng Yin*, Jiashu Xu*, Tianxin Zu, Mohammad Soleymani.

Correspondence to: 
  - Yufeng Yin (yin@ict.usc.edu)

## Introduction
This is the official Pytorch implementation for '''X-Norm: Exchanging Normalization Parameters for Bimodal Fusion.'''

This repo contains the following methods for multimodal fusion:
 - Late fusion
 - Early fusion
 - [Misa](https://github.com/declare-lab/MISA) [1]
 - [MulT](https://github.com/yaohungt/Multimodal-Transformer) [2]
 - Gradient-Blending [3]
 - X-Norm (our method)

## Overivew
We present '''X-Norm''', a novel, simple and efficient method for bimodal fusion that generates and exchanges limited but meaningful normalization parameters between the modalities implicitly aligning the feature spaces.

### Overview for X-Norm
![Alt text](/figures/X-Norm.pdf?raw=true "X-Norm")

### Architecture for NormExchange layer
![Alt text](/figures/NormExchange.pdf?raw=true "NormExchange")

## Usage
### Prerequisites

### Datasets

### Run the Code

## Citation
If you find this work or code is helpful in your research, please cite:
```
Coming soon
```

## Reference
[1] Hazarika, Devamanyu, Roger Zimmermann, and Soujanya Poria. "Misa: Modality-invariant and-specific representations for multimodal sentiment analysis." Proceedings of the 28th ACM international conference on multimedia. 2020.

[2] Tsai, Yao-Hung Hubert, et al. "Multimodal transformer for unaligned multimodal language sequences." Proceedings of the conference. Association for Computational Linguistics. Meeting. Vol. 2019. NIH Public Access, 2019.

[3] Wang, Weiyao, Du Tran, and Matt Feiszli. "What makes training multi-modal classification networks hard?." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
