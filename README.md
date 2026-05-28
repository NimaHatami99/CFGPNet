# CFGPNet: Cross-Attention-Based Fused Gradient Programmed Network Framework for Multispectral Object Detection

## 📖 Introduction
This repository contains the official implementation of **CFGPNet**, a framework designed to improve multispectral object detection through **Extensively improved GELAN**, **Computation efficient attention**, **Attention selection and Aggregation fusion**, and **Programmable gradient information**.  
This approach strengthens feature representation while preserving computational efficiency, enhances cross-modal feature interaction and reduces redundant information transfer between visible and thermal branches, combines dense feature aggregation with selective attention-based emphasis, and also improves gradient delivery and optimization quality without altering the inference pathway. 

### CFGPNet-c, -m:

<p align="center">
  <a href="docs/dualyolo2-c.png">
    <img src="docs/dualyolo2-c.png" alt="CFGPNet-c, -m" width="90%">
  </a>
</p>

### CFGPNet-e:

<p align="center">
  <a href="docs/dualyolo2-e.png">
    <img src="docs/dualyolo2-e.png" alt="CFGPNet-e" width="90%">
  </a>
</p>

### RepViTCSPELAN4: 

<p align="center">
  <a href="docs/RepViTCSPELAN4.png">
    <img src="docs/RepViTCSPELAN4.png" alt="RepViTCSPELAN4" width="40%">
  </a>
</p> 

### MFE: 

<p align="center">
  <a href="docs/shuffle.png">
    <img src="docs/shuffle.png" alt="MFE" width="90%">
  </a>
</p> 

### CEA: 

<p align="center">
  <a href="docs/EEMA.png">
    <img src="docs/EEMA.png" alt="CEA" width="40%">
  </a>
</p> 

### ASAF: 

<p align="center">
  <a href="docs/FeatFuse.png">
    <img src="docs/FeatFuse.png" alt="ASAF" width="90%">
  </a>
</p> 

--- 

## Performance 

### Quantitative comparison on the **FLIR** dataset

| **Modality** | **Model** | **mAP50** | **mAP50:95** | **Weights (M)** |
|:--:|:--:|:--:|:--:|:--:|
| **T** | [Fast R-CNN (2015)](https://arxiv.org/abs/1504.08083) | 74.2 | 38.0 | -- |
| **T** | [SSD (2016)](https://arxiv.org/abs/1504.08083) | 65.1 | 30.2 | -- |
| **T** | [RetinaNet (2017)](https://arxiv.org/abs/1504.08083) | 64.9 | 28.6 | -- |
| **T** | [YOLOv5 (2022)](https://arxiv.org/abs/1504.08083) | 74.2 | 37.2 | -- |
| **T** | [YOLOv8 (2023)](https://arxiv.org/abs/1504.08083) | 74.2 | 37.7 | -- |
| **T** | [YOLOv11 (2024)](https://arxiv.org/abs/1504.08083) | 75.3 | 38.9 | -- |
| **T** | [YOLOX (2021)](https://arxiv.org/abs/1504.08083) | 79.5 | 41.6 | -- |
| **T** | [DINO (2022)](https://arxiv.org/abs/1504.08083) | 78.7 | 40.3 | -- |
| **T** | [MobileFomer (2022)](https://arxiv.org/abs/1504.08083) | 72.9 | 36.0 | -- |
| **T** | [EfficientViT (2023)](https://arxiv.org/abs/1504.08083) | 73.1 | 36.6 | -- |
| **RGB** | [Fast R-CNN (2015)](https://arxiv.org/abs/1504.08083) | 67.6 | 30.1 | -- |
| **RGB** | [SSD (2016)](https://arxiv.org/abs/1504.08083) | 59.3 | 23.2 | -- |
| **RGB** | [RetinaNet (2017)](https://arxiv.org/abs/1504.08083) | 59.1 | 23.0 | -- |
| **RGB** | [YOLOv5 (2022)](https://arxiv.org/abs/1504.08083) | 68.4 | 31.3 | -- |
| **RGB** | [YOLOv8 (2023)](https://arxiv.org/abs/1504.08083) | 68.2 | 31.2 | -- |
| **RGB** | [YOLOv11 (2024)](https://arxiv.org/abs/1504.08083) | 68.9 | 31.9 | -- |
| **RGB** | [YOLOX (2021)](https://arxiv.org/abs/1504.08083) | 72.5 | 35.1 | -- |
| **RGB** | [DINO (2022)](https://arxiv.org/abs/1504.08083) | 65.3 | 30.5 | -- |
| **RGB** | [SuperYOLO (2023)](https://arxiv.org/abs/1504.08083) | 72.5 | -- | -- |
| **RGB** | [MobileFomer (2022)](https://arxiv.org/abs/1504.08083) | 66.9 | 30.2 | -- |
| **RGB** | [EfficientViT (2023)](https://arxiv.org/abs/1504.08083) | 67.3 | 30.7 | -- |
| **RGB-T** | [MMI-Det (2024)](https://arxiv.org/abs/1504.08083) | 79.8 | 40.5 | 207.6 |
| **RGB-T** | [CFT (2021)](https://arxiv.org/abs/1504.08083) | 78.7 | 40.2 | 196.9 |
| **RGB-T** | [ICAFusion (2024)](https://arxiv.org/abs/1504.08083) | 79.2 | 41.4 | 120.2 |
| **RGB-T** | [CSSA (2023)](https://arxiv.org/abs/1504.08083) | 79.2 | 41.3 | -- |
| **RGB-T** | [LRAF-Net (2023)](https://arxiv.org/abs/1504.08083) | <ins>80.5</ins> | 42.8 | -- |
| **RGB-T** | [C<sup>2</sup>DFF-Net (2025)](https://arxiv.org/abs/1504.08083) | 76.9 | 40.8 | 6.6 |
| **RGB-T** | [YOLO-Adaptor (2024)](https://arxiv.org/abs/1504.08083) | 80.1 | -- | -- |
| **RGB-T** | [CrossFormer (2024)](https://arxiv.org/abs/1504.08083) | 79.3 | 42.1 | -- |
| **RGB-T** | [EI<sup>2</sup>Det (2025)](https://arxiv.org/abs/1504.08083) | 80.2 | -- | 127.7 |
| **RGB-T** | [CAMDet (2025)](https://arxiv.org/abs/1504.08083) | 76.3 | 37.1 | -- |
| **RGB-T** | [DHANet (2025)](https://arxiv.org/abs/1504.08083) | 74.3 | -- | -- |
| **RGB-T** | [MRD-YOLO (2024)](https://arxiv.org/abs/1504.08083) | 76.5 | 40.9 | -- |
| **RGB-T** | [DFF (2025)](https://arxiv.org/abs/1504.08083) | 80.1 | 38.5 | 120.9 |
| **RGB-T** | [LCMA (2026)](https://arxiv.org/abs/1504.08083) | <ins>80.5</ins> | 42.2 | -- |
| **RGB-T** | [M2I2HA (2026)](https://arxiv.org/abs/1504.08083) | 72.5 | 37.8 | 37.6 |
| **RGB-T** | [JFDet (2026)](https://arxiv.org/abs/1504.08083) | 76.3 | -- | -- |
| **RGB-T** | [MCOR (2025)](https://arxiv.org/abs/1504.08083) | 78.2 | 39.9 | -- |
| **RGB-T** | [ERFF (2026)](https://arxiv.org/abs/1504.08083) | <ins>80.6</ins> | -- | -- |
| **RGB-T** | [ADCA-Net (2026)](https://arxiv.org/abs/1504.08083) | 78.9 | -- | 34.0 |
| **RGB-T** | [DLRMamba (2026)](https://arxiv.org/abs/1504.08083) | 80.0 | -- | -- |
| **RGB-T** | [FCAT (2026)](https://arxiv.org/abs/1504.08083) | 79.9 | 42.7 | 85.2 |
| **RGB-T** | [PMDet (2026)](https://arxiv.org/abs/1504.08083) | 80.4 | 42.8 | 277.1 |
| **RGB-T** | [CFGPNet-m](https://arxiv.org/abs/1504.08083) | 80.0 | <ins>43.1</ins> | 21.0 |
| **RGB-T** | [CFGPNet-c](https://arxiv.org/abs/1504.08083) | 79.8 | <ins>43.6</ins> | 71.7 |
| **RGB-T** | [CFGPNet-e](https://arxiv.org/abs/1504.08083) | <ins>80.7</ins> | <ins>45.0</ins> | 180.9 |

> **Notes:** The three best results are underlined.

## Framework specifications

### CFGPNet model scales

**Input size:** 640×640

| **Model** | **#Param. (M)** | **GFLOPs (G)** | **FPS** | **Model Size (MB)** | **#Layers** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **CFGPNet-m** | 21.0 | 94.6 | 1463.4 | 41.6 | 1165 |
| **CFGPNet-c** | 71.7 | 362.2 | 386.1 | 138.8 | 1165 |
| **CFGPNet-e** | 180.9 | 560.7 | 136.9 | 349.6 | 1806 |

> **Notes:** #Param. and GFLOPs are reported for a single forward pass at `1×6×640×640`. FPS is measured by timing the forward pass only, excluding data loading and preprocessing.
