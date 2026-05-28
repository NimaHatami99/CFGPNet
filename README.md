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

### FLIR: 



## Framework specifications

### CFGPNet model scales

**Input size:** 640×640

| **Model** | **#Param. (M)** | **GFLOPs (G)** | **FPS** | **Model Size (MB)** | **#Layers** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **CFGPNet-m** | 21.0 | 94.6 | 1463.4 | 41.6 | 1165 |
| **CFGPNet-c** | 71.7 | 362.2 | 386.1 | 138.8 | 1165 |
| **CFGPNet-e** | 180.9 | 560.7 | 136.9 | 349.6 | 1806 |

> **Notes:** #Param. and GFLOPs are reported for a single forward pass at `1×6×640×640`. FPS is measured by timing the forward pass only, excluding data loading and preprocessing.
