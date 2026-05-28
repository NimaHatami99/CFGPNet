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
| **T** | [Fast R-CNN (2015)](https://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html) | 74.2 | 38.0 | -- |
| **T** | [SSD (2016)](https://doi.org/10.1007/978-3-319-46448-0_2) | 65.1 | 30.2 | -- |
| **T** | [RetinaNet (2017)](https://ieeexplore.ieee.org/document/8417976) | 64.9 | 28.6 | -- |
| **T** | [YOLOv5 (2022)](https://zenodo.org/records/7002879) | 74.2 | 37.2 | -- |
| **T** | [YOLOv8 (2023)](https://github.com/ultralytics/ultralytics) | 74.2 | 37.7 | -- |
| **T** | [YOLOv11 (2024)](https://arxiv.org/abs/2410.17725) | 75.3 | 38.9 | -- |
| **T** | [YOLOX (2021)](https://arxiv.org/abs/2107.08430) | 79.5 | 41.6 | -- |
| **T** | [DINO (2022)](https://arxiv.org/abs/2203.03605) | 78.7 | 40.3 | -- |
| **T** | [MobileFormer (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Mobile-Former_Bridging_MobileNet_and_Transformer_CVPR_2022_paper.html) | 72.9 | 36.0 | -- |
| **T** | [EfficientViT (2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html) | 73.1 | 36.6 | -- |
| **RGB** | [Fast R-CNN (2015)](https://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html) | 67.6 | 30.1 | -- |
| **RGB** | [SSD (2016)](https://doi.org/10.1007/978-3-319-46448-0_2) | 59.3 | 23.2 | -- |
| **RGB** | [RetinaNet (2017)](https://ieeexplore.ieee.org/document/8417976) | 59.1 | 23.0 | -- |
| **RGB** | [YOLOv5 (2022)](https://zenodo.org/records/7002879) | 68.4 | 31.3 | -- |
| **RGB** | [YOLOv8 (2023)](https://github.com/ultralytics/ultralytics) | 68.2 | 31.2 | -- |
| **RGB** | [YOLOv11 (2024)](https://arxiv.org/abs/2410.17725) | 68.9 | 31.9 | -- |
| **RGB** | [YOLOX (2021)](https://arxiv.org/abs/2107.08430) | 72.5 | 35.1 | -- |
| **RGB** | [DINO (2022)](https://arxiv.org/abs/2203.03605) | 65.3 | 30.5 | -- |
| **RGB** | [SuperYOLO (2023)](https://doi.org/10.1109/TGRS.2023.3258666) | 72.5 | -- | -- |
| **RGB** | [MobileFormer (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Mobile-Former_Bridging_MobileNet_and_Transformer_CVPR_2022_paper.html) | 66.9 | 30.2 | -- |
| **RGB** | [EfficientViT (2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html) | 67.3 | 30.7 | -- |
| **RGB-T** | [MMI-Det (2024)](https://doi.org/10.1109/TCSVT.2024.3418965) | 79.8 | 40.5 | 207.6 |
| **RGB-T** | [CFT (2021)](https://arxiv.org/abs/2111.00273) | 78.7 | 40.2 | 196.9 |
| **RGB-T** | [ICAFusion (2024)](https://www.sciencedirect.com/science/article/pii/S0031320323006118) | 79.2 | 41.4 | 120.2 |
| **RGB-T** | [CSSA (2023)](https://openaccess.thecvf.com/content/CVPR2023W/PBVS/html/Cao_Multimodal_Object_Detection_by_Channel_Switching_and_Spatial_Attention_CVPRW_2023_paper.html) | 79.2 | 41.3 | -- |
| **RGB-T** | [LRAF-Net (2023)](https://doi.org/10.1109/TNNLS.2023.3266452) | <ins>80.5</ins> | 42.8 | -- |
| **RGB-T** | [C<sup>2</sup>DFF-Net (2025)](https://doi.org/10.1109/TGRS.2025.3614295) | 76.9 | 40.8 | 6.6 |
| **RGB-T** | [YOLO-Adaptor (2024)](https://doi.org/10.1109/TIV.2024.3393015) | 80.1 | -- | -- |
| **RGB-T** | [CrossFormer (2024)](https://www.sciencedirect.com/science/article/pii/S016786552400045X) | 79.3 | 42.1 | -- |
| **RGB-T** | [EI<sup>2</sup>Det (2025)](https://doi.org/10.1109/TCSVT.2025.3539625) | 80.2 | -- | 127.7 |
| **RGB-T** | [CAMDet (2025)](https://doi.org/10.1109/ICASSP49660.2025.10889505) | 76.3 | 37.1 | -- |
| **RGB-T** | [DHANet (2025)](https://doi.org/10.1109/TGRS.2025.3578675) | 74.3 | -- | -- |
| **RGB-T** | [MRD-YOLO (2024)](https://www.mdpi.com/1424-8220/24/10/3222) | 76.5 | 40.9 | -- |
| **RGB-T** | [DFF (2025)](https://www.mdpi.com/2076-3417/15/11/5857) | 80.1 | 38.5 | 120.9 |
| **RGB-T** | [LCMA (2026)](https://www.mdpi.com/2079-9292/15/3/498) | <ins>80.5</ins> | 42.2 | -- |
| **RGB-T** | [M2I2HA (2026)](https://arxiv.org/abs/2601.14776) | 72.5 | 37.8 | 37.6 |
| **RGB-T** | [JFDet (2026)](https://www.mdpi.com/2072-4292/18/1/176) | 76.3 | -- | -- |
| **RGB-T** | [MCOR (2025)](https://openaccess.thecvf.com/content/WACV2025/html/Jang_Multispectral_Object_Detection_Enhanced_by_Cross-Modal_Information_Complementary_and_Cosine_WACV_2025_paper.html) | 78.2 | 39.9 | -- |
| **RGB-T** | [ERFF (2026)](https://www.sciencedirect.com/science/article/pii/S1566253525011728) | <ins>80.6</ins> | -- | -- |
| **RGB-T** | [ADCA-Net (2026)](https://www.sciencedirect.com/science/article/pii/S0957417426009255) | 78.9 | -- | 34.0 |
| **RGB-T** | [DLRMamba (2026)](https://arxiv.org/abs/2603.06920) | 80.0 | -- | -- |
| **RGB-T** | [FCAT (2026)](https://www.mdpi.com/2072-4292/18/5/826) | 79.9 | 42.7 | 85.2 |
| **RGB-T** | [PMDet (2026)](https://www.mdpi.com/2072-4292/18/7/1068) | 80.4 | 42.8 | 277.1 |
| **RGB-T** | CFGPNet-m | 80.0 | <ins>43.1</ins> | 21.0 |
| **RGB-T** | CFGPNet-c | 79.8 | <ins>43.6</ins> | 71.7 |
| **RGB-T** | CFGPNet-e | <ins>80.7</ins> | <ins>45.0</ins> | 180.9 |

> **Notes:** The three best results are underlined.

<p align="center">
  <a href="docs/FLIR_map_param_diagram.png">
    <img src="docs/FLIR_map_param_diagram.png" alt="FLIR performance" width="60%">
  </a>
</p>

### Quantitative comparison on the **M3FD** dataset

| **Modality** | **Model** | **mAP50** | **mAP50:95** | **Weights (M)** |
|:--:|:--:|:--:|:--:|:--:|
| **T** | [Fast R-CNN (2015)](https://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html) | 78.5 | 49.0 | -- |
| **T** | [SSD (2016)](https://doi.org/10.1007/978-3-319-46448-0_2) | 76.9 | 46.6 | -- |
| **T** | [RetinaNet (2017)](https://ieeexplore.ieee.org/document/8417976) | 79.3 | 49.3 | -- |
| **T** | [YOLOv5 (2022)](https://zenodo.org/records/7002879) | 80.5 | 50.6 | -- |
| **T** | [YOLOv8 (2023)](https://github.com/ultralytics/ultralytics) | 81.3 | 51.0 | -- |
| **T** | [YOLOv11 (2024)](https://arxiv.org/abs/2410.17725) | 81.2 | 50.8 | -- |
| **T** | [YOLOX (2021)](https://arxiv.org/abs/2107.08430) | 80.6 | 50.3 | -- |
| **T** | [MobileFormer (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Mobile-Former_Bridging_MobileNet_and_Transformer_CVPR_2022_paper.html) | 79.3 | 49.5 | -- |
| **T** | [EfficientViT (2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html) | 80.7 | 50.3 | -- |
| **RGB** | [Fast R-CNN (2015)](https://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html) | 80.8 | 51.1 | -- |
| **RGB** | [SSD (2016)](https://doi.org/10.1007/978-3-319-46448-0_2) | 79.4 | 48.6 | -- |
| **RGB** | [RetinaNet (2017)](https://ieeexplore.ieee.org/document/8417976) | 81.9 | 51.4 | -- |
| **RGB** | [YOLOv5 (2022)](https://zenodo.org/records/7002879) | 83.2 | 52.3 | -- |
| **RGB** | [YOLOv8 (2023)](https://github.com/ultralytics/ultralytics) | 84.3 | 53.4 | -- |
| **RGB** | [YOLOv11 (2024)](https://arxiv.org/abs/2410.17725) | 83.7 | 54.3 | -- |
| **RGB** | [YOLOX (2021)](https://arxiv.org/abs/2107.08430) | 82.2 | 50.8 | -- |
| **RGB** | [MobileFormer (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Mobile-Former_Bridging_MobileNet_and_Transformer_CVPR_2022_paper.html) | 82.1 | 51.3 | -- |
| **RGB** | [EfficientViT (2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html) | 84.0 | 52.7 | -- |
| **RGB-T** | [MMI-Det (2024)](https://doi.org/10.1109/TCSVT.2024.3418965) | 83.5 | 51.9 | 207.6 |
| **RGB-T** | [CFT (2021)](https://arxiv.org/abs/2111.00273) | 85.0 | 54.5 | 196.9 |
| **RGB-T** | [ICAFusion (2024)](https://www.sciencedirect.com/science/article/pii/S0031320323006118) | 85.1 | 53.5 | 120.2 |
| **RGB-T** | [TF-YOLO (2023)](https://www.mdpi.com/2032-6653/14/12/352) | 87.9 | -- | -- |
| **RGB-T** | [EI<sup>2</sup>Det (2025)](https://doi.org/10.1109/TCSVT.2025.3539625) | 86.2 | 55.5 | 127.7 |
| **RGB-T** | [MRD-YOLO (2024)](https://www.mdpi.com/1424-8220/24/10/3222) | 86.6 | 59.3 | -- |
| **RGB-T** | [DFF (2025)](https://www.mdpi.com/2076-3417/15/11/5857) | 84.2 | 52.4 | 120.9 |
| **RGB-T** | [MCOR (2025)](https://openaccess.thecvf.com/content/WACV2025/html/Jang_Multispectral_Object_Detection_Enhanced_by_Cross-Modal_Information_Complementary_and_Cosine_WACV_2025_paper.html) | 87.2 | 57.3 | -- |
| **RGB-T** | [TINet (2023)](https://doi.org/10.1109/TIM.2023.3251414) | 85.0 | 49.5 | -- |
| **RGB-T** | [TarDAL (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.html) | 80.7 | 50.1 | -- |
| **RGB-T** | [MMFN (2025)](https://doi.org/10.1109/TCSVT.2024.3454631) | 86.2 | -- | 176.4 |
| **RGB-T** | [CrossModalNet (2026)](https://www.sciencedirect.com/science/article/pii/S0957417425032920) | 87.3 | 55.6 | 92.8 |
| **RGB-T** | [LEFuse (2025)](https://www.sciencedirect.com/science/article/pii/S0925231225002644) | 78.9 | 48.5 | -- |
| **RGB-T** | [IVHL (2025)](https://doi.org/10.1109/TMM.2025.3639945) | 83.4 | 55.8 | 271.0 |
| **RGB-T** | [FD2Net (2025)](https://ojs.aaai.org/index.php/AAAI/article/view/32507) | 83.5 | -- | -- |
| **RGB-T** | [CARNet (2026)](https://www.sciencedirect.com/science/article/pii/S0957417425034803) | 84.3 | -- | 271.7 |
| **RGB-T** | [DMFusion-YOLOv8 (2025)](https://doi.org/10.1007/s11760-025-05014-6) | 84.2 | 55.4 | 13.2 |
| **RGB-T** | [FreDFT (2025)](https://arxiv.org/abs/2511.10046) | 88.4 | 59.7 | -- |
| **RGB-T** | [MAFTNet (2026)](https://doi.org/10.1109/JSEN.2025.3649961) | 75.6 | 51.1 | 30.5 |
| **RGB-T** | [MCFFSQR (2025)](https://www.sciencedirect.com/science/article/pii/S1350449525005250) | 80.8 | 53.6 | 77.0 |
| **RGB-T** | [EFAF (2025)](https://doi.org/10.1109/JSTARS.2025.3648007) | 88.8 | 58.0 | 395.0 |
| **RGB-T** | [SCIA (2025)](https://www.sciencedirect.com/science/article/pii/S0893608025014145) | 88.6 | 59.9 | -- |
| **RGB-T** | [PFI-Net (2025)](https://www.sciencedirect.com/science/article/pii/S0031320325016668) | 82.9 | 53.2 | 14.9 |
| **RGB-T** | [RSC-MD (2025)](https://arxiv.org/abs/2511.15433) | 85.2 | 59.5 | -- |
| **RGB-T** | [SCVI (2026)](https://www.sciencedirect.com/science/article/pii/S0925231226000858) | 68.8 | 40.1 | -- |
| **RGB-T** | [WtCAFNet (2025)](https://www.sciencedirect.com/science/article/pii/S0165168425005626) | 85.2 | 58.2 | 59.9 |
| **RGB-T** | [DLRMamba (2026)](https://arxiv.org/abs/2603.06920) | 76.6 | -- | -- |
| **RGB-T** | [DWSF-Net (2026)](https://doi.org/10.1109/TMM.2026.3668686) | 86.9 | -- | 289.7 |
| **RGB-T** | [SAFF (2026)](https://doi.org/10.1109/TGRS.2026.3674467) | 87.4 | 60.1 | -- |
| **RGB-T** | [DDIF (2026)](https://doi.org/10.1109/TIP.2026.3671618) | 80.8 | 52.1 | 62.8 |
| **RGB-T** | [CHEF-Det (2026)](https://www.sciencedirect.com/science/article/pii/S1051200426002137) | 82.4 | 54.8 | -- |
| **RGB-T** | [IVFDNet (2026)](https://www.sciencedirect.com/science/article/pii/S0925231226010374) | <ins>88.9</ins> | -- | -- |
| **RGB-T** | [ACSE-Net (2026)](https://link.springer.com/article/10.1007/s44443-026-00732-4) | 86.3 | 59.4 | 55.0 |
| **RGB-T** | [SFCNet (2026)](https://www.researchgate.net/publication/404236009_Modality-Aware_Fusion_and_Selection_for_Robust_Multispectral_Pedestrian_Detection) | 80.4 | 53.5 | -- |
| **RGB-T** | [MDAFN (2026)](https://doi.org/10.1109/TVT.2026.3678921) | 87.0 | 54.6 | -- |
| **RGB-T** | [OARE (2026)](https://doi.org/10.1109/TMM.2026.3678036) | 88.4 | -- | -- |
| **RGB-T** | [DF-Net (2026)](https://ieeexplore.ieee.org/document/11514987) | 80.0 | 52.6 | -- |
| **RGB-T** | [WD-FQDet (2026)](https://arxiv.org/abs/2605.13621) | 73.7 | 46.4 | 60.7 |
| **RGB-T** | CFGPNet-m | 87.8 | <ins>60.3</ins> | 21.0 |
| **RGB-T** | CFGPNet-c | <ins>89.0</ins> | <ins>62.2</ins> | 71.7 |
| **RGB-T** | CFGPNet-e | <ins>89.9</ins> | <ins>63.4</ins> | 180.9 | 

<p align="center">
  <a href="docs/M3FD_map_param_diagram.png">
    <img src="docs/M3FD_map_param_diagram.png" alt="M3FD performance" width="60%">
  </a>
</p>

## Framework specifications

### CFGPNet model scales

**Input size:** 640×640

| **Model** | **#Param. (M)** | **GFLOPs (G)** | **FPS** | **Model Size (MB)** | **#Layers** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **CFGPNet-m** | 21.0 | 94.6 | 1463.4 | 41.6 | 1165 |
| **CFGPNet-c** | 71.7 | 362.2 | 386.1 | 138.8 | 1165 |
| **CFGPNet-e** | 180.9 | 560.7 | 136.9 | 349.6 | 1806 |

> **Notes:** #Param. and GFLOPs are reported for a single forward pass at `1×6×640×640`. FPS is measured by timing the forward pass only, excluding data loading and preprocessing.
