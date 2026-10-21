<h1 align="center">Zero-OVCD</h1>

<h3 align="center">
Bridging Training-Free Foundation Models and Pseudo-Label Learning for Open-Vocabulary Change Detection
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2608.11663">arXiv</a>
</p>

## Overview

**Zero-OVCD** is a two-stage framework for open-vocabulary change detection (OVCD) in remote sensing imagery that requires **no manual annotations throughout the complete pipeline**. It bridges training-free vision foundation model inference with noise-aware pseudo-label learning, enabling category-specific change detection from natural-language prompts.

Existing training-free OVCD pipelines are susceptible to errors caused by inaccurate candidate masks, ambiguous semantic assignments, and error accumulation across sequential inference stages. Zero-OVCD addresses these limitations by progressively refining the intermediate predictions rather than directly treating staged inference results as final change masks.

![Overview of the Zero-OVCD framework](assets/Question.png)

## Framework Overview

Zero-OVCD consists of two stages: **training-free change pseudo-label generation** and **noise-aware pseudo-label learning**.

In **Stage I**, frozen vision foundation models are used to generate high-quality category-specific change pseudo-labels. The pipeline progressively reduces different types of inference errors through three complementary modules:

- **Mask Refinement Module (MRM):** refines multi-source candidate masks by exploiting the complementarity between automatic and text-guided semantic masks.
- **Similarity-Based Multiscale Fusion Module (SMFM):** performs multiscale semantic verification and background-aware margin filtering to improve the reliability of category assignments.
- **Mask Correction and Completion Module (MCCM):** removes residual false-positive proposals and recovers potentially missed change regions using cross-temporal target-category responses.

In **Stage II**, the Stage I pseudo-labels are used to train a category-specific change detector. A two-phase noise-aware learning strategy performs checkpoint-based pseudo-label updating and high-agreement sample selection to mitigate residual pseudo-label noise. Final predictions are obtained by voting over selected checkpoint predictions together with the corresponding Stage I pseudo-labels.

![Overview of the Zero-OVCD framework](assets/Zero-OVCD.png)

> **Code:** Currently, only the Stage II training configurations are provided. The full source code will be made available upon paper acceptance.

## Datasets

We evaluate Zero-OVCD on four representative remote sensing change detection benchmarks. Please download the datasets from the following links:

| **Dataset** | **Evaluation Setting** | **Download** |
|:-----------:|:----------------------:|:------------:|
| LEVIR-CD | Building Change Detection | [Link](https://drive.google.com/file/d/1yRAjetmKfWCImCRV-97wzgA8ym_1g1hV/view?usp=drive_link) |
| WHU-CD | Building Change Detection | [Link](https://drive.google.com/file/d/1SnGTGmZO72bAa4O4cBjGPjUbXASp2ZnW/view?usp=drive_link) |
| S2Looking | Building Change Detection | [Link](https://huggingface.co/datasets/PENGYUANKE/S2Looking/resolve/main/S2Looking.zip) |
| SECOND | Category-wise OVCD (6 classes) | [Link](https://drive.google.com/file/d/1w-4HQdKqaD0fpWboeVt2WSEM49pcPldc/view?usp=drive_link) |

### Usage Convention

The released datasets contain bitemporal images and ground-truth change annotations. The ground-truth annotations are used only for quantitative evaluation and are not used as supervision or inference inputs in Zero-OVCD.

In **Stage I**, the spatially aligned bitemporal images together with the target-category text prompts are used to generate category-specific change pseudo-labels.

In **Stage II**, the bitemporal images and the pseudo-labels generated in Stage I are used for noise-aware pseudo-label learning, and the final output is the predicted category-specific change mask.

## Frozen Vision Foundation Models

Zero-OVCD employs three frozen vision foundation models in Stage I. Note that **SegEarth-OV3 shares the same `sam3.pt` checkpoint with SAM3**, so only two unique pretrained checkpoints are required.

| **Model** | **Backbone / Variant** | **Checkpoint Used in Zero-OVCD** | **Official Model Page** |
|:---------:|:----------------------:|:--------------------------------:|:-----------------------:|
| SAM3 | PE-L+/14 | `sam3.pt` | [Link](https://huggingface.co/facebook/sam3/tree/main) |
| DINOv3 | ViT-L/16 (SAT-493M) | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | [Link](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) |
| SegEarth-OV3 | SAM3-based PE-L+/14 | `sam3.pt` (shared with SAM3) | Same as SAM3 |

## Stage II Reproducibility Configuration

Detailed configurations for the two Stage II training phases are provided in:

- **Warm-up Training:** [`configs/stage2/warmup_training/changerex_warmup.py`](configs/stage2/warmup_training/changerex_warmup.py)
- **High-Agreement Sample Training:** [`configs/stage2/high_agreement_sample_training/changerex_high_agreement.py`](configs/stage2/high_agreement_sample_training/changerex_high_agreement.py)

### Initialization

ChangerEx uses a ResNet-18 V1c backbone initialized from the pretrained `resnet18_v1c.pth` weights. Both Stage II training phases use the same initialization setting, while the high-agreement sample training phase is independently reinitialized without loading a warm-up checkpoint.

The pretrained ResNet-18 V1c weights (`resnet18_v1c.pth`, 44.7 MB) can be downloaded from [Google Drive](https://drive.google.com/file/d/16aVDA0B1CBx9Pk8tzBA8b--RzL5VVGpR/view?usp=drive_link) and should be placed at:

```text
checkpoint/resnet18_v1c.pth
