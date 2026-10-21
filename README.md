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

> **Code:** The full source code will be made available upon paper acceptance. Stage II reproducibility configurations and input/output conventions are provided below.

## Datasets

We evaluate Zero-OVCD on four representative remote sensing change detection benchmarks. Please download the datasets from the following links:

| **Dataset** | **Evaluation Setting** | **Download** |
|:-----------:|:----------------------:|:------------:|
| LEVIR-CD | Building Change Detection | [Link](https://drive.google.com/file/d/1yRAjetmKfWCImCRV-97wzgA8ym_1g1hV/view?usp=drive_link) |
| WHU-CD | Building Change Detection | [Link](https://drive.google.com/file/d/1SnGTGmZO72bAa4O4cBjGPjUbXASp2ZnW/view?usp=drive_link) |
| S2Looking | Building Change Detection | [Link](https://huggingface.co/datasets/PENGYUANKE/S2Looking/resolve/main/S2Looking.zip) |
| SECOND | Category-wise OVCD (6 classes) | [Link](https://drive.google.com/file/d/1w-4HQdKqaD0fpWboeVt2WSEM49pcPldc/view?usp=drive_link) |

## Frozen Vision Foundation Models

Zero-OVCD employs three frozen vision foundation models in Stage I. Note that **SegEarth-OV3 shares the same `sam3.pt` checkpoint with SAM3**, so only two unique pretrained checkpoints are required.

| **Model** | **Backbone / Variant** | **Checkpoint Used in Zero-OVCD** | **Official Model Page** |
|:---------:|:----------------------:|:--------------------------------:|:-----------------------:|
| SAM3 | PE-L+/14 | `sam3.pt` | [Link](https://huggingface.co/facebook/sam3/tree/main) |
| DINOv3 | ViT-L/16 (SAT-493M) | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | [Link](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) |
| SegEarth-OV3 | SAM3-based PE-L+/14 | `sam3.pt` (shared with SAM3) | Same as SAM3 |

## Stage II Reproducibility Configuration

The configuration files for the two training phases of Stage II are provided below:

- **Warm-up Training:** [`configs/stage2/warmup_training/changerex_warmup.py`](configs/stage2/warmup_training/changerex_warmup.py)
- **High-Agreement Sample Training:** [`configs/stage2/high_agreement_sample_training/changerex_high_agreement.py`](configs/stage2/high_agreement_sample_training/changerex_high_agreement.py)

### Initialization

ChangerEx uses a ResNet-18 V1c backbone initialized from the pretrained `resnet18_v1c.pth` weights. Both Stage II training phases use the same initialization setting. The high-agreement sample training phase independently reinitializes ChangerEx and does not load any checkpoint from the warm-up training phase.

The configuration files expect the pretrained backbone at:

```text
checkpoint/resnet18_v1c.pth
```

### Random Seed

No fixed global random seed is manually specified in the released configurations. Sampler seed handling uses `DistSamplerSeedHook`.

### Data Augmentation

The same training-time augmentation settings are used in both Stage II phases:

- random rotation within `[-20°, 20°]` with `rotate_prob=0.5`;
- random flipping with `flip_prob=0.5`;
- random cropping with dataset-specific crop sizes;
- bitemporal image-order exchange with probability `0.5`;
- photometric distortion with `brightness_delta=10`, `contrast_range=(0.8, 1.2)`, `saturation_range=(0.8, 1.2)`, and `hue_delta=10`.

Dataset-specific crop sizes, OHEM `min_kept` values, and validation/test resize scales are specified directly in the corresponding configuration files.

### Training Settings

Warm-up training is performed for 20,000 iterations. ChangerEx is then independently reinitialized and trained for 60,000 iterations in the high-agreement sample training phase. Both phases use AdamW with a learning rate of `0.001`, weight decay of `0.05`, polynomial learning-rate decay, a training batch size of `8`, and validation every `500` iterations.

### Input/Output Convention

The Stage II configuration files use the following data organization:

```text
<DATA_ROOT>/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

`A/` and `B/` contain spatially aligned bitemporal images, while `label/` contains the corresponding pseudo-labels with matching filenames.

For **Warm-up Training**, the original Stage I pseudo-labels are used as supervision.

For **High-Agreement Sample Training**, the selected high-agreement samples and their updated pseudo-labels are used for training, while the corresponding high-agreement validation subset is used for pseudo-label-based validation.
