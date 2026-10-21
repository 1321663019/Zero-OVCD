# Copyright (c) Zero-OVCD authors.
# Unified Stage II reproducibility configuration for Zero-OVCD.
#
# Usage:
#   1. Set `dataset_name` to one of:
#      'LEVIR-CD', 'WHU-CD', 'S2Looking', or 'SECOND'.
#   2. Adjust `data_root` in `dataset_settings` if your local dataset path differs.
#   3. Place the Stage II backbone initialization checkpoint at:
#      checkpoints/resnet18_v1c.pth
#
# Reproducibility notes:
#   - The paper-default training batch size is 8.
#   - The paper-default learning rate is 0.001.
#   - `ignore_index` is unified to 0 in this reproducibility config.
#   - OHEM `min_kept` is 30,000 for WHU-CD and 100,000 for the other datasets.
#     WHU-CD uses 256x256 training patches, whereas the other datasets use
#     512x512 training crops.
#   - No fixed global random seed is specified here. MMEngine generates the
#     runtime seed at launch and records it in the corresponding training log.
#   - For exact reproduction of the reported experiments, keep batch_size=8.
#     It may be adjusted when GPU memory is insufficient, but such a run is no
#     longer an exact reproduction of the paper setting.

# -----------------------------------------------------------------------------
# Dataset selection
# -----------------------------------------------------------------------------

dataset_name = 'LEVIR-CD'  # 'LEVIR-CD' | 'WHU-CD' | 'S2Looking' | 'SECOND'

dataset_settings = {
    'LEVIR-CD': dict(
        dataset_type='LEVIR_CD_Dataset',
        data_root='data/LEVIR-CD',
        crop_size=(512, 512),
        test_size=(1024, 1024),
        min_kept=100000,
        save_best='F1_Prec_mean',
    ),
    'WHU-CD': dict(
        dataset_type='WHU_CD_Dataset',
        data_root='data/WHU-CD',
        crop_size=(256, 256),
        test_size=(256, 256),
        min_kept=30000,
        save_best='mFscore',
    ),
    'S2Looking': dict(
        dataset_type='S2Looking_Dataset',
        data_root='data/S2Looking',
        crop_size=(512, 512),
        test_size=(1024, 1024),
        min_kept=100000,
        save_best='F1_Prec_mean',
    ),
    'SECOND': dict(
        dataset_type='SECOND_Dataset',
        # The supplied SECOND configuration is category-wise (water shown here).
        # Replace "water" with the prepared target-category folder when needed.
        data_root='data/SECOND/water',
        crop_size=(512, 512),
        test_size=(512, 512),
        min_kept=100000,
        save_best='F1_Prec_mean',
    ),
}

if dataset_name not in dataset_settings:
    raise ValueError(
        f'Unsupported dataset_name={dataset_name!r}. '
        f'Choose from {list(dataset_settings.keys())}.'
    )

dataset_cfg = dataset_settings[dataset_name]
dataset_type = dataset_cfg['dataset_type']
data_root = dataset_cfg['data_root']
crop_size = dataset_cfg['crop_size']
test_size = dataset_cfg['test_size']
min_kept = dataset_cfg['min_kept']
save_best = dataset_cfg['save_best']

# -----------------------------------------------------------------------------
# Common Stage II settings reported in the paper
# -----------------------------------------------------------------------------

train_batch_size = 8
learning_rate = 0.001
ignore_index = 0

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[
        123.675, 116.28, 103.53,
        123.675, 116.28, 103.53,
    ],
    std=[
        58.395, 57.12, 57.375,
        58.395, 57.12, 57.375,
    ],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    test_cfg=dict(size_divisor=32),
)

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------

default_scope = 'opencd'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=100,
        log_metric_by_epoch=False,
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_best=save_best,
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
launcher = 'none'

# -----------------------------------------------------------------------------
# ChangerEx model
# -----------------------------------------------------------------------------

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='DIEncoderDecoder',
    pretrained='checkpoints/resnet18_v1c.pth',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='IA_ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        interaction_cfg=(
            None,
            dict(type='SpatialExchange', p=0.5),
            dict(type='ChannelExchange', p=0.5),
            dict(type='ChannelExchange', p=0.5),
        ),
    ),
    decode_head=dict(
        type='Changer',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        sampler=dict(
            type='mmseg.OHEMPixelSampler',
            thresh=0.7,
            min_kept=min_kept,
        ),
        loss_decode=[
            dict(
                type='SymmetricCELoss',
                ignore_index=ignore_index,
                loss_name='loss_sce',
                loss_weight=1.0,
            ),
            dict(
                type='LovaszLoss',
                loss_name='loss_lovasz',
                loss_weight=1.0,
                classes='present',
                per_image=False,
                reduction='none',
            ),
            dict(
                type='DiceLoss',
                ignore_index=ignore_index,
                loss_name='loss_dice',
                loss_weight=1.0,
            ),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# -----------------------------------------------------------------------------
# Optimizer and learning-rate schedule
# -----------------------------------------------------------------------------

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
)

# Kept for compatibility with project code that reads `optimizer` directly.
optimizer = dict(
    type='AdamW',
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)

param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=1000,
        by_epoch=False,
        start_factor=1e-6,
    ),
    dict(
        type='PolyLR',
        begin=1000,
        end=60000,
        by_epoch=False,
        power=1.0,
        eta_min=1e-6,
    ),
]

# The project uses the custom AIO2loops_WHUCD training loop for Stage II.
# The paper reports 20,000 warm-up iterations followed by independent
# reinitialization and 60,000 high-agreement-sample training iterations.
train_cfg = dict(
    type='AIO2loops_WHUCD',
    max_iters=60000,
    val_interval=500,
)

val_cfg = dict(type='ValLoop_AIO2_WHUCD')
test_cfg = dict(type='TestLoop')

# -----------------------------------------------------------------------------
# Data pipelines
# -----------------------------------------------------------------------------

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(
        type='MultiImgRandomRotFlip',
        degree=(-20, 20),
        flip_prob=0.5,
        rotate_prob=0.5,
    ),
    dict(
        type='MultiImgRandomCrop',
        crop_size=crop_size,
        cat_max_ratio=0.75,
    ),
    dict(
        type='MultiImgExchangeTime',
        prob=0.5,
    ),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10,
    ),
    dict(type='MultiImgPackSegInputs'),
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(
        type='MultiImgResize',
        scale=test_size,
        keep_ratio=True,
    ),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]

# -----------------------------------------------------------------------------
# Dataloaders
#
# Expected Stage II directory convention:
#   <data_root>/
#       train/A/
#       train/B/
#       train/label/
#       val/A/
#       val/B/
#       val/label/
#       test/A/
#       test/B/
#       test/label/
#
# A and B are the two temporal images. "label" contains the corresponding
# Stage I pseudo-label masks used by Stage II.
# -----------------------------------------------------------------------------

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='train/A',
            img_path_to='train/B',
            seg_map_path='train/label',
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='val/A',
            img_path_to='val/B',
            seg_map_path='val/label',
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='test/A',
            img_path_to='test/B',
            seg_map_path='test/label',
        ),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type='mmseg.IoUMetric',
    iou_metrics=['mFscore', 'mIoU'],
)

test_evaluator = dict(
    type='mmseg.IoUMetric',
    iou_metrics=['mFscore', 'mIoU'],
)

# -----------------------------------------------------------------------------
# Test-time augmentation
# -----------------------------------------------------------------------------

img_ratios = [0.75, 1.0, 1.25]

tta_model = dict(type='mmseg.SegTTAModel')

tta_pipeline = [
    dict(type='MultiImgLoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='MultiImgResize', scale_factor=0.75, keep_ratio=True),
                dict(type='MultiImgResize', scale_factor=1.0, keep_ratio=True),
                dict(type='MultiImgResize', scale_factor=1.25, keep_ratio=True),
            ],
            [
                dict(
                    type='MultiImgRandomFlip',
                    prob=0.0,
                    direction='horizontal',
                ),
                dict(
                    type='MultiImgRandomFlip',
                    prob=1.0,
                    direction='horizontal',
                ),
            ],
            [dict(type='MultiImgLoadAnnotations')],
            [dict(type='MultiImgPackSegInputs')],
        ],
    ),
]

# -----------------------------------------------------------------------------
# Visualization / work directory
# -----------------------------------------------------------------------------

vis_backends = [dict(type='CDLocalVisBackend')]

visualizer = dict(
    type='CDLocalVisualizer',
    name='visualizer',
    alpha=1.0,
    vis_backends=vis_backends,
)

safe_dataset_name = dataset_name.lower().replace('-', '_')
work_dir = f'work_dirs/stage2/{safe_dataset_name}'
