# Zero-OVCD Stage II configuration
#
# Construction rule:
# 1) Settings explicitly stated in the paper follow the paper.
# 2) Settings shared by the supplied original configs are retained.
# 3) Dataset-dependent crop size, OHEM min_kept, and validation/test resize scale
#    are specified separately for each dataset.
# 4) ignore_index is unified to 0.
# 5) Other inconsistent or unclear settings are intentionally omitted.

# Phase: warm-up training
# Paper-aligned settings:
#   max_iters = 20000
#   AdamW, lr = 0.001, weight_decay = 0.05
#   train batch_size = 8
#   validation interval = 500
#
# No fixed global seed is manually specified. The original configs consistently
# use DistSamplerSeedHook.


dataset_name = 'LEVIR-CD'  # 'LEVIR-CD' | 'WHU-CD' | 'S2Looking' | 'SECOND'

dataset_settings = {
    'LEVIR-CD': dict(
        dataset_type='LEVIR_CD_Dataset',
        crop_size=(512, 512),
        min_kept=100000,
        test_scale=(1024, 1024),
    ),
    'WHU-CD': dict(
        dataset_type='WHU_CD_Dataset',
        crop_size=(256, 256),
        min_kept=30000,
        test_scale=(256, 256),
    ),
    'S2Looking': dict(
        dataset_type='S2Looking_Dataset',
        crop_size=(512, 512),
        min_kept=100000,
        test_scale=(1024, 1024),
    ),
    'SECOND': dict(
        dataset_type='SECOND_Dataset',
        crop_size=(512, 512),
        min_kept=100000,
        test_scale=(512, 512),
    ),
}

if dataset_name not in dataset_settings:
    raise ValueError(
        f'Unsupported dataset_name={dataset_name!r}. '
        f'Choose from {list(dataset_settings.keys())}.'
    )

dataset_cfg = dataset_settings[dataset_name]
dataset_type = dataset_cfg['dataset_type']
crop_size = dataset_cfg['crop_size']
min_kept = dataset_cfg['min_kept']
test_scale = dataset_cfg['test_scale']

# Replace this placeholder with the prepared data directory for the current phase.
# For SECOND, point it to the prepared target-category directory.
data_root = '<PATH_TO_PREPARED_STAGE2_DATA>'

ignore_index = 0

data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675, 116.28, 103.53,
        123.675, 116.28, 103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    std=[
        58.395, 57.12, 57.375,
        58.395, 57.12, 57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='DualInputSegDataPreProcessor',
)

norm_cfg = dict(requires_grad=True, type='SyncBN')

model = dict(
    backbone=dict(
        contract_dilation=True,
        depth=18,
        dilations=(1, 1, 1, 1),
        interaction_cfg=(
            None,
            dict(p=0.5, type='SpatialExchange'),
            dict(p=0.5, type='ChannelExchange'),
            dict(p=0.5, type='ChannelExchange'),
        ),
        norm_cfg=norm_cfg,
        norm_eval=False,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 2, 2),
        style='pytorch',
        type='IA_ResNetV1c',
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        align_corners=False,
        channels=128,
        dropout_ratio=0.1,
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        loss_decode=[
            dict(
                ignore_index=ignore_index,
                loss_name='loss_sce',
                loss_weight=1.0,
                type='SymmetricCELoss',
            ),
            dict(
                classes='present',
                loss_name='loss_lovasz',
                loss_weight=1.0,
                per_image=False,
                reduction='none',
                type='LovaszLoss',
            ),
            dict(
                ignore_index=ignore_index,
                loss_name='loss_dice',
                loss_weight=1.0,
                type='DiceLoss',
            ),
        ],
        norm_cfg=norm_cfg,
        num_classes=2,
        sampler=dict(
            min_kept=min_kept,
            thresh=0.7,
            type='mmseg.OHEMPixelSampler',
        ),
        type='Changer',
    ),
    pretrained='checkpoint/resnet18_v1c.pth',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='DIEncoderDecoder',
)

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        type='CheckpointHook',
    ),
    logger=dict(
        interval=100,
        log_metric_by_epoch=False,
        type='LoggerHook',
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
)

default_scope = 'opencd'

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)

img_ratios = [0.75, 1.0, 1.25]
load_from = None
log_processor = dict(by_epoch=False)
resume = False

optim_wrapper = dict(
    optimizer=dict(
        betas=(0.9, 0.999),
        lr=0.001,
        type='AdamW',
        weight_decay=0.05,
    ),
    type='OptimWrapper',
)

optimizer = dict(
    betas=(0.9, 0.999),
    lr=0.001,
    type='AdamW',
    weight_decay=0.05,
)

param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=1000,
        start_factor=1e-6,
        type='LinearLR',
    ),
    dict(
        begin=1000,
        by_epoch=False,
        end=20000,
        eta_min=1e-6,
        power=1.0,
        type='PolyLR',
    ),
]

train_cfg = dict(
    max_iters=20000,
    type='AIO2loops_WHUCD',
    val_interval=500,
)

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(
        degree=(-20, 20),
        flip_prob=0.5,
        rotate_prob=0.5,
        type='MultiImgRandomRotFlip',
    ),
    dict(
        cat_max_ratio=0.75,
        crop_size=crop_size,
        type='MultiImgRandomCrop',
    ),
    dict(prob=0.5, type='MultiImgExchangeTime'),
    dict(
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        hue_delta=10,
        saturation_range=(0.8, 1.2),
        type='MultiImgPhotoMetricDistortion',
    ),
    dict(type='MultiImgPackSegInputs'),
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(
        keep_ratio=True,
        scale=test_scale,
        type='MultiImgResize',
    ),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path_from='train/A',
            img_path_to='train/B',
            seg_map_path='train/label',
        ),
        data_root=data_root,
        pipeline=train_pipeline,
        type=dataset_type,
    ),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'),
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='val/A',
            img_path_to='val/B',
            seg_map_path='val/label',
        ),
        data_root=data_root,
        pipeline=test_pipeline,
        type=dataset_type,
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
)

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='test/A',
            img_path_to='test/B',
            seg_map_path='test/label',
        ),
        data_root=data_root,
        pipeline=test_pipeline,
        type=dataset_type,
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
)

val_evaluator = dict(
    iou_metrics=['mFscore', 'mIoU'],
    type='mmseg.IoUMetric',
)

test_evaluator = dict(
    iou_metrics=['mFscore', 'mIoU'],
    type='mmseg.IoUMetric',
)

val_cfg = dict(type='ValLoop_AIO2_WHUCD')
test_cfg = dict(type='TestLoop')

tta_model = dict(type='mmseg.SegTTAModel')

tta_pipeline = [
    dict(backend_args=None, type='MultiImgLoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scale_factor=0.75,
                    type='MultiImgResize',
                ),
                dict(
                    keep_ratio=True,
                    scale_factor=1.0,
                    type='MultiImgResize',
                ),
                dict(
                    keep_ratio=True,
                    scale_factor=1.25,
                    type='MultiImgResize',
                ),
            ],
            [
                dict(
                    direction='horizontal',
                    prob=0.0,
                    type='MultiImgRandomFlip',
                ),
                dict(
                    direction='horizontal',
                    prob=1.0,
                    type='MultiImgRandomFlip',
                ),
            ],
            [dict(type='MultiImgLoadAnnotations')],
            [dict(type='MultiImgPackSegInputs')],
        ],
        type='TestTimeAug',
    ),
]
