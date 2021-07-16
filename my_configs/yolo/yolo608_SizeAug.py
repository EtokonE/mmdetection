# Изменены размеры якорных боксов и убраны лишние аугментации

from define_anno import TRAIN_FILES, TEST_FILES, VAL_FILES, data_root

print(f'TRAIN FILES: {TRAIN_FILES}')
print(f'VAL FILES: {VAL_FILES}')
print(f'TEST FILES: {TEST_FILES}')
num_classes = 1
dataset_type = 'MyDataset'
checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'
resume_from = None #'/home/dron_maks/mmdetection/experiment/yolo/yolo608_base/epoch_5.pth'
workflow = [('train', 1)]
work_dir = './experiment/yolo/yolo608_SizeAug'
model = dict(
    type='YOLOV3',
    pretrained='open-mmlab://darknet53',
    backbone=dict(type='Darknet', depth=53, out_indices=(3, 4, 5)),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=num_classes,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(80, 90), (156, 198), (200, 200)],
                        [(20, 51), (45, 20), (50, 79)],
                        [(6, 8), (10, 10), (16, 16)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.3),
        max_per_img=100))
dataset_type = dataset_type
data_root = data_root
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='Expand', mean=[0, 0, 0], to_rgb=True, ratio_range=(1, 2)),
    dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(
        type='Normalize',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        #flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=11,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=TRAIN_FILES,
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Expand', mean=[0, 0, 0], to_rgb=True,
                ratio_range=(1, 2)),
            dict(
                type='Resize',
                img_scale=[(320, 320), (608, 608)],
                keep_ratio=True),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type=dataset_type,
        ann_file=VAL_FILES,
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(608, 608),
                #flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type=dataset_type,
        ann_file=TEST_FILES,
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(608, 608),
                #flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        samples_per_gpu=56,))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.01,
    step=[5, 7])
runner = dict(type='EpochBasedRunner', max_epochs=7)
evaluation = dict(interval=1, metric='mAP')

