from define_anno import TRAIN_FILES, TEST_FILES, VAL_FILES, data_root


input_size = 512
model = dict(
    type='SingleStageDetector',  # The name of detector
    pretrained='open-mmlab://vgg16_caffe', # The ImageNet pretrained backbone to be loaded
    backbone=dict( # The config of backbone
        type='SSDVGG', # The type of the backbone, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py#L288 for more details.
        input_size=512,
        depth=16, 
        with_last_pool=False, 
        ceil_mode=True, #  when True, will use ceil instead of floor to compute the output shape (nn.AvgPool2d)
        out_indices=(3, 4), # The index of output feature maps produced in each stages
        out_feature_indices=(22, 34),
        l2_norm_scale=20), # Default 20
    neck=None,
    bbox_head=dict(   # Config of box head in the RoIHead.
        type='SSDHead', # Type of the bbox head, Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L177 for implementation details.
        in_channels=(512, 1024, 512, 256, 256, 256, 256), 
        num_classes=1, #  Number of categories excluding the background category.
        anchor_generator=dict( # Config dict for anchor generator !!!IMPORTANT!!!
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=512,
            basesize_ratio_range=(0.1, 0.9), #  Ratio range of anchors.
            strides=[2, 4, 8, 16, 64, 125, 512], # Strides of anchors in multiple feature levels. !!!CHANGE!!!
            ratios=[[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]), # !!!CHANGE!!! The list of ratios between the height and width of anchors in a single level.
        bbox_coder=dict(  # Box coder used in the second stage.
            type='DeltaXYWHBBoxCoder', # Type of box coder.
            target_means=[0.0, 0.0, 0.0, 0.0], # Means used to encode and decode box
            target_stds=[0.1, 0.1, 0.2, 0.2])),  # Standard variance for encoding and decoding. It is smaller since the boxes are more accurate. [0.1, 0.1, 0.2, 0.2] is a conventional setting.
    train_cfg=dict(  # Config of training hyperparameters for rpn and rcnn
        assigner=dict(  # Config of assigner
            type='MaxIoUAssigner',  # Type of assigner, MaxIoUAssigner is used for many common detectors. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10 for more details.
            pos_iou_thr=0.3, # change # IoU >= threshold 0.7 will be taken as positive samples
            neg_iou_thr=0.15, # change # IoU < threshold 0.3 will be taken as negative samples
            min_pos_iou=0.3, # The minimal IoU threshold to take boxes as positive samples
            ignore_iof_thr=-1, # IoF threshold for ignoring bboxes
            gt_max_assign_all=False),
        smoothl1_beta=1.0,
        allowed_border=-1, # The border allowed after padding for valid anchors.
        pos_weight=-1,  # The weight of positive samples during training.
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict( # Config for testing hyperparameters for rpn and rcnn
        nms=dict(type='nms', iou_threshold=0.7), # change #Type of nms # NMS threshold
        min_bbox_size=0, # The allowed minimal box size
        score_thr=0.02,  # Threshold to filter out boxes
        max_per_img=100))  # Max number of detections of each image
cudnn_benchmark = True
dataset_type = 'MyDataset'
#data_root = '/home/user/Documents/Kalinin/Data/full_data/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True) # Image normalization config to normalize the input images
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True), # First pipeline to load images from file path
    dict(type='LoadAnnotations', with_bbox=True),  # Second pipeline to load annotations for current image

    dict(type='Resize', img_scale=(512, 512), keep_ratio=True), # Augmentation pipeline that resize the images and their annotations
    dict(
        type='Normalize', 
        mean=[123.675, 116.28, 103.53],
        std=[1, 1, 1],
        to_rgb=True), # Augmentation pipeline that normalize the input images
    dict(type='RandomFlip', flip_ratio=0.1), # Augmentation pipeline that flip the images and their annotations (SMALL)
    dict(type='DefaultFormatBundle'), # Default format bundle to gather data in the pipeline
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])   # Pipeline that decides which keys in the data should be passed to the detector
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleF lipAug', # An encapsulation that encapsulates the testing augmentations
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[1, 1, 1],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=32, # Batch size of a single GPU
    workers_per_gpu=3, # Worker to pre-fetch data for each single GPU
    train=dict(
            type=dataset_type,
            ann_file= TRAIN_FILES, 
            img_prefix= data_root,
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[1, 1, 1],
                    to_rgb=True),
                dict(type='RandomFlip', flip_ratio=0.1),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
    val=dict(
        type=dataset_type,
        ann_file= VAL_FILES, 
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[1, 1, 1],
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
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='SGD', lr=0.02/8., momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1.0/1e10,
    step=[6, 12, 22])
runner = dict(type='EpochBasedRunner', max_epochs=15)
checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20200308-038c5591.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = './experiment/ssd/my_ssd_aug'


