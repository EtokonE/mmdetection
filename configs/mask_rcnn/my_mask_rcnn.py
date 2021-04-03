#_base_ = 'faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
_base_ = 'faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'


dataset_type = 'MyDataset'
data_root = '/home/user/Documents/Kalinin/Data/full_data/'

data = dict(
        train = dict(
                type = dataset_type,
                ann_file = data_root +  'ch02_20200605121548-part 00000.json',
                img_prefix = data_root ,
                ),


        test = dict(
            type = dataset_type,
            ann_file = data_root + 'ch02_20200605114152-part 00000.json',
            img_prefix = data_root,
            ),

        val = dict(
            type = dataset_type,
            ann_file = data_root + 'ch01_20200605121410-part 00000.json',
            img_prefix = data_root
            ),
        )

model = dict(roi_head = dict( bbox_head=dict(num_classes=1) ) )

work_dir = './experiment/mask_rcnn'

optimizer = dict(lr = 0.02/8,)

lr_config = dict(warmup_ratio = 1.0 / 1e10)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
evaluation = dict(metric='mAP')

checkpoint_config = dict(interval=1)

#load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth'
