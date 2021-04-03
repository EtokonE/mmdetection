_base_ = 'ssd512_coco.py'

dataset_type = 'MyDataset'
data_root = '/home/user/Documents/Kalinin/Data/full_data/'

data = dict(
        train = dict(dataset=dict(
                type = dataset_type,
                ann_file = data_root +  'ch02_20200605121548-part 00000.json',
                img_prefix = data_root +  'video_ch02_20200605121548',
                ),
            ),

        test = dict(
            type = dataset_type,
            ann_file = data_root + 'ch02_20200605114152-part 00000.json',
            img_prefix = data_root + 'video_ch02_20200605114152',
            ),

        val = dict(
            type = dataset_type,
            ann_file = data_root + 'ch01_20200605121410-part 00000.json',
            img_prefix = data_root + 'video_ch01_20200605121410',
            ),
        )

model = dict(
        bbox_head = dict(
            num_classes = 1,))

work_dir = './experiment'

optimizer = dict(lr = 0.02/8,)

lr_config = dict(warmup_ratio = 1.0 / 1e10)
log_config = dict(interval=1)
evaulation = dict(metric='mAP', interval=1,)
checkpoint_config = dict(interval=1)
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20200308-038c5591.pth'

