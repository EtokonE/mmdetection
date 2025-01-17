# Загружаем предобученные веса классификатора


#import sys
#sys.path.insert(1, '/home/user/server/home/d.grushevskaya1/projects/dron_maks/mmdetection')
from define_anno import TRAIN_FILES, TEST_FILES, VAL_FILES, data_root

print(f'TRAIN FILES: {TRAIN_FILES}')
print(f'VAL FILES: {VAL_FILES}')
print(f'TEST FILES: {TEST_FILES}')
# --------------------------

# Размер входного изображения
input_size = 512
# ----------------------------
model = dict(
    # ---------------------------------------------------

    # Название детектора
    type='SingleStageDetector',
    # ---------------------------------------------------

    # Откуда будем загружать предобученные веса
    #pretrained='./experiment/weights_yolo_dasha/checkpoint.pth',
    pretrained='./model.pth',
    # ---------------------------------------------------

    # Определяем конфигурацию классификатора ->
    # https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/models/backbones/ssd_vgg.html
    backbone=dict(
        # ---------------------------------------------------

        # Тип сети
        type='SSDVGG',
        # ---------------------------------------------------

        # Размер входных данных
        input_size=512,
        # ---------------------------------------------------

        # Глубина сети
        depth=16,
        # ---------------------------------------------------

        # Убирает пулинг в последнем слое VGG перед дополнительными слоями SSD
        # https://mmcv.readthedocs.io/en/stable/_modules/mmcv/cnn/vgg.html
        with_last_pool=False,
        # ---------------------------------------------------

        # Используется в nn.MaxPool2d.
        # В случае режима ceil дополнительные столбцы и строки добавляются как справа, так и снизу.
        # (Не сверху и не слева).
        # Это не обязательно должна быть одна дополнительная колонка. Это также зависит от величины шага.
        ceil_mode=True,
        # ---------------------------------------------------

        # VGG имеет 5 ступеней до FC слоёв. => Берем выходы из двух последних слоев
        out_indices=(3, 4),
        # ---------------------------------------------------

        # Проверить, но : В реализации индексируются -> Свертка, Активация, Пулинг =>
        # Мы берем карты активации из VGG в соответствии с этими индексами
        out_feature_indices=(22, 34),
        # ---------------------------------------------------

        # Используется в слое нормализации, отвечающем за предсказание с 'half' точностью. FP32->FP16
        l2_norm_scale=20),
    # ---------------------------------------------------

    # Шея
    neck=None,
    # ---------------------------------------------------

    # Конфигурация головы
    bbox_head=dict(
        type='SSDHead',
        # ---------------------------------------------------

        # Глубина карт активации, используемых для предсказаний боксов
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        # ---------------------------------------------------

        # Количество классов в датасете. Без учета background
        num_classes=1,
        # ---------------------------------------------------

        # Конфигурация дефолт-боксов
        anchor_generator=dict(
            type='SSDAnchorGenerator',


            # Следует ли сначала умножать масштабы при генерации
            # дефолт боксов. Если True, боксы в той же строке будут иметь
            # одинаковые масштабы. По умолчанию True в версии 2.0.
            scale_major=False,
            # ---------------------------------------------------

            # Размер входного тензора
            input_size=512,
            # ---------------------------------------------------

            # Диапазон соотношения дефолт-боксов ???
            # Для SSD_512 должен начинаться с 0.1
            basesize_ratio_range=(0.1, 0.9),
            # ---------------------------------------------------

            # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py
            # Шаги дефолт-боксов на нескольких уровнях функций по порядку (w, h)
            # По умолчанию base_sizes дефолт-боксов установлен в None =>
            # В этом случае за основные размеры дефолт-боксов отвечают как раз strides
            # (Если strides не квадратные, берется самый короткий шаг.)
            strides=[8, 16, 32, 64, 128, 256, 512],
            # ---------------------------------------------------

            # Соотношение между высотой и шириной. Логично (0.25, 0.5, 1, 2, 3)
            # 1 - квадрат, <1 - ширина больше высоты, > 1 - наоборот
            # Для нас разумно 0,5 и 1
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]),
        # ---------------------------------------------------

        # Кодирует bbox (x1, y1, x2, y2) в дельту (dx, dy, dw, dh) и
        # Декодирует дельта (dx, dy, dw, dh) обратно в исходный bbox (x1 , y1, x2, y2).
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    # ---------------------------------------------------

    # Конфигурация гиперпараметров обучения
    train_cfg=dict(
        # ---------------------------------------------------


        assigner=dict(
            # Этот метод соотносит каждый *Истинный* бокс с каждым *дефолт-боксом*
            # И присваивает: *(-1)* // *либо ??полуположительное число??*
            # -1 означает отрицательный sample
            # Полуположительное число - это индекс (отсчитываемый от 0) назначенного gt.

            # Назначение выполняется в следующих шагах, порядок имеет значение:

            # 1. Присвоить всем боксам background
            # 2. Присвоить 0 всем дефолт-боксам IOU которых < neg_iou_thr
            # 3. Для каждого *бокса*, если его IOU с ближайшим истинным боксом >= pos_iou_thr
            #    соотнести его с этим боксом
            # 4. Каждый истинный бокс соотнести с ближайшим к себе дефолт-боксом(мб > 1)
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,

            # ---------------------------------------------------
            # Минимальный iou для того, чтобы bbox считался положительным.
            # Положительные образцы могут иметь меньший IoU, чем pos_iou_thr из-за 4-го шага
            min_pos_iou=0.0,
            # ---------------------------------------------------

            # Порог IoF для игнорирования bbox-ов (если указан `gt_bboxes_ignore`).
            # Отрицательные значения не означают игнорирование любых bboxes
            ignore_iof_thr=-1,
            # ---------------------------------------------------

            # Слудует ли назначать всем боксам с наибольшим перекрытием с некоторым истинным боксом именно к этому боксу
            # Имеется в виду что на шаге 3 может может возникнуть следующая ситуация:
            # bbox_A имеет iou = 0.9 и 0.8 с истинными боксами b_1 и b_2 соответственно =>
            # => b_1 будет определен как наилучший для bbox_A. Однако если b_2 имеет gt_argmax_overlaps = A =>
            # => (максимальное перекрытие именно с этим боксом), то assign_gt_inds бокса А будет перезаписан на b_2??
            # (However, if GT bbox 2's gt_argmax_overlaps=A, bbox A's assigned_gt_inds will be overwritten to be bbox B)
            gt_max_assign_all=False),
        # ---------------------------------------------------

        # Порог кусочной ф-ции L1 => 0.5x^2 if |x| < 1.0
        smoothl1_beta=1.0,
        # ---------------------------------------------------

        # Максимальная дистанция, на которую дефолт-бокс может вылезать за границу изображения
        # Устанавливается положительным числом
        allowed_border=-1,
        # ---------------------------------------------------

        # Вес положительных образцов во время обучения.???
        # Если здесь <= 0, то pos_weight = 1.0
        pos_weight=-1,
        # ---------------------------------------------------

        # Предпологаю что это N_hard_negatives
        # Negative - все не сматченные по порогу с истинными боксы, их в разы больше =>
        # Нам нужно определить в каком соотношении будем добавлять их в лосс
        # https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/models/dense_heads/ssd_head.html
        neg_pos_ratio=3,
        # ---------------------------------------------------

        debug=False),
    # ---------------------------------------------------
    test_cfg=dict(

        # Нам необходимо вычислить iou между всеми предсказанными боксами. Для этого берем все у которых взаимное
        # Перекрытие не меньше указанного порога и оставляем самый уверенный по скору класса (Можно увеличить)
        nms=dict(type='nms', iou_threshold=0.3),
        # ---------------------------------------------------

        min_bbox_size=0,
        score_thr=0.02,  # Минимальный порог для визуализации боксов
        max_per_img=200))

cudnn_benchmark = True  # True для алгоритмов с фиксированным размером входных данных
dataset_type = 'MyDataset'  # Тип датасета
# ---------------------------------------------------

# Конфигурация для нормализации входных изображений
# Средние значения, используемые для предварительно обученных моделей backbone
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
# ---------------------------------------------------


# https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/datasets/pipelines/transforms.html
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),  # Загрузка изображений из пути к файлу
    dict(type='LoadAnnotations', with_bbox=True),  # Загрузка аннотаций к текущему изображению
    dict(
        # Фотометрическое искажение пременяется к изображению последовательно с вероятность 0,5
        # 1.случайная яркость
        # 2.случайный контраст (режим 0)
        # 3.преобразовать цвет из BGR в HSV
        # 4.случайная насыщенность
        # 5.случайный оттенок
        # 6.преобразовать цвет из HSV в BGR
        # 7.случайный контраст (режим 1)
        # 8.произвольно поменять местами каналы
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
       type='Expand',  # Рандомный поворот
       mean=[123.675, 116.28, 103.53],
       to_rgb=True,
       ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',  # Произвольная обрезка изображения и боксов и минимальныи IoU
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),  # Изменяем размер входного изображения
    dict(
        type='Normalize',  # Нормализация
        mean=[123.675, 116.28, 103.53],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='RandomFlip', flip_ratio=0.5),  # Поворот
    dict(type='DefaultFormatBundle'),  # Пакет формата по умолчанию для сбора данных в конвейере
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])  # Собираем в пайплайн необходимые ключи
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',  # Инкапсуляция тестовых аугментаций
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
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
    samples_per_gpu=32,  # Batch size of a single GPU
    workers_per_gpu=3,  # Worker to pre-fetch data for each single GPU
    train=dict(
            type=dataset_type,
            ann_file=TRAIN_FILES,
            img_prefix=data_root,
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                dict(
                    type='Expand',
                    mean=[123.675, 116.28, 103.53],
                    to_rgb=True,
                    ratio_range=(1, 4)),
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                    min_crop_size=0.3),
                dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[1, 1, 1],
                    to_rgb=True),
                dict(type='RandomFlip', flip_ratio=0.5),
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
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
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
        ann_file=str(TEST_FILES[0]), 
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
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
    step=[])
runner = dict(type='EpochBasedRunner', max_epochs=2)
checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20200308-038c5591.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = './experiment/ssd/weights_yolo_dasha'
