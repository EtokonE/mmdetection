import os
import cv2
import numpy as np

from mmdet.apis import init_detector, inference_detector
import mmcv
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Предварительная разметка видео. Результатом является csv файл с номером кадра и боксами на нем')
    parser.add_argument('config', help='Конфиг по которому обучалась модель')
    parser.add_argument('checkpoint', help='Файл с весами предобученной модели')
    parser.add_argument('workdir', help='Папка с видеофайлами')
    parser.add_argument('video', help='Название видео')
    parser.add_argument('outdir', help='Директория, куда сохраняем файл с результатами')
    parser.add_argument('--iou_thr', default=0.3, help='Порог, после которого бокс попадает в файл')
    args = parser.parse_args()
    return args

def layout_video(config, checkpoint, work_dir, video, outdir, iou_thr):

    csv_file = str(video) + '_layout.csv'
    model = init_detector(config, checkpoint, device='cuda:0')

    video_reader = mmcv.VideoReader(os.path.join(work_dir, video))

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    layout = []
    unident = np.empty(5)
    unident[:] = np.nan
    count = 0

    for i, frame in enumerate(mmcv.track_iter_progress(video_reader)):
        result = inference_detector(model, frame)
        for i in range(len(result[0])):
            if result[0][0][4] < iou_thr:
                layout.append(np.insert(unident, 0, i))
                break
            elif result[0][i][4] < iou_thr:
                break
            elif result[0][i][4] >= iou_thr:
                layout.append(np.insert(result[0][i], 0, i))
        if i % 500 == 0:
            print(i)
            layout_df = pd.DataFrame(layout, columns=['frame', 'x1', 'y1', 'x2', 'y2', 'score'])
            layout_df.to_csv(os.path.join(work_dir, csv_file))
            print('test run')
            break

    layout_df = pd.DataFrame(layout, columns=['frame', 'x1', 'y1', 'x2', 'y2', 'score'])
    layout_df.to_csv(os.path.join(work_dir, csv_file))

if __name__ == '__main__':
    print('start')
    args = parse_args()
    layout_video(config=args.config,
                 checkpoint=args.checkpoint,
                 work_dir=args.workdir,
                 video=args.video,
                 outdir=args.outdir,
                 iou_thr=args.iou_thr)