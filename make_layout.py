import glob, os
import pandas as pd
import numpy as np

import mmcv
from mmdet.apis import init_detector, inference_detector

import argparse

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
    print(os.path.join(work_dir, video))
    video_reader = mmcv.VideoReader(os.path.join(work_dir, video))
    print(video_reader._frame_cnt)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    layout = []
    count = 0

    for frame in mmcv.track_iter_progress(video_reader):
        result = inference_detector(model, frame)
        for i in range(len(result[0])):
            if result[0][i][4] < iou_thr:
                break
            elif result[0][i][4] >= iou_thr:
                layout.append(np.insert(result[0][i][:4], 0, count))
        count += 1

    layout_df = pd.DataFrame(layout, columns=['frame', 'x', 'y', 'x2', 'y2'])
    layout_df['w'] = abs(layout_df['x2'] - layout_df['x'])
    layout_df['h'] = abs(layout_df['y2'] - layout_df['y'])
    layout_df['logs'] = np.nan
    layout_df = layout_df.drop(columns=['x2', 'y2'])
    layout_df = layout_df.astype({'frame' : 'int32', 'x' : 'int32', 'y' : 'int32', 'w' : 'int32', 'h' : 'int32'})
    layout_df.to_csv(os.path.join(work_dir, csv_file), index=False)

if __name__ == '__main__':
    args = parse_args()
    if args.video == '*.mp4':
        for some_video in os.listdir(args.workdir):
            if some_video.endswith(".mp4"):
        #for some_video in glob.glob(os.path.join(args.workdir, args.video)):
                print(f'Started to process video {some_video}')
                layout_video(config=args.config,
                            checkpoint=args.checkpoint,
                            work_dir=args.workdir,
                            video=some_video,
                            outdir=args.outdir,
                            iou_thr=args.iou_thr)

    else:       
        layout_video(config=args.config,
                     checkpoint=args.checkpoint,
                     work_dir=args.workdir,
                     video=args.video,
                     outdir=args.outdir,
                     iou_thr=args.iou_thr)
