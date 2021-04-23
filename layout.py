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
    model = init_detector(config, checkpoint)

    vidcap = cv2.VideoCapture(os.path.join(work_dir, video))
    frame = 0
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    layout = []
    unident = np.empty(5)
    unident[:] = np.nan
    frame = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            result = inference_detector(model, image)
            for i in range(len(result[0])):
                if result[0][0][4] < iou_thr:
                    layout.append(np.insert(unident, 0, count))
                    break
                elif b[0][i][4] >= iou_thr:
                    layout.append(np.insert(result[0][i], 0, count))

    layout_df = pd.DataFrame(layout, columns=['frame', 'x1', 'y1', 'x2', 'y2', 'score'])
    layout_df.to_csv(os.path.join(work_dir, csv_file))


'''
def video_to_frames(work_dir, video, out_dir):
    vidcap = cv2.VideoCapture(os.path.join(work_dir, video))
    count = 0
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            filename = (video[:-4] + f'-{str(count)}.png')
            cv2.imwrite(os.path.join(out_dir, filename), image)
            count += 1
        else:
            break

        #cv2.destroyAllWindows()
        vidcap.release()
        print(success, image.shape)
        break

work_dir = '/home/max/Рабочий стол/Папка/Video/'
video = 'ch02_20210325114122.mp4'
out = '/home/max/Рабочий стол/Папка/Video/ch02_20210325114122'
count = 1
#video_to_frames(work_dir, video, out)
#print(video[:-4] + f'-{str(count)}')



def inference():
    config_file = 'experiment/my_ssd512_full.py'
    checkpoint_file = 'experiment/epoch_9.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file)
    # test a single image and show the results
    img = 'experiment/ch02_20210325114122-0.png'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    print('redult')
    # visualize the results in a new window
    model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file='experiment/result.jpg')
    print(result[0][0])
'''
if __name__ == '__main__':
    print('start')
    args = parse_args()
    layout_video(config=args.config,
                 checkpoint=args.checkpoint,
                 work_dir=args.workdir,
                 video=args.video,
                 outdir=args.outdir,
                 iou_thr=args.iou_thr)
