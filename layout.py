import os
import cv2
from mmdet.apis import init_detector, inference_detector
import mmcv


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
    config_file = 'experiment/my_ssd_full.py'
    checkpoint_file = 'experiment/epoch_9.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file)

    # test a single image and show the results
    img = 'experiment/ch02_20210325114122-0.png'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    # visualize the results in a new window
    model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file='experiment/result.jpg')
    print(result)
