import asyncio
from argparse import ArgumentParser
import timeit
from time import time

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('source', help='Image file or Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
                '--out', type=str, default='./speedTestVideoResult.mp4v', help='Output video file')    
    parser.add_argument(
        '--image_test', 
        action='store_true', 
        help='start speed test on single frame')
    parser.add_argument(
        '--video_test',
        action='store_true', 
        help='start speed test on video')
    args = parser.parse_args()
    return args


args = parse_args()
model = init_detector(args.config, args.checkpoint, device=args.device)
if args.image_test:
    img = args.source


def videoSpeedTest(video, out, model, score_thr):
    import mmcv
    import cv2
    
    video_reader = mmcv.VideoReader(video)

    # Закодировать mp4v -> four-character code 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    video_writer = cv2.VideoWriter(
                    out, fourcc, video_reader.fps, 
                    (video_reader.width, video_reader.height))
    
    inference_time = []
    resultProcessing_time = []
    writeVideo_time = []
    frameProcess_time = []

    # Работа с кадром
    for frame in mmcv.track_iter_progress(video_reader):
            # Все операции обработки кадра
            # ==============================================================
            start_frameProcess = time()
            # ==============================================================

            
            # Проход через нейросеть
            # ==============================================================
            start_inference = time()

            result = inference_detector(model, frame)

            end_inference = time()
            inference_time.append(end_inference - start_inference)
            # ==============================================================

                    
            # Интерпретация работы сети
            # ==============================================================
            start_resultProcessing = time()
            
            frame = model.show_result(frame, result, score_thr=score_thr)

            end_resultProcessing = time()
            resultProcessing_time.append(end_resultProcessing - start_resultProcessing)
            # ==============================================================

            
            # Запись кадра
            # ==============================================================
            start_writeVideo = time()
            
            video_writer.write(frame)

            end_writeVideo = time()
            writeVideo_time.append(end_writeVideo - start_writeVideo)
            # ==============================================================

            
            # Итого на один кадр
            # ==============================================================
            end_frameProcess = time()
            frameProcess_time.append(end_frameProcess - start_frameProcess)
            # ==============================================================

            
    video_writer.release()
    cv2.destroyAllWindows()

    print(f'Обработка кадра в среднем: {sum(frameProcess_time) / len(frameProcess_time)} sec' )
    print(f'Проход через нейросеть: {sum(inference_time) / len(inference_time)} sec' )
    print(f'Обработка результата: {sum(resultProcessing_time) / len(resultProcessing_time)} sec' )
    print(f'Обработка результата: {sum(writeVideo_time) / len(writeVideo_time)} sec' )


    
def main(args):
    if args.image_test:
        print('===============================================')
        print(timeit.repeat('inference_detector(model, img)', 
                            'from __main__ import inference_detector, model, img', 
                            repeat=20, 
                            number=1))
        print('===============================================')
    
    elif args.video_test:
        print('===============================================')
        videoSpeedTest(args.source, args.out, model, args.score_thr)
        print('===============================================')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

