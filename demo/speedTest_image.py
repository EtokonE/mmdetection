import asyncio
from argparse import ArgumentParser
import timeit

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument('--speed_test', action='store_true', help='start speed test on one frame')
    args = parser.parse_args()
    return args

args = parse_args()
model = init_detector(args.config, args.checkpoint, device=args.device)
img = args.img

'''
def speed_test():
    args = parse_args()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    img = args.img
    #return inference_detector(model, img)
    #print(timeit.repeat('inference_detector(model, img)', 'from __main__ import inference_detector, model, img', repeat=20, number=1))
    print(timeit.repeat('inference_detector(model, img)', 'from __main__ import inference_detector', repeat=20, number=1))
'''
def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    #show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    model.show_result(args.img, result, score_thr=args.score_thr,font_size=5 , out_file='result_.jpg')


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    elif args.speed_test:
        print(timeit.repeat('inference_detector(model, img)', 'from __main__ import inference_detector, model, img', repeat=20, number=1)

    else:
        main(args)
