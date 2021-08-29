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
        '--speed_test', 
        action='store_true', 
        help='start speed test on one frame')

    args = parser.parse_args()
    return args

args = parse_args()
model = init_detector(args.config, args.checkpoint, device=args.device)
img = args.img

"""
def speed_test():
    args = parse_args()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    img = args.img
    #return inference_detector(model, img)
    #print(timeit.repeat('inference_detector(model, img)', 'from __main__ import inference_detector, model, img', repeat=20, number=1))
    print(timeit.repeat('inference_detector(model, img)', 'from __main__ import inference_detector', repeat=20, number=1))
"""
def main(args):
    print(timeit.repeat('inference_detector(model, img)', 
                        'from __main__ import inference_detector, model, img', 
                        repeat=20, 
                        number=1))


if __name__ == '__main__':
    args = parse_args()
    main(args)

