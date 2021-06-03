from mmdet.apis import init_detector
from mmdet.models import build_detector
from mmcv import Config
import torch
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('out_file', default='./saved_model.pt', help='path to out file')
    args = parser.parse_args()
    return args

    
def main(args):
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model)
    print(model)
    torch.save(model, args.out_file)    
    print('Model saved')


if __name__ == '__main__':
    args = parse_args()
    main(args)


