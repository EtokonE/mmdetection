from mmdet.apis import init_detector
from mmdet.models import build_detector
from mmcv import Config
import torch
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args

    
def main(args):
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg) 
    print(model)
    torch.save(model.state_dict(), './saved_model_state.pt')    
    print('Model saved')


if __name__ == '__main__':
    args = parse_args()
    main(args)


