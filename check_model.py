from mmdet.apis import init_detector
from mmdet.models import build_detector
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
    #model = init_detector(args.config, args.checkpoint, device=args.device)
    model = build_detector(args.config, test_cfg=args.config.get('test_cfg'))
    print(model)
    torch.save(model.state_dict(), './saved_model_state.pt')    
    print('Model saved')


if __name__ == '__main__':
    args = parse_args()
    main(args)


