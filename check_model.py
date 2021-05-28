from mmcv.cnn import VGG
import torch
model = VGG(depth=16)
print(model)
torch.save(model, 'model.pth')
