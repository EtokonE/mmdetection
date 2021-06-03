import torch
model = torch.load('./saved_model.pt')
model.eval()
print(model)
