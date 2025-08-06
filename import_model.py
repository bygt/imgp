import torch

print("Model indiriliyor...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.eval()
print("Model başarıyla yüklendi.")