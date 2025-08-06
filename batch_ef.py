import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os

# Klasör yolları
image_dir = "images"      # Görsellerin bulunduğu klasör
vector_dir = "vectors"    # .npy vektörlerin kaydedileceği klasör

# DINOv2 modelini yükle
print("Model yükleniyor...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.eval()
print("Model yüklendi.")

# Görsel işleme adımları
transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# vectors klasörü yoksa oluştur
os.makedirs(vector_dir, exist_ok=True)

# Görsel klasörünü tara
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

print(f"{len(image_files)} görsel bulundu.\n")

for filename in image_files:
    image_path = os.path.join(image_dir, filename)
    vector_path = os.path.join(vector_dir, os.path.splitext(filename)[0] + ".npy")

    # Zaten vektör varsa atla
    if os.path.exists(vector_path):
        print(f"Atlandı (zaten var): {filename}")
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            features = model(input_tensor)

        vector = features.squeeze().cpu().numpy()
        np.save(vector_path, vector)
        print(f"✓ Kaydedildi: {filename} -> {vector_path}")
    except Exception as e:
        print(f"Hata ({filename}):", e)
