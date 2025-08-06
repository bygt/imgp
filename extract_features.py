
import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os

# Ayarlar
image_path = "exmp.jpg"                    # Girdi görseli
vector_dir = "vectors"                     # .npy dosyalarının saklanacağı klasör
vector_path = os.path.join(                # exmp.npy olarak kaydedecek
    vector_dir, os.path.splitext(os.path.basename(image_path))[0] + ".npy"
)

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

# Görseli aç
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Özellik çıkar
with torch.no_grad():
    features = model(input_tensor)

vector = features.squeeze().cpu().numpy()
print("Feature vector shape:", vector.shape)

# vectors klasörü yoksa oluştur
os.makedirs(vector_dir, exist_ok=True)

# .npy olarak kaydet
np.save(vector_path, vector)
print(f"Vektör kaydedildi: {vector_path}")
