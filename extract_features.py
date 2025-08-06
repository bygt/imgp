import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

# Klasör yolları
TEMP_LIB_DIR = "temp_lib"   # Yeni eklenecek klasör (library gibi)
IMAGES_DIR = "images"
VECTOR_DIR = "vectors"

# Model ve transform (mevcut batch_extract_features.py’den aynısı)
print("Model yükleniyor...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.eval()
print("Model yüklendi.")

transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def extract_and_move(filename):
    temp_path = os.path.join(TEMP_LIB_DIR, filename)
    vector_path = os.path.join(VECTOR_DIR, os.path.splitext(filename)[0] + ".npy")
    image_path = os.path.join(IMAGES_DIR, filename)

    # Eğer zaten vektör varsa atla
    if os.path.exists(vector_path):
        print(f"Atlandı (vektör zaten var): {filename}")
        return

    try:
        image = Image.open(temp_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            features = model(input_tensor)

        vector = features.squeeze().cpu().numpy()
        np.save(vector_path, vector)
        print(f"✓ Kaydedildi: {filename} -> {vector_path}")

        # Görseli temp_lib'den images klasörüne taşı
        shutil.move(temp_path, image_path)
        print(f"✓ Taşındı: {filename} -> {IMAGES_DIR}")

    except Exception as e:
        print(f"Hata ({filename}):", e)

if __name__ == "__main__":
    if not os.path.exists(TEMP_LIB_DIR):
        print(f"{TEMP_LIB_DIR} klasörü bulunamadı!")
    else:
        files = [f for f in os.listdir(TEMP_LIB_DIR) if f.lower().endswith(".jpg")]
        if not files:
            print(f"{TEMP_LIB_DIR} içinde işlenecek dosya yok.")
        for file in files:
            extract_and_move(file)
