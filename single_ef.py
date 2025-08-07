# single_ef.py

import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import clip

# Folder paths
TEMP_LIB_DIR = "temp_lib"
IMAGES_DIR = "images"
VECTOR_DIR = "vectors"

print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DINOv2
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
dino_model.eval()

# Load CLIP
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()
print("Models loaded.")

# DINO transform
dino_transform = transforms.Compose([
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

    if os.path.exists(vector_path):
        print(f"Skipped (vector already exists): {filename}")
        return

    try:
        image = Image.open(temp_path).convert("RGB")

        dino_input = dino_transform(image).unsqueeze(0).to(device)
        clip_input = clip_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            dino_feat = dino_model(dino_input).squeeze().cpu().numpy()
            clip_feat = clip_model.encode_image(clip_input).squeeze().cpu().numpy()

        dino_feat /= np.linalg.norm(dino_feat)
        clip_feat /= np.linalg.norm(clip_feat)

        combined_feat = np.concatenate([dino_feat, clip_feat])

        np.save(vector_path, combined_feat)
        print(f"✓ Saved: {filename} -> {vector_path}")

        shutil.move(temp_path, image_path)
        print(f"✓ Moved: {filename} -> {IMAGES_DIR}")

    except Exception as e:
        print(f"Error ({filename}):", e)

if __name__ == "__main__":
    if not os.path.exists(TEMP_LIB_DIR):
        print(f"{TEMP_LIB_DIR} folder not found!")
    else:
        files = [f for f in os.listdir(TEMP_LIB_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            print(f"No files to process in {TEMP_LIB_DIR}.")
        for file in files:
            extract_and_move(file)
