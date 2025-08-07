# batch_ef.py

import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os

# Load CLIP
import clip

# Folder paths
image_dir = "images"
vector_dir = "vectors"

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

os.makedirs(vector_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"{len(image_files)} images found.\n")

for filename in image_files:
    image_path = os.path.join(image_dir, filename)
    vector_path = os.path.join(vector_dir, os.path.splitext(filename)[0] + ".npy")

    if os.path.exists(vector_path):
        print(f"Skipped (already exists): {filename}")
        continue

    try:
        # Load image for DINO
        image = Image.open(image_path).convert("RGB")
        dino_input = dino_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            dino_feat = dino_model(dino_input).squeeze().cpu().numpy()

        # Load image for CLIP
        clip_input = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_feat = clip_model.encode_image(clip_input)
            clip_feat = clip_feat.squeeze().cpu().numpy()

        # Normalize features separately
        dino_feat /= np.linalg.norm(dino_feat)
        clip_feat /= np.linalg.norm(clip_feat)

        # Concatenate features
        combined_feat = np.concatenate([dino_feat, clip_feat])

        np.save(vector_path, combined_feat)
        print(f"✓ Saved: {filename} -> {vector_path}")

    except Exception as e:
        print(f"Error ({filename}):", e)
