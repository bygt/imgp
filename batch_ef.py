# batch_ef.py

import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import time

# Load CLIP
import clip
from background_removal import preprocess_image_for_clothing

# Folder paths
image_dir = "static/images"
vector_dir = "vectors"

print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DINOv2 - force offline mode
try:
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', source='local', verbose=False).to(device)
except:
    # Try cached version
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', force_reload=False, verbose=False).to(device)
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

def process_single_image(args):
    """Tek bir görseli işle"""
    filename, image_dir, vector_dir = args
    
    # Global modelleri kullan
    global dino_model, clip_model, clip_preprocess, dino_transform
    
    image_path = os.path.join(image_dir, filename)
    vector_path = os.path.join(vector_dir, os.path.splitext(filename)[0] + ".npy")
    
    if os.path.exists(vector_path):
        return f"Skipped (already exists): {filename}"
    
    try:
        # Load image without background removal for vector creation
        processed_image = preprocess_image_for_clothing(image_path, use_background_removal=False)
        
        # Load image for DINO
        dino_input = dino_transform(processed_image).unsqueeze(0).to(device)
        with torch.no_grad():
            dino_feat = dino_model(dino_input).squeeze().cpu().numpy()
        
        # Load image for CLIP with clothing focus
        clip_input = clip_preprocess(processed_image).unsqueeze(0).to(device)
        
        # Kıyafet odaklı text promptları
        text_prompts = [
            "clothing item",
            "fashion garment", 
            "shirt",
            "dress",
            "pants",
            "jacket",
            "sweater",
            "blouse"
        ]
        
        with torch.no_grad():
            # Görsel özellikler
            image_features = clip_model.encode_image(clip_input)
            
            # Text özellikler
            text_tokens = clip.tokenize(text_prompts).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features_mean = text_features.mean(dim=0, keepdim=True)
            
            # Birleştir
            alpha = 0.7  # Görsel özellik ağırlığı
            beta = 0.3   # Text özellik ağırlığı
            combined_features = alpha * image_features + beta * text_features_mean
            
            clip_feat = combined_features.squeeze().cpu().numpy()
        
        # Normalize features separately
        dino_feat /= np.linalg.norm(dino_feat)
        clip_feat /= np.linalg.norm(clip_feat)
        
        # Concatenate features and normalize
        combined_feat = np.concatenate([dino_feat, clip_feat]).astype("float32")
        combined_feat /= np.linalg.norm(combined_feat) + 1e-12
        
        np.save(vector_path, combined_feat)
        return f"✓ Saved: {filename} -> {vector_path}"
        
    except Exception as e:
        return f"Error ({filename}): {e}"

# Sıralı işle
def process_images_sequential(image_files, image_dir, vector_dir):
    """Görselleri sıralı işle"""
    print("Processing images sequentially...")
    
    start_time = time.time()
    results = []
    
    for filename in image_files:
        result = process_single_image((filename, image_dir, vector_dir))
        results.append(result)
        print(result)
    
    end_time = time.time()
    print(f"\n⏱️ Total processing time: {end_time - start_time:.2f} seconds")
    print(f"🚀 Speed: {len(image_files) / (end_time - start_time):.2f} images/second")

# Sıralı işle
if __name__ == '__main__':
    print("🚀 Starting sequential processing...")
    process_images_sequential(image_files, image_dir, vector_dir)
