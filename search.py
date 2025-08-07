import numpy as np
import faiss
import os
from PIL import Image
from torchvision import transforms
import torch
import clip  # transformers yerine clip modülü kullanılacak

INDEX_FILE = "faiss.index"
FILENAMES_FILE = "filenames.txt"
UPLOADS_DIR = "uploads"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DINOv2 model
def load_dino_model():
    print("Loading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    model.eval()
    print("DINOv2 model loaded.")
    return model

# Load CLIP model
def load_clip_model():
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    print("CLIP model loaded.")
    return clip_model, clip_preprocess

def extract_dino_features(model, image_path):
    transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)
    vector = features.squeeze(0).cpu().numpy().astype('float32')
    vector /= np.linalg.norm(vector)
    return vector

def extract_clip_features(clip_model, clip_preprocess, image_path):
    image = Image.open(image_path).convert("RGB")
    clip_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(clip_input)
    vector = features.squeeze(0).cpu().numpy().astype('float32')
    vector /= np.linalg.norm(vector)
    return vector

def main():
    if not os.path.exists(UPLOADS_DIR):
        print(f"{UPLOADS_DIR} folder not found!")
        return

    files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(files) == 0:
        print("No images found in uploads folder for searching.")
        return
    if len(files) > 1:
        print("Multiple images found in uploads folder, using the first one:", files[0])

    query_image_path = os.path.join(UPLOADS_DIR, files[0])

    dino_model = load_dino_model()
    clip_model, clip_preprocess = load_clip_model()

    dino_vec = extract_dino_features(dino_model, query_image_path)
    clip_vec = extract_clip_features(clip_model, clip_preprocess, query_image_path)

    # Combine DINO and CLIP vectors (concatenate)
    query_vector = np.concatenate((dino_vec, clip_vec)).astype('float32')

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)

    with open(FILENAMES_FILE, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f.readlines()]

    distances, indices = index.search(np.array([query_vector]), 5)
    similarities = distances[0]

    print("\n🔍 Top similar images:")
    for rank, (i, sim) in enumerate(zip(indices[0], similarities), start=1):
        print(f"{rank}. {filenames[i]} - Similarity: {sim:.4f}")

    # Clean uploads folder
    for f in files:
        os.remove(os.path.join(UPLOADS_DIR, f))
    print("Uploads folder cleaned.")

if __name__ == "__main__":
    main()
