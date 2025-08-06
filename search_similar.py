# search_similar.py

import numpy as np
import faiss
import os
from PIL import Image
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF
import dinov2.models.vision_transformer as vits
from dinov2.models import build_model
from dinov2.transforms import make_classification_eval_transform

INDEX_FILE = "faiss.index"
FILENAMES_FILE = "filenames.txt"
VECTOR_DIM = 1024

def load_model():
    print("Model yükleniyor...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    print("Model yüklendi.")
    return model

def extract_features(model, image_path):
    image = Image.open(image_path).convert("RGB")
    transform = make_classification_eval_transform()
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor).squeeze(0).numpy()
    return features.astype("float32")

def search(query_vector, index, top_k=5):
    D, I = index.search(np.array([query_vector]), top_k)
    return I[0], D[0]

def main():
    query_image_path = "uploads/query.jpg"  # örnek path, ihtiyacına göre değiştir
    model = load_model()
    query_vector = extract_features(model, query_image_path)

    print("FAISS index yükleniyor...")
    index = faiss.read_index(INDEX_FILE)

    with open(FILENAMES_FILE, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f.readlines()]

    top_indices, distances = search(query_vector, index)

    print("\n🔍 En benzer görseller:")
    for rank, (i, d) in enumerate(zip(top_indices, distances), start=1):
        print(f"{rank}. {filenames[i]} - Mesafe: {d:.4f}")

if __name__ == "__main__":
    main()
