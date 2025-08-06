import numpy as np
import faiss
import os
from PIL import Image
from torchvision import transforms
import torch

INDEX_FILE = "faiss.index"
FILENAMES_FILE = "filenames.txt"
UPLOADS_DIR = "uploads"

def load_model():
    print("Model yükleniyor...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    print("Model yüklendi.")
    return model

def extract_features(model, image_path):
    transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor)
    return features.squeeze(0).cpu().numpy().astype('float32')

def search(query_vector, index, top_k=5):
    D, I = index.search(np.array([query_vector]), top_k)
    return I[0], D[0]

def main():
    # uploads klasöründeki tek görseli bul
    files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if len(files) == 0:
        print("Uploads klasöründe arama için görsel bulunamadı.")
        return
    if len(files) > 1:
        print("Uploads klasöründe birden fazla görsel var, ilkini kullanacağım:", files[0])

    query_image_path = os.path.join(UPLOADS_DIR, files[0])

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

    # Arama sonrası uploads klasörünü temizle
    for f in files:
        os.remove(os.path.join(UPLOADS_DIR, f))
    print("Uploads klasörü temizlendi.")

if __name__ == "__main__":
    main()
