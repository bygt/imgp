import numpy as np
import faiss
import os
from PIL import Image
from torchvision import transforms
import torch
import clip
from background_removal import preprocess_image_for_clothing
from config import BACKGROUND_REMOVAL, CLIP_PROMPTING, MODEL_CONFIG

INDEX_FILE = "faiss.index"
FILENAMES_FILE = "filenames.txt"
UPLOADS_DIR = "static/uploads"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model tanımları (ilk başta None)
_dino_model = None
_clip_model = None
_clip_preprocess = None
_faiss_index = None
_filenames = None
_index_mtime = None
_filenames_mtime = None

# DINO model yükle
def load_dino_model():
    global _dino_model
    if _dino_model is None:
        try:
            print("Loading DINOv2 model...")
            _dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
            _dino_model.eval()
            print("DINOv2 model loaded.")
        except Exception as e:
            print(f"DINOv2 loading failed: {e}")
            raise
    return _dino_model

# CLIP model yükle
def load_clip_model():
    global _clip_model, _clip_preprocess
    if _clip_model is None or _clip_preprocess is None:
        try:
            print("Loading CLIP model...")
            _clip_model, _clip_preprocess = clip.load("ViT-L/14", device=device)
            _clip_model.eval()
            print("CLIP model loaded.")
        except Exception as e:
            print(f"CLIP loading failed: {e}")
            raise
    return _clip_model, _clip_preprocess

def get_index_and_filenames():
    global _faiss_index, _filenames, _index_mtime, _filenames_mtime
    try:
        current_index_mtime = os.path.getmtime(INDEX_FILE)
        current_filenames_mtime = os.path.getmtime(FILENAMES_FILE)
    except Exception:
        current_index_mtime = None
        current_filenames_mtime = None

    need_reload = (
        _faiss_index is None
        or _filenames is None
        or _index_mtime != current_index_mtime
        or _filenames_mtime != current_filenames_mtime
    )

    if need_reload:
        _faiss_index = faiss.read_index(INDEX_FILE)
        with open(FILENAMES_FILE, "r", encoding="utf-8") as f:
            _filenames = [line.strip() for line in f]
        _index_mtime = current_index_mtime
        _filenames_mtime = current_filenames_mtime

    return _faiss_index, _filenames

# DINO ile vektör çıkar
def extract_dino_features(model, image_path, use_bg_removal=True):
    transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Arka plan kaldırma ile önişleme
    if use_bg_removal:
        image = preprocess_image_for_clothing(image_path, use_background_removal=True)
    else:
        image = Image.open(image_path).convert("RGB")
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)
    vector = features.squeeze(0).cpu().numpy().astype('float32')
    vector /= np.linalg.norm(vector)
    return vector

# CLIP ile vektör çıkar (kıyafet odaklı)
def extract_clip_features(clip_model, clip_preprocess, image_path, use_bg_removal=True):
    # Arka plan kaldırma ile önişleme
    if use_bg_removal:
        image = preprocess_image_for_clothing(image_path, use_background_removal=True)
    else:
        image = Image.open(image_path).convert("RGB")
    
    clip_input = clip_preprocess(image).unsqueeze(0).to(device)
    
    # Kıyafet odaklı text promptları
    text_prompts = CLIP_PROMPTING['clothing_prompts']
    
    with torch.no_grad():
        # Görsel özellikler
        image_features = clip_model.encode_image(clip_input)
        
        # Text özellikler
        text_tokens = clip.tokenize(text_prompts).to(device)
        text_features = clip_model.encode_text(text_tokens)
        
        # Text özelliklerinin ortalamasını al
        text_features_mean = text_features.mean(dim=0, keepdim=True)
        alpha = CLIP_PROMPTING['image_weight']  # Görsel özellik ağırlığı
        beta = CLIP_PROMPTING['text_weight']    # Text özellik ağırlığı
        
        combined_features = alpha * image_features + beta * text_features_mean
        
    vector = combined_features.squeeze(0).cpu().numpy().astype('float32')
    vector /= np.linalg.norm(vector)
    return vector

# 🔍 Ana fonksiyon
def search_similar_images(image_path, top_k: int = 50):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Query image not found: {image_path}")

    try:
        dino_model = load_dino_model()
        clip_model, clip_preprocess = load_clip_model()

        # Arama sırasında query fotoğun arka planını kaldır
        dino_vec = extract_dino_features(dino_model, image_path, use_bg_removal=True)
        clip_vec = extract_clip_features(clip_model, clip_preprocess, image_path, use_bg_removal=True)
        query_vector = np.concatenate((dino_vec, clip_vec)).astype('float32')
        query_vector /= np.linalg.norm(query_vector) + 1e-12

        index, filenames = get_index_and_filenames()

        print(f"Query vektör boyutu: {len(query_vector)}")
        print(f"FAISS index boyutu: {index.d}")
        
        if len(query_vector) != index.d:
            raise ValueError(f"Vektör boyutu uyumsuzluğu: Query={len(query_vector)}, Index={index.d}")

        distances, indices = index.search(np.array([query_vector]), top_k)
        results = [(filenames[i], float(dist)) for i, dist in zip(indices[0], distances[0])]
        return results
    except Exception as e:
        print(f"Search function error: {type(e).__name__}: {str(e)}")
        raise

# CLI
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
    results = search_similar_images(query_image_path)

    print("\n🔍 Top similar images:")
    for rank, (filename, sim) in enumerate(results, start=1):
        print(f"{rank}. {filename} - Similarity: {sim:.4f}")

    # Clean uploads
    for f in files:
        os.remove(os.path.join(UPLOADS_DIR, f))
    print("Uploads folder cleaned.")

if __name__ == "__main__":
    main()
