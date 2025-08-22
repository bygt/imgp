import numpy as np
import faiss
import os
from PIL import Image
from torchvision import transforms
import torch
import clip

from config import BACKGROUND_REMOVAL, MODEL_CONFIG

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
    
    image = Image.open(image_path).convert("RGB")
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)
    vector = features.squeeze(0).cpu().numpy().astype('float32')
    vector /= np.linalg.norm(vector)
    return vector

def create_sliding_windows(image_path, grid_size=3, overlap=0.3):
    """
    Görseli grid_size x grid_size parçaya böl
    overlap: Parçalar arası örtüşme oranı (0.3 = %30)
    """
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # Parça boyutlarını hesapla
        window_width = int(width / grid_size)
        window_height = int(height / grid_size)
        
        # Örtüşme miktarını hesapla
        overlap_x = int(window_width * overlap)
        overlap_y = int(window_height * overlap)
        
        windows = []
        positions = []
        
        for row in range(grid_size):
            for col in range(grid_size):
                # Parça pozisyonunu hesapla
                x = col * (window_width - overlap_x)
                y = row * (window_height - overlap_y)
                
                # Görsel sınırlarını kontrol et
                x = max(0, min(x, width - window_width))
                y = max(0, min(y, height - window_height))
                
                # Parçayı kes
                window = image.crop((x, y, x + window_width, y + window_height))
                
                # Parçayı yeniden boyutlandır (CLIP için)
                window = window.resize((224, 224), Image.Resampling.LANCZOS)
                
                windows.append(window)
                positions.append((x, y, x + window_width, y + window_height))
        
        return windows, positions
    except Exception as e:
        print(f"Sliding window error: {e}")
        return [], []

# CLIP ile vektör çıkar (sadece görsel)
def extract_clip_features(clip_model, clip_preprocess, image_path, use_bg_removal=False):
    # Normal görsel işleme (background removal yok)
    image = Image.open(image_path).convert("RGB")
    
    clip_input = clip_preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Sadece görsel özellikler
        image_features = clip_model.encode_image(clip_input)
        vector = image_features.squeeze(0).cpu().numpy().astype('float32')
        vector /= np.linalg.norm(vector)
        return vector

# Sliding window ile CLIP vektör çıkar
def extract_clip_features_sliding_window(clip_model, clip_preprocess, image_path, grid_size=3):
    """
    Görseli parçalara bölerek her parça için CLIP vektörü çıkar
    En yüksek similarity'yi döndür
    """
    try:
        windows, positions = create_sliding_windows(image_path, grid_size=grid_size)
        if not windows:
            # Fallback: normal CLIP
            return extract_clip_features(clip_model, clip_preprocess, image_path)
        
        best_vector = None
        best_score = -1
        
        # Her parça için CLIP vektörü çıkar
        for i, window in enumerate(windows):
            # PIL Image'i tensor'a çevir
            clip_input = clip_preprocess(window).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(clip_input)
                vector = image_features.squeeze(0).cpu().numpy().astype('float32')
                vector /= np.linalg.norm(vector)
                
                # Bu parça için similarity hesapla (basit bir metrik)
                if best_vector is not None:
                    similarity = np.dot(vector, best_vector)
                    if similarity > best_score:
                        best_score = similarity
                        best_vector = vector
                else:
                    best_vector = vector
                    best_score = 0
        
        return best_vector
    except Exception as e:
        print(f"Sliding window CLIP error: {e}")
        # Fallback: normal CLIP
        return extract_clip_features(clip_model, clip_preprocess, image_path)

# 🔍 Ana fonksiyon
def search_similar_images(image_path, top_k: int = 50, use_sliding_window: bool = False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Query image not found: {image_path}")

    try:
        dino_model = load_dino_model()
        clip_model, clip_preprocess = load_clip_model()

        # Arama sırasında query fotoğun arka planını kaldır
        dino_vec = extract_dino_features(dino_model, image_path, use_bg_removal=True)
        
        # Sliding window kullanılıyorsa CLIP için sliding window, yoksa normal
        if use_sliding_window:
            print("🔍 Using sliding window search for better cropped image matching...")
            clip_vec = extract_clip_features_sliding_window(clip_model, clip_preprocess, image_path, grid_size=3)
        else:
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
    results = search_similar_images(query_image_path, use_sliding_window=True)

    print("\n🔍 Top similar images:")
    for rank, (filename, sim) in enumerate(results, start=1):
        print(f"{rank}. {filename} - Similarity: {sim:.4f}")

    # Clean uploads
    for f in files:
        os.remove(os.path.join(UPLOADS_DIR, f))
    print("Uploads folder cleaned.")

if __name__ == "__main__":
    main()
