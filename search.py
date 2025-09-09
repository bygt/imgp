import numpy as np
import faiss
import os
from PIL import Image
from torchvision import transforms
import torch
import clip
import time

from database import get_db_manager
import config

# Veritabanı manager'ı
db_manager = get_db_manager()

# Eski dosya tabanlı sistem için sabitler (geriye uyumluluk)
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

def get_database_vectors():
    """Veritabanından görsel vektörlerini al ve FAISS index formatına dönüştür"""
    try:
        # Veritabanından vektörleri al
        db_results = db_manager.get_image_vectors()
        
        if not db_results:
            print("Veritabanında görsel vektörü bulunamadı!")
            return None, []
        
        # Vektörleri numpy array'lere dönüştür
        vectors = []
        filenames = []
        
        for result in db_results:
            try:
                # Binary vektörleri numpy array'e dönüştür
                dino_vec = np.frombuffer(result['dino_vector'], dtype=np.float32)
                clip_vec = np.frombuffer(result['clip_vector'], dtype=np.float32)
                
                # Birleştirilmiş vektörü oluştur
                combined_vec = np.concatenate((dino_vec, clip_vec)).astype('float32')
                combined_vec /= np.linalg.norm(combined_vec) + 1e-12
                
                vectors.append(combined_vec)
                filenames.append(result['image_path'])
                
            except Exception as e:
                print(f"Vektör dönüştürme hatası: {e}")
                continue
        
        if not vectors:
            print("Hiçbir geçerli vektör bulunamadı!")
            return None, []
        
        # FAISS index oluştur
        vectors_array = np.array(vectors, dtype=np.float32)
        dimension = vectors_array.shape[1]
        
        # FAISS index oluştur (L2 distance için)
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(vectors_array)
        
        print(f"Veritabanından {len(vectors)} vektör yüklendi. Boyut: {dimension}")
        return faiss_index, filenames
        
    except Exception as e:
        print(f"Veritabanı vektör yükleme hatası: {e}")
        return None, []

def get_index_and_filenames():
    """FAISS index ve filenames dosyalarını yükle"""
    try:
        # FAISS index dosyasını yükle
        if not os.path.exists(INDEX_FILE):
            print(f"❌ FAISS index dosyası bulunamadı: {INDEX_FILE}")
            print("💡 Önce 'python faissb.py' çalıştırın!")
            return None, []
        
        if not os.path.exists(FILENAMES_FILE):
            print(f"❌ Filenames dosyası bulunamadı: {FILENAMES_FILE}")
            print("💡 Önce 'python faissb.py' çalıştırın!")
            return None, []
        
        # FAISS index'i yükle
        index = faiss.read_index(INDEX_FILE)
        
        # Filenames dosyasını yükle
        with open(FILENAMES_FILE, 'r', encoding='utf-8') as f:
            filenames = [line.strip() for line in f.readlines()]
        
        print(f"✅ FAISS index yüklendi: {index.ntotal} vektör, {len(filenames)} dosya")
        return index, filenames
        
    except Exception as e:
        print(f"❌ FAISS index yükleme hatası: {e}")
        print("💡 Önce 'python faissb.py' çalıştırın!")
        return None, []

# DINO ile vektör çıkar
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
def extract_clip_features(clip_model, clip_preprocess, image_path):
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
def search_similar_images(image_path, top_k: int = None, use_sliding_window: bool = False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Query image not found: {image_path}")

    start_time = time.time()
    
    # Default top_k değerini config'den al
    if top_k is None:
        top_k = config.SEARCH_CONFIG['default_top_k']
    
    try:
        dino_model = load_dino_model()
        clip_model, clip_preprocess = load_clip_model()

        # DINO vektörü çıkar
        dino_vec = extract_dino_features(dino_model, image_path)
        
        # Sliding window kullanılıyorsa CLIP için sliding window, yoksa normal
        if use_sliding_window:
            print("🔍 Using sliding window search for better cropped image matching...")
            clip_vec = extract_clip_features_sliding_window(clip_model, clip_preprocess, image_path, grid_size=3)
        else:
            clip_vec = extract_clip_features(clip_model, clip_preprocess, image_path)
        
        query_vector = np.concatenate((dino_vec, clip_vec)).astype('float32')
        query_vector /= np.linalg.norm(query_vector) + 1e-12

        # FAISS index'ini yükle
        index, filenames = get_index_and_filenames()
        
        if index is None:
            raise ValueError("FAISS index yüklenemedi! Önce 'python faissb.py' çalıştırın.")

        print(f"Query vektör boyutu: {len(query_vector)}")
        print(f"FAISS index boyutu: {index.d}")
        print(f"Veritabanından {len(filenames)} görsel yüklendi")
        
        if len(query_vector) != index.d:
            raise ValueError(f"Vektör boyutu uyumsuzluğu: Query={len(query_vector)}, Index={index.d}")

        # FAISS ile arama yap
        distances, indices = index.search(np.array([query_vector]), top_k)
        
        # Sonuçları hazırla
        results = []
        
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(filenames):
                similarity_score = 1 / (1 + dist)  
                similarity_percentage = similarity_score * 100
                print(similarity_score, similarity_percentage)
                # URL oluştur - sadece dosya adını al ve .JPG ekle
                filename = os.path.basename(filenames[idx])  # Sadece dosya adını al
                filename_with_jpg = f"{filename}.JPG"  # Sonuna .JPG ekle
                image_url = f"https://uniteksverse.blob.core.windows.net/files/{filename_with_jpg}?v=51526.426"
                results.append({
                    'filename': filename,  # Sadece dosya adı
                    'similarity': similarity_score,  # 0-1 aralığında (eski format için)
                    'similarity_percentage': similarity_percentage,  # Yüzde olarak
                    'url': image_url
                })
        
        
        # Arama süresini hesapla
        search_duration = int((time.time() - start_time) * 1000)
        print(f"🔍 Arama tamamlandı! Süre: {search_duration}ms, Sonuç: {len(results)} görsel")
        
        # Arama geçmişini veritabanına kaydet
        try:
            _save_search_history(image_path, os.path.basename(image_path), top_k, search_duration, len(results))
        except Exception as e:
            print(f"Arama geçmişi kaydedilemedi: {e}")
        
        return results
        
    except Exception as e:
        print(f"Search function error: {type(e).__name__}: {str(e)}")
        raise

def _save_search_history(query_image_path, query_image_name, top_k, search_duration_ms, results_count):
    """Arama geçmişini veritabanına kaydet"""
    try:
        query = """
        INSERT INTO search_history 
        (query_image_path, query_image_name, top_k, search_duration_ms, results_count, search_type)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        db_manager.execute_query(query, (
            query_image_path,
            query_image_name,
            top_k,
            search_duration_ms,
            results_count,
            'combined'
        ))
        
    except Exception as e:
        print(f"Arama geçmişi kaydetme hatası: {e}")

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
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['filename']} - Similarity: {result['similarity_percentage']:.1f}%")
        print(f"   URL: {result['url']}")

    # Clean uploads
    for f in files:
        os.remove(os.path.join(UPLOADS_DIR, f))
    print("Uploads folder cleaned.")

if __name__ == "__main__":
    main()
