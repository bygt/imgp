#!/usr/bin/env python3
"""
Vector Processor - 61k görseli 100'er 100'er işler
"""

import requests
import os
import numpy as np
import torch
import time
from batch_ef import dino_model, clip_model, clip_preprocess, dino_transform
import shutil

# Klasörler
TEMP_DIR = "temp_lib"
VECTORS_DIR = "vectors"
BATCH_SIZE = 100
API_URL = "http://127.0.0.1:5000/api/core-media"

def create_directories():
    """Gerekli klasörleri oluştur"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(VECTORS_DIR, exist_ok=True)
    print(f"✅ Klasörler oluşturuldu: {TEMP_DIR}, {VECTORS_DIR}")

def get_image_urls():
    """API'den görsel URL'lerini al"""
    try:
        print("🔍 API'den görsel URL'leri alınıyor...")
        response = requests.get(API_URL, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                filtered_files = data.get('filtered_files', [])
                print(f"✅ {len(filtered_files)} görsel URL'si alındı")
                return filtered_files
            else:
                print(f"❌ API hatası: {data.get('error')}")
                return []
        else:
            print(f"❌ HTTP hatası: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ API bağlantı hatası: {e}")
        return []

def download_image(url, filename):
    """Tek bir görseli indir"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            filepath = os.path.join(TEMP_DIR, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
        else:
            print(f"❌ İndirme hatası {filename}: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ İndirme hatası {filename}: {e}")
        return None

def extract_vectors(image_path):
    """Görselden vektörleri çıkar - batch_ef modellerini kullan"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Görseli yükle
        from PIL import Image
        processed_image = Image.open(image_path)
        
        # DINO vektörü
        dino_input = dino_transform(processed_image).unsqueeze(0).to(device)
        with torch.no_grad():
            dino_feat = dino_model(dino_input).squeeze().cpu().numpy()
        
        # CLIP vektörü
        clip_input = clip_preprocess(processed_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(clip_input)
            clip_feat = image_features.squeeze().cpu().numpy()
        
        # Normalize
        dino_feat /= np.linalg.norm(dino_feat)
        clip_feat /= np.linalg.norm(clip_feat)
        
        # Birleştir
        combined_feat = np.concatenate([dino_feat, clip_feat]).astype("float32")
        combined_feat /= np.linalg.norm(combined_feat) + 1e-12
        
        return dino_feat, clip_feat, combined_feat
        
    except Exception as e:
        print(f"❌ Vektör çıkarma hatası: {e}")
        return None, None, None

def save_vectors(dino_feat, clip_feat, combined_feat, filename):
    """Vektörleri dosyaya kaydet"""
    try:
        base_name = os.path.splitext(filename)[0]
        combined_path = os.path.join(VECTORS_DIR, f"{base_name}.npy")
        
        # Eğer dosya zaten varsa atla
        if os.path.exists(combined_path):
            print(f"✅ Bulundu ve atlandı: {filename}")
            return True
        
        # Sadece combined vector'ı kaydet
        np.save(combined_path, combined_feat)
        
        print(f"✅ {filename}")
        return True
    except Exception as e:
        print(f"❌ Vektör kaydetme hatası {filename}: {e}")
        return False

def clean_temp_dir():
    """Temp klasörünü temizle"""
    try:
        if os.path.exists(TEMP_DIR):
            # Dosyaları tek tek sil
            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"❌ Dosya silme hatası {filename}: {e}")
            print(f"🧹 {TEMP_DIR} temizlendi")
    except Exception as e:
        print(f"❌ Temizleme hatası: {e}")

def process_batch(batch_data, batch_num, total_batches):
    """Bir batch'i işle"""
    print(f"\n🔄 Batch {batch_num}/{total_batches} işleniyor ({len(batch_data)} görsel)...")
    
    batch_start_time = time.time()
    success_count = 0
    error_count = 0
    
    for item in batch_data:
        try:
            image_url = item['url']
            
            # URL'den filename çıkar
            filename = image_url.split('/')[-1].split('?')[0]  # ?v=51526.426 kısmını çıkar
            
            # ÖNCE vektör var mı kontrol et
            base_name = os.path.splitext(filename)[0]
            combined_path = os.path.join(VECTORS_DIR, f"{base_name}.npy")
            
            if os.path.exists(combined_path):
                print(f"✅ Bulundu ve atlandı: {filename}")
                success_count += 1
                continue
            
            # Görseli indir
            image_path = download_image(image_url, filename)
            if not image_path:
                error_count += 1
                continue
            
            # Vektörleri çıkar
            dino_feat, clip_feat, combined_feat = extract_vectors(image_path)
            if dino_feat is None:
                print(f"✔️ Oluşturuluyor: {filename}")
                success_count += 1
                continue
            
            # Vektörleri kaydet
            if save_vectors(dino_feat, clip_feat, combined_feat, filename):
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"❌ Batch işleme hatası: {e}")
            error_count += 1
            continue
    
    batch_time = time.time() - batch_start_time
    print(f"✅ Batch {batch_num} tamamlandı: {success_count} başarılı, {error_count} hata, {batch_time:.2f}s")
    
    return success_count, error_count

def main():
    """Ana işlem fonksiyonu"""
    print("🚀 Batch Vector Processing Başlıyor...")
    print(f"📁 Vektör klasör: {VECTORS_DIR}")
    
    # Klasörleri oluştur
    create_directories()
    
    # Görsel URL'lerini al
    image_urls = get_image_urls()
    if not image_urls:
        print("❌ Görsel URL'leri alınamadı!")
        return
    
    total_images = len(image_urls)
    total_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"📊 Toplam: {total_images} görsel, {total_batches} batch")
    
    # Batch'leri işle
    total_success = 0
    total_errors = 0
    start_time = time.time()
    
    for i in range(0, total_images, BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        batch_data = image_urls[i:i + BATCH_SIZE]
        
        # Batch'i işle
        success, errors = process_batch(batch_data, batch_num, total_batches)
        total_success += success
        total_errors += errors
        
        # Temp klasörünü temizle
        clean_temp_dir()
        
        # Progress göster
        progress = (batch_num / total_batches) * 100
        print(f"📈 İlerleme: {progress:.1f}% ({batch_num}/{total_batches})")
    
    # Sonuçları göster
    total_time = time.time() - start_time
    print("\n🎉 İşlem Tamamlandı!")
    print(f"✅ Başarılı: {total_success}")
    print(f"❌ Hata: {total_errors}")
    print(f"⏱️ Toplam süre: {total_time:.2f} saniye")
    if total_time > 0:
        print(f"🚀 Hız: {total_success / total_time:.2f} görsel/saniye")
    else:
        print("🚀 Hız: Hesaplanamadı")
    
    # Temp klasörünü temizle
    clean_temp_dir()

if __name__ == "__main__":
    main()
