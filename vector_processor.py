#!/usr/bin/env python3
"""
Vector Processor - 61k görseli 100'er 100'er işler
"""

import requests
import os
import tempfile
import numpy as np
import torch
import time
import json
from PIL import Image
from batch_ef import dino_model, clip_model, clip_preprocess, dino_transform
from background_removal import preprocess_image_for_clothing
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
    """batch_ef'teki process_single_image fonksiyonunu kullan"""
    try:
        # batch_ef'teki fonksiyonu çağır
        from batch_ef import process_single_image
        
        # Geçici dosya adı oluştur
        temp_filename = os.path.basename(image_path)
        temp_dir = os.path.dirname(image_path)
        
        # batch_ef fonksiyonunu çağır
        result = process_single_image((temp_filename, temp_dir, VECTORS_DIR))
        
        if "✓ Saved:" in result:
            # Vektör dosyalarını oku
            base_name = os.path.splitext(temp_filename)[0]
            dino_path = os.path.join(VECTORS_DIR, f"{base_name}_dino.npy")
            clip_path = os.path.join(VECTORS_DIR, f"{base_name}_clip.npy")
            combined_path = os.path.join(VECTORS_DIR, f"{base_name}_combined.npy")
            
            # Eğer dosyalar varsa oku
            if os.path.exists(combined_path):
                combined_feat = np.load(combined_path)
                return None, None, combined_feat  # Sadece combined döndür
            else:
                print(f"❌ Vektör dosyası bulunamadı: {combined_path}")
                return None, None, None
        else:
            print(f"❌ batch_ef hatası: {result}")
            return None, None, None
            
    except Exception as e:
        print(f"❌ Vektör çıkarma hatası: {e}")
        return None, None, None

def save_vectors(dino_feat, clip_feat, combined_feat, filename):
    """Vektörleri dosyaya kaydet - batch_ef zaten kaydediyor"""
    try:
        # batch_ef zaten vektörleri kaydediyor, sadece kontrol et
        base_name = os.path.splitext(filename)[0]
        combined_path = os.path.join(VECTORS_DIR, f"{base_name}_combined.npy")
        
        if os.path.exists(combined_path):
            return True
        else:
            print(f"❌ Vektör dosyası bulunamadı: {combined_path}")
            return False
            
    except Exception as e:
        print(f"❌ Vektör kaydetme hatası {filename}: {e}")
        return False

def clean_temp_dir():
    """Temp klasörünü temizle"""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR, exist_ok=True)
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
            image_id = item['id']
            image_url = item['url']
            filename = f"image_{image_id}.jpg"
            
            # Görseli indir
            image_path = download_image(image_url, filename)
            if not image_path:
                error_count += 1
                continue
            
            # Vektörleri çıkar
            dino_feat, clip_feat, combined_feat = extract_vectors(image_path)
            if dino_feat is None:
                error_count += 1
                continue
            
            # Vektörleri kaydet
            if save_vectors(dino_feat, clip_feat, combined_feat, filename):
                success_count += 1
            else:
                error_count += 1
            
            # Geçici dosyayı sil
            try:
                os.remove(image_path)
            except:
                pass
                
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
    print(f"📊 Batch boyutu: {BATCH_SIZE}")
    print(f"📁 Temp klasör: {TEMP_DIR}")
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
    print(f"\n🎉 İşlem Tamamlandı!")
    print(f"✅ Başarılı: {total_success}")
    print(f"❌ Hata: {total_errors}")
    print(f"⏱️ Toplam süre: {total_time:.2f} saniye")
    print(f"🚀 Hız: {total_success / total_time:.2f} görsel/saniye")
    
    # Temp klasörünü temizle
    clean_temp_dir()

if __name__ == "__main__":
    main()
