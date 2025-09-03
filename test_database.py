#!/usr/bin/env python3
"""
MSSQL Veritabanı Bağlantı Test Scripti
Bu script veritabanı bağlantısını test eder ve temel işlemleri gösterir.
"""

import os
import sys
import numpy as np
from database import get_db_manager
from config import DATABASE_CONFIG

def test_database_connection():
    """Veritabanı bağlantısını test et"""
    print("🔌 Veritabanı bağlantısı test ediliyor...")
    
    try:
        db_manager = get_db_manager()
        
        # Bağlantıyı test et
        if db_manager.test_connection():
            print("✅ Veritabanı bağlantısı başarılı!")
            return True
        else:
            print("❌ Veritabanı bağlantısı başarısız!")
            return False
            
    except Exception as e:
        print(f"❌ Veritabanı bağlantı hatası: {e}")
        return False

def test_basic_operations():
    """Temel veritabanı işlemlerini test et"""
    print("\n🔧 Temel veritabanı işlemleri test ediliyor...")
    
    try:
        db_manager = get_db_manager()
        
        # Test verisi ekle
        print("📝 Test verisi ekleniyor...")
        test_dino_vector = np.random.rand(1024).astype(np.float32)
        test_clip_vector = np.random.rand(768).astype(np.float32)
        test_combined_vector = np.random.rand(1792).astype(np.float32)
        
        result = db_manager.insert_image_vector(
            image_path="/test/images/test1.jpg",
            image_name="Test Image 1",
            dino_vector=test_dino_vector,
            clip_vector=test_clip_vector,
            combined_vector=test_combined_vector,
            metadata='{"category": "test", "description": "Test image"}',
            file_size=1024000,
            image_width=1920,
            image_height=1080,
            file_extension="jpg"
        )
        
        print(f"✅ Test verisi eklendi! ID: {result}")
        
        # Verileri oku
        print("📖 Veriler okunuyor...")
        vectors = db_manager.get_image_vectors()
        print(f"✅ {len(vectors)} vektör okundu")
        
        # Test verisini sil
        if vectors:
            first_id = vectors[0]['image_id']
            print(f"🗑️ Test verisi siliniyor (ID: {first_id})...")
            delete_result = db_manager.delete_image_vector(first_id)
            print(f"✅ Test verisi silindi! Etkilenen satır: {delete_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Temel işlemler test hatası: {e}")
        return False

def show_database_config():
    """Veritabanı konfigürasyonunu göster"""
    print("\n⚙️ Veritabanı Konfigürasyonu:")
    print("=" * 50)
    
    for key, value in DATABASE_CONFIG.items():
        if key == 'password':
            print(f"{key:20}: {'*' * len(str(value))}")
        else:
            print(f"{key:20}: {value}")
    
    print("\n💡 Environment Variables:")
    print("=" * 50)
    env_vars = [
        'DB_SERVER', 'DB_NAME', 'DB_USERNAME', 'DB_PASSWORD',
        'DB_PORT', 'DB_TRUSTED_CONNECTION'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        if var == 'DB_PASSWORD' and value != 'Not set':
            value = '*' * len(value)
        print(f"{var:20}: {value}")

def main():
    """Ana test fonksiyonu"""
    print("🚀 MSSQL Veritabanı Test Scripti")
    print("=" * 50)
    
    # Konfigürasyonu göster
    show_database_config()
    
    # Bağlantıyı test et
    if not test_database_connection():
        print("\n❌ Veritabanı bağlantısı başarısız! Lütfen konfigürasyonu kontrol edin.")
        print("\n🔧 Kontrol edilecek noktalar:")
        print("1. SQL Server çalışıyor mu?")
        print("2. Veritabanı adı doğru mu?")
        print("3. Kullanıcı adı ve şifre doğru mu?")
        print("4. ODBC Driver yüklü mü?")
        print("5. Firewall ayarları doğru mu?")
        return False
    
    # Temel işlemleri test et
    if not test_basic_operations():
        print("\n❌ Temel işlemler test hatası!")
        return False
    
    print("\n✅ Tüm testler başarılı! Veritabanı hazır.")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Test iptal edildi.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        sys.exit(1)
