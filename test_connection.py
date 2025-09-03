#!/usr/bin/env python3
"""
Sadece MSSQL Veritabanı Bağlantısını Test Et
"""

import os
from dotenv import load_dotenv
import pymssql

# .env dosyasını yükle
load_dotenv()

def test_connection():
    """Sadece bağlantıyı test et"""
    print("🔌 Veritabanı bağlantısı test ediliyor...")
    
    try:
        # Environment variables'dan bilgileri al
        server = os.getenv('DB_SERVER', 'forkis')
        database = os.getenv('DB_NAME', 'imgp_db')
        username = os.getenv('DB_USERNAME', 'uni_developer')
        password = os.getenv('DB_PASSWORD', '')
        port = int(os.getenv('DB_PORT', '1433'))
        
        print(f"📡 Bağlanmaya çalışılıyor: {server}:{port}/{database}")
        print(f"👤 Kullanıcı: {username}")
        
        # Bağlantıyı dene
        connection = pymssql.connect(
            server=server,
            port=port,
            database=database,
            user=username,
            password=password
        )
        
        print("✅ Bağlantı başarılı!")
        
        # Basit bir sorgu çalıştır
        cursor = connection.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()
        print(f"📊 SQL Server Versiyonu: {version[0]}")
        
        cursor.close()
        connection.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Bağlantı hatası: {e}")
        return False

if __name__ == "__main__":
    test_connection()
