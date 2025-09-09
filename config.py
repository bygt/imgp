# config.py
# Kıyafet odaklı görsel arama konfigürasyonu
import os

# MSSQL Veritabanı ayarları
DATABASE_CONFIG = {
    'server': os.getenv('DB_SERVER', 'localhost'),
    'database': os.getenv('DB_NAME', 'imgp_db'),
    'username': os.getenv('DB_USERNAME', 'your_username'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
    'port': int(os.getenv('DB_PORT', '1433')),
    'timeout': int(os.getenv('DB_TIMEOUT', '30'))
}

# Arka plan kaldırma ayarları
BACKGROUND_REMOVAL = {
    'enabled': False,  # Vektör oluştururken kapalı, sadece arama sırasında aktif
    #'model': 'u2net',  # 'u2net', 'u2net_human_seg', 'u2netp', 'silueta'
    'model': 'u2net',  # SSL sorunu için u2net kullanıyoruz
    'fallback_to_original': True  # Hata durumunda orijinal görüntüyü kullan
}

# CLIP text prompting ayarları
CLIP_PROMPTING = {
    'enabled': True,
    'image_weight': 0.7,  # Görsel özellik ağırlığı (0.0-1.0)
    'text_weight': 0.3,   # Text özellik ağırlığı (0.0-1.0)
    'clothing_prompts': [
        "clothing item",
        "fashion garment",
        "shirt", 
        "dress",
        "pants",
        "jacket",
        "sweater",
        "blouse",
        "t-shirt",
        "skirt",
        "coat",
        "trousers"
    ]
}

# Model ayarları
MODEL_CONFIG = {
    'dino_model': 'dinov2_vitl14',
    'clip_model': 'ViT-L/14',
    'device': 'auto'  # 'auto', 'cuda', 'cpu'
}

# Arama ayarları
SEARCH_CONFIG = {
    'default_top_k': 100,
    'max_top_k': 500
}
