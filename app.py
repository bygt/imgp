from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import requests
import tempfile
import numpy as np
import torch
from PIL import Image
from batch_ef import dino_model, clip_model, clip_preprocess, dino_transform
from background_removal import preprocess_image_for_clothing

# Mitigate OpenMP runtime conflict on Windows (Torch/FAISS). Use at your own risk.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from werkzeug.utils import secure_filename
from search import search_similar_images
from config import BACKGROUND_REMOVAL, CLIP_PROMPTING
from database import get_db_manager

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Veritabanı manager'ı
db_manager = get_db_manager()

def download_and_process_image(image_url, media_id):
    """Görseli indir ve geçici dosya olarak kaydet"""
    try:
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            # Geçici dosya oluştur
            temp_dir = tempfile.gettempdir()
            temp_filename = f"temp_image_{media_id}.jpg"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            return temp_path
        else:
            print(f"Failed to download image: {response.status_code}")
            return None
    except Exception as e:
        print(f"Download error: {e}")
        return None

def extract_vectors_from_image(image_path):
    """batch_ef kullanarak görselden vektörleri çıkar"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Görseli yükle ve preprocess
        processed_image = preprocess_image_for_clothing(image_path, use_background_removal=False)
        
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
        
        # VARBINARY için bytes'a çevir
        dino_bytes = dino_feat.tobytes()
        clip_bytes = clip_feat.tobytes()
        combined_bytes = combined_feat.tobytes()
        
        return dino_bytes, clip_bytes, combined_bytes
        
    except Exception as e:
        print(f"Vector extraction error: {e}")
        raise

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Klasörü garanti et
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        file = request.files.get("image")
        threshold_str = request.form.get("threshold", "0")
        sliding_window_enabled = request.form.get("sliding_window_enabled") == "true"
        try:
            threshold = max(0, min(100, int(threshold_str)))
        except Exception:
            threshold = 0
        if file and file.filename:
            try:
                for f in os.listdir(app.config['UPLOAD_FOLDER']):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            except Exception as e:
                print(f"Dosya temizleme hatası: {e}")

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # top_k geniş tutulur; UI tarafında eşik filtrelemesi yapılacak
            try:
                results = search_similar_images(filepath, top_k=200, use_sliding_window=sliding_window_enabled)
            except Exception as e:
                print(f"Search error: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                results = []

            # Ensure image filenames point to actual image files (not .npy)
            def resolve_image_filename(name: str) -> str:
                base, ext = os.path.splitext(name)
                image_dir = os.path.join('static', 'images')
                preferred_exts = ['.jpg', '.jpeg', '.png']
                # If already an image extension and exists, return as-is
                if ext.lower() in preferred_exts and os.path.exists(os.path.join(image_dir, name)):
                    return name
                # Try resolve by base name
                for e in preferred_exts:
                    cand = base + e
                    if os.path.exists(os.path.join(image_dir, cand)):
                        return cand
                # Fallback to original
                return name

            results = [(resolve_image_filename(name), score) for name, score in results]

            return render_template("index.html", results=results, uploaded_image=filename, threshold=threshold)

    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.svg', mimetype='image/svg+xml')

@app.route('/api/core-media', methods=['GET'])
def get_core_media():
    """Core_Media tablosundan -0 ile biten dosyaları al ve URL'leri oluştur"""
    try:
        # Filtrelenmiş sorgu - Tüm veriler tek sayfada
        filtered_query = """
        SELECT Id, FileName FROM [Core_Media] 
        WHERE FileName LIKE '%-0.JPG%'
        ORDER BY Id
        """
        
        filtered_files = db_manager.execute_query(filtered_query)
        
        # Her dosya için sadece ID ve URL oluştur
        files_with_urls = []
        for file_data in filtered_files:
            filename = file_data.get('FileName', '')
            file_id = file_data.get('Id', '')
            if filename and file_id:
                url = f"https://uniteksverse.blob.core.windows.net/files/{filename}?v=51526.426"
                files_with_urls.append({
                    'id': file_id,
                    'url': url
                })
        
        return jsonify({
            'total_count': len(files_with_urls),
            'success': True,
            'table_name': 'Core_Media',
            'database': 'Universe',
            'filtered_files': files_with_urls
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/process-vectors', methods=['POST'])
def process_vectors():
    """Batch processing ile vektörleri çıkar ve Product_Vector tablosuna kaydet"""
    try:
        # Core_Media'dan -0.JPG ile biten dosyaları al
        media_query = """
        SELECT cm.Id as media_id, cm.FileName, cpm.ProductId as product_id
        FROM [Core_Media] cm
        INNER JOIN [Catalog_ProductMedia] cpm ON cm.Id = cpm.MediaId
        WHERE cm.FileName LIKE '%-0.JPG%'
        ORDER BY cm.Id
        """
        
        media_files = db_manager.execute_query(media_query)
        
        if not media_files:
            return jsonify({
                'success': False,
                'error': 'No media files found'
            }), 404
        
        # Batch processing için hazırla
        batch_size = 100  # Her seferde 100 dosya işle
        total_files = len(media_files)
        processed_count = 0
        error_count = 0
        
        results = []
        
        # Batch halinde işle
        for i in range(0, total_files, batch_size):
            batch = media_files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_files + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            for media_file in batch:
                try:
                    media_id = media_file['media_id']
                    file_name = media_file['FileName']
                    product_id = media_file['product_id']
                    
                    # Görseli indir
                    image_url = f"https://uniteksverse.blob.core.windows.net/files/{file_name}?v=51526.426"
                    image_path = download_and_process_image(image_url, media_id)
                    
                    if image_path:
                        try:
                            # batch_ef kullanarak vektörleri çıkar
                            dino_vector, clip_vector, combined_vector = extract_vectors_from_image(image_path)
                            
                            # Vektörleri Product_Vector tablosuna kaydet
                            update_query = """
                            UPDATE [Product_Vector] 
                            SET dino_vector = ?, clip_vector = ?, combined_vector = ?, status = 'completed'
                            WHERE product_id = ? AND media_id = ?
                            """
                            
                            db_manager.execute_query(update_query, (dino_vector, clip_vector, combined_vector, product_id, media_id))
                            processed_count += 1
                            
                            # Görseli sil
                            os.remove(image_path)
                            
                        except Exception as e:
                            error_count += 1
                            print(f"Vector extraction error for {file_name}: {e}")
                            continue
                    else:
                        error_count += 1
                        print(f"Failed to download {file_name}")
                        continue
                    
                    results.append({
                        'media_id': media_id,
                        'product_id': product_id,
                        'file_name': file_name,
                        'status': 'pending'
                    })
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error processing {media_file.get('FileName', 'unknown')}: {e}")
                    continue
        
        return jsonify({
            'success': True,
            'message': f'Batch processing completed',
            'total_files': total_files,
            'processed_count': processed_count,
            'error_count': error_count,
            'batch_size': batch_size,
            'results': results[:10]  # İlk 10 sonucu göster
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/table/<table_name>', methods=['GET'])
def get_table_info(table_name):
    """Belirli bir tablonun yapısını ve verilerini göster"""
    try:

       
        
        return jsonify({
            'success': True,
            'table_name': table_name,
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Optional warmup to reduce first-request latency
from search import load_dino_model, load_clip_model

def warmup():
    try:
        load_dino_model()
        load_clip_model()
        # get_index_and_filenames() - FAISS index yükleme şimdilik devre dışı
        print("Models loaded successfully (FAISS index disabled)")
    except Exception as e:
        print(f"Warmup error: {e}")

if __name__ == "__main__":
    # Warm once: if reloader child or when reloader is disabled
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("WERKZEUG_RUN_MAIN") is None:
        warmup()
    app.run(debug=False, use_reloader=True)
