from flask import Flask, render_template, request, send_from_directory
import os

# Mitigate OpenMP runtime conflict on Windows (Torch/FAISS). Use at your own risk.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from werkzeug.utils import secure_filename
from search import search_similar_images
from config import BACKGROUND_REMOVAL, CLIP_PROMPTING

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Klasörü garanti et
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        file = request.files.get("image")
        threshold_str = request.form.get("threshold", "0")
        try:
            threshold = max(0, min(100, int(threshold_str)))
        except Exception:
            threshold = 0
        if file and file.filename:
            # Önce önceki upload'ları temizle
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
                results = search_similar_images(filepath, top_k=200)
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

# Optional warmup to reduce first-request latency
from search import load_dino_model, load_clip_model, get_index_and_filenames

def warmup():
    try:
        load_dino_model()
        load_clip_model()
        get_index_and_filenames()
    except Exception as e:
        print(f"Warmup error: {e}")

if __name__ == "__main__":
    # Warm once: if reloader child or when reloader is disabled
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("WERKZEUG_RUN_MAIN") is None:
        warmup()
    # Disable reloader to avoid double processes and confusing reloads on actions
    app.run(debug=True, use_reloader=False)
