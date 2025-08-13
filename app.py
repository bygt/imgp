from flask import Flask, render_template, request, send_from_directory
import os

# Mitigate OpenMP runtime conflict on Windows (Torch/FAISS). Use at your own risk.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from werkzeug.utils import secure_filename
from search import search_similar_images

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Önce uploads klasöründeki dosyaları temizle
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Dosya silme hatası: {e}")

        file = request.files.get("image")
        threshold_str = request.form.get("threshold", "0")
        try:
            threshold = max(0, min(100, int(threshold_str)))
        except Exception:
            threshold = 0
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # top_k geniş tutulur; UI tarafında eşik filtrelemesi yapılacak
            results = search_similar_images(filepath, top_k=200)

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
