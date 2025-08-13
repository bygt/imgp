from flask import Flask, render_template, request, send_from_directory
import os
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
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            results = search_similar_images(filepath)

            return render_template("index.html", results=results, uploaded_image=filename)

    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.svg', mimetype='image/svg+xml')

# Optional warmup to reduce first-request latency
from search import load_dino_model, load_clip_model, get_index_and_filenames

@app.before_first_request
def warmup():
    try:
        load_dino_model()
        load_clip_model()
        get_index_and_filenames()
    except Exception as e:
        print(f"Warmup error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
