from flask import Flask, render_template, request
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
            results = [(os.path.splitext(name)[0], score) for name, score in results]

            return render_template("index.html", results=results, uploaded_image=filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
