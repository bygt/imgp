# IMGSearch — Lightning‑fast Visual Similarity Search (DINOv2 + CLIP + FAISS)

Find images that look alike in milliseconds. Upload an image, and get the most similar items from your catalog via a sleek web UI powered by Flask.

## Features
- DINOv2 + CLIP image embeddings combined and normalized for robust similarity
- FAISS index for scalable, fast nearest‑neighbor search
- Clean UI with drag‑and‑drop, global paste (Ctrl+V), live similarity threshold slider
- Windows‑friendly setup and scripts to build vectors and index
- Git‑safe: `static/images` folder is kept, contents are ignored by git

## How it works
1. Precompute dataset embeddings with `batch_ef.py` (DINOv2 + CLIP), save to `vectors/*.npy`.
2. Build a FAISS index with `faissb.py` and write `filenames.txt` (image names with extensions).
3. At query time, an uploaded image is embedded similarly and searched against the FAISS index.
4. Results are rendered instantly; the slider filters results client‑side without extra requests.

## Getting started
- Requirements: Python 3.10+, a modern CPU (GPU optional), internet (to download models on first run)

```bash
# 1) Create and activate a virtual environment (Windows PowerShell)
python -m venv venv
./venv/Scripts/Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) Put catalog images here (git-ignored, folder kept)
#    static/images/

# 4) Extract embeddings for all images
python batch_ef.py

# 5) Build FAISS index and filenames mapping
python faissb.py

# 6) Run the web app
python app.py
```
Then open `http://127.0.0.1:5000/`.

## Usage
- Drag & drop an image, click to select, or simply paste from clipboard anywhere on the page.
- Move the similarity slider to hide items below the chosen threshold (no page reloads).

## Project structure
```
app.py                 # Flask app
search.py              # Model loading, feature extraction, FAISS search
batch_ef.py            # Bulk embedding extraction for dataset (static/images -> vectors)
single_ef.py           # Embedding extraction for one-off files (temp_lib -> static/images)
faissb.py              # Build FAISS index + filenames.txt (with image extensions)
static/
  images/              # Your catalog images (ignored by git, kept via .gitkeep)
  uploads/             # Uploaded query images
  style.css
templates/
  index.html           # UI
vectors/               # Saved combined feature vectors (*.npy)
filenames.txt          # One image filename (with extension) per vector row in index order
faiss.index            # FAISS index file
```

## Scripts
- `batch_ef.py`: Processes all images in `static/images`, saves combined normalized vectors to `vectors/`.
- `single_ef.py`: Processes files from `temp_lib/` individually, moves them into `static/images` and saves vectors.
- `faissb.py`: Loads vectors, builds `faiss.index`, writes `filenames.txt` with real image filenames (extensions included).

## Notes & configuration
- Combined vector = concat(DINOv2, CLIP‑image), then L2‑normalize. Index uses inner‑product to approximate cosine similarity.
- `filenames.txt` must contain real image names with extensions (e.g., `shoe_001.jpg`). If you change images/vectors, rebuild the index.
- `static/images` is tracked as a folder but its contents are git‑ignored via `.gitignore` + `.gitkeep`.

## Troubleshooting
- First run is slow: models download and load once. Subsequent runs are faster.
- Windows OpenMP warning (Torch/FAISS): The app sets `KMP_DUPLICATE_LIB_OK=TRUE` by default. Remove if you prefer strict safety.
- Images don’t render:
  - Ensure files exist in `static/images` and are readable.
  - Rebuild mapping: `python faissb.py` (this writes `filenames.txt` with extensions).
  - Hard refresh the browser (Ctrl+F5).
- Index not updating after rebuild: the app auto‑reloads index/mapping when the underlying files change (mtime‑based cache).

## Roadmap (nice‑to‑have)
- Text/Tag‑aware re‑ranking (combine image similarity with CLIP text embeddings)
- Paging and “load more” for very large result sets
- Optional server‑side threshold filtering and API endpoints

— Enjoy fast visual search! ✨