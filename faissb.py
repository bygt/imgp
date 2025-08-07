# faissb.py

import os
import faiss
import numpy as np

VECTOR_DIR = "vectors"
INDEX_FILE = "faiss.index"
FILENAMES_FILE = "filenames.txt"

def load_vectors(vector_dir):
    vectors = []
    filenames = []

    for filename in os.listdir(vector_dir):
        if filename.endswith(".npy"):
            path = os.path.join(vector_dir, filename)
            vector = np.load(path)
            vectors.append(vector.astype("float32"))
            filenames.append(filename)
    
    return np.stack(vectors), filenames

def build_faiss_index():
    print("Loading vectors...")
    vectors, filenames = load_vectors(VECTOR_DIR)

    print(f"{len(vectors)} vectors loaded. Building FAISS index...")
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity on normalized vectors
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    
    with open(FILENAMES_FILE, "w", encoding="utf-8") as f:
        for name in filenames:
            f.write(name + "\n")

    print("FAISS index created and saved.")

if __name__ == "__main__":
    build_faiss_index()
