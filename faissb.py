# build_faiss_index.py

import os
import faiss as faiss
import numpy as np

VECTOR_DIR = "vectors"
INDEX_FILE = "faiss.index"

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
    print("Vektörler yükleniyor...")
    vectors, filenames = load_vectors(VECTOR_DIR)

    print(f"{len(vectors)} vektör yüklendi. FAISS index oluşturuluyor...")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    
    # isimleri de ayrı kaydediyoruz
    with open("filenames.txt", "w", encoding="utf-8") as f:
        for name in filenames:
            f.write(name + "\n")

    print("FAISS index oluşturuldu ve kaydedildi.")

if __name__ == "__main__":
    build_faiss_index()
