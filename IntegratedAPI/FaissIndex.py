# build_faiss_index.py
import faiss
import numpy as np

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    embeddings = np.load('book_embeddings.npy')
    faiss_index = build_faiss_index(embeddings)
    faiss.write_index(faiss_index, 'faiss_index.index')
