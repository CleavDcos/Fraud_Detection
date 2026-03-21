from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    with open("data/phishing_knowledge.json") as f:
        return json.load(f)

def create_index():
    data = load_data()
    texts = [item["text"] for item in data]

    embeddings = model.encode(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, data, embeddings

index, data, embeddings = create_index()

def search_similar(query, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append(data[idx])

    return results