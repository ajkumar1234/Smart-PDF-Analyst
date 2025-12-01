# embeddings.py
import os
import faiss
import pickle
from typing import List, Dict
from openai import OpenAI

EMBED_DIM = 1536  # OpenAI text-embedding-3-small dimension; adjust if different

def get_openai_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    client = OpenAI(api_key=api_key)
    # OpenAI python client usage may vary; here's a general approach:
    embeddings = []
    for t in texts:
        resp = client.embeddings.create(model="text-embedding-3-small", input=t)
        embeddings.append(resp.data[0].embedding)
    return embeddings

def build_or_load_faiss(docs: List[Dict], api_key: str, index_path="faiss.index", meta_path="metadatas.pkl"):
    # If index already exists, load it
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadatas = pickle.load(f)
        return index, metadatas

    # otherwise build
    texts = []
    metadatas = []
    for doc in docs:
        from ingestion import chunk_text
        chunks = chunk_text(doc["text"])
        for c in chunks:
            texts.append(c["text"])
            metadatas.append({"doc_name": doc["name"], "chunk_id": c["id"]})

    embeddings = get_openai_embeddings(texts, api_key=api_key)
    import numpy as np
    xb = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(xb)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

    return index, {"texts": texts, "metadatas": metadatas}
