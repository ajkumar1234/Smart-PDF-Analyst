# retriever.py
import faiss
import pickle
import numpy as np
from openai import OpenAI

def _load_meta(meta_path="metadatas.pkl"):
    with open(meta_path, "rb") as f:
        m = pickle.load(f)
    return m["texts"], m["metadatas"]

def _query_index(index, query_embedding, top_k=4):
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return I[0], D[0]

def get_query_embedding(query: str, api_key: str):
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=query)
    return resp.data[0].embedding

def get_answer(query: str, index, meta, api_key: str, top_k=4):
    client = OpenAI(api_key=api_key)
    q_emb = get_query_embedding(query, api_key)
    idxs, dists = _query_index(index, q_emb, top_k=top_k)

    texts, metadatas = meta["texts"], meta["metadatas"]
    retrieved = []
    context = ""
    for i in idxs:
        retrieved.append({"text": texts[i], "meta": metadatas[i]})
        context += texts[i] + "\n---\n"

    prompt = f"""
You are a helpful assistant. Use the following CONTEXT to answer the question. Only use information from the context.
CONTEXT:
{context}

QUESTION:
{query}

Answer concisely and mention the source chunk ids.
"""
    # call OpenAI Chat/Completion
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
    answer = resp.choices[0].message.content
    return answer, retrieved

def summarize_document(index, meta, api_key: str):
    # Summarize concatenated top chunks (or use an LLM with a summary prompt)
    texts = meta["texts"]
    summary_input = "\n\n".join(texts[:10])  # take first 10 chunks as sample
    client = OpenAI(api_key=api_key)
    prompt = f"Summarize the following document into a concise report:\n\n{summary_input}"
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
    return resp.choices[0].message.content
