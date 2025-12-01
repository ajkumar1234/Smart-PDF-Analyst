# app.py
import os
import streamlit as st
from ingestion import extract_text_from_pdf
from embeddings import build_or_load_faiss, embed_texts
from retriever import get_answer, summarize_document
from utils import save_uploaded_file

st.set_page_config(page_title="Smart PDF Analyst", layout="wide")
st.title("Smart PDF Analyst — RAG + QA for PDFs")

# Load API key from env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY in your environment (.env) to use embeddings & LLM.")
    st.stop()

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    docs = []
    for up in uploaded_files:
        path = save_uploaded_file(up)
        text = extract_text_from_pdf(path)
        docs.append({"path": path, "text": text, "name": up.name})

    # Build or load vector store
    faiss_index, metadatas = build_or_load_faiss(docs, api_key=OPENAI_API_KEY)

    st.success("Document(s) processed and indexed.")

    st.sidebar.header("Actions")
    action = st.sidebar.selectbox("Choose", ["Chat with documents", "Generate summary report"])
    if action == "Chat with documents":
        query = st.text_input("Ask something about the uploaded documents")
        if st.button("Get Answer") and query:
            with st.spinner("Retrieving answer..."):
                answer, sources = get_answer(query, faiss_index, metadatas, api_key=OPENAI_API_KEY)
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Source Chunks")
            for i, s in enumerate(sources):
                st.markdown(f"**Chunk {i+1}** — {s['meta']}")
                st.write(s['text'])
    else:
        if st.button("Create Summary Report"):
            with st.spinner("Generating summary..."):
                summary = summarize_document(faiss_index, metadatas, api_key=OPENAI_API_KEY)
            st.subheader("Document Summary")
            st.write(summary)
            # Optionally: create downloadable PDF/MD (left to you)
