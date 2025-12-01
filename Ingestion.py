# ingestion.py
import pdfplumber
import os
import uuid
from typing import List, Dict

def extract_text_from_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Returns list of chunks: {'id': str, 'text': str, 'meta': {...}}
    """
    chunks = []
    start = 0
    doc_len = len(text)
    while start < doc_len:
        end = min(start + chunk_size, doc_len)
        chunk_text = text[start:end]
        chunks.append({"id": str(uuid.uuid4()), "text": chunk_text})
        start += chunk_size - overlap
    return chunks
