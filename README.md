# Smart-PDF-Analyst


Smart PDF Analyst is a Retrieval-Augmented Generation (RAG) powered app that lets users upload PDFs and ask natural language questions. The system extracts text, creates embeddings, stores them in a vector store (FAISS), retrieves context, and uses an LLM to answer queries and generate summaries.

## Features
- Upload multiple PDFs
- Semantic search over document chunks
- LLM-powered answers grounded in source chunks
- Summarization / report generation
- Streamlit interface for easy demo

## Tech Stack
- Python, Streamlit
- OpenAI embeddings (or SentenceTransformers)
- FAISS vector store
- pdfplumber for extraction

## Setup (local)
1. Clone repo:
   ```bash
   git clone <repo-url>
   cd smart-pdf-analyst
