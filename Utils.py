# utils.py
import os
from pathlib import Path

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(st_file):
    out_path = Path(UPLOAD_DIR) / st_file.name
    with open(out_path, "wb") as f:
        f.write(st_file.getbuffer())
    return str(out_path)
