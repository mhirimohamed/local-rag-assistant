
import os
# config.py
QDRANT_PATH = "./qdrant_data"         # stockage local persistant pour Qdrant (mode 'path')
QDRANT_COLLECTION = "mess_kb"         # nom de la collection

# Modèles FastEmbed côté Qdrant pour l'hybride
#QDRANT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"   # rapide, EN, dense
QDRANT_DENSE_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' # Choix rapide, FR, dense
#QDRANT_DENSE_MODEL = "intfloat/multilingual-e5-large"   # Choix optimal performance, FR/EN, dense
QDRANT_SPARSE_MODEL = "Qdrant/bm25"            # sparse (BM25)

# UI / RAG par défaut
DEFAULT_TOPK = 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

APP_TITLE = "PriviaDOC"
APP_DESC = "Un assistant IA avec réponses entièrement privées"


import base64
from pathlib import Path
import mimetypes

def data_uri_from_file(path: str) -> str:
    p = Path(path)
    mime = "image/jpeg"  # fallback
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"

bg_uri = data_uri_from_file("background2.png")  # <- put your filename here

CUSTOM_CSS = f"""
/* Put the background on the Gradio container */
.gradio-container {{
  background: url('{bg_uri}') no-repeat center center fixed;
  background-size: cover;
}}
footer{{display:none !important}}
"""
