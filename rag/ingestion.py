
# rag/ingestion.py
import os
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document

from .qdrant_store import ensure_qdrant_client
from config import QDRANT_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP

import gradio as gr


def _lc_load_and_split(files: List[gr.File]) -> List[Document]:
    """
    Charge des fichiers .txt, .pdf, .docx et les découpe en chunks.
    - .pdf  -> PyPDFLoader
    - .docx -> Docx2txtLoader
    - .txt  -> TextLoader (UTF-8, puis fallback)
    En cas d'échec de loader, tente un fallback minimal en lecture brute (UTF-8 ignore).
    """
    docs: List[Document] = []
    if not files:
        return docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for f in files:
        path = getattr(f, "name", None) or getattr(f, "path", None)
        if not path:
            continue
        path = os.path.abspath(path)
        if not os.path.exists(path):
            continue

        ext = os.path.splitext(path)[1].lower()
        pages: List[Document] = []

       # try:
        if ext == ".pdf":
            loader = PyPDFLoader(path)
            pages = loader.load()

        elif ext == ".docx":
            loader = Docx2txtLoader(path)
            pages = loader.load()

        elif ext == ".txt":
            # Essai encodage utf-8 d'abord
            try:
                loader = TextLoader(path, encoding="utf-8")
                pages = loader.load()
            except Exception:
                # Fallback sans encodage explicite
                loader = TextLoader(path)
                pages = loader.load()

        # else:
        #     # Extension non listée : on tente comme du texte UTF-8, puis fallback
        #     try:
        #         loader = TextLoader(path, encoding="utf-8")
        #         pages = loader.load()
        #     except Exception:
        #         loader = TextLoader(path)
        #         pages = loader.load()

      #  except Exception:
            # Fallback ultime : lecture brute en UTF-8 (ignore) pour ne pas bloquer tout le lot
      #      try:
      #          with open(path, "rb") as fh:
      #              raw = fh.read().decode("utf-8", errors="ignore")
      #          pages = [Document(page_content=raw, metadata={"source": os.path.basename(path)})]
      #      except Exception:
                # On ignore silencieusement ce fichier problématique
      #          continue

        # Découpage + normalisation de la métadonnée "source"
        chunks = splitter.split_documents(pages)

        unique_chunks = []
        seen = set()

        for chunk in chunks:
            text = chunk.page_content.strip()
            h = hash(text)
            if h not in seen:
                seen.add(h)
                unique_chunks.append(chunk)

        for c in unique_chunks:
            c.metadata = c.metadata or {}
            # On ne remplace pas si déjà présent, mais on garantit la présence de 'source'
            c.metadata["source"] = c.metadata.get("source") or os.path.basename(path)

        docs.extend(chunks)

    return docs


def ingest_into_qdrant(files: List[gr.File]) -> Tuple[str, List[str]]:
    """
    Charge et découpe (LangChain), puis indexe dans Qdrant (hybride dense+sparse via FastEmbed).
    Retourne un message de statut + la liste des sources disponibles (pour filtres).
    """

    if not files:
        return "Aucun fichier sélectionné.", []

    docs = _lc_load_and_split(files)

    if not docs:
        return "Aucun contenu extrait des fichiers fournis.", []

    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]

    sources = sorted({m.get("source", "inconnu") for m in metas})
    client = ensure_qdrant_client()
    client.add(
        collection_name=QDRANT_COLLECTION,
        documents=texts,
        metadata=metas,
        batch_size=64,
        parallel=None,
    )

    msg = (
        f"✅ Base de connaissance crée.\n\n"
        f"- Chunks (pargraphes) indexés: {len(texts)}\n"
        f"- Vous pouvez maintenant interroger la base de connaissance."
    )
    return msg, sources
