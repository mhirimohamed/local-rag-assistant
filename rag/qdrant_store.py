
# rag/qdrant_store.py
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as http_models
import os 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from config import (
    QDRANT_PATH,
    QDRANT_COLLECTION,
    QDRANT_DENSE_MODEL,
    QDRANT_SPARSE_MODEL,
)

def ensure_qdrant_client() -> QdrantClient:
    """
    Qdrant en mode local (persistant), modèles FastEmbed pour hybride (dense + BM25).
    """
 
    client = QdrantClient(path=QDRANT_PATH)  #location=":memory:"
    client.set_model(QDRANT_DENSE_MODEL)
    client.set_sparse_model(QDRANT_SPARSE_MODEL)
    return client

def _build_qdrant_filter(selected_sources: List[str]) -> Optional[http_models.Filter]:
    if not selected_sources:
        return None
    return http_models.Filter(
        must=[
            http_models.FieldCondition(
                key="source",
                match=http_models.MatchAny(any=selected_sources),
            )
        ]
    )

def qdrant_hybrid_search(
    query_text: str,
    top_k: int,
    selected_sources: List[str],
) -> List[Dict[str, Any]]:
    """
    Recherche hybride (dense + sparse) avec fusion côté client (RRF) quand
    un modèle sparse est configuré via set_sparse_model.
    """
    
    client = ensure_qdrant_client()
    q_filter = _build_qdrant_filter(selected_sources)

    result = client.query(
        collection_name=QDRANT_COLLECTION,
        query_text=query_text,
        limit=top_k*4,
        query_filter=q_filter,
    )
 
    hits = []
    for point in result:
        hits.append({
            "text": point.metadata['document'] or "",
            "source": point.metadata['source'] ,
            "score": point.score})
    return hits


def format_context_for_prompt(hits: List[Dict[str, Any]], user_message, top_k) -> str:
    """
    Construit un bloc CONTEXTE pour le prompt (avec numérotation et source).
    """
    if not hits:
        return ""

    # Reranking  
    model_path = "models/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def rerank_score(query: str, passage: str) -> float:
        inputs = tokenizer(
            query, passage,
            return_tensors="pt",
            truncation=True,
            max_length=512)
        with torch.no_grad():
            score = model(**inputs).logits[0].item()
        return score

    for h in hits:
        text = h.get("text", "")
        h["rerank_score"] = rerank_score(user_message, text)

    hits = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    hits = hits[:top_k] 

    for h in hits:
        print("hist :", h)

    lines = []
    for i, h in enumerate(hits, 1):
        src = h.get("source", "inconnu")
        snippet = (h.get("text") or "").strip()
        if len(snippet) > 1000:
            snippet = snippet[:1000] + "..."
        lines.append(f"[{i}] Source: {snippet}")
#        lines.append(f"[{i}] Source: {src}\n{snippet}")
    return "\n\n".join(lines)
