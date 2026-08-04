
# ui/handlers.py
from typing import Generator, List, Optional, Tuple

from llm.ollama_chat import build_messages_from_history, stream_ollama_chat
from rag.qdrant_store import qdrant_hybrid_search, format_context_for_prompt


def _find_last_user_index(history: List[dict]) -> Optional[int]:
    for i in range(len(history) - 1, -1, -1):
        if isinstance(history[i], dict) and history[i].get("role") == "user":
            return i
    return None


def on_clear_chat() -> Tuple[List[dict], str]:
    return [], ""

def on_send_submit_rag(
    user_message: str,
    history: List[dict],
    system_prompt: str,
    model: str,
    temperature: float,
    seed: Optional[float],
    rag_enabled: bool,
    top_k: int,
    selected_sources: List[str],
) -> Generator[Tuple[str, List[dict]], None, None]:
    """
    Soumission 'Envoyer' avec option RAG hybride.
    - Si RAG désactivé ou indisponible -> chat LLM pur.
    - Sinon -> récupère les top-K passages, injecte CONTEXTE dans le prompt.
    """
    
    if not user_message or not user_message.strip():
        yield user_message, history
        return

    if not model:
        new_history = (history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "⚠️ Aucun modèle sélectionné. Merci d’en choisir un."},
        ]
        yield "", new_history
        return

    # 1) Prépare l’affichage immédiat côté UI (user + assistant vide/notice)
    augmented_user = user_message       # restera tel quel si pas de contexte
    citations_note = ""                 # note affichée si contexte injecté
    running_history = (history or []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""},  # placeholder pour le streaming
    ]
    yield "", running_history

    # 2) Si RAG activé, tente une recherche hybride (avec fallback silencieux)
    if rag_enabled:

        hits = qdrant_hybrid_search(user_message, max(1, int(top_k)), selected_sources or [])
        context_block = format_context_for_prompt(hits, user_message, top_k)

        if context_block:
            augmented_user = (" Vous êtes un assistant RAG. Répondez de manière concise et précise uniquement en vous appuyant sur le CONTEXTE ci‑dessous.\n"
                              f"CONTEXTE:\n{context_block}\n\n"
                              f"QUESTION:\n{user_message}")
            
           # citations_note = "\n\n_RAG: contexte injecté avec sources ... _"
            citations_note = ""

            # Affiche tout de suite la note côté UI (sans bloquer le streaming)
            running_history[-1]["content"] = citations_note
            yield "", running_history
        # sinon: aucun contexte -> on reste en LLM pur

    # 3) Streaming vers Ollama (LLM pur OU prompt augmenté si RAG a fourni du contexte)
    seed_int: Optional[int] = None
    if seed is not None:
        try:
            seed_int = int(seed)
        except Exception:
            seed_int = None

    messages = build_messages_from_history(history, augmented_user, system_prompt)
    try:
        for partial in stream_ollama_chat(messages, model=model, temperature=temperature, seed=seed_int):
            # Concatène la note une seule fois (au besoin)
            if citations_note and citations_note not in partial:
                running_history[-1]["content"] = partial + citations_note
            else:
                running_history[-1]["content"] = partial
            yield "", running_history
    except Exception as e:
        running_history[-1]["content"] = f"[Error] {e}"
        yield "", running_history
