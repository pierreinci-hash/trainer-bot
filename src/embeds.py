from __future__ import annotations
from typing import Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def make_embeddings_with_fallback(api_key: str, model: str) -> Tuple[object, bool]:
    """
    Versucht OpenAI-Embeddings. Bei Verbindungs-/DNS-/Proxy-Problemen fällt
    automatisch auf lokale HuggingFace-Embeddings zurück.
    Returns: (embeddings_object, used_local_fallback: bool)
    """
    # 1) OpenAI versuchen (mit Mini-Healthcheck, damit Verbindungsfehler sofort auffallen)
    try:
        embs = OpenAIEmbeddings(api_key=api_key, model=model)
        # Healthcheck (kostet minimal): löst Netzwerkfehler früh aus
        _ = embs.embed_query("healthcheck")
        return embs, False
    except Exception:
        # 2) Lokaler Fallback
        local = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Healthcheck lokal (sollte immer gehen)
        _ = local.embed_query("healthcheck")
        return local, True