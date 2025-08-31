# src/embeds.py
from langchain_openai import OpenAIEmbeddings

class StrictEmbeddingError(Exception):
    """Eigener Fehler, wenn OpenAI-Embeddings nicht erreichbar sind."""
    pass

def make_embeddings_strict(api_key: str, model: str):
    """
    Baut IMMER OpenAI-Embeddings und testet sie sofort mit einem kleinen String.
    Falls kein Ergebnis: wirft StrictEmbeddingError.
    """
    try:
        emb = OpenAIEmbeddings(openai_api_key=api_key, model=model)
        # Soforttest (ping)
        _ = emb.embed_query("ping")
        return emb
    except Exception as e:
        raise StrictEmbeddingError(
            f"❌ OpenAI-Embeddings konnten nicht initialisiert werden.\n"
            f"API-Key oder Netzwerk prüfen.\nDetails: {e}"
        )
