# src/settings.py
from __future__ import annotations
import os
from pathlib import Path

def _get(key: str, default: str | None = None) -> str | None:
    """
    Robust: zuerst ENV, dann (falls verfügbar) st.secrets.
    So crasht die App nicht, wenn in der Cloud noch keine Secrets hinterlegt sind.
    """
    # 1) Environment
    val = os.environ.get(key)
    if val not in (None, ""):
        return val

    # 2) Streamlit secrets (falls vorhanden/konfiguriert)
    try:
        import streamlit as st  # Import hier, um lokale Nutzung ohne Streamlit zu erlauben
        try:
            return st.secrets.get(key, default)  # type: ignore[attr-defined]
        except Exception:
            return default
    except Exception:
        return default

# ---- OpenAI / Modelle ----
OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = _get("OPENAI_BASE_URL", "")  # Bei OpenAI leer lassen
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = _get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ---- Pfade ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
LOGS_DIR = DATA_DIR / "logs"
STORAGE_DIR = BASE_DIR / "storage"
PERSIST_DIR = STORAGE_DIR / "chroma"

# Verzeichnisse sicherstellen (Cloud & lokal)
for p in [DATA_DIR, PDF_DIR, LOGS_DIR, STORAGE_DIR, PERSIST_DIR]:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ---- RAG: Chunking Defaults (von ingest.py erwartet) ----
# Du kannst diese Werte bei Bedarf anpassen:
DEFAULT_CHUNK_SIZE = int(os.environ.get("DEFAULT_CHUNK_SIZE", "1000"))       # Zeichen pro Chunk
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("DEFAULT_CHUNK_OVERLAP", "200"))  # Überlappung in Zeichen