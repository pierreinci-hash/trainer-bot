# src/settings.py
from __future__ import annotations
import os
from pathlib import Path

# Versuche, Streamlit-Secrets zu lesen (in Cloud gesetzt)
try:
    import streamlit as st
    _SECRETS = st.secrets  # type: ignore
except Exception:
    _SECRETS = {}

def _get(key: str, default: str | None = None) -> str | None:
    # Reihenfolge: ENV > Streamlit Secrets > Default
    return os.environ.get(key) or _SECRETS.get(key) or default

# ---- OpenAI / Modelle ----
OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = _get("OPENAI_BASE_URL", "")  # bei OpenAI leer lassen
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
    p.mkdir(parents=True, exist_ok=True)