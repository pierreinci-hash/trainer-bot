# src/ingest.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.faiss import FAISS

from settings import (
    PDF_DIR, PERSIST_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
)

# Kleiner Marker, um zu wissen, welches Backend wir nutzen
BACKEND_MARKER = PERSIST_DIR.parent / "index_backend.txt"  # storage/index_backend.txt
FAISS_DIR = PERSIST_DIR.parent / "faiss"                   # storage/faiss/


def collect_pdfs(pdf_dir: Path | str = PDF_DIR) -> List[str]:
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdfs = [str(p) for p in pdf_dir.glob("*.pdf")]
    return pdfs


def load_and_split(pdfs: List[str]):
    docs = []
    for p in pdfs:
        try:
            loader = PyPDFLoader(p)
            docs.extend(loader.load())
        except Exception as e:
            # PDF kann defekt sein – einfach überspringen
            print(f"[WARN] PDF konnte nicht geladen werden: {p} – {e}")
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # Quelle/Seitennummern sind bereits in metadata (PyPDFLoader setzt 'source' und 'page')
    return chunks


def _write_backend_marker(name: str):
    BACKEND_MARKER.parent.mkdir(parents=True, exist_ok=True)
    with open(BACKEND_MARKER, "w", encoding="utf-8") as f:
        f.write(name.strip())


def _read_backend_marker() -> str | None:
    if not BACKEND_MARKER.exists():
        return None
    try:
        return BACKEND_MARKER.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


def build_vectorstore(docs, embeddings=None) -> Tuple[str, int]:
    """
    Erstellt einen Vektorindex.
    1) Versucht Chroma (persistent).
    2) Fällt bei Fehlern automatisch auf FAISS (Datei-basiert) zurück.
    Gibt (backend_name, anzahl_chunks) zurück.
    """
    n = len(docs) if docs else 0
    if not docs:
        return ("none", 0)

    # --- Versuch A: Chroma persistent ---
    try:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(PERSIST_DIR),
        )
        _write_backend_marker("chroma")
        return ("chroma", n)
    except Exception as e:
        print(f"[WARN] Chroma fehlgeschlagen, versuche FAISS. Grund: {e}")

    # --- Versuch B: FAISS (Fallback, robust in Cloud) ---
    try:
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(str(FAISS_DIR))  # schreibt index-Dateien nach storage/faiss
        _write_backend_marker("faiss")
        return ("faiss", n)
    except Exception as e:
        raise RuntimeError(f"Weder Chroma noch FAISS konnten erstellt werden: {e}")


def load_vectorstore_any(embeddings):
    """
    Lädt den existierenden Index (Chroma oder FAISS), abhängig vom Marker.
    Gibt den VectorStore zurück oder None, falls keiner existiert.
    """
    backend = _read_backend_marker()
    if backend == "chroma" and PERSIST_DIR.exists():
        try:
            return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)
        except Exception as e:
            print(f"[WARN] Chroma laden fehlgeschlagen: {e}")

    if backend == "faiss" and FAISS_DIR.exists():
        try:
            return FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"[WARN] FAISS laden fehlgeschlagen: {e}")

    # Falls kein Marker da ist, aber Chroma/FAISS-Verzeichnisse existieren: heuristisch
    if PERSIST_DIR.exists():
        try:
            return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)
        except Exception:
            pass
    if FAISS_DIR.exists():
        try:
            return FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
        except Exception:
            pass

    return None