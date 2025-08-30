import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from settings import PDF_DIR, PERSIST_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from embeds import make_embeddings_with_fallback

def collect_pdfs(pdf_dir: str) -> List[str]:
    p = Path(pdf_dir)
    return [str(f) for f in sorted(p.glob("**/*.pdf"))]

def load_and_split(paths: List[str]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    all_docs = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Metadaten setzen (Quelle/Seite)
        for d in docs:
            meta = d.metadata or {}
            meta["source"] = os.path.basename(path)
            d.metadata = meta
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)
    return all_docs

def build_vectorstore(docs):
    """
    Erstellt/aktualisiert Chroma mit OpenAI-Embeddings; fällt bei
    Verbindungsproblemen automatisch auf lokale Embeddings zurück.
    """
    embeddings, used_local = make_embeddings_with_fallback(OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL)
    if used_local:
        print("⚠️  Hinweis: OpenAI-Embeddings nicht erreichbar – nutze lokalen Fallback (all-MiniLM-L6-v2).")
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vs.persist()
    return vs

def main():
    print(f"PDF-Verzeichnis: {PDF_DIR}")
    pdfs = collect_pdfs(PDF_DIR)
    if not pdfs:
        print("Keine PDFs gefunden. Lege Dateien in 'data/pdfs' ab.")
        return
    print(f"{len(pdfs)} PDF(s) gefunden. Starte Ingest ...")
    docs = load_and_split(pdfs)
    print(f"{len(docs)} Text-Chunks erzeugt. Erstelle Vektorindex ...")
    build_vectorstore(docs)
    print(f"Fertig. Persistenzordner: {PERSIST_DIR}")

if __name__ == "__main__":
    main()