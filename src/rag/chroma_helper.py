import os
import sys
import shutil
import tempfile
from pathlib import Path
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_NAME = "documents"

def get_writable_chroma_path():
    """Returns guaranteed-writable Chroma directory in process temp storage."""
    tmp_chroma = Path(tempfile.gettempdir()) / "rag_chroma_store_v2"
    tmp_chroma.mkdir(parents=True, exist_ok=True)
    return str(tmp_chroma)

def get_writable_documents_dir():
    """Returns guaranteed-writable documents directory for PDF uploads."""
    tmp_docs = Path(tempfile.gettempdir()) / "rag_decision_documents_v2"
    tmp_docs.mkdir(parents=True, exist_ok=True)
    
    # Sync initial sample documents from repo if available
    repo_docs = PROJECT_ROOT / "data" / "documents"
    if repo_docs.exists():
        for pdf in repo_docs.glob("*.pdf"):
            dst = tmp_docs / pdf.name
            if not dst.exists():
                try:
                    shutil.copy2(pdf, dst)
                except Exception:
                    pass
    return tmp_docs

_client = None
_collection = None

def get_chroma_client_and_collection():
    """Singleton getter for Chroma client and collection in writable storage."""
    global _client, _collection
    if _client is None or _collection is None:
        c_path = get_writable_chroma_path()
        _client = chromadb.PersistentClient(path=c_path)
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _client, _collection
