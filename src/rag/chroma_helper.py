import os
import sys
import shutil
import tempfile
from pathlib import Path
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_NAME = "documents"

def is_dir_writable(path):
    """Check if directory is writable."""
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / f".test_write_{os.getpid()}"
        test_file.write_text("ok")
        if test_file.exists():
            test_file.unlink()
        return True
    except Exception:
        return False

def get_writable_chroma_path():
    """Returns guaranteed-writable Chroma directory (handles Streamlit Cloud read-only mounts)."""
    repo_chroma = PROJECT_ROOT / "data" / "chroma"
    if is_dir_writable(repo_chroma):
        return str(repo_chroma)
    
    tmp_chroma = Path(tempfile.gettempdir()) / "rag_decision_chroma"
    tmp_chroma.mkdir(parents=True, exist_ok=True)
    
    # Copy baseline chroma files if needed
    if repo_chroma.exists() and not (tmp_chroma / "chroma.sqlite3").exists():
        try:
            for item in repo_chroma.glob("*"):
                if item.is_dir():
                    shutil.copytree(item, tmp_chroma / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, tmp_chroma / item.name)
        except Exception:
            pass
            
    return str(tmp_chroma)

def get_writable_documents_dir():
    """Returns guaranteed-writable documents directory for PDF uploads."""
    repo_docs = PROJECT_ROOT / "data" / "documents"
    if is_dir_writable(repo_docs):
        return repo_docs
    tmp_docs = Path(tempfile.gettempdir()) / "rag_decision_documents"
    tmp_docs.mkdir(parents=True, exist_ok=True)
    return tmp_docs

_client = None
_collection = None

def get_chroma_client_and_collection():
    """Singleton getter for Chroma client and collection."""
    global _client, _collection
    if _client is None or _collection is None:
        c_path = get_writable_chroma_path()
        _client = chromadb.PersistentClient(path=c_path)
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _client, _collection
