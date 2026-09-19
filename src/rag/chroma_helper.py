import os
import shutil
import tempfile
from pathlib import Path
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "data" / "documents"
COLLECTION_NAME = "documents"

def make_writable_recursively(path):
    """Ensure all directories and files inside path have full read-write permissions."""
    try:
        os.chmod(path, 0o777)
    except Exception:
        pass
    for root, dirs, files in os.walk(path):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o777)
            except Exception:
                pass
        for f in files:
            try:
                os.chmod(os.path.join(root, f), 0o666)
            except Exception:
                pass

def get_writable_chroma_path():
    """
    Returns a guaranteed read-write path for ChromaDB.
    Always creates a fully writable directory in temp storage to avoid read-only mount locks.
    """
    fallback_dir = Path(tempfile.gettempdir()) / "rag_chroma_storage"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy pre-existing seed database if present and not yet initialized
    if DEFAULT_CHROMA_DIR.exists() and not (fallback_dir / "chroma.sqlite3").exists():
        try:
            shutil.copytree(DEFAULT_CHROMA_DIR, fallback_dir, dirs_exist_ok=True)
        except Exception:
            pass
            
    make_writable_recursively(fallback_dir)
    return str(fallback_dir)

def get_chroma_client_and_collection():
    """Returns (client, collection, chroma_path)."""
    chroma_path = get_writable_chroma_path()
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return client, collection, chroma_path

def get_documents_dir():
    """
    Returns a guaranteed read-write directory for storing uploaded PDFs.
    """
    docs_dir = Path(tempfile.gettempdir()) / "rag_documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy default documents from repository if available
    if DEFAULT_DOCS_DIR.exists():
        for pdf_file in DEFAULT_DOCS_DIR.glob("*.pdf"):
            dest_file = docs_dir / pdf_file.name
            if not dest_file.exists():
                try:
                    shutil.copy2(pdf_file, dest_file)
                except Exception:
                    pass
                    
    make_writable_recursively(docs_dir)
    return docs_dir
