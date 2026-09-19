import os
import shutil
import tempfile
from pathlib import Path
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "documents"

def get_writable_chroma_path():
    """
    Returns a guaranteed read-write path for ChromaDB.
    On Streamlit Cloud / read-only filesystems, uses /tmp and copies seed database.
    """
    try:
        DEFAULT_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        test_file = DEFAULT_CHROMA_DIR / ".write_test"
        with open(test_file, "w") as f:
            f.write("test")
        test_file.unlink()
        return str(DEFAULT_CHROMA_DIR)
    except Exception:
        pass

    # Writable fallback (e.g. Streamlit Cloud /tmp)
    fallback_dir = Path(tempfile.gettempdir()) / "rag_chroma_storage"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy pre-existing seed database if present
    if DEFAULT_CHROMA_DIR.exists() and not (fallback_dir / "chroma.sqlite3").exists():
        try:
            shutil.copytree(DEFAULT_CHROMA_DIR, fallback_dir, dirs_exist_ok=True)
        except Exception:
            pass
            
    return str(fallback_dir)

def get_chroma_client_and_collection():
    """Returns (client, collection, chroma_path)."""
    chroma_path = get_writable_chroma_path()
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return client, collection, chroma_path
