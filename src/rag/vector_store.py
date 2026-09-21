from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

import chromadb

from src.rag.loader import load_pdf
from src.rag.chunker import chunk_documents
from src.rag.embeddings import create_embeddings


from src.rag.chroma_helper import get_chroma_client_and_collection, get_writable_chroma_path, COLLECTION_NAME

CHROMA_PATH = get_writable_chroma_path()
client, collection = get_chroma_client_and_collection()



# ==================================================
# INDEX ONE PDF
# ==================================================

def index_pdf(pdf_path):
    global client, collection
    client, collection = get_chroma_client_and_collection()
    pdf_path = Path(pdf_path)


    print(
        f"\nLoading: {pdf_path.name}"
    )

    # ----------------------------------------------
    # LOAD PDF
    # ----------------------------------------------

    pages = load_pdf(
        pdf_path
    )

    if not pages:

        raise ValueError(
            "No readable text found in the PDF."
        )

    print(
        f"Extracted {len(pages)} pages."
    )


    # ----------------------------------------------
    # CHUNK DOCUMENT
    # ----------------------------------------------

    chunks = chunk_documents(
        pages
    )

    if not chunks:

        raise ValueError(
            "No text chunks were created."
        )

    print(
        f"Created {len(chunks)} chunks."
    )


    # ----------------------------------------------
    # REMOVE OLD VERSION OF SAME DOCUMENT
    # ----------------------------------------------

    existing = collection.get(
        where={
            "source": pdf_path.name
        }
    )

    existing_ids = existing.get(
        "ids",
        []
    )

    if existing_ids:

        collection.delete(
            ids=existing_ids
        )

        print(
            f"Removed {len(existing_ids)} old chunks."
        )


    # ----------------------------------------------
    # CREATE EMBEDDINGS
    # ----------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "Creating embeddings..."
    )

    embeddings = create_embeddings(
        texts
    )


    # ----------------------------------------------
    # PREPARE CHROMADB DATA
    # ----------------------------------------------

    ids = []

    documents = []

    metadatas = []

    for i, chunk in enumerate(
        chunks
    ):

        ids.append(
            f"{pdf_path.stem}_chunk_{i}"
        )

        documents.append(
            chunk["text"]
        )

        metadatas.append({

            "page": str(
                chunk["page"]
            ),

            "source": pdf_path.name

        })


    # ----------------------------------------------
    # STORE IN CHROMADB
    # ----------------------------------------------

    collection.add(

        ids=ids,

        documents=documents,

        embeddings=[
            embedding.tolist()
            for embedding in embeddings
        ],

        metadatas=metadatas
    )


    print(
        "\nDocument indexed successfully."
    )

    print(
        f"Document : {pdf_path.name}"
    )

    print(
        f"Chunks   : {len(chunks)}"
    )

    print(
        f"Database : {CHROMA_PATH}"
    )

    return {
        "source": pdf_path.name,
        "pages": len(pages),
        "chunks": len(chunks)
    }


# ==================================================
# MAIN TEST
# ==================================================

if __name__ == "__main__":

    pdf_folder = (
        PROJECT_ROOT
        / "data"
        / "documents"
    )

    pdf_files = list(
        pdf_folder.glob("*.pdf")
    )

    if not pdf_files:

        print(
            "No PDF found."
        )

        sys.exit()


    for pdf_path in pdf_files:

        index_pdf(
            pdf_path
        )