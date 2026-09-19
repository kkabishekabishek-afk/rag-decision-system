from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from src.rag.loader import load_pdf


def chunk_documents(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks


if __name__ == "__main__":

    pdf_folder = Path(
        "data/documents"
    )

    pdf_files = list(
        pdf_folder.glob("*.pdf")
    )

    if not pdf_files:

        print(
            "No PDF found."
        )

        exit()

    pages = load_pdf(
        pdf_files[0]
    )

    chunks = chunk_documents(
        pages
    )

    print(
        f"\nTotal chunks: {len(chunks)}"
    )

    for i, chunk in enumerate(
        chunks[:5],
        start=1
    ):

        print(
            f"\n--- Chunk {i} ---"
        )

        print(
            f"Page: {chunk['page']}"
        )

        print(
            chunk["text"]
        )