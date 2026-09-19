import chromadb

from sentence_transformers import (
    SentenceTransformer
)


EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)

from src.rag.chroma_helper import get_chroma_client_and_collection

client, collection, CHROMA_PATH = get_chroma_client_and_collection()
COLLECTION_NAME = "documents"
TOP_K = 3

print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Embedding model loaded."
)


def retrieve(question):

    question_embedding = (
        embedding_model
        .encode(question)
        .tolist()
    )

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=TOP_K
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    retrieved = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        retrieved.append({

            "text": document,

            "page": metadata["page"],

            "source": metadata["source"]

        })

    return retrieved