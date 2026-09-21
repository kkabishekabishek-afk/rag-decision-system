import chromadb

from sentence_transformers import (
    SentenceTransformer
)


EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)

CHROMA_PATH = "data/chroma"

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


client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
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