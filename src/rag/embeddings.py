from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Embedding model loaded."
)


def create_embeddings(texts):

    embeddings = embedding_model.encode(
        texts
    )

    return embeddings


if __name__ == "__main__":

    texts = [
        "Abishek is an MCA student.",
        "Abishek knows Python and Java."
    ]

    embeddings = create_embeddings(
        texts
    )

    print(
        f"\nEmbedding size: {len(embeddings[0])}"
    )