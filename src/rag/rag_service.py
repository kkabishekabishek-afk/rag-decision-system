import chromadb

from sentence_transformers import (
    SentenceTransformer
)

from src.rag.llm_client import call_llm


# ==========================================
# CONFIGURATION
# ==========================================

EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)

OLLAMA_MODEL = (
    "llama3.2:latest"
)

CHROMA_PATH = (
    "data/chroma"
)

COLLECTION_NAME = (
    "documents"
)

TOP_K = 3


# ==========================================
# EMBEDDING MODEL
# ==========================================

print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Embedding model loaded."
)


# ==========================================
# CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ==========================================
# RETRIEVER
# ==========================================

def retrieve(
    question,
    source=None
):

    # ======================================
    # CREATE QUESTION EMBEDDING
    # ======================================

    question_embedding = (
        embedding_model
        .encode(question)
        .tolist()
    )


    # ======================================
    # SEARCH CHROMADB
    # ======================================

    if source:

        print(
            f"\nFiltering retrieval to: {source}"
        )

        results = collection.query(

            query_embeddings=[
                question_embedding
            ],

            n_results=TOP_K,

            where={
                "source": source
            }
        )

    else:

        print(
            "\nSearching all documents."
        )

        results = collection.query(

            query_embeddings=[
                question_embedding
            ],

            n_results=TOP_K
        )


    # ======================================
    # CHECK RESULTS
    # ======================================

    if not results["documents"]:
        return []


    if not results["documents"][0]:
        return []


    documents = (
        results["documents"][0]
    )

    metadatas = (
        results["metadatas"][0]
    )


    # ======================================
    # FORMAT RESULTS
    # ======================================

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


# ==========================================
# CREATE CONTEXT
# ==========================================

def create_context(
    retrieved_documents
):

    context = ""


    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context += f"""
SOURCE {i}

Page: {document['page']}

Source: {document['source']}

{document['text']}

--------------------------------
"""


    return context


# ==========================================
# LLM
# ==========================================

def ask_llm(
    question,
    context
):

    prompt = f"""
You are a document-grounded
AI assistant.

Answer the question ONLY using
the supplied document context.

Rules:

1. Do not invent information.

2. Do not use outside knowledge.

3. Include all relevant information.

4. If the answer is not present,
   say:

"I could not find this information
in the document."

5. Treat the supplied context as
the ONLY source of truth.

6. Do not use information from
previous questions or documents.

DOCUMENT CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""


    return call_llm(
        prompt,
        model=OLLAMA_MODEL,
        temperature=0.0
    )



# ==========================================
# SIMPLE QUESTION FUNCTION
# ==========================================

def ask_question(
    question,
    source=None
):

    documents = retrieve(
        question,
        source=source
    )


    context = create_context(
        documents
    )


    answer = ask_llm(
        question,
        context
    )


    return {

        "question": question,

        "answer": answer,

        "sources": documents

    }


# ==========================================
# CLI TEST
# ==========================================

if __name__ == "__main__":

    source = input(
        "\nEnter PDF filename "
        "(leave empty for all documents): "
    ).strip()


    question = input(
        "\nAsk a question: "
    )


    result = ask_question(
        question,
        source=source if source else None
    )


    print(
        "\n=============================="
    )

    print(
        "ANSWER"
    )

    print(
        "=============================="
    )

    print(
        result["answer"]
    )


    print(
        "\n=============================="
    )

    print(
        "SOURCES"
    )

    print(
        "=============================="
    )


    for source in result["sources"]:

        print(
            f"Page {source['page']} - "
            f"{source['source']}"
        )