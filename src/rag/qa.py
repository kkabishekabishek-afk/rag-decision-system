import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:latest"

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "documents"

TOP_K = 3


# ==========================================
# LOAD EMBEDDING MODEL ONCE
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ==========================================
# LOAD CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ==========================================
# RETRIEVE DOCUMENTS
# ==========================================

def retrieve_documents(question):

    # Convert question into embedding
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K
    )

    return results


# ==========================================
# CREATE CONTEXT
# ==========================================

def create_context(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
Page: {metadata['page']}
Document: {metadata['source']}

Content:
{document}
"""
        )

    return "\n".join(context_parts), metadatas


# ==========================================
# GENERATE ANSWER USING OLLAMA
# ==========================================

def generate_answer(question, context):

    prompt = f"""
You are a precise document question-answering assistant.

Answer the user's question using ONLY the information
contained in the document context.

RULES:

1. Do not invent information.

2. Do not use outside knowledge.

3. If the document contains a list, include ALL items
   that directly answer the question.

4. Never omit an item from a list.

5. If the answer is not present in the context, say:
"I could not find this information in the document."

6. Keep the answer concise.

DOCUMENT CONTEXT:
========================
{context}
========================

USER QUESTION:
{question}

FINAL ANSWER:
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ==========================================
# MAIN PROGRAM
# ==========================================

print("\n======================================")
print("       LOCAL PDF RAG SYSTEM")
print("======================================")

print("\nEmbedding model : all-MiniLM-L6-v2")
print("LLM             : llama3.2:latest")
print("Vector database : ChromaDB")

print("\nSystem ready.")
print("Type 'exit' to stop.")


while True:

    question = input(
        "\nAsk a question about the PDF: "
    )

    # Stop program
    if question.lower().strip() == "exit":
        print("\nExiting...")
        break

    if not question.strip():
        print("Please enter a question.")
        continue

    # ======================================
    # RETRIEVAL
    # ======================================

    results = retrieve_documents(
        question
    )

    # ======================================
    # CREATE CONTEXT
    # ======================================

    context, metadatas = create_context(
        results
    )

    # ======================================
    # GENERATE ANSWER
    # ======================================

    print("\nGenerating answer...")

    answer = generate_answer(
        question,
        context
    )

    # ======================================
    # DISPLAY ANSWER
    # ======================================

    print("\n======================================")
    print("             FINAL ANSWER")
    print("======================================")

    print(answer)

    # ======================================
    # DISPLAY SOURCES
    # ======================================

    pages = sorted(
        set(
            metadata["page"]
            for metadata in metadatas
        )
    )

    print("\nSource pages:", pages)