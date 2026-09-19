from pathlib import Path
import sys


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ==========================================
# RAG SERVICE
# ==========================================

from src.rag.rag_service import retrieve


# ==========================================
# RETRIEVER AGENT
# ==========================================

def retriever_agent(
    question,
    source
):

    print(
        "\n[RETRIEVER AGENT]"
    )

    print(
        f"Searching only: {source}"
    )


    # ======================================
    # RETRIEVE FROM ACTIVE DOCUMENT ONLY
    # ======================================

    documents = retrieve(
        question,
        source=source
    )


    print(
        f"Retrieved {len(documents)} "
        "relevant chunks."
    )

    return documents


# ==========================================
# CLI TEST
# ==========================================

if __name__ == "__main__":

    question = input(
        "\nAsk the Retriever Agent: "
    )

    source = input(
        "Enter PDF filename: "
    )


    results = retriever_agent(
        question,
        source
    )


    print(
        "\n=============================="
    )

    print(
        "RETRIEVED INFORMATION"
    )

    print(
        "=============================="
    )


    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            f"Page: {result['page']}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            result["text"]
        )