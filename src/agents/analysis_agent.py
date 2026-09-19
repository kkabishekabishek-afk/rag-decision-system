from pathlib import Path
import sys

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.rag.llm_client import call_llm

OLLAMA_MODEL = (
    "llama3.2:latest"
)


def analysis_agent(
    question,
    retrieved_documents
):

    print(
        "\n[ANALYSIS AGENT]"
    )

    print(
        "Analyzing retrieved information..."
    )

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

    prompt = f"""
You are the Analysis Agent in a general-purpose document-grounded RAG decision support system.
Analyze ONLY the retrieved document context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

IMPORTANT RULES:
1. Use ONLY information present in the retrieved context.
2. Do NOT use outside knowledge or invent numbers/facts.
3. Extract key facts, metrics, figures, financial values, entity details, and evidence directly from the document.
4. For financial, loan, or income documents:
   - Identify stated income (Gross, Net, Total), revenues, expenses, liabilities, and reporting periods (Annual, Monthly, Quarterly).
   - If annual figures are given, explicitly compute and state the monthly equivalent (e.g., Annual Income / 12 = Monthly Income).
5. For professional profiles or resumes:
   - Extract explicitly listed technical skills, qualifications, tools, and project evidence objectively.
6. Clearly separate facts directly stated in the document from derived numerical calculations.
7. If requested information is not available in the context, clearly state that it is unavailable.

Provide a concise, clearly formatted, and structured analysis.

ANALYSIS:
"""

    return call_llm(prompt, model=OLLAMA_MODEL, temperature=0.0)