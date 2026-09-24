from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.rag.llm_client import call_llm

OLLAMA_MODEL = "llama3.2:latest"


def build_context(
    retrieved_documents
):

    context = ""

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context += f"""
SOURCE {i}

Page:
{document.get("page", "Unknown")}

Source:
{document.get("source", "Unknown")}

Content:
{document.get("text", "")}

--------------------------------
"""

    return context


def verification_agent(
    question,
    retrieved_documents,
    decision,
    analysis=None,
    risk=None,
    solution=None
):

    print(
        "\n[VERIFICATION AGENT]"
    )

    context = build_context(
        retrieved_documents
    )

    prompt = f"""
You are the Verification Agent.

Verify whether the proposed answer correctly
answers the user's question.

USER QUESTION:

{question}


DOCUMENT:

{context}


ANALYSIS:

{analysis or ""}


PROPOSED ANSWER:

{decision}


RULES:

1. Check factual claims against the document.

2. User-provided values are allowed.

3. Do not flag a value merely because it
   does not appear in the document if the
   user explicitly supplied it.

4. Check arithmetic.

5. Check units and time periods.

6. Check whether the answer directly answers
   the question.

7. Missing information is not negative evidence.

8. Do not require the document to contain
   information explicitly supplied by the user.

9. Flag hallucinated numbers.

10. Flag unsupported conclusions.

11. If the answer is supported, return:

STATUS: VERIFIED

ISSUES:
None

CORRECTION:
None

12. If an important error exists, return:

STATUS: NEEDS_CORRECTION

ISSUES:
- Explain the exact problem.

CORRECTION:
- Explain what must be corrected.

Do not rewrite the entire answer.
"""

    return call_llm(
        prompt,
        model=OLLAMA_MODEL,
        temperature=0.0,
        agent_type="VERIFICATION"
    )


def revise_answer(
    question,
    retrieved_documents,
    original_answer,
    verification,
    analysis=None,
    risk=None,
    solution=None
):

    print(
        "\n[CORRECTION AGENT]"
    )

    context = build_context(
        retrieved_documents
    )

    prompt = f"""
You are the Correction Agent.

Produce the corrected final answer.

QUESTION:

{question}


DOCUMENT:

{context}


ANALYSIS:

{analysis or ""}


ORIGINAL ANSWER:

{original_answer}


VERIFICATION:

{verification}


RULES:

1. Keep supported facts.

2. Keep explicit user-provided values.

3. Correct arithmetic errors.

4. Correct unit or time-period errors.

5. Remove unsupported claims.

6. Never treat missing information as
   negative evidence.

7. Do not invent numbers.

8. Answer the user's actual question.

9. Be concise.

Return ONLY the corrected final answer.
"""

    return call_llm(
        prompt,
        model=OLLAMA_MODEL,
        temperature=0.0,
        agent_type="CORRECTION"
    )


if __name__ == "__main__":

    print(
        "Verification Agent loaded successfully."
    )