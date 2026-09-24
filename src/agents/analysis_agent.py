from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.rag.llm_client import call_llm

OLLAMA_MODEL = "llama3.2:latest"


def analysis_agent(
    question,
    retrieved_documents
):

    print(
        "\n[ANALYSIS AGENT]"
    )

    context = ""

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context += f"""
SOURCE {i}

Page: {document.get("page", "Unknown")}

Source: {document.get("source", "Unknown")}

{document.get("text", "")}

--------------------------------
"""


    prompt = f"""
You are the Analysis Agent in a
general-purpose document-grounded
decision-support system.

Analyze the user's question together
with the retrieved document evidence.

USER QUESTION:

{question}


DOCUMENT EVIDENCE:

{context}


IMPORTANT RULES:

1. Use the document as the factual source.

2. The user's question may contain additional
   values or conditions that are NOT in the
   document.

3. Do NOT reject a question simply because
   a value supplied by the user is not in
   the document.

4. Treat explicit values supplied in the
   user's question as USER-PROVIDED INPUT.

5. Clearly distinguish:

   DOCUMENT FACTS
   USER-PROVIDED INPUT
   MISSING INFORMATION
   INFERENCES

6. Extract important numerical values when
   present.

7. Preserve units.

8. Preserve the time period of values when
   the document provides one.

9. If the document says a value is annual,
   monthly, quarterly, etc., preserve that.

10. Do NOT assume a missing time period.

11. If the question requires arithmetic,
    identify the values required for the
    calculation.

12. Calculations may use:
    - document values
    - explicit user-provided values

13. Do not invent missing numbers.

14. Missing information is not negative evidence.

15. Do not make the final decision.

16. Keep the analysis concise.

OUTPUT FORMAT:

DOCUMENT FACTS:

- Relevant facts from the document.

USER-PROVIDED INPUT:

- Values or conditions explicitly supplied
  in the question.

CALCULATION REQUIREMENTS:

- Explain what calculation is required,
  if any.

MISSING INFORMATION:

- Information required but genuinely unavailable.

REASONING:

- Explain how the document facts and user
  inputs relate to the question.

Do not provide a final recommendation.
"""

    return call_llm(
        prompt,
        model=OLLAMA_MODEL,
        temperature=0.0,
        agent_type="ANALYSIS"
    )


if __name__ == "__main__":

    print(
        "Analysis Agent loaded successfully."
    )