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


from src.rag.llm_client import call_llm

OLLAMA_MODEL = "llama3.2:latest"


# ==========================================
# RISK AGENT
# ==========================================

def risk_agent(
    question,
    retrieved_documents,
    analysis
):

    print(
        "\n[RISK AGENT]"
    )

    print(
        "Analyzing risks and uncertainty..."
    )


    # ======================================
    # DOCUMENT CONTEXT
    # ======================================

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


    # ======================================
    # PROMPT
    # ======================================

    prompt = f"""
You are the Risk Analysis Agent in a
general-purpose document-grounded
RAG decision-support system.

The system can analyze ANY type of
document.

Do not assume the document is a resume,
business report, financial report,
technical document, research paper,
policy, contract, or any other specific
type.

Your job is ONLY to identify genuine
risks, limitations, missing information,
and uncertainty that affect the user's
question or decision.

==================================================
USER QUESTION
==================================================

{question}


==================================================
DOCUMENT EVIDENCE
==================================================

{context}


==================================================
ANALYSIS AGENT OUTPUT
==================================================

{analysis}


==================================================
STRICT RULES
==================================================

1. Use ONLY the supplied document evidence
   and the analysis derived from it.

2. Do NOT use outside knowledge.

3. Do NOT invent facts.

4. Do NOT give solutions or recommendations.

5. Do NOT treat missing information as
   a negative fact.

6. Distinguish between:

   DOCUMENTED FACT
   Information explicitly stated in the
   document.

   MISSING INFORMATION
   Information relevant to the question
   that is not provided.

   ACTUAL RISK
   A documented fact that creates a genuine
   limitation, problem, or risk.

   UNCERTAINTY
   Something that cannot be reliably
   determined from the available evidence.

7. If information is unavailable, say:

   "The available documents do not provide
   enough information to determine this."

8. Never assume that missing information
   means something does not exist.

9. Do not make unsupported assumptions about
   people, organizations, systems, projects,
   performance, capabilities, resources,
   finances, or outcomes.

10. Do not create numerical ratings,
    scores, percentages, probabilities,
    rankings, or measurements.

11. Only identify a risk when there is a
    reasonable connection between the
    evidence and the user's question.

12. Do not force a risk simply because the
    system expects one.

13. If no meaningful risk can be identified,
    explicitly say:

    "No meaningful risk identified from
    the available document evidence."

14. For financial, loan, or purchasing decisions:
    - Evaluate Debt-to-Income (DTI) impact and monthly repayment affordability.
    - Evaluate cash flow insolvency if monthly EMI exceeds or heavily burdens net income.
    - Note risks of operational costs (maintenance, fuel, interest rates) and lack of emergency buffers.

15. Keep the analysis concise, structured, and evidence-grounded.

==================================================
OUTPUT FORMAT
==================================================


DOCUMENTED FACTS:

- List only important facts relevant
  to the risk analysis.


MISSING INFORMATION:

- List important information that is
  unavailable.

- Do NOT describe missing information
  as a negative fact.


ACTUAL RISKS:

- List only genuine risks supported
  by the evidence.

- If none exist, state:

  "No meaningful risk identified from
  the available document evidence."


UNCERTAINTY:

- Explain what cannot be determined
  reliably from the available evidence.


RISK ANALYSIS:

Give a short overall explanation of
how the identified risks and uncertainties
affect the user's question.

==================================================
FINAL RULE
==================================================

Do not provide solutions.

Do not provide recommendations.

Do not make the final decision.

Those tasks belong to later agents.
"""


    # ======================================
    # CALL OLLAMA
    # ======================================

    return call_llm(prompt, model=OLLAMA_MODEL, temperature=0.0)


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "Risk Agent loaded successfully."
    )