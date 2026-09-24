from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.rag.llm_client import call_llm

OLLAMA_MODEL = "llama3.2:latest"


def decision_agent(
    question,
    retrieved_documents,
    analysis,
    risk_analysis,
    solution
):

    evidence = ""

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        evidence += f"""
SOURCE {i}

Document:
{document.get("source", "Unknown")}

Page:
{document.get("page", "Unknown")}

Content:
{document.get("text", "")}

--------------------------------
"""


    prompt = f"""
You are the Decision Agent in a
general-purpose document-grounded
decision-support system.

Your job is to directly answer the
user's question using:

1. Document evidence
2. Explicit user-provided information
3. Calculations required by the question

Do NOT require every value in the question
to appear in the document.

==================================================
USER QUESTION
==================================================

{question}


==================================================
DOCUMENT EVIDENCE
==================================================

{evidence}


==================================================
ANALYSIS
==================================================

{analysis}


==================================================
RISK ANALYSIS
==================================================

{risk_analysis}


==================================================
POSSIBLE SOLUTIONS
==================================================

{solution}


==================================================
IMPORTANT RULES
==================================================

1. DOCUMENT FACTS

Use the document as the source for facts
about the subject of the document.


2. USER INPUT

If the user explicitly provides a value
in the question, such as:

- price
- EMI
- salary
- cost
- loan amount
- interest rate
- duration
- budget
- quantity

you MAY use that value in the reasoning.

Do NOT say:

"The value is not in the document"

when the value was explicitly supplied
by the user.


3. CALCULATIONS

If the user asks a question requiring
calculation, perform the calculation.

Examples:

monthly amount × 12

annual amount ÷ 12

revenue - expense

profit / revenue

loan amount = price - down payment

Do arithmetic carefully.


4. TIME PERIOD

Pay close attention to:

annual
monthly
quarterly
weekly
daily

Never silently change the period.

If the document does not establish the
period required for the decision, clearly
state the uncertainty.


5. DO NOT INVENT NUMBERS

Only use:

- numbers from the document
- numbers explicitly supplied by the user

Do not invent missing values.


6. MISSING INFORMATION

Missing information is not negative evidence.

Do not say:

"X does not exist."

Instead say:

"The available information does not
establish X."


7. DIRECTLY ANSWER THE QUESTION

Do not produce a generic essay.

The first part of the answer should
directly answer the user's question.


8. EXPLAIN THE CALCULATION

When a calculation is required, show
the important calculation briefly.


9. DISTINGUISH CONFIDENCE

If the decision depends on an unstated
assumption, explicitly identify it.


10. DOCUMENT-ONLY FACTUAL GROUNDING

Do not introduce external financial,
medical, legal, technical or other
domain facts unless they are supplied
by the document or user.


==================================================
DECISION PROCESS
==================================================

STEP 1:
Understand exactly what the user wants.

STEP 2:
Extract relevant document values.

STEP 3:
Extract explicit user-provided values.

STEP 4:
Determine whether calculation is required.

STEP 5:
Perform the calculation if the required
values are available.

STEP 6:
Check units and time periods.

STEP 7:
Determine whether the available evidence
supports a clear answer.

STEP 8:
If something essential is missing,
state exactly what is missing.

==================================================
OUTPUT FORMAT
==================================================

ANSWER:

Give the direct answer first.

CALCULATION:

Show the relevant calculation if required.

DOCUMENT EVIDENCE:

List the document facts used.

USER INPUT:

List important values supplied by the user.

UNCERTAINTY:

Only list genuinely missing information.

CONCLUSION:

Give a short direct conclusion.

==================================================
EXAMPLE
==================================================

If the document says:

Annual net income = $32,000

and the user asks:

"Can I buy a laptop with an EMI of
$13,300 per month?"

The correct reasoning is:

Annual EMI = $13,300 × 12
Annual EMI = $159,600

Then compare the relevant periods.

Do NOT reject the question merely because
$13,300 is not written in the document.

However, if the document does not establish
whether $32,000 is annual or monthly,
state that this affects the conclusion.

==================================================
FINAL RULE
==================================================

Be direct.

Do not repeat the same evidence multiple times.

Do not produce unnecessary sections.

Answer the actual question.
"""


    return call_llm(
        prompt,
        model=OLLAMA_MODEL,
        temperature=0.0,
        agent_type="DECISION"
    )


if __name__ == "__main__":

    print(
        "Decision Agent loaded successfully."
    )