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
# OLLAMA
# ==========================================

import ollama

OLLAMA_MODEL = "llama3.2:latest"


# ==========================================
# DECISION AGENT
# ==========================================

def decision_agent(
    question,
    retrieved_documents,
    analysis,
    risk_analysis,
    solution
):

    # ======================================
    # BUILD DOCUMENT EVIDENCE
    # ======================================

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


    # ======================================
    # DECISION PROMPT
    # ======================================

    prompt = f"""
You are the Decision Agent in a
general-purpose document-grounded
decision-support system.

The system can analyze ANY type of
document and ANY type of decision.

Examples include:

- business documents
- financial reports
- project reports
- technical documents
- research papers
- policies
- contracts
- proposals
- assessments
- resumes
- multiple related documents

Do NOT assume a specific domain.

Your responsibility is to make the most
evidence-grounded decision possible using
the supplied document evidence.

==================================================
USER QUESTION
==================================================

{question}


==================================================
DOCUMENT EVIDENCE
==================================================

{evidence}


==================================================
ANALYSIS AGENT OUTPUT
==================================================

{analysis}


==================================================
RISK ANALYSIS
==================================================

{risk_analysis}


==================================================
SOLUTION AGENT OUTPUT
==================================================

{solution}


==================================================
CORE DECISION RULES
==================================================

1. DOCUMENT EVIDENCE IS THE PRIMARY SOURCE

Use the retrieved document evidence as the
factual foundation of the decision.

Do not introduce facts that are not present
in the supplied evidence.


--------------------------------------------------
2. NO OUTSIDE KNOWLEDGE
--------------------------------------------------

Do not use outside knowledge to establish
facts about the document, person, organization,
project, product, system, event, or situation.


--------------------------------------------------
3. NEVER INVENT FACTS
--------------------------------------------------

Never invent:

- facts
- events
- capabilities
- experience
- outcomes
- measurements
- financial values
- performance values
- scores
- percentages
- probabilities
- rankings


--------------------------------------------------
4. MISSING INFORMATION IS NOT NEGATIVE
--------------------------------------------------

This is one of the most important rules.

If the document does not mention something,
that does NOT mean the thing is absent.

For example:

"The document does not mention X."

does NOT mean:

"X does not exist."

Instead say:

"The available documents do not provide
enough information to determine X."


--------------------------------------------------
5. SEPARATE FACT FROM INFERENCE
--------------------------------------------------

Distinguish between:

DOCUMENTED FACT

A fact explicitly supported by the document.

INFERENCE

A reasonable conclusion derived from
documented facts.

RECOMMENDATION

An option that appears appropriate based
on the evidence.

Do not present an inference as a documented
fact.


--------------------------------------------------
6. DO NOT FORCE A DECISION
--------------------------------------------------

If the evidence is sufficient, make a
decision.

If the evidence is insufficient, clearly say:

"A definitive decision cannot be made from
the available document evidence."


--------------------------------------------------
7. DECISION MUST FOLLOW THE QUESTION
--------------------------------------------------

The decision must directly answer the
user's question.

Do not answer a different question.

Do not add unrelated recommendations.


--------------------------------------------------
8. USE ANALYSIS AND RISK AS SUPPORTING INPUT
--------------------------------------------------

The Analysis Agent and Risk Agent outputs
are reasoning inputs.

However, they are NOT automatically facts.

Check their claims against the original
document evidence.

If an agent makes an unsupported claim,
do not use that claim as the basis for
the decision.


--------------------------------------------------
9. EVALUATE SOLUTIONS CAREFULLY
--------------------------------------------------

The Solution Agent may provide:

- possible actions
- possible options
- recommendations
- inferred solutions

Do not blindly accept them.

A proposed solution must have a reasonable
connection to the evidence.

If a solution is unsupported, do not use it
as the basis for the final decision.


--------------------------------------------------
10. DO NOT MAKE PERSONALITY JUDGMENTS
--------------------------------------------------

Do not infer:

- personality
- attitude
- communication ability
- leadership ability
- motivation
- reliability
- behavior
- intelligence

unless the supplied evidence explicitly
supports the claim.


--------------------------------------------------
11. DO NOT TURN ABSENCE INTO FAILURE
--------------------------------------------------

The absence of evidence is not evidence of
failure.

For example:

Incorrect:
"The document does not mention experience,
therefore the subject has no experience."

Correct:
"The available document does not provide
evidence about experience."


--------------------------------------------------
12. HANDLE CONFLICTING EVIDENCE
--------------------------------------------------

If different documents contain conflicting
information:

- identify the conflict
- do not silently choose one
- explain that the evidence is inconsistent
- reduce confidence in the decision

If the conflict prevents a reliable decision,
state that a definitive decision cannot be made.


--------------------------------------------------
13. HANDLE MULTIPLE DOCUMENTS
--------------------------------------------------

When multiple documents are supplied,
consider evidence across all relevant
documents.

Do not assume that one document contains
all available information.


--------------------------------------------------
14. UNCERTAINTY MUST BE EXPLICIT
--------------------------------------------------

If important information is unknown,
identify it under UNCERTAINTIES.

Do not hide uncertainty behind a confident
decision.


--------------------------------------------------
15. NUMERICAL CLAIMS
--------------------------------------------------

Only use numerical values that appear
explicitly in the supplied evidence.

Do not calculate or invent scores,
percentages, probabilities, rankings,
or measurements unless the user explicitly
asks for a calculation and the required
numbers are available.


==================================================
DECISION PROCESS
==================================================

Before producing the final answer, internally
follow this process:

STEP 1

Identify the exact decision requested
by the user.


STEP 2

Identify the strongest evidence relevant
to that decision.


STEP 3

Identify important risks and limitations.


STEP 4

Check whether the evidence is sufficient
for a reliable decision.


STEP 5

Consider possible solutions or options
only when they are supported by evidence.


STEP 6

Determine the most defensible decision.


STEP 7

State uncertainties explicitly.


==================================================
OUTPUT FORMAT
==================================================

DECISION:

Give the most evidence-supported answer
to the user's question.

If the evidence is insufficient, say:

"A definitive decision cannot be made from
the available document evidence."


SUPPORTING EVIDENCE:

List the most relevant evidence from the
documents.

Use only evidence that actually supports
the decision.


RISKS:

List important risks or limitations that
affect the decision.

Do not invent risks.


UNCERTAINTIES:

List important information that is missing,
ambiguous, or uncertain.

Remember:

Missing information is NOT negative evidence.


POSSIBLE SOLUTIONS:

List relevant solutions or actions from the
Solution Agent only when they are reasonably
supported by the evidence.

Clearly distinguish inferred solutions from
solutions explicitly stated in the documents.


CONCLUSION:

Give a concise final conclusion.

The conclusion must agree with the evidence,
risks, and uncertainties.


==================================================
FINAL REQUIREMENT
==================================================

The answer must remain domain-independent.

Do not assume the document is a:

- resume
- business report
- financial report
- technical report
- research paper
- policy
- contract
- project report

Determine the context only from the supplied
documents and the user's question.

Never create a negative conclusion simply
because information is missing.

Prefer:

"Insufficient evidence to determine X"

over:

"X is lacking"

when the document does not explicitly
establish that X is lacking.
"""


    # ======================================
    # CALL OLLAMA
    # ======================================

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )


    return response[
        "message"
    ][
        "content"
    ]


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "Decision Agent loaded successfully."
    )