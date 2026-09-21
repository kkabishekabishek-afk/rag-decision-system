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
# BUILD DOCUMENT CONTEXT
# ==========================================

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


# ==========================================
# VERIFICATION AGENT
# ==========================================

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

    print(
        "Checking final answer against "
        "document evidence..."
    )


    # ======================================
    # CONTEXT
    # ======================================

    context = build_context(
        retrieved_documents
    )


    analysis_text = (
        analysis
        if analysis
        else "Not provided."
    )

    risk_text = (
        risk
        if risk
        else "Not provided."
    )

    solution_text = (
        solution
        if solution
        else "Not provided."
    )


    # ======================================
    # VERIFICATION PROMPT
    # ======================================

    prompt = f"""
You are the final Evidence Verification
Agent in a general-purpose multi-agent
RAG decision-support system.

Your responsibility is to determine whether
the proposed answer is actually supported
by the supplied document evidence.

The system may analyze ANY domain.

Do not assume the document is a resume,
business report, financial report, technical
document, research paper, policy, contract,
project report, or any other specific type.


==================================================
USER QUESTION
==================================================

{question}


==================================================
DOCUMENT EVIDENCE
==================================================

{context}


==================================================
ANALYSIS AGENT
==================================================

{analysis_text}


==================================================
RISK AGENT
==================================================

{risk_text}


==================================================
SOLUTION AGENT
==================================================

{solution_text}


==================================================
PROPOSED DECISION
==================================================

{decision}


==================================================
CRITICAL VERIFICATION RULES
==================================================

RULE 1 — EVERY FACTUAL CLAIM MUST HAVE
EVIDENCE

For every important factual claim in the
proposed answer, ask:

"Can I find direct evidence for this
claim in the supplied documents?"

If NO:

Flag it as:

UNSUPPORTED CLAIM


--------------------------------------------------
RULE 2 — MISSING INFORMATION IS NOT
NEGATIVE EVIDENCE
--------------------------------------------------

This is mandatory.

If the document does not mention X:

That means:

"X is not established by the available
document evidence."

It does NOT mean:

"X does not exist."

It does NOT mean:

"X is lacking."

It does NOT mean:

"X is weak."

It does NOT mean:

"X is insufficient."


Example:

Document:
No communication information is provided.

Incorrect answer:

"The subject has poor communication skills."

This MUST be flagged.


--------------------------------------------------
RULE 3 — ABSENCE OF EVIDENCE IS NOT
EVIDENCE OF ABSENCE
--------------------------------------------------

Never accept reasoning such as:

"The document does not mention X,
therefore X is absent."

Flag this as:

UNSUPPORTED NEGATIVE INFERENCE


--------------------------------------------------
RULE 4 — INTEREST IS NOT CAPABILITY
--------------------------------------------------

Do not treat statements such as:

"I am interested in X."

"I want to learn X."

"My goal is X."

as proof of:

"Has experience in X."

"Is skilled in X."

"Is proficient in X."

"Can perform X."

If the answer makes this conversion,
flag it.


--------------------------------------------------
RULE 5 — PROJECT/COURSE/KEYWORD CLAIMS
MUST BE SUPPORTED
--------------------------------------------------

A keyword appearing in a document does
not automatically prove proficiency.

Do not accept:

keyword → expertise

interest → expertise

goal → experience

course → professional capability

unless the document provides sufficient
evidence for that conclusion.


--------------------------------------------------
RULE 6 — MISSING EXPERIENCE

If experience is not documented:

Do NOT allow:

"Lacks experience."

Instead:

"The available document does not provide
enough evidence to determine the level of
experience."


--------------------------------------------------
RULE 7 — PERSONAL ATTRIBUTES

Do not infer unsupported:

- communication ability
- teamwork
- leadership
- personality
- motivation
- reliability
- intelligence
- attitude
- work ethic

unless directly supported by evidence.


--------------------------------------------------
RULE 8 — SUITABILITY / ELIGIBILITY /
APPROVAL DECISIONS

If the question asks whether something is:

- suitable
- appropriate
- eligible
- acceptable
- recommended
- ready
- safe
- viable
- qualified

the answer must identify the actual
document evidence supporting that judgment.

Do not accept a positive or negative
decision based only on:

- interest
- generic statements
- missing information
- assumptions
- unrelated evidence.


--------------------------------------------------
RULE 9 — SOLUTIONS

A proposed solution may be logically
inferred from evidence.

However:

The answer must NOT claim:

"The document recommends X"

unless the document actually recommends X.

If X is inferred, it must be presented
as an inferred option.


--------------------------------------------------
RULE 10 — RISKS

A risk must have a reasonable connection
to the evidence.

Do not accept generic risks simply because
they sound reasonable.

For example:

"Communication may be a risk"

is NOT valid merely because communication
information is missing.


--------------------------------------------------
RULE 11 — NUMERICAL CLAIMS

Flag invented:

- scores
- percentages
- rankings
- probabilities
- measurements
- financial values
- performance values

unless directly supported by evidence.


--------------------------------------------------
RULE 12 — OVERCONFIDENCE

Flag conclusions that are stronger than
the evidence.

Example:

Evidence:
"Limited information is available."

Answer:
"This proves the subject is unsuitable."

This is NOT supported.


--------------------------------------------------
RULE 13 — CONTRADICTIONS

Compare the proposed answer with the
documents.

Flag genuine contradictions.

Do not flag simple wording differences
that preserve the same meaning.


--------------------------------------------------
RULE 14 — RELEVANCE

The final answer must actually answer
the user's question.

Do not reject an answer merely because
it contains useful supporting information.


==================================================
VERIFICATION PROCEDURE
==================================================

Perform these checks:

CHECK 1

Identify the main decision or conclusion.


CHECK 2

Identify every important factual claim
supporting that decision.


CHECK 3

Compare each claim against the document.


CHECK 4

Check whether missing information has
been incorrectly treated as a negative.


CHECK 5

Check whether inference has been presented
as fact.


CHECK 6

Check whether recommendations are actually
supported by the evidence.


CHECK 7

Check whether the conclusion is stronger
than the evidence.


CHECK 8

Check whether the answer directly answers
the user's question.


==================================================
IMPORTANT
==================================================

Do NOT search for reasons to reject a
correct answer.

Only flag genuine problems.

However, if the answer contains even one
important unsupported factual claim that
affects the decision, it MUST be marked:

NEEDS_CORRECTION


==================================================
RETURN FORMAT
==================================================

If the answer is fully supported:

STATUS:
VERIFIED

ISSUES:
None

CORRECTION:
None


If there is a genuine problem:

STATUS:
NEEDS_CORRECTION

ISSUES:

- State the exact unsupported claim.
- Explain why the document does not support it.
- Identify whether it is:
  * unsupported claim
  * unsupported negative inference
  * missing-information error
  * unsupported capability inference
  * unsupported recommendation
  * contradiction
  * overconfidence
  * hallucination
  * irrelevant reasoning

CORRECTION:

State exactly what must be changed.

Do NOT rewrite the complete answer.


==================================================
FINAL PRINCIPLE
==================================================

The goal is evidence-grounded reasoning.

The verifier must protect the final answer
from hallucinations and unsupported
conclusions.

A conclusion of:

"A definitive decision cannot be made
from the available document evidence."

is completely valid when evidence is
insufficient.

Do not force a positive or negative decision.
"""


    return call_llm(
        prompt,
        model=OLLAMA_MODEL,
        temperature=0.0
    )



# ==========================================
# CORRECTION AGENT
# ==========================================

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

    print(
        "Revising the answer using "
        "verification feedback..."
    )


    # ======================================
    # CONTEXT
    # ======================================

    context = build_context(
        retrieved_documents
    )


    analysis_text = (
        analysis
        if analysis
        else "Not provided."
    )

    risk_text = (
        risk
        if risk
        else "Not provided."
    )

    solution_text = (
        solution
        if solution
        else "Not provided."
    )


    # ======================================
    # CORRECTION PROMPT
    # ======================================

    prompt = f"""
You are the Correction Agent in a
general-purpose document-grounded
decision-support system.

Your job is to produce the final corrected
answer using the document evidence and the
Verification Agent feedback.

==================================================
USER QUESTION
==================================================

{question}


==================================================
DOCUMENT EVIDENCE
==================================================

{context}


==================================================
ANALYSIS
==================================================

{analysis_text}


==================================================
RISK ANALYSIS
==================================================

{risk_text}


==================================================
POSSIBLE SOLUTIONS
==================================================

{solution_text}


==================================================
ORIGINAL ANSWER
==================================================

{original_answer}


==================================================
VERIFICATION FEEDBACK
==================================================

{verification}


==================================================
CORRECTION RULES
==================================================

1. Use document evidence as the factual
   foundation.

2. Remove unsupported factual claims.

3. Remove hallucinated information.

4. Remove unsupported negative claims.

5. Never convert missing information into
   negative evidence.

6. Do not say something is lacking merely
   because the document does not mention it.

7. Do not convert interest into expertise.

8. Do not convert goals into experience.

9. Do not convert keywords into proficiency.

10. Do not infer personality or personal
    attributes without evidence.

11. Preserve valid evidence-supported
    conclusions.

12. Preserve reasonable inference when it
    is clearly presented as inference.

13. Do not present an inferred solution as
    an explicit document recommendation.

14. Remove generic recommendations that are
    not relevant to the evidence.

15. Do not invent facts.

16. Do not invent numerical values.

17. Do not introduce outside knowledge.

18. If evidence is insufficient, explicitly
    state that a definitive conclusion cannot
    be made.

19. Keep the final answer directly relevant
    to the user's question.

20. The final answer must remain
    domain-independent.


==================================================
IMPORTANT CORRECTION EXAMPLE
==================================================

INCORRECT:

"The document does not mention communication,
therefore communication is a weakness."


CORRECT:

"The available document does not provide
enough information to determine communication
ability."


INCORRECT:

"The subject is interested in X, therefore
the subject is skilled in X."


CORRECT:

"The document indicates interest in X, but
does not provide sufficient evidence to
determine proficiency in X."


==================================================
FINAL ANSWER REQUIREMENT
==================================================

Return ONLY the corrected final answer.

Do not include:

- verification status
- verification issues
- correction explanation
- internal reasoning
- agent names
- meta commentary
"""


    return call_llm(
        prompt,
        model=OLLAMA_MODEL,
        temperature=0.0
    )