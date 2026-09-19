import re
from src.rag.llm_client import call_llm

OLLAMA_MODEL = "llama3.2:latest"


def rule_based_classification(question):
    """
    Fast and deterministic classification for common query patterns.
    Returns SIMPLE, ANALYTICAL, or DECISION.
    """

    q = question.lower().strip()

    # -------------------------------------------------
    # DECISION
    # -------------------------------------------------
    decision_patterns = [
        r"\bis .* suitable\b",
        r"\bis .* a good fit\b",
        r"\bshould .* focus\b",
        r"\bshould .* choose\b",
        r"\bshould .* pursue\b",
        r"\bwhich .* should\b",
        r"\bwhich .* is better\b",
        r"\bwhich .* would be best\b",
        r"\bwhat .* should .* do\b",
        r"\bwhat .* is the best\b",
        r"\brecommend\b",
        r"\bsuitable\b",
        r"\bfit for\b",
        r"\brisk\b",
        r"\badvantage\b",
        r"\bdisadvantage\b",
        r"\bdecision\b",
    ]

    for pattern in decision_patterns:
        if re.search(pattern, q):
            return "DECISION"


    # -------------------------------------------------
    # ANALYTICAL
    # -------------------------------------------------
    analytical_patterns = [
        r"\bstrongest\b",
        r"\bweaknesses\b",
        r"\bstrengths\b",
        r"\banalyze\b",
        r"\banalyse\b",
        r"\banalysis\b",
        r"\bsummarize\b",
        r"\bsummarise\b",
        r"\bsummary\b",
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bexplain how\b",
        r"\bhow .* demonstrate\b",
        r"\bhow .* relate\b",
        r"\bhow .* connected\b",
        r"\bidentify .* skills\b",
        r"\bidentify .* strengths\b",
        r"\btechnical background\b",
        r"\btechnical profile\b",
        r"\balign\b",
        r"\bpatterns\b",
        r"\boverall\b",
    ]

    for pattern in analytical_patterns:
        if re.search(pattern, q):
            return "ANALYTICAL"


    # -------------------------------------------------
    # SIMPLE
    # -------------------------------------------------
    simple_patterns = [
        r"^what is\b",
        r"^what are\b",
        r"^where\b",
        r"^when\b",
        r"^who\b",
        r"^which college\b",
        r"^which university\b",
        r"^what programming languages\b",
        r"^what technologies\b",
        r"^what skills\b",
        r"^what tools\b",
        r"^what percentage\b",
        r"^how much\b",
    ]

    for pattern in simple_patterns:
        if re.search(pattern, q):
            return "SIMPLE"


    # -------------------------------------------------
    # Ambiguous → use LLM
    # -------------------------------------------------

    return None


def llm_classification(question):

    prompt = f"""
You are a strict Router Agent.

Classify the user's question into exactly ONE category.

SIMPLE:
A question asking for one or more directly stated facts
from the document.

Examples:
- What is Abishek's BCA percentage?
- Where did Abishek complete his BCA?
- What programming languages does Abishek know?

ANALYTICAL:
A question requiring interpretation, comparison,
organization, summarization, or identifying relationships
between multiple pieces of information.

Examples:
- What are Abishek's strongest technical skills?
- Summarize Abishek's projects.
- Compare Abishek's education and skills.
- How do Abishek's projects demonstrate his skills?

DECISION:
A question requiring judgment, recommendation,
suitability, risk assessment, or choosing between options.

Examples:
- Is Abishek suitable for a software engineering role?
- Should Abishek focus on cloud or software development?
- What are the risks of selecting Abishek?

IMPORTANT:
Return ONLY ONE word:

SIMPLE
ANALYTICAL
DECISION

USER QUESTION:
{question}
"""

    response_text = call_llm(prompt, model=OLLAMA_MODEL, temperature=0.0)
    result = response_text.strip().upper()

    # Exact matching only
    if result == "DECISION":
        return "DECISION"

    if result == "ANALYTICAL":
        return "ANALYTICAL"

    if result == "SIMPLE":
        return "SIMPLE"

    # Safe fallback
    return "SIMPLE"


def classify_query(question):

    # First try deterministic rules
    category = rule_based_classification(question)

    if category is not None:
        return category

    # Only ambiguous questions go to Ollama
    return llm_classification(question)


if __name__ == "__main__":

    print("\n==============================")
    print("SELF-ADAPTIVE ROUTER")
    print("==============================")

    while True:

        question = input(
            "\nEnter question (or exit): "
        )

        if question.lower().strip() == "exit":
            break

        category = classify_query(question)

        print(
            f"\nQuery type: {category}"
        )