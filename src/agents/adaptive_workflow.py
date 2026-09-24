from pathlib import Path
import sys

# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# AGENTS
# ==========================================

from src.agents.router_agent import classify_query
from src.agents.retriever_agent import retriever_agent
from src.agents.analysis_agent import analysis_agent
from src.agents.risk_agent import risk_agent
from src.agents.solution_agent import solution_agent
from src.agents.decision_agent import decision_agent
from src.agents.verification_agent import (
    verification_agent,
    revise_answer
)

from src.rag.rag_service import (
    ask_llm,
    create_context
)


# ==========================================
# FAST DECISION DETECTION
# ==========================================

def is_calculation_decision(question):

    q = question.lower()

    calculation_words = [
        "can i buy",
        "can i afford",
        "afford",
        "emi",
        "loan",
        "installment",
        "installment",
        "per month",
        "monthly",
        "per year",
        "annual",
        "cost",
        "price",
        "budget",
        "salary",
        "income",
        "expense",
        "profit",
        "loss",
        "calculate",
        "how much",
        "how many",
        "worth",
    ]

    return any(
        word in q
        for word in calculation_words
    )


# ==========================================
# BUILD AGENT ACTIVITY
# ==========================================

def activity(
    agent,
    status,
    description
):

    return {
        "agent": agent,
        "status": status,
        "description": description
    }


# ==========================================
# ADAPTIVE WORKFLOW
# ==========================================

def run_workflow(
    question,
    source
):

    activities = []

    # ======================================
    # STEP 1 — ROUTER
    # ======================================

    query_type = classify_query(
        question
    )

    activities.append(
        activity(
            "🔀 Router Agent",
            "completed",
            f"Classified query as {query_type}."
        )
    )

    print(
        f"\n[ROUTER] Query type: {query_type}"
    )


    # ======================================
    # STEP 2 — RETRIEVER
    # ======================================

    print(
        "\n[RETRIEVER AGENT]"
    )

    documents = retriever_agent(
        question,
        source
    )

    activities.append(
        activity(
            "🔎 Retriever Agent",
            "completed",
            f"Retrieved {len(documents)} relevant "
            f"chunks from {source}."
        )
    )


    # ======================================
    # NO DOCUMENT FOUND
    # ======================================

    if not documents:

        answer = (
            "I could not find relevant information "
            "in the active document to answer this question."
        )

        activities.append(
            activity(
                "🧠 Analysis Agent",
                "skipped",
                "No relevant document evidence was retrieved."
            )
        )

        return {
            "answer": answer,
            "query_type": query_type,
            "documents": [],
            "source": source,
            "activities": activities
        }


    # ======================================
    # SIMPLE QUERY
    # ======================================

    if query_type == "SIMPLE":

        activities.append(
            activity(
                "🧠 Analysis Agent",
                "skipped",
                "Not required for a direct information query."
            )
        )

        activities.append(
            activity(
                "⚠️ Risk Agent",
                "skipped",
                "Risk analysis is not required."
            )
        )

        activities.append(
            activity(
                "💡 Solution Agent",
                "skipped",
                "No solution generation is required."
            )
        )

        activities.append(
            activity(
                "🎯 Decision Agent",
                "skipped",
                "No decision is required."
            )
        )

        activities.append(
            activity(
                "✅ Verification Agent",
                "skipped",
                "Direct document answer does not require "
                "decision verification."
            )
        )

        activities.append(
            activity(
                "✏️ Correction Agent",
                "skipped",
                "No correction required."
            )
        )

        context = create_context(
            documents
        )

        answer = ask_llm(
            question,
            context
        )

        activities.append(
            activity(
                "💬 Answer Generator",
                "completed",
                "Generated a document-grounded answer."
            )
        )

        return {
            "answer": answer,
            "query_type": query_type,
            "documents": documents,
            "source": source,
            "activities": activities
        }


    # ======================================
    # ANALYTICAL QUERY
    # ======================================

    if query_type == "ANALYTICAL":

        print(
            "\n[ANALYSIS AGENT]"
        )

        analysis = analysis_agent(
            question,
            documents
        )

        activities.append(
            activity(
                "🧠 Analysis Agent",
                "completed",
                "Analyzed the retrieved document evidence."
            )
        )

        activities.append(
            activity(
                "⚠️ Risk Agent",
                "skipped",
                "Risk analysis is not required."
            )
        )

        activities.append(
            activity(
                "💡 Solution Agent",
                "skipped",
                "Solution generation is not required."
            )
        )

        activities.append(
            activity(
                "🎯 Decision Agent",
                "skipped",
                "No decision is required."
            )
        )

        activities.append(
            activity(
                "✅ Verification Agent",
                "skipped",
                "No decision verification required."
            )
        )

        activities.append(
            activity(
                "✏️ Correction Agent",
                "skipped",
                "No correction required."
            )
        )

        return {
            "answer": analysis,
            "query_type": query_type,
            "documents": documents,
            "analysis": analysis,
            "source": source,
            "activities": activities
        }


    # ======================================
    # DECISION QUERY
    # ======================================

    if query_type == "DECISION":

        # ==================================
        # ANALYSIS
        # ==================================

        print(
            "\n[ANALYSIS AGENT]"
        )

        analysis = analysis_agent(
            question,
            documents
        )

        activities.append(
            activity(
                "🧠 Analysis Agent",
                "completed",
                "Extracted relevant document facts and "
                "identified information needed to answer "
                "the decision."
            )
        )


        # ==================================
        # FAST CALCULATION DECISION
        # ==================================

        if is_calculation_decision(question):

            print(
                "\n[FAST DECISION MODE]"
            )

            activities.append(
                activity(
                    "⚠️ Risk Agent",
                    "skipped",
                    "Skipped because this is a calculation-"
                    "based decision and no separate risk "
                    "analysis is required."
                )
            )

            activities.append(
                activity(
                    "💡 Solution Agent",
                    "skipped",
                    "Skipped to avoid unnecessary LLM generation."
                )
            )


            # ==============================
            # DECISION
            # ==============================

            print(
                "\n[DECISION AGENT]"
            )

            decision = decision_agent(
                question,
                documents,
                analysis,
                "Not required for this calculation-based decision.",
                "Not required for this calculation-based decision."
            )

            activities.append(
                activity(
                    "🎯 Decision Agent",
                    "completed",
                    "Used document values and user-provided "
                    "values to calculate and answer the decision."
                )
            )


            # ==============================
            # VERIFICATION
            # ==============================

            print(
                "\n[VERIFICATION AGENT]"
            )

            verification = verification_agent(
                question,
                documents,
                decision,
                analysis,
                "",
                ""
            )

            activities.append(
                activity(
                    "✅ Verification Agent",
                    "completed",
                    "Checked the decision against the "
                    "document evidence and calculation."
                )
            )


            needs_correction = (
                "NEEDS_CORRECTION"
                in verification.upper()
            )


            # ==============================
            # CORRECTION
            # ==============================

            if needs_correction:

                print(
                    "\n[CORRECTION AGENT]"
                )

                final_answer = revise_answer(
                    question,
                    documents,
                    decision,
                    verification,
                    analysis,
                    "",
                    ""
                )

                activities.append(
                    activity(
                        "✏️ Correction Agent",
                        "completed",
                        "Corrected unsupported or inaccurate "
                        "parts of the proposed answer."
                    )
                )

            else:

                final_answer = decision

                activities.append(
                    activity(
                        "✏️ Correction Agent",
                        "skipped",
                        "Verification found no important "
                        "correction required."
                    )
                )


            return {
                "answer": final_answer,
                "query_type": query_type,
                "documents": documents,
                "analysis": analysis,
                "risk": "",
                "solution": "",
                "decision": decision,
                "verification": verification,
                "source": source,
                "activities": activities
            }


        # ==================================
        # FULL DECISION WORKFLOW
        # ==================================

        print(
            "\n[RISK AGENT]"
        )

        risk = risk_agent(
            question,
            documents,
            analysis
        )

        activities.append(
            activity(
                "⚠️ Risk Agent",
                "completed",
                "Identified evidence-supported risks, "
                "limitations and uncertainties."
            )
        )


        print(
            "\n[SOLUTION AGENT]"
        )

        solution = solution_agent(
            question,
            documents,
            analysis,
            risk
        )

        activities.append(
            activity(
                "💡 Solution Agent",
                "completed",
                "Generated evidence-grounded possible "
                "solutions and options."
            )
        )


        print(
            "\n[DECISION AGENT]"
        )

        decision = decision_agent(
            question,
            documents,
            analysis,
            risk,
            solution
        )

        activities.append(
            activity(
                "🎯 Decision Agent",
                "completed",
                "Produced the evidence-grounded decision."
            )
        )


        print(
            "\n[VERIFICATION AGENT]"
        )

        verification = verification_agent(
            question,
            documents,
            decision,
            analysis,
            risk,
            solution
        )

        activities.append(
            activity(
                "✅ Verification Agent",
                "completed",
                "Checked the final decision for unsupported "
                "claims and contradictions."
            )
        )


        needs_correction = (
            "NEEDS_CORRECTION"
            in verification.upper()
        )


        if needs_correction:

            print(
                "\n[CORRECTION AGENT]"
            )

            final_answer = revise_answer(
                question,
                documents,
                decision,
                verification,
                analysis,
                risk,
                solution
            )

            activities.append(
                activity(
                    "✏️ Correction Agent",
                    "completed",
                    "Revised the answer using verification feedback."
                )
            )

        else:

            final_answer = decision

            activities.append(
                activity(
                    "✏️ Correction Agent",
                    "skipped",
                    "Verification found no important correction required."
                )
            )


        return {
            "answer": final_answer,
            "query_type": query_type,
            "documents": documents,
            "analysis": analysis,
            "risk": risk,
            "solution": solution,
            "decision": decision,
            "verification": verification,
            "source": source,
            "activities": activities
        }


    # ======================================
    # FALLBACK
    # ======================================

    context = create_context(
        documents
    )

    answer = ask_llm(
        question,
        context
    )

    return {
        "answer": answer,
        "query_type": query_type,
        "documents": documents,
        "source": source,
        "activities": activities
    }


# ==========================================
# CLI TEST
# ==========================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "SELF-ADAPTIVE MULTI-AGENT RAG"
    )

    print(
        "======================================"
    )

    source = input(
        "\nEnter PDF filename: "
    ).strip()

    if not source:

        print(
            "\nERROR: PDF filename is required."
        )

        sys.exit()

    while True:

        question = input(
            "\nEnter your question (or exit): "
        ).strip()

        if question.lower() in [
            "exit",
            "quit"
        ]:

            break

        if not question:

            continue

        try:

            result = run_workflow(
                question,
                source
            )

            print(
                "\n======================================"
            )

            print(
                "FINAL RESPONSE"
            )

            print(
                "======================================"
            )

            print(
                result.get(
                    "answer",
                    "No answer returned."
                )
            )

        except Exception as e:

            print(
                "\nERROR:"
            )

            print(
                str(e)
            )