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
# AGENTS
# ==========================================

from src.agents.router_agent import (
    classify_query
)

from src.agents.retriever_agent import (
    retriever_agent
)

from src.agents.analysis_agent import (
    analysis_agent
)

from src.agents.risk_agent import (
    risk_agent
)

from src.agents.solution_agent import (
    solution_agent
)

from src.agents.decision_agent import (
    decision_agent
)

from src.agents.verification_agent import (
    verification_agent,
    revise_answer
)


# ==========================================
# RAG
# ==========================================

from src.rag.rag_service import (
    ask_llm,
    create_context
)


# ==========================================
# ADAPTIVE WORKFLOW
# ==========================================

def run_workflow(
    question,
    source
):

    # ======================================
    # STEP 1
    # QUERY ROUTER
    # ======================================

    query_type = classify_query(
        question
    )

    print(
        "\n[ROUTER]"
    )

    print(
        f"Query type: {query_type}"
    )


    # ======================================
    # STEP 2
    # RETRIEVER
    # ======================================

    print(
        "\n[RETRIEVER AGENT]"
    )

    print(
        f"Active document: {source}"
    )

    documents = retriever_agent(
        question,
        source
    )

    print(
        f"Retrieved {len(documents)} "
        "relevant chunks."
    )


    # ======================================
    # SIMPLE QUERY
    # ======================================

    if query_type == "SIMPLE":

        print(
            "\n[ANSWER GENERATOR]"
        )

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
            "source": source
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

        return {
            "answer": analysis,
            "query_type": query_type,
            "documents": documents,
            "analysis": analysis,
            "source": source
        }


    # ======================================
    # DECISION QUERY
    # ======================================

    if query_type == "DECISION":

        # ----------------------------------
        # ANALYSIS
        # ----------------------------------

        print(
            "\n[ANALYSIS AGENT]"
        )

        analysis = analysis_agent(
            question,
            documents
        )


        # ----------------------------------
        # RISK
        # ----------------------------------

        print(
            "\n[RISK AGENT]"
        )

        risk = risk_agent(
            question,
            documents,
            analysis
        )


        # ----------------------------------
        # SOLUTION
        # ----------------------------------

        print(
            "\n[SOLUTION AGENT]"
        )

        solution = solution_agent(
            question,
            documents,
            analysis,
            risk
        )


        # ----------------------------------
        # DECISION
        # ----------------------------------

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


        # ----------------------------------
        # VERIFICATION
        # ----------------------------------

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


        # ==================================
        # CHECK VERIFICATION RESULT
        # ==================================

        needs_correction = (
            "NEEDS_CORRECTION"
            in verification.upper()
        )


        # ==================================
        # CORRECTION
        # ==================================

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

        else:

            final_answer = decision


        # ==================================
        # RETURN RESULT
        # ==================================

        return {
            "answer": final_answer,
            "query_type": query_type,
            "documents": documents,
            "analysis": analysis,
            "risk": risk,
            "solution": solution,
            "decision": decision,
            "verification": verification,
            "source": source
        }


    # ======================================
    # UNKNOWN QUERY TYPE
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
        "source": source
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
        "DECISION SUPPORT SYSTEM"
    )

    print(
        "======================================"
    )


    # ======================================
    # SELECT ACTIVE DOCUMENT
    # ======================================

    source = input(
        "\nEnter PDF filename: "
    ).strip()


    if not source:

        print(
            "\nERROR: PDF filename is required."
        )

        sys.exit()


    # ======================================
    # QUESTION LOOP
    # ======================================

    while True:

        question = input(
            "\nEnter your question "
            "(or exit): "
        ).strip()


        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "\nExiting..."
            )

            break


        if not question:

            continue


        try:

            result = run_workflow(
                question,
                source
            )


            # ==================================
            # QUERY TYPE
            # ==================================

            print(
                "\n======================================"
            )

            print(
                "QUERY TYPE"
            )

            print(
                "======================================"
            )

            print(
                result.get(
                    "query_type",
                    "UNKNOWN"
                )
            )


            # ==================================
            # ACTIVE DOCUMENT
            # ==================================

            print(
                "\n======================================"
            )

            print(
                "ACTIVE DOCUMENT"
            )

            print(
                "======================================"
            )

            print(
                result.get(
                    "source",
                    source
                )
            )


            # ==================================
            # FINAL RESPONSE
            # ==================================

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