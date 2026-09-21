import streamlit as st
from pathlib import Path

from src.agents.adaptive_workflow import run_workflow
from src.rag.vector_store import index_pdf
from src.rag.chroma_helper import get_writable_documents_dir



# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Self-Adaptive Multi-Agent RAG",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# ACTIVE DOCUMENT STATE
# ============================================================

if "active_document" not in st.session_state:
    st.session_state.active_document = None


# ============================================================
# TITLE
# ============================================================

st.title(
    "🤖 Self-Adaptive Multi-Agent RAG"
)

st.caption(
    "Intelligent Document Analysis and Decision Support "
    "using RAG, Multi-Agent Architecture and Ollama"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Architecture")

    st.markdown(
        """
        **Adaptive Workflow**

        📄 Document Upload  
        ↓  
        🧩 Document Processing  
        ↓  
        🗄️ ChromaDB Knowledge Base  
        ↓  
        🔀 Query Router  
        ↓  
        🔎 Retriever  
        ↓  
        🧠 Analysis Agent  
        ↓  
        ⚠️ Risk Agent  
        ↓  
        💡 Solution / Decision  
        ↓  
        ✅ Verification  
        ↓  
        ✏️ Correction  
        ↓  
        💬 Final Answer
        """
    )

    st.divider()

    st.subheader("Agents")

    st.write("🔀 Router")
    st.write("🔎 Retriever")
    st.write("🧠 Analysis")
    st.write("⚠️ Risk")
    st.write("💡 Solution")
    st.write("🎯 Decision")
    st.write("✅ Verification")
    st.write("✏️ Correction")

    st.divider()
    with st.expander("⚙️ Optional Settings"):
        st.caption("Optional Groq API key for high-speed cloud inference:")
        groq_k = st.text_input("Groq API Key", value=st.session_state.get("GROQ_API_KEY", ""), type="password", placeholder="gsk_...")
        if groq_k:
            st.session_state["GROQ_API_KEY"] = groq_k


# ============================================================
# DOCUMENT KNOWLEDGE BASE
# ============================================================

st.header("📚 Document Knowledge Base")

st.write(
    "Upload a PDF document once. "
    "The uploaded document becomes the active document. "
    "All questions will use only the active document."
)


# ============================================================
# ACTIVE DOCUMENT DISPLAY
# ============================================================

if st.session_state.active_document:

    st.success(
        f"📄 Active document: "
        f"{st.session_state.active_document}"
    )

else:

    st.info(
        "No active document. "
        "Please upload and process a PDF."
    )


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


if uploaded_file is not None:

    st.info(
        f"Selected document: {uploaded_file.name}"
    )

    if st.button(
        "📥 Upload & Process Document",
        use_container_width=True
    ):

        documents_folder = get_writable_documents_dir()
        pdf_path = documents_folder / uploaded_file.name


        try:

            # ------------------------------------------------
            # Save PDF
            # ------------------------------------------------

            with open(
                pdf_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )


            # ------------------------------------------------
            # Index PDF
            # ------------------------------------------------

            with st.spinner(
                "Processing document..."
            ):

                result = index_pdf(
                    pdf_path
                )


            # ------------------------------------------------
            # SET ACTIVE DOCUMENT
            # ------------------------------------------------

            st.session_state.active_document = (
                result["source"]
            )


            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            st.success(
                "✅ Document indexed successfully!"
            )

            st.success(
                f"📄 Active document: "
                f"{result['source']}"
            )


            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Document",
                    result["source"]
                )

            with col2:

                st.metric(
                    "Pages",
                    result["pages"]
                )

            with col3:

                st.metric(
                    "Chunks",
                    result["chunks"]
                )


        except Exception as e:

            st.error(
                "❌ Document processing failed."
            )

            st.exception(e)


# ============================================================
# SEPARATOR
# ============================================================

st.divider()


# ============================================================
# ASK QUESTION
# ============================================================

st.header("💬 Ask Questions")

if st.session_state.active_document:

    st.write(
        "Ask questions about the active document:"
    )

    st.info(
        f"🔒 Questions are restricted to: "
        f"{st.session_state.active_document}"
    )

else:

    st.write(
        "Upload and process a PDF before asking questions."
    )


question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: What are the major risks "
        "identified in the document?"
    ),
    height=100
)


ask_button = st.button(
    "🔍 Ask Question",
    use_container_width=True
)


# ============================================================
# RUN WORKFLOW
# ============================================================

if ask_button:

    # --------------------------------------------------------
    # CHECK QUESTION
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question."
        )


    # --------------------------------------------------------
    # CHECK ACTIVE DOCUMENT
    # --------------------------------------------------------

    elif not st.session_state.active_document:

        st.warning(
            "Please upload and process a PDF first."
        )


    # --------------------------------------------------------
    # RUN WORKFLOW
    # --------------------------------------------------------

    else:

        try:

            active_document = (
                st.session_state.active_document
            )


            # ------------------------------------------------
            # Run workflow
            # ------------------------------------------------

            with st.spinner(
                "Running adaptive multi-agent workflow..."
            ):

                result = run_workflow(
                    question,
                    active_document
                )


            # =================================================
            # WORKFLOW TYPE
            # =================================================

            st.divider()

            st.header(
                "🔄 Adaptive Workflow"
            )


            st.info(
                f"📄 Active Document: "
                f"{active_document}"
            )


            query_type = result.get(
                "query_type",
                "UNKNOWN"
            )


            if query_type == "SIMPLE":

                st.success(
                    "🔀 Query Type: SIMPLE"
                )

            elif query_type == "ANALYTICAL":

                st.success(
                    "🔀 Query Type: ANALYTICAL"
                )

            elif query_type == "DECISION":

                st.success(
                    "🔀 Query Type: DECISION"
                )

            else:

                st.warning(
                    f"🔀 Query Type: {query_type}"
                )


            # =================================================
            # AGENT STATUS
            # =================================================

            st.subheader(
                "Agent Execution"
            )


            if query_type == "SIMPLE":

                st.write(
                    "🔀 Router Agent — ✅ Completed"
                )

                st.write(
                    "🔎 Retriever Agent — ✅ Completed"
                )

                st.write(
                    "💬 Answer Generator — ✅ Completed"
                )

                st.write(
                    "🧠 Analysis Agent — ⏭️ Skipped"
                )

                st.write(
                    "⚠️ Risk Agent — ⏭️ Skipped"
                )

                st.write(
                    "💡 Solution Agent — ⏭️ Skipped"
                )

                st.write(
                    "🎯 Decision Agent — ⏭️ Skipped"
                )

                st.write(
                    "✅ Verification Agent — ⏭️ Skipped"
                )

                st.write(
                    "✏️ Correction Agent — ⏭️ Skipped"
                )


            elif query_type == "ANALYTICAL":

                st.write(
                    "🔀 Router Agent — ✅ Completed"
                )

                st.write(
                    "🔎 Retriever Agent — ✅ Completed"
                )

                st.write(
                    "🧠 Analysis Agent — ✅ Completed"
                )

                st.write(
                    "⚠️ Risk Agent — ⏭️ Skipped"
                )

                st.write(
                    "💡 Solution Agent — ⏭️ Skipped"
                )

                st.write(
                    "🎯 Decision Agent — ⏭️ Skipped"
                )

                st.write(
                    "✅ Verification Agent — ⏭️ Skipped"
                )

                st.write(
                    "✏️ Correction Agent — ⏭️ Skipped"
                )


            elif query_type == "DECISION":

                st.write(
                    "🔀 Router Agent — ✅ Completed"
                )

                st.write(
                    "🔎 Retriever Agent — ✅ Completed"
                )

                st.write(
                    "🧠 Analysis Agent — ✅ Completed"
                )

                st.write(
                    "⚠️ Risk Agent — ✅ Completed"
                )

                st.write(
                    "💡 Solution Agent — ✅ Completed"
                )

                st.write(
                    "🎯 Decision Agent — ✅ Completed"
                )

                st.write(
                    "✅ Verification Agent — "
                    "✅ Completed"
                )


                verification = result.get(
                    "verification",
                    ""
                )


                if (
                    "NEEDS_CORRECTION"
                    in str(
                        verification
                    ).upper()
                ):

                    st.write(
                        "✏️ Correction Agent — "
                        "✅ Completed"
                    )

                else:

                    st.write(
                        "✏️ Correction Agent — "
                        "⏭️ Skipped"
                    )


            # =================================================
            # FINAL ANSWER
            # =================================================

            st.divider()

            st.header(
                "💬 Final Answer"
            )


            answer = result.get(
                "answer",
                ""
            )


            if answer:

                st.markdown(
                    answer
                )

            else:

                st.warning(
                    "No answer was returned."
                )


            # =================================================
            # RETRIEVED SOURCES
            # =================================================

            documents = result.get(
                "documents",
                []
            )


            if documents:

                st.divider()

                st.header(
                    "📚 Retrieved Sources"
                )


                for i, document in enumerate(
                    documents,
                    start=1
                ):

                    source_name = document.get(
                        "source",
                        "Unknown"
                    )

                    page_number = document.get(
                        "page",
                        "?"
                    )


                    with st.expander(
                        f"Source {i} — "
                        f"{source_name} "
                        f"(Page {page_number})"
                    ):

                        st.write(
                            document.get(
                                "text",
                                ""
                            )
                        )


            # =================================================
            # ANALYSIS
            # =================================================

            if query_type == "DECISION":

                analysis = result.get(
                    "analysis",
                    ""
                )


                if analysis:

                    st.divider()

                    st.header(
                        "🧠 Analysis"
                    )

                    st.write(
                        analysis
                    )


            # =================================================
            # RISK
            # =================================================

            if query_type == "DECISION":

                risk = result.get(
                    "risk",
                    ""
                )


                if risk:

                    st.divider()

                    st.header(
                        "⚠️ Risk Analysis"
                    )

                    st.write(
                        risk
                    )


            # =================================================
            # SOLUTION
            # =================================================

            if query_type == "DECISION":

                solution = result.get(
                    "solution",
                    ""
                )


                if solution:

                    st.divider()

                    st.header(
                        "💡 Possible Solutions"
                    )

                    st.write(
                        solution
                    )


            # =================================================
            # DECISION
            # =================================================

            if query_type == "DECISION":

                decision = result.get(
                    "decision",
                    ""
                )


                if decision:

                    st.divider()

                    st.header(
                        "🎯 Decision"
                    )

                    st.write(
                        decision
                    )


            # =================================================
            # VERIFICATION
            # =================================================

            if query_type == "DECISION":

                verification = result.get(
                    "verification",
                    ""
                )


                if verification:

                    st.divider()

                    st.header(
                        "✅ Verification"
                    )

                    st.write(
                        verification
                    )


            # =================================================
            # SYSTEM COMPLETE
            # =================================================

            st.success(
                "✅ Workflow completed successfully."
            )


        except Exception as e:

            st.error(
                "❌ Error while running the workflow."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Python • Streamlit • ChromaDB • "
    "Sentence Transformers • Ollama • "
    "Adaptive Multi-Agent RAG"
)