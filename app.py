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
# SESSION STATE
# ============================================================

if "active_document" not in st.session_state:
    st.session_state.active_document = None

if "agent_activity" not in st.session_state:
    st.session_state.agent_activity = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Architecture")

    st.markdown(
        """
**Adaptive Workflow**

📄 Document  
↓  
🧩 Processing  
↓  
🗄️ ChromaDB  
↓  
🔀 Router  
↓  
🔎 Retriever  
↓  
🧠 Analysis  
↓  
⚠️ Risk  
↓  
💡 Solution  
↓  
🎯 Decision  
↓  
✅ Verification  
↓  
✏️ Correction  
↓  
💬 Final Answer
"""
    )

    st.divider()

    # ========================================================
    # ACTIVE DOCUMENT
    # ========================================================

    st.subheader("📄 Active Document")

    if st.session_state.active_document:

        st.success(
            st.session_state.active_document
        )

    else:

        st.info(
            "No document selected."
        )


    st.divider()

    # ========================================================
    # AGENT ACTIVITY
    # ========================================================

    st.subheader("🤖 Agent Activity")

    activities = (
        st.session_state.agent_activity
    )

    if activities:

        for item in activities:

            agent = item.get(
                "agent",
                "Agent"
            )

            status = item.get(
                "status",
                "unknown"
            )

            description = item.get(
                "description",
                ""
            )

            if status == "completed":

                st.markdown(
                    f"**{agent}**"
                )

                st.caption(
                    f"✅ {description}"
                )

            elif status == "skipped":

                st.markdown(
                    f"**{agent}**"
                )

                st.caption(
                    f"⏭️ {description}"
                )

            else:

                st.markdown(
                    f"**{agent}**"
                )

                st.caption(
                    description
                )

    else:

        st.caption(
            "Agent activity will appear here "
            "after a question is processed."
        )


    st.divider()

    # ========================================================
    # SIMPLE SETTINGS
    # ========================================================

    with st.expander(
        "⚙️ Optional Settings"
    ):

        st.caption(
            "Workflow settings"
        )

        st.checkbox(
            "Strict document grounding",
            value=True,
            disabled=True
        )

        st.checkbox(
            "Show agent activity",
            value=True,
            disabled=True
        )

        st.caption(
            "Adaptive routing automatically selects "
            "the required agents."
        )


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🤖 Self-Adaptive Multi-Agent RAG"
)

st.caption(
    "Document-grounded intelligent decision support "
    "using RAG, multi-agent reasoning and Ollama"
)


# ============================================================
# DOCUMENT KNOWLEDGE BASE
# ============================================================

st.header(
    "📚 Document Knowledge Base"
)

st.write(
    "Upload a PDF once. It becomes the active document "
    "for subsequent questions."
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

        documents_folder = (
            get_writable_documents_dir()
        )

        pdf_path = (
            documents_folder
            / uploaded_file.name
        )

        try:

            with open(
                pdf_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )


            with st.spinner(
                "Processing document..."
            ):

                result = index_pdf(
                    pdf_path
                )


            st.session_state.active_document = (
                result["source"]
            )

            st.session_state.agent_activity = []

            st.session_state.last_result = None

            st.success(
                "✅ Document indexed successfully."
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
# ACTIVE DOCUMENT
# ============================================================

if st.session_state.active_document:

    st.success(
        f"🔒 Active document: "
        f"{st.session_state.active_document}"
    )


st.divider()


# ============================================================
# ASK QUESTION
# ============================================================

st.header(
    "💬 Ask Questions"
)

if st.session_state.active_document:

    st.caption(
        f"Questions are restricted to: "
        f"{st.session_state.active_document}"
    )

else:

    st.info(
        "Upload and process a PDF before asking questions."
    )


question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: According to the income in the "
        "document, can I buy a laptop with an EMI "
        "of ₹13,300 per month?"
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

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not st.session_state.active_document:

        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        active_document = (
            st.session_state.active_document
        )

        try:

            with st.spinner(
                "Running adaptive workflow..."
            ):

                result = run_workflow(
                    question,
                    active_document
                )


            # ================================================
            # SAVE RESULT
            # ================================================

            st.session_state.last_result = result

            st.session_state.agent_activity = (
                result.get(
                    "activities",
                    []
                )
            )


            # ================================================
            # QUERY TYPE
            # ================================================

            st.divider()

            query_type = result.get(
                "query_type",
                "UNKNOWN"
            )

            st.subheader(
                "🔄 Workflow"
            )

            st.info(
                f"Query Type: **{query_type}**"
            )


            # ================================================
            # FINAL ANSWER
            # ================================================

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


            # ================================================
            # SOURCES
            # ================================================

            documents = result.get(
                "documents",
                []
            )

            if documents:

                st.divider()

                st.header(
                    "📚 Evidence"
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


            # ================================================
            # TECHNICAL DETAILS
            # ================================================

            if query_type == "DECISION":

                with st.expander(
                    "🧠 Analysis Details"
                ):

                    st.write(
                        result.get(
                            "analysis",
                            "Not available."
                        )
                    )


                risk = result.get(
                    "risk",
                    ""
                )

                if risk:

                    with st.expander(
                        "⚠️ Risk Details"
                    ):

                        st.write(
                            risk
                        )


                solution = result.get(
                    "solution",
                    ""
                )

                if solution:

                    with st.expander(
                        "💡 Solution Details"
                    ):

                        st.write(
                            solution
                        )


                decision = result.get(
                    "decision",
                    ""
                )

                if decision:

                    with st.expander(
                        "🎯 Decision Details"
                    ):

                        st.write(
                            decision
                        )


                verification = result.get(
                    "verification",
                    ""
                )

                if verification:

                    with st.expander(
                        "✅ Verification Details"
                    ):

                        st.write(
                            verification
                        )


            st.success(
                "✅ Workflow completed."
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