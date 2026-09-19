import os
import sys
import time
from pathlib import Path
import streamlit as st

# ============================================================
# PROJECT ROOT CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.adaptive_workflow import run_workflow
from src.rag.vector_store import index_pdf, collection, CHROMA_PATH

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Self-Adaptive Multi-Agent RAG | Decision Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODERN CUSTOM CSS (Glassmorphism, Neon Accents, Dark Theme)
# ============================================================
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1400px;
    }

    /* Gradient Header Hero */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.15), transparent 40%),
                    radial-gradient(circle at bottom left, rgba(56, 189, 248, 0.1), transparent 40%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.98rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* System Status Badges */
    .badge-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 14px;
    }

    .tech-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #CBD5E1;
    }

    .tech-pill.status-online {
        border-color: rgba(16, 185, 129, 0.3);
        background: rgba(16, 185, 129, 0.1);
        color: #34D399;
    }

    /* Card Styling */
    .glass-card {
        background: rgba(19, 27, 46, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        backdrop-filter: blur(10px);
        transition: all 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
    }

    /* Workflow Agent Pipeline Visualizer */
    .pipeline-container {
        display: flex;
        align-items: center;
        gap: 8px;
        overflow-x: auto;
        padding: 16px 8px;
        margin: 16px 0 24px 0;
        scrollbar-width: thin;
    }

    .agent-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-width: 105px;
        padding: 10px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        text-align: center;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: #94A3B8;
        transition: all 0.3s ease;
    }

    .agent-step.completed {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.2), rgba(79, 70, 229, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.5);
        color: #EEF2FF;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
    }

    .agent-step.skipped {
        opacity: 0.45;
        border-style: dashed;
    }

    .agent-arrow {
        color: #475569;
        font-weight: bold;
        font-size: 1.1rem;
    }

    /* Executive Decision Callout */
    .decision-callout {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9));
        border-left: 5px solid #6366F1;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin: 16px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* Risk Box */
    .risk-callout {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    /* Solution Box */
    .solution-callout {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    /* Source Citation Card */
    .source-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }

    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        transform: translateY(-1px);
    }

    /* Metric Card Customization */
    [data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d1322;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "active_document" not in st.session_state:
    # Auto-detect existing PDFs in data/documents
    docs_dir = Path("data") / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    existing_docs = [f.name for f in docs_dir.glob("*.pdf")]
    st.session_state.active_document = existing_docs[0] if existing_docs else None

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ============================================================
# HERO HEADER SECTION
# ============================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Self-Adaptive Multi-Agent Decision Engine</div>
    <div class="hero-subtitle">
        Enterprise Document Intelligence & Autonomous Strategic Deliberation powered by RAG, ChromaDB Vector Store, and Multi-Agent Orchestration.
    </div>
    <div class="badge-container">
        <span class="tech-pill status-online">● ChromaDB Online</span>
        <span class="tech-pill status-online">● Ollama llama3.2 Active</span>
        <span class="tech-pill">🧠 8-Agent Collective</span>
        <span class="tech-pill">🔍 MiniLM-L6-v2 Embeddings</span>
        <span class="tech-pill">🛡️ Autonomous Verification Loop</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR - CONTROL PANEL & ARCHITECTURE
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Existing Documents Selector
    docs_dir = Path("data") / "documents"
    available_docs = [f.name for f in docs_dir.glob("*.pdf")]
    
    if available_docs:
        selected_doc = st.selectbox(
            "📁 Select Active Knowledge Source",
            options=available_docs,
            index=available_docs.index(st.session_state.active_document) if st.session_state.active_document in available_docs else 0,
            help="Choose an already indexed PDF document from the knowledge base."
        )
        if selected_doc != st.session_state.active_document:
            st.session_state.active_document = selected_doc
            st.rerun()
    else:
        st.info("No documents found in knowledge base.")

    st.markdown("---")
    st.markdown("### 📤 Upload New Document")
    uploaded_file = st.file_uploader(
        "Upload PDF for Vector Indexing",
        type=["pdf"],
        help="Upload a PDF. It will be parsed, chunked, embedded, and stored in ChromaDB."
    )

    if uploaded_file is not None:
        if st.button("🚀 Process & Index PDF", use_container_width=True):
            save_path = docs_dir / uploaded_file.name
            with st.spinner("Processing & embedding document..."):
                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    result = index_pdf(save_path)
                    st.session_state.active_document = result["source"]
                    st.success(f"✅ Indexed {result['source']} ({result['chunks']} chunks, {result['pages']} pages)")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Indexing error: {str(e)}")

    st.markdown("---")
    st.markdown("### 🧬 Multi-Agent Topology")
    st.markdown("""
    <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.6;">
        <b>🔀 Router Agent</b>: Intent & Complexity Classifier<br>
        <b>🔎 Retriever Agent</b>: Semantic Vector Grounding<br>
        <b>🧠 Analysis Agent</b>: Deep Synthesizer & Extractor<br>
        <b>⚠️ Risk Agent</b>: Vulnerability & Risk Auditing<br>
        <b>💡 Solution Agent</b>: Actionable Recommendations<br>
        <b>🎯 Decision Agent</b>: Strategic Executive Verdict<br>
        <b>✅ Verification Agent</b>: Grounded Fact Checker<br>
        <b>✏️ Correction Agent</b>: Autonomous Hallucination Fixer
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Query History", use_container_width=True):
        st.session_state.query_history = []
        st.session_state.last_result = None
        st.rerun()

# ============================================================
# MAIN WORKSPACE
# ============================================================

# Active Document Status Banner
doc_status_col1, doc_status_col2, doc_status_col3 = st.columns([2, 1, 1])

with doc_status_col1:
    if st.session_state.active_document:
        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 12px 18px; display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.4rem;">📄</span>
            <div>
                <div style="font-size: 0.75rem; color: #818CF8; font-weight: 700; text-transform: uppercase;">Active Grounding Document</div>
                <div style="font-size: 0.95rem; color: #F1F5F9; font-weight: 600;">{st.session_state.active_document}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No active document selected. Please upload or select a PDF from the sidebar.")

with doc_status_col2:
    try:
        total_chunks = len(collection.get(where={"source": st.session_state.active_document})["ids"]) if st.session_state.active_document else 0
    except Exception:
        total_chunks = "N/A"
    st.metric(label="Active Chunks", value=str(total_chunks), delta="Indexed in Chroma")

with doc_status_col3:
    st.metric(label="Inference LLM", value="Llama 3.2", delta="Local Ollama")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# QUERY INTERACTION DECK
# ============================================================
st.markdown("### 💬 Decision Deliberation Deck")

# Quick Question Starter Chips
st.markdown("<span style='font-size: 0.8rem; color: #94A3B8; font-weight: 600;'>QUICK PROMPTS:</span>", unsafe_allow_html=True)
quick_cols = st.columns(4)

prompt_choice = None
with quick_cols[0]:
    if st.button("🎯 Assess Role Fit", use_container_width=True):
        prompt_choice = f"Is the subject in {st.session_state.active_document} suitable for a Senior Engineering / Leadership role? Provide justification."
with quick_cols[1]:
    if st.button("⚠️ Analyze Risks", use_container_width=True):
        prompt_choice = "What are the major risks, weaknesses, or potential concerns identified in this document?"
with quick_cols[2]:
    if st.button("📊 Executive Summary", use_container_width=True):
        prompt_choice = "Provide a comprehensive structured summary of the key findings, achievements, and core details in this document."
with quick_cols[3]:
    if st.button("💡 Strategic Next Steps", use_container_width=True):
        prompt_choice = "What actionable solutions or recommendations should be prioritized based on this document?"

default_question = prompt_choice if prompt_choice else ""

user_query = st.text_area(
    "Enter your strategic question or deliberation prompt:",
    value=default_question,
    placeholder="e.g. Evaluate the suitability of the candidate for software architecture, highlighting potential risks and mitigation plans...",
    height=90,
    help="Enter any fact-retrieval, analytical, or strategic decision query. The router agent will dynamically orchestrate the appropriate agents."
)

col_ask, col_space = st.columns([1, 3])
with col_ask:
    submit_query = st.button("🔍 Execute Multi-Agent Deliberation", use_container_width=True)

# ============================================================
# RUN MULTI-AGENT WORKFLOW
# ============================================================
if submit_query:
    if not user_query.strip():
        st.warning("⚠️ Please provide a valid prompt or question.")
    elif not st.session_state.active_document:
        st.error("❌ Please select or upload an active document first.")
    else:
        active_doc = st.session_state.active_document
        
        # Live Progress Animation
        with st.status("🤖 Orchestrating Self-Adaptive Multi-Agent Workflow...", expanded=True) as status_box:
            st.write("🔀 **Router Agent**: Classifying query intent and routing path...")
            time.sleep(0.3)
            
            st.write("🔎 **Retriever Agent**: Querying ChromaDB for high-dimensional semantic chunks...")
            time.sleep(0.3)
            
            try:
                start_time = time.time()
                result = run_workflow(user_query, active_doc)
                duration = round(time.time() - start_time, 2)
                
                query_type = result.get("query_type", "UNKNOWN")
                
                if query_type == "DECISION":
                    st.write("🧠 **Analysis Agent**: Synthesizing deep context...")
                    st.write("⚠️ **Risk Agent**: Performing risk & vulnerability matrix analysis...")
                    st.write("💡 **Solution Agent**: Formulating strategic alternatives...")
                    st.write("🎯 **Decision Agent**: Synthesizing final strategic resolution...")
                    st.write("✅ **Verification Agent**: Grounding check & hallucination verification completed.")
                elif query_type == "ANALYTICAL":
                    st.write("🧠 **Analysis Agent**: Synthesizing structured multi-faceted analysis...")
                else:
                    st.write("💬 **Answer Generator**: Formulated concise factual response.")
                
                status_box.update(label=f"✅ Workflow Completed in {duration}s — Route: {query_type}", state="complete", expanded=False)
                
                result["duration"] = duration
                result["question"] = user_query
                st.session_state.last_result = result
                st.session_state.query_history.insert(0, {
                    "question": user_query,
                    "query_type": query_type,
                    "result": result,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                
            except Exception as e:
                status_box.update(label="❌ Multi-Agent Workflow Failed", state="error", expanded=True)
                st.error(f"Error during execution: {str(e)}")
                st.exception(e)

# ============================================================
# RESULTS DISPLAY & AGENT REASONING VISUALIZATION
# ============================================================
if st.session_state.last_result:
    res = st.session_state.last_result
    q_type = res.get("query_type", "UNKNOWN")
    
    st.markdown("---")
    
    # --------------------------------------------------------
    # INTERACTIVE AGENT PIPELINE VISUALIZER
    # --------------------------------------------------------
    st.markdown("#### ⚡ Dynamic Agent Execution Pipeline")
    
    is_simple = (q_type == "SIMPLE")
    is_analytical = (q_type == "ANALYTICAL")
    is_decision = (q_type == "DECISION")
    
    verification_text = str(res.get("verification", ""))
    had_correction = "NEEDS_CORRECTION" in verification_text.upper()
    
    steps_html = f"""
    <div class="pipeline-container">
        <div class="agent-step completed">
            <span style="font-size: 1.2rem;">🔀</span>
            <span>Router</span>
            <span style="font-size: 0.65rem; color: #34D399;">● COMPLETED</span>
        </div>
        <div class="agent-arrow">→</div>
        <div class="agent-step completed">
            <span style="font-size: 1.2rem;">🔎</span>
            <span>Retriever</span>
            <span style="font-size: 0.65rem; color: #34D399;">● {len(res.get('documents', []))} CHUNKS</span>
        </div>
        <div class="agent-arrow">→</div>
        <div class="agent-step {'completed' if (is_analytical or is_decision) else 'skipped'}">
            <span style="font-size: 1.2rem;">🧠</span>
            <span>Analysis</span>
            <span style="font-size: 0.65rem; color: {'#34D399' if (is_analytical or is_decision) else '#64748B'};">● {'COMPLETED' if (is_analytical or is_decision) else 'SKIPPED'}</span>
        </div>
        <div class="agent-arrow">→</div>
        <div class="agent-step {'completed' if is_decision else 'skipped'}">
            <span style="font-size: 1.2rem;">⚠️</span>
            <span>Risk</span>
            <span style="font-size: 0.65rem; color: {'#34D399' if is_decision else '#64748B'};">● {'COMPLETED' if is_decision else 'SKIPPED'}</span>
        </div>
        <div class="agent-arrow">→</div>
        <div class="agent-step {'completed' if is_decision else 'skipped'}">
            <span style="font-size: 1.2rem;">💡</span>
            <span>Solution</span>
            <span style="font-size: 0.65rem; color: {'#34D399' if is_decision else '#64748B'};">● {'COMPLETED' if is_decision else 'SKIPPED'}</span>
        </div>
        <div class="agent-arrow">→</div>
        <div class="agent-step {'completed' if is_decision else 'skipped'}">
            <span style="font-size: 1.2rem;">🎯</span>
            <span>Decision</span>
            <span style="font-size: 0.65rem; color: {'#34D399' if is_decision else '#64748B'};">● {'COMPLETED' if is_decision else 'SKIPPED'}</span>
        </div>
        <div class="agent-arrow">→</div>
        <div class="agent-step {'completed' if is_decision else 'skipped'}">
            <span style="font-size: 1.2rem;">✅</span>
            <span>Verifier</span>
            <span style="font-size: 0.65rem; color: {'#34D399' if is_decision else '#64748B'};">● {'PASSED' if (is_decision and not had_correction) else ('AUDITED' if is_decision else 'SKIPPED')}</span>
        </div>
        <div class="agent-arrow">→</div>
        <div class="agent-step {'completed' if had_correction else 'skipped'}">
            <span style="font-size: 1.2rem;">✏️</span>
            <span>Correction</span>
            <span style="font-size: 0.65rem; color: {'#F59E0B' if had_correction else '#64748B'};">● {'APPLIED' if had_correction else 'SKIPPED'}</span>
        </div>
    </div>
    """
    st.markdown(steps_html, unsafe_allow_html=True)

    # --------------------------------------------------------
    # STRUCTURED OUTPUT TABS
    # --------------------------------------------------------
    tab_titles = ["🎯 Executive Summary & Verdict", "📚 Source Citations"]
    if is_decision:
        tab_titles.insert(1, "🧠 Multi-Agent Deliberation (Analysis, Risk, Solution)")
        tab_titles.insert(2, "🛡️ Quality Verification Audit")
    elif is_analytical:
        tab_titles.insert(1, "🧠 Deep Analysis")

    tabs = st.tabs(tab_titles)

    # TAB 1: EXECUTIVE VERDICT / ANSWER
    with tabs[0]:
        st.markdown(f"""
        <div class="decision-callout">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #818CF8; letter-spacing: 0.05em; text-transform: uppercase;">
                    ROUTE: {q_type} DECISION FLOW
                </span>
                <span style="font-size: 0.8rem; color: #94A3B8;">
                    ⏱️ Latency: {res.get('duration', '0.0')}s
                </span>
            </div>
            <div style="font-size: 1.05rem; color: #F8FAFC; line-height: 1.6;">
                {res.get('answer', 'No answer generated.')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Download Report Option
        report_content = f"""# Multi-Agent Strategic Decision Report
**Active Document:** {res.get('source')}
**Query Type:** {q_type}
**User Prompt:** {res.get('question')}
**Execution Time:** {res.get('duration', 'N/A')}s

---

## Executive Verdict / Answer
{res.get('answer')}

"""
        if is_decision:
            report_content += f"""
## Deep Deliberation Breakdown

### 🧠 Analysis
{res.get('analysis')}

### ⚠️ Risk Assessment
{res.get('risk')}

### 💡 Proposed Solutions & Recommendations
{res.get('solution')}

### 🎯 Strategic Decision Rationale
{res.get('decision')}

### 🛡️ Quality & Grounding Verification
{res.get('verification')}
"""
        st.download_button(
            label="📥 Export Analysis Brief (Markdown)",
            data=report_content,
            file_name=f"decision_report_{int(time.time())}.md",
            mime="text/markdown"
        )

    # TAB 2 (DECISION / ANALYTICAL): DEEP DELIBERATION
    if is_decision:
        with tabs[1]:
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.markdown("#### 🧠 Context Analysis")
                st.markdown(f"""
                <div class="glass-card">
                    {res.get('analysis', 'No analysis details.')}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### ⚠️ Risk Matrix & Vulnerabilities")
                st.markdown(f"""
                <div class="risk-callout">
                    {res.get('risk', 'No risks identified.')}
                </div>
                """, unsafe_allow_html=True)

            with col_d2:
                st.markdown("#### 💡 Strategic Solutions & Countermeasures")
                st.markdown(f"""
                <div class="solution-callout">
                    {res.get('solution', 'No solutions proposed.')}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 🎯 Core Decision Formulation")
                st.markdown(f"""
                <div class="glass-card">
                    {res.get('decision', 'No decision record.')}
                </div>
                """, unsafe_allow_html=True)

        # TAB 3 (DECISION): VERIFICATION AUDIT
        with tabs[2]:
            st.markdown("#### 🛡️ Verifier Agent Audit Trail")
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-weight: 700; color: {'#F59E0B' if had_correction else '#34D399'}; margin-bottom: 8px;">
                    {'⚠️ Autonomous Correction Triggered & Applied' if had_correction else '✅ Verified Consistent with Document Grounding'}
                </div>
                {res.get('verification', 'Verification clean.')}
            </div>
            """, unsafe_allow_html=True)

    elif is_analytical:
        with tabs[1]:
            st.markdown("#### 🧠 Comprehensive Analysis")
            st.markdown(f"""
            <div class="glass-card">
                {res.get('analysis', 'No detailed analysis.')}
            </div>
            """, unsafe_allow_html=True)

    # CITATIONS TAB
    citations_tab_idx = -1
    with tabs[citations_tab_idx]:
        st.markdown("#### 📚 Grounded Document Chunks (ChromaDB)")
        docs = res.get("documents", [])
        if docs:
            for idx, doc_item in enumerate(docs, start=1):
                src_name = doc_item.get("source", "Unknown Document")
                page_no = doc_item.get("page", "1")
                txt = doc_item.get("text", "")
                
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-header">
                        <span>📑 Source Chunk #{idx} — {src_name}</span>
                        <span style="background: rgba(99, 102, 241, 0.2); color: #A5B4FC; padding: 2px 8px; border-radius: 6px;">Page {page_no}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5; white-space: pre-wrap;">
{txt}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No explicit source chunks returned.")

# ============================================================
# QUERY HISTORY
# ============================================================
if len(st.session_state.query_history) > 1:
    with st.expander("🕒 Session Query History"):
        for item in st.session_state.query_history[1:]:
            st.markdown(f"**[{item['timestamp']}] ({item['query_type']})** `{item['question']}`")
            st.markdown(f"> {item['result'].get('answer', '')[:200]}...")
            st.markdown("---")

# ============================================================
# MODERN FOOTER
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 20px; color: #64748B; font-size: 0.8rem;">
    ⚡ <b>Self-Adaptive Multi-Agent RAG Decision System</b> • Powered by Streamlit, ChromaDB, Sentence-Transformers & Ollama Llama 3.2
</div>
""", unsafe_allow_html=True)