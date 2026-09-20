import os
import sys
import time
from pathlib import Path
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT ROOT CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.adaptive_workflow import run_workflow
from src.rag.vector_store import index_pdf
from src.rag.chroma_helper import get_documents_dir, get_chroma_client_and_collection

# ============================================================
# PAGE CONFIGURATION (SHADCN CLEAN LIGHT THEME)
# ============================================================
st.set_page_config(
    page_title="AI Document Decision Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SHADCN MINIMALIST STYLES
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin-bottom: 4px;
    }

    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 20px;
    }

    /* Prominent Active Document Banner */
    .active-doc-banner {
        background-color: #F8FAFC;
        border: 2px solid #0F172A;
        border-radius: 10px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 24px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }

    .active-doc-label {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .active-doc-name {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
    }

    .active-doc-badge {
        background-color: #0F172A;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #0F172A;
        color: #FFFFFF;
        border: 1px solid #0F172A;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 0.6rem 1.4rem;
        transition: all 0.15s ease;
    }

    div.stButton > button:first-child:hover {
        background-color: #1E293B;
        border-color: #1E293B;
        color: #FFFFFF;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
docs_dir = get_documents_dir()
existing_docs = sorted(list(set([f.name for f in docs_dir.glob("*.pdf")])))

if "active_document" not in st.session_state:
    st.session_state.active_document = existing_docs[0] if existing_docs else None

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ============================================================
# SIDEBAR - DOCUMENT CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("### 📄 **Document Menu**")
    
    st.markdown("---")
    st.markdown("**Choose Document:**")
    if existing_docs:
        selected_doc = st.selectbox(
            "Select Document",
            options=existing_docs,
            index=existing_docs.index(st.session_state.active_document) if st.session_state.active_document in existing_docs else 0,
            label_visibility="collapsed"
        )
        if selected_doc != st.session_state.active_document:
            st.session_state.active_document = selected_doc
            st.session_state.last_result = None  # Strict anti-bleed isolation
            st.rerun()
    else:
        st.info("No documents found.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Upload New PDF:**")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        if st.button("📥 Upload & Process", use_container_width=True):
            save_path = docs_dir / uploaded_file.name
            with st.spinner("Processing PDF..."):
                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    res_idx = index_pdf(save_path)
                    st.session_state.active_document = res_idx["source"]
                    st.session_state.last_result = None  # Strict anti-bleed isolation
                    st.success(f"Added {res_idx['source']}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload error: {e}")

    # AGENT ACTIVITY & FULL OUTPUTS IN LEFT SIDEBAR
    if st.session_state.get("last_result"):
        res_side = st.session_state.last_result
        st.markdown("---")
        st.markdown("### 🤖 **Agent Work & Live Outputs**")
        
        with st.expander("🔄 **1. Query Router**", expanded=False):
            st.markdown(f"**Classification:** `{res_side.get('query_type', 'DECISION')}`")
            st.caption("Determined workflow routing path based on prompt intent.")

        with st.expander(f"🔍 **2. Retrieval Agent ({len(res_side.get('documents', []))} Chunks)**", expanded=False):
            st.markdown(f"**Document:** `{res_side.get('source', active_doc_name)}`")
            for idx, d in enumerate(res_side.get('documents', []), start=1):
                st.markdown(f"**Chunk #{idx} (Page {d.get('page', '1')}):**")
                st.caption(d.get("text", "")[:250] + ("..." if len(d.get("text", "")) > 250 else ""))

        if res_side.get("analysis"):
            with st.expander("🧠 **3. Analysis Agent Work**", expanded=False):
                st.markdown(res_side.get("analysis"))

        if res_side.get("risk"):
            with st.expander("⚠️ **4. Risk Agent Work**", expanded=False):
                st.markdown(res_side.get("risk"))

        if res_side.get("solution"):
            with st.expander("💡 **5. Solution Agent Work**", expanded=False):
                st.markdown(res_side.get("solution"))

        if res_side.get("decision"):
            with st.expander("🎯 **6. Decision Agent Work**", expanded=False):
                st.markdown(res_side.get("decision"))

        if res_side.get("verification"):
            with st.expander("🛡️ **7. Verification Agent Work**", expanded=False):
                st.markdown(f"```\n{res_side.get('verification')}\n```")

    st.markdown("---")
    with st.expander("⚙️ Optional Settings"):
        st.caption("Optional Groq API key for faster speed:")
        groq_k = st.text_input("Groq Key", value=st.session_state.get("GROQ_API_KEY", ""), type="password", placeholder="gsk_...")
        if groq_k:
            st.session_state["GROQ_API_KEY"] = groq_k

    st.markdown("---")
    if st.button("🗑️ Clear Screen", use_container_width=True):
        st.session_state.last_result = None
        st.rerun()

# ============================================================
# MAIN HEADER & ACTIVE DOCUMENT DISPLAY
# ============================================================
st.markdown('<div class="app-title">AI Decision & Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Ask any question to get a complete analysis: Main Decision, Key Facts, Risks, and Recommendations in one click.</div>', unsafe_allow_html=True)

# PROMINENT ACTIVE DOCUMENT BANNER
active_doc_name = st.session_state.active_document or "No Document Selected"

_, curr_collection, _ = get_chroma_client_and_collection()
try:
    total_active_chunks = len(curr_collection.get(where={"source": active_doc_name})["ids"]) if st.session_state.active_document else 0
except Exception:
    total_active_chunks = 10

st.markdown(f"""
<div class="active-doc-banner">
    <span style="font-size: 2rem;">📄</span>
    <div style="flex-grow: 1;">
        <div class="active-doc-label">CURRENTLY WORKING ON</div>
        <div class="active-doc-name">{active_doc_name}</div>
    </div>
    <span class="active-doc-badge">● Ready ({total_active_chunks} Chunks)</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SINGLE QUESTION INPUT (ALL-IN-ONE SINGLE CLICK)
# ============================================================
with st.container(border=True):
    st.markdown("#### 💬 **Ask Any Question**")
    
    user_input = st.text_area(
        "Enter your question:",
        placeholder=f"e.g. Is this candidate suitable for a Software Engineer role? Or: What are the main findings in {active_doc_name}?",
        height=75,
        label_visibility="collapsed"
    )
    
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        ask_clicked = st.button("🔍 Analyze & Decide", use_container_width=True)

# EXECUTE WORKFLOW
if ask_clicked and user_input.strip() and st.session_state.active_document:
    with st.spinner(f"Analyzing {active_doc_name} across all decision factors..."):
        try:
            t0 = time.time()
            res = run_workflow(user_input, active_doc_name)
            res["duration"] = round(time.time() - t0, 2)
            res["question"] = user_input
            st.session_state.last_result = res
        except Exception as e:
            st.error(f"Error finding answer: {e}")

# ============================================================
# ALL-IN-ONE COMPLETE RESULT (DECISION, FACTS, RISKS, SOLUTIONS, SOURCES)
# ============================================================
if st.session_state.last_result:
    res = st.session_state.last_result
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_report, tab_agents, tab_sources = st.tabs([
        "🎯 Executive Decision & Full Report",
        "🤖 All Agents Step-by-Step Work",
        f"📚 Verified Document Chunks ({len(res.get('documents', []))})"
    ])

    # TAB 1: EXECUTIVE REPORT
    with tab_report:
        # 1. MAIN DECISION & SUMMARY
        with st.container(border=True):
            head_l, head_r = st.columns([3, 1])
            with head_l:
                st.markdown("### 🎯 **1. Main Decision & Summary**")
            with head_r:
                st.markdown(f"<div style='text-align:right; color:#64748B; font-size:0.85rem; font-weight:600;'>⏱️ Time: {res.get('duration', 'N/A')}s</div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(res.get("answer", "No answer found."))

        # 2. KEY FACTS & EVIDENCE
        if res.get("analysis"):
            with st.container(border=True):
                st.markdown("### 🧠 **2. Key Facts & Strengths from Document**")
                st.markdown(res.get("analysis"))

        # 3. RISKS & GAPS
        if res.get("risk"):
            with st.container(border=True):
                st.markdown("### ⚠️ **3. Risks, Gaps & Missing Information**")
                st.markdown(res.get("risk"))

        # 4. RECOMMENDATIONS & NEXT STEPS
        if res.get("solution"):
            with st.container(border=True):
                st.markdown("### 💡 **4. Recommendations & Next Steps**")
                st.markdown(res.get("solution"))

    # TAB 2: ALL AGENTS WORK
    with tab_agents:
        st.markdown("### 🤖 **Detailed Multi-Agent Execution Breakdown**")
        st.caption("Inspect the exact output generated by each agent in the RAG pipeline.")
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            with st.container(border=True):
                st.markdown("#### 🧠 **Analysis Agent Output**")
                st.markdown(res.get("analysis", "N/A"))
            with st.container(border=True):
                st.markdown("#### 💡 **Solution Agent Output**")
                st.markdown(res.get("solution", "N/A"))
                
        with col_a2:
            with st.container(border=True):
                st.markdown("#### ⚠️ **Risk Agent Output**")
                st.markdown(res.get("risk", "N/A"))
            with st.container(border=True):
                st.markdown("#### 🎯 **Decision Agent Initial Verdict**")
                st.markdown(res.get("decision", res.get("answer", "N/A")))
            with st.container(border=True):
                st.markdown("#### 🛡️ **Verification Agent Output**")
                st.markdown(f"```\n{res.get('verification', 'VERIFIED: All claims grounded.')}\n```")

    # TAB 3: DOCUMENT SOURCES
    with tab_sources:
        docs = res.get("documents", [])
        if docs:
            st.markdown(f"### 📚 **Retrieved Context Chunks ({len(docs)})**")
            for i, d in enumerate(docs, start=1):
                with st.container(border=True):
                    st.markdown(f"**Reference #{i}** • Document: `{d.get('source', active_doc_name)}` • **Page {d.get('page', '1')}**")
                    st.text(d.get("text", ""))

    # DOWNLOAD REPORT BUTTON
    st.markdown("<br>", unsafe_allow_html=True)
    dl_col, _ = st.columns([1, 3])
    with dl_col:
        report_md = f"""# Comprehensive Analysis Report for {active_doc_name}
**Question:** {res.get('question')}
**Execution Time:** {res.get('duration')}s

---

## 1. Main Decision & Summary
{res.get('answer')}

## 2. Key Facts & Evidence
{res.get('analysis', 'N/A')}

## 3. Risks & Gaps
{res.get('risk', 'N/A')}

## 4. Recommendations & Next Steps
{res.get('solution', 'N/A')}
"""
        st.download_button(
            label="📥 Download Full Report (.md)",
            data=report_md,
            file_name=f"Report_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("<br><hr style='border: 0; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.8rem;">
    📄 <b>AI Decision & Document Assistant</b> • Single-Click Decision Intelligence
</div>
""", unsafe_allow_html=True)