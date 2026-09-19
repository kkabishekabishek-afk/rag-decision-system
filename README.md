---
title: Self Adaptive Multi Agent RAG Decision System
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
---

# ⚡ Self-Adaptive Multi-Agent RAG Decision System

Enterprise-grade document intelligence and autonomous strategic decision support system powered by Multi-Agent deliberation, ChromaDB vector retrieval, and Streamlit.

## 🧬 Multi-Agent Topology
* **🔀 Router Agent**: Classifies query complexity into `SIMPLE`, `ANALYTICAL`, or `DECISION`.
* **🔎 Retriever Agent**: Fetches top relevant chunks from ChromaDB with semantic embeddings.
* **🧠 Analysis Agent**: Synthesizes and contextualizes extracted facts.
* **⚠️ Risk Agent**: Audits potential operational, strategic, or domain risks.
* **💡 Solution Agent**: Generates actionable alternatives and solutions.
* **🎯 Decision Agent**: Formulates clear executive resolutions.
* **✅ Verification Agent**: Performs grounding checks to eliminate hallucinations.
* **✏️ Correction Agent**: Autonomously corrects discrepancies when verified.

---

## 🚀 Hugging Face Spaces (100% Free Streamlit SDK)

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space)
2. Select SDK: **Streamlit** (Free CPU Basic • 16GB RAM)
3. Push this repository to your Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>
   git push space master:main
   ```
4. Hugging Face will deploy it 24/7 for free!
