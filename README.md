---
title: Self Adaptive Multi Agent RAG Decision System
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ⚡ Self-Adaptive Multi-Agent RAG Decision System

Enterprise-grade document intelligence and autonomous strategic decision support system powered by Multi-Agent deliberation, ChromaDB vector retrieval, and Ollama LLMs.

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

## 🚀 Hugging Face Spaces Deployment Instructions

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space)
2. Select **Docker** (or **Streamlit**)
3. Clone or push this repository to your Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>
   git push space main
   ```
4. Hugging Face Spaces will automatically build the container and deploy your app 24/7.
