# 💊 Antidiabetic Drug Safety RAG Assistant

An AI-powered pharmacovigilance intelligence tool built with **LlamaIndex**, **Groq LLM**, and **Streamlit**. 
It answers safety questions about **SGLT2 inhibitors** and **GLP-1 receptor agonists** using official regulatory drug labeling, and every answer is traceable to the exact source passage.

🔗 Live Web Application**: [faers-rag-assistant.streamlit.app](http://localhost:8501/) 
---
## 📚 Data Scope

The knowledge base consists of regulatory product information for the two target drug classes:

| Source | Document type | Regulator |
|--------|---------------|-----------|
| **USPI** | U.S. Prescribing Information | FDA |
| **SmPC** | Summary of Product Characteristics | EMA |

- **Drug classes:** SGLT2 inhibitors (e.g.,Canagliflozin, Dapagliflozin, Empagliflozin) and GLP-1 receptor agonists (e.g., Semaglutide, Liraglutide, Tirzepatide).
- **Sections of interest:** contraindications, warnings and precautions, adverse reactions, drug interactions, and use in special populations.

> **Note:** This tool does not analyze individual FAERS case reports. It reflects what the approved labeling states, which makes it useful for comparing FDA and EMA safety language across products.

---

## 🌟 Key Features

- **HyDE & Re-Ranking Architecture:** Uses **Hypothetical Document Embeddings (HyDE)** to expand short or vague clinical queries, then re-ranks candidates with a **SentenceTransformer Reranker** (`BAAI/bge-reranker-large`) for higher contextual precision. 
- **Strict Source Attribution:** Shows the retrieved chunks, file names, page numbers, and similarity scores in expandable containers so every claim can be checked against the label.
- **Dynamic RAG Parameter Tuning:** Sidebar controls adjust retrieval depth (`similarity_top_k`) and the system prompt persona in real time.
- **Streamlit Web Interface:** Persistent chat session state, one-click "New Chat" reset, and cached heavy resources via `@st.cache_resource`. The 👍/👎 buttons are UI placeholders and do not store feedback yet.

---

## 🏗️ Project Architecture & Directory Structure

```text
faers_rag/
├── .vscode/                # VS Code workspace settings & launch configurations
├── app.py                  # Streamlit UI layer & caching orchestration
├── src/
│   ├── engine.py           # Core RAG engine (HyDE, memory, retriever, reranker)
│   ├── model_loader.py     # Initializers for Groq LLM & HuggingFace embeddings
│   └── config.py           # Core application settings and RAG parameters
├── local_storage/          # Persistent vector store index files (Git ignored)
├── .env                    # API keys and local environment variables (Git ignored)
├── .gitignore              # Excludes local_storage, notebooks, and environment files
├── environment.yml         # Conda environment definition
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
