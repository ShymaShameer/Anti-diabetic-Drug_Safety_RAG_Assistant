# 💊 FAERS Pharmacovigilance RAG Assistant

An AI-powered pharmacovigilance intelligence tool built with **LlamaIndex**, **Groq LLM**, and **Streamlit**. This platform analyzes FDA Adverse Event Reporting System (FAERS) data and drug labeling for targeted therapeutic classes—specifically **SGLT2 inhibitors** and **GLP-1 receptor agonists**—to provide transparent, source-attributed safety insights.

---

## 🌟 Key Features

- **HyDE & Re-Ranking Architecture**: Uses **Hypothetical Document Embeddings (HyDE)** for query expansion paired with a **SentenceTransformer Reranker** (`BAAI/bge-reranker-large`) to maximize contextual precision.
- **Strict Source Attribution**: Renders source document chunks, page metadata, file names, and similarity scores in expandable UI containers for factual validation.
- **Dynamic RAG Parameter Tuning**: Interactive sidebar controls allow real-time adjustments to retrieval depth (`similarity_top_k`) and system prompt personas.
- **Streamlit Web Interface**: Features an optimized chat session state, instant message resetting ("New Chat"), non-blocking feedback placeholders (👍/👎), and resource caching via `@st.cache_resource`.

---

## 🏗️ Project Architecture & Directory Structure

```text
faers_rag/
├── app.py                  # Streamlit UI layer & caching orchestration
├── src/
│   ├── engine.py           # Core RAG engine (HyDE, memory, retriever, reranker)
│   ├── model_loader.py     # Initializers for Groq LLM & HuggingFace embeddings
│   └── vector_store.py     # Vector database loading & persistence
├── .gitignore              # Ignores large indexes, local storage, & .env files
├── environment.yml         # Conda environment definition
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
