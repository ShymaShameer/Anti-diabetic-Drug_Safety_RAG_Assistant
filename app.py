import logging
import streamlit as st

# 1. Silence background HTTP and LlamaIndex logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)

from src.model_loader import initialise_llm, get_embedding_model
from src.engine import get_chat_engine

# --- Page Setup ---
st.set_page_config(
    page_title="FAERS Pharmacovigilance Assistant",
    page_icon="💊",
    layout="wide"
)

# --- Sidebar: Configuration Panel & Controls ---
with st.sidebar:
    st.header("⚙️ RAG Configuration")
    
    # 1. Slider to control retrieval top_k
    similarity_top_k = st.slider(
        "Similarity Top K",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of relevant document text chunks to retrieve for context."
    )

    # 2. Dropdown to select system prompt persona
    prompt_preset = st.selectbox(
        "System Prompt Preset",
        options=[
            "Standard Pharmacovigilance Analyst",
            "Concise Safety Summary",
            "Strict Document Verification"
        ],
        index=0
    )

    prompt_mapping = {
        "Standard Pharmacovigilance Analyst": (
            "You are an AI assistant specialized in pharmacovigilance and drug safety. "
            "Provide balanced, accurate information based on the retrieved FAERS data and drug labeling."
        ),
        "Concise Safety Summary": (
            "Provide direct, bulleted summaries emphasizing high-level safety signals and key facts."
        ),
        "Strict Document Verification": (
            "Answer strictly using the provided context chunks. If information is missing, explicitly state "
            "that the retrieved document context is insufficient."
        )
    }
    selected_system_prompt = prompt_mapping[prompt_preset]

    st.divider()

    # 3. Control: New Chat Button
    if st.button("🗑️ New Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 Welcome to the Drug Safety Analysis Assistant. "
                    "You can ask me questions about FAERS data, adverse events "
                    "associated with SGLT2 inhibitors and GLP-1 receptor agonists."
                )
            }
        ]
        st.rerun()

st.title("💊  Drug Information Assistant")
st.caption("FAERS Pharmacovigilance & Safety Query System")

# --- Engine & Model Caching ---
@st.cache_resource
def load_models():
    llm = initialise_llm()
    embed_model = get_embedding_model()
    return llm, embed_model

@st.cache_resource
def get_cached_chat_engine(similarity_top_k: int, system_prompt: str):
    llm, embed_model = load_models()
    return get_chat_engine(
        llm=llm,
        embed_model=embed_model,
        similarity_top_k=similarity_top_k,
        system_prompt=system_prompt
    )

# Retrieve cached engine instance without rebuilding on every rerun
chat_engine = get_cached_chat_engine(similarity_top_k, selected_system_prompt)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to the Drug Safety Analysis Assistant. "
                "You can ask me questions about FAERS data, adverse events "
                "associated with SGLT2 inhibitors and GLP-1 receptor agonists."
            )
        }
    ]

# --- Render Chat History ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if stored in message history
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Show Sources"):
                for src_idx, source in enumerate(msg["sources"], 1):
                    file_name = source.metadata.get("file_name", "Unknown File")
                    page_num = source.metadata.get("page_label", "N/A")
                    score = f"{source.score:.4f}" if hasattr(source, "score") and source.score is not None else "N/A"
                    
                    st.markdown(f"**Source {src_idx}** — File: `{file_name}` | Page: `{page_num}` | Score: `{score}`")
                    st.info(source.text)

        # User Feedback Buttons for Assistant responses
        if msg["role"] == "assistant" and idx > 0:
            col1, col2, _ = st.columns([0.05, 0.05, 0.9])
            with col1:
                st.button("👍", key=f"thumb_up_{idx}")
            with col2:
                st.button("👎", key=f"thumb_down_{idx}")

# --- Handle User Input ---
if prompt := st.chat_input("Ask about FAERS signals (e.g., 'What is Invokana?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Antidiabetic drug data..."):
            response = chat_engine.chat(prompt)
            response_text = str(response)
            
            # Extract source chunks from LlamaIndex response object
            source_nodes = getattr(response, "source_nodes", [])

            st.markdown(response_text)

            # Display source chunks for the current turn
            if source_nodes:
                with st.expander("🔍 Show Sources"):
                    for src_idx, source in enumerate(source_nodes, 1):
                        file_name = source.metadata.get("file_name", "Unknown File")
                        page_num = source.metadata.get("page_label", "N/A")
                        score = f"{source.score:.4f}" if hasattr(source, "score") and source.score is not None else "N/A"
                        
                        st.markdown(f"**Source {src_idx}** — File: `{file_name}` | Page: `{page_num}` | Score: `{score}`")
                        st.info(source.text)

    # Save to history with sources attached
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": source_nodes
    })
    st.rerun()