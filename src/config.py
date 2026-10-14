from pathlib import Path

# --- LLM Model Configuration ---
LLM_MODEL: str = "openai/gpt-oss-120b"
LLM_MAX_NEW_TOKENS: int = 450
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_FREQUENCY_PENALTY: float = 0.2
LLM_PRESENCE_PENALTY: float = 0.2
LLM_REPETITION_PENALTY: float = 1.03


LLM_SYSTEM_PROMPT: str = (
    "You are a helpful assistant.Answer questions only using the provided documents.If the answer isn't in the documents, say you don't know."
)



# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
#EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 20
RERANKER_TOP_N = 3  
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3" 
# The size of each text chunk in tokens
CHUNK_SIZE: int = 350
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 60


# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900


# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"

BEST_RERANKER_STRATEGY: dict[str, int] = {'retriever_k': 10, 'reranker_n': 2}