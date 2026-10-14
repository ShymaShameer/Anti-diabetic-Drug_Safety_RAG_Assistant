# for cleaner output suppressing the log messages
import logging

# Silence HTTP request logs from httpx and urllib3
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Silence internal LlamaIndex verbose logs
logging.getLogger("llama_index").setLevel(logging.WARNING)


from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage
 )


from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatMemoryBuffer,ChatSummaryMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.readers.file import PyMuPDFReader # for advanced PDF parsing
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import TransformQueryEngine # Query rewriting
from llama_index.core.indices.query.query_transform import HyDEQueryTransform 
from llama_index.core.retrievers import TransformRetriever
from llama_index.core.base.base_retriever import BaseRetriever

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    RERANKER_TOP_N,
    RERANKER_MODEL_NAME,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
    BEST_RERANKER_STRATEGY,
)
from src.model_loader import (
    get_embedding_model,
    initialise_llm,
    #initialise_hyde_llm
)    

def _create_new_vector_store(
        embed_model: HuggingFaceEmbedding
) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from documents."""

    print(
        "Creating new vector store from all files in the 'data' directory..."
    )

# Use PyMuPDFReader for robust extraction of PDF text and image descriptions/metadata
    file_extractor = {".pdf": PyMuPDFReader()}

    # To read all the files in the specified directory.
    documents: list[Document] = SimpleDirectoryReader(
        input_dir=DATA_PATH,
        required_exts=[".pdf"],          # Ensures only PDFs are loaded
        file_extractor=file_extractor,
        recursive=True
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. Cannot create vector store."
        )

    # To break the long document into smaller, more manageable chunks.
    text_splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # This is the core of the vector store. It takes the text chunks,
    # uses the embedding model to convert them to vectors, and stores them.
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model
    )

    # To save the newly created index to disk for future use.
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print("Vector store created and saved.")
    return index

def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Loads the vector store from disk if it exists;
    otherwise, creates a new one.
    """

    # Create the parent directory if it doesn't exist.
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # Check if the directory contains any files.
    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        # We must provide the embed_model when loading the index.
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model
        )
    else:
        # If the directory is empty,
        # call our internal function to build the index.
        return _create_new_vector_store(embed_model)


def get_chat_engine(
    llm: Groq,
    embed_model: HuggingFaceEmbedding,
    similarity_top_k: int = 3,
    system_prompt: str | None = None
) -> CondensePlusContextChatEngine:
    """Initialises and returns the main conversational RAG chat engine with HyDE."""

    vector_index: VectorStoreIndex = get_vector_store(embed_model)

    # 1. Use the dynamic similarity_top_k parameter from Streamlit sidebar
    base_retriever: BaseRetriever = vector_index.as_retriever(
        similarity_top_k=similarity_top_k
    )

    # 'include_original=True' ensures the system searches with BOTH
    # the hallucinated answer and the user's real question.
    hyde = HyDEQueryTransform(
        include_original=True,
        llm=llm,
    )

    # Query transformation wrapper
    hyde_retriever = TransformRetriever(
        retriever=base_retriever,
        query_transform=hyde
    )

    reranker = SentenceTransformerRerank(
        top_n=RERANKER_TOP_N,
        model=RERANKER_MODEL_NAME
    )

    memory = ChatSummaryMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT,
        llm=llm, 
    )

    # 2. Use the dynamic system_prompt if passed, otherwise fall back to LLM_SYSTEM_PROMPT
    active_system_prompt = system_prompt if system_prompt is not None else LLM_SYSTEM_PROMPT

    chat_engine = CondensePlusContextChatEngine(
        retriever=hyde_retriever,
        llm=llm,
        memory=memory,
        system_prompt=active_system_prompt,
        node_postprocessors=[reranker]
    )

    return chat_engine