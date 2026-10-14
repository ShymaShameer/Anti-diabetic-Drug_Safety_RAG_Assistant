import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import (
    LLM_MODEL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_CACHE_PATH,
)

# Load environment variables from .env file
load_dotenv()


def _get_groq_api_key() -> str:
    """Helper to validate and retrieve the Groq API key."""
    api_key: str | None = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure it's set in your .env file."
        )
    return api_key


def initialise_llm() -> Groq:
    """Initialises the primary Groq LLM with core parameters from config."""
    return Groq(
        api_key=_get_groq_api_key(),
        model=LLM_MODEL,
    )


#def initialise_hyde_llm() -> Groq:
    #"""Initializes a lightweight, fast Groq LLM for HyDE query expansion."""
    #return Groq(
    #    api_key=_get_groq_api_key(),
     #   model="llama-3.1-8b-instant",  # Fast model for low-latency query expansion
     #   temperature=0.1,
    #)


def get_embedding_model() -> HuggingFaceEmbedding:
    """Initialises and returns the HuggingFace embedding model."""
    # Ensure cache directory exists
    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix()
    )