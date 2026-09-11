"""
Central configuration for the customer support agent.
Change the model names here if you have different Ollama models pulled.

Recommended models to pull before running:
    ollama pull llama3.1          # or llama3.2 / qwen2.5 - must support tool calling
    ollama pull nomic-embed-text   # for RAG embeddings
"""

import os

# --- Ollama connection ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat/completion model. Must support Ollama "tools" (function calling).
# Good options: "llama3.1", "llama3.2", "qwen2.5", "mistral-nemo"
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1")

# Embedding model for RAG
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PRODUCTS_FILE = os.path.join(DATA_DIR, "products.json")
ORDERS_FILE = os.path.join(DATA_DIR, "orders.json")
TICKETS_FILE = os.path.join(DATA_DIR, "tickets.json")
POLICIES_DIR = os.path.join(DATA_DIR, "policies")

KB_INDEX_DIR = os.path.join(DATA_DIR, "kb_index")
KB_EMBEDDINGS_FILE = os.path.join(KB_INDEX_DIR, "embeddings.npy")
KB_METADATA_FILE = os.path.join(KB_INDEX_DIR, "metadata.json")

# --- RAG chunking ---
CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 30
RAG_TOP_K = 3

# --- Company info (used in system prompts) ---
COMPANY_NAME = "NovaTech Audio"
SUPPORT_CATEGORIES = ["ORDER", "REFUND", "PRODUCT", "COMPLAINT", "TECHNICAL", "OTHER"]
