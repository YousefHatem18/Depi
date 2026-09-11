"""
Simple, dependency-light RAG pipeline:
    1. Load policy .md files
    2. Chunk them (word-based sliding window with overlap)
    3. Embed each chunk with Ollama (nomic-embed-text)
    4. Store vectors + metadata on disk (numpy + json)
    5. Cosine-similarity search at query time
    6. (generate_answer) feed retrieved chunks to the chat model as context

Run this file directly to (re)build the index:
    python rag.py
"""

import os
import json
import glob

import numpy as np

import config
from ollama_client import embed, chat


# ---------------------------------------------------------------------------
# 1-2. Load + chunk documents
# ---------------------------------------------------------------------------

def _load_documents():
    """Return list of {"source": filename, "text": full file content}."""
    docs = []
    for path in sorted(glob.glob(os.path.join(config.POLICIES_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": os.path.basename(path), "text": text})
    return docs


def _chunk_text(text, chunk_size=config.CHUNK_SIZE_WORDS, overlap=config.CHUNK_OVERLAP_WORDS):
    """Word-based sliding-window chunking with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunks():
    """Load all policy docs and split into (source, chunk_text) pairs."""
    all_chunks = []
    for doc in _load_documents():
        for chunk in _chunk_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": chunk})
    return all_chunks


# ---------------------------------------------------------------------------
# 3-4. Build and persist the embedding index
# ---------------------------------------------------------------------------

def build_index(verbose=True):
    """Embed every chunk and save vectors + metadata to disk."""
    os.makedirs(config.KB_INDEX_DIR, exist_ok=True)
    chunks = build_chunks()
    if not chunks:
        raise RuntimeError(f"No policy documents found in {config.POLICIES_DIR}")

    vectors = []
    metadata = []
    for i, c in enumerate(chunks):
        if verbose:
            print(f"Embedding chunk {i + 1}/{len(chunks)} from {c['source']}...")
        vec = embed(c["text"])
        vectors.append(vec)
        metadata.append({"source": c["source"], "text": c["text"]})

    matrix = np.array(vectors, dtype=np.float32)
    np.save(config.KB_EMBEDDINGS_FILE, matrix)
    with open(config.KB_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"Indexed {len(chunks)} chunks -> {config.KB_EMBEDDINGS_FILE}")
    return len(chunks)


def index_exists():
    return os.path.exists(config.KB_EMBEDDINGS_FILE) and os.path.exists(config.KB_METADATA_FILE)


def ensure_index(verbose=True):
    """Build the index only if it doesn't already exist on disk."""
    if not index_exists():
        build_index(verbose=verbose)


# ---------------------------------------------------------------------------
# 5. Vector search
# ---------------------------------------------------------------------------

def _cosine_similarity(query_vec, matrix):
    query_vec = np.array(query_vec, dtype=np.float32)
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return m_norm @ q_norm


def search(query, top_k=config.RAG_TOP_K):
    """Return the top_k most relevant chunks for a query, with similarity scores."""
    ensure_index(verbose=False)

    matrix = np.load(config.KB_EMBEDDINGS_FILE)
    with open(config.KB_METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_vec = embed(query)
    scores = _cosine_similarity(query_vec, matrix)
    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "source": metadata[idx]["source"],
            "text": metadata[idx]["text"],
            "score": float(scores[idx]),
        })
    return results


# ---------------------------------------------------------------------------
# 6. Generate a grounded answer using retrieved context
# ---------------------------------------------------------------------------

def generate_answer(query, top_k=config.RAG_TOP_K):
    """
    Full RAG answer: retrieve relevant policy chunks, then ask the chat model
    to answer using ONLY that retrieved context.
    """
    results = search(query, top_k=top_k)
    context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    )

    system_prompt = (
        f"You are a policy assistant for {config.COMPANY_NAME}. Answer the user's question "
        "using ONLY the context below, which was retrieved from the official company policy "
        "documents. If the context doesn't fully answer the question, say what is missing. "
        "Always prefer this retrieved context over any general knowledge you might have.\n\n"
        f"Context:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    reply = chat(messages)
    return {"answer": reply.get("content", ""), "sources": results}


if __name__ == "__main__":
    n = build_index()
    print(f"Done. {n} chunks indexed.")
    demo = search("What if my headphones arrived broken?")
    for r in demo:
        print(f"\n--- {r['source']} (score={r['score']:.3f}) ---\n{r['text'][:200]}...")
