"""
Minimal client for talking to a local Ollama server.
No external SDK required - just `requests`.
"""

import requests
import config


class OllamaError(RuntimeError):
    pass


def chat(messages, tools=None, model=None, temperature=0.3):
    """
    Call POST /api/chat on the local Ollama server.

    messages: list of {"role": "system"|"user"|"assistant"|"tool", "content": str, ...}
    tools: optional list of tool schemas (OpenAI-style function calling format)
    Returns the raw 'message' dict from Ollama's response, e.g.
        {"role": "assistant", "content": "...", "tool_calls": [...] }
    """
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model or config.OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            "Could not reach Ollama at "
            f"{config.OLLAMA_BASE_URL}. Is 'ollama serve' running, and did you "
            f"'ollama pull {config.OLLAMA_CHAT_MODEL}'?"
        ) from e
    except requests.exceptions.HTTPError as e:
        raise OllamaError(f"Ollama chat call failed: {e} - {resp.text}") from e

    data = resp.json()
    return data.get("message", {"role": "assistant", "content": ""})


def embed(text, model=None):
    """
    Call POST /api/embeddings on the local Ollama server.
    Returns a list[float] embedding vector.
    """
    url = f"{config.OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": model or config.OLLAMA_EMBED_MODEL, "prompt": text}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            f"Could not reach Ollama at {config.OLLAMA_BASE_URL}. Is 'ollama serve' running, "
            f"and did you 'ollama pull {config.OLLAMA_EMBED_MODEL}'?"
        ) from e
    except requests.exceptions.HTTPError as e:
        raise OllamaError(f"Ollama embeddings call failed: {e} - {resp.text}") from e

    data = resp.json()
    embedding = data.get("embedding")
    if not embedding:
        raise OllamaError(f"No embedding returned for text: {text[:60]!r}")
    return embedding
