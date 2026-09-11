# NovaTech Audio — AI Customer Support Agent

A Gradio-based customer support chatbot with multi-tool reasoning and a RAG
knowledge base, powered by a local Ollama model.

## What's inside

| File | Purpose |
|---|---|
| `app.py` | Gradio UI (Chat / Tickets / Knowledge Base / Demo Data tabs) |
| `agent.py` | Classification, summarization, and the multi-tool reasoning loop |
| `tools.py` | The 5 required tools + `search_company_policy` (RAG) tool + JSON schemas |
| `rag.py` | Chunking, embeddings, vector index, retrieval, RAG answer generation |
| `ollama_client.py` | Thin `requests`-based wrapper around the local Ollama HTTP API |
| `config.py` | Model names, file paths, chunking params — edit here to tune things |
| `data/products.json` | Demo product catalog |
| `data/orders.json` | Demo orders |
| `data/tickets.json` | Created tickets get appended here at runtime |
| `data/policies/*.md` | The company knowledge base documents used for RAG |

## 1. Install Ollama models

You said you already have Ollama installed. Pull two models:

```bash
ollama pull llama3.1          # chat model — must support tool/function calling
ollama pull nomic-embed-text  # embedding model for RAG
```

`llama3.1`, `llama3.2`, `qwen2.5`, and `mistral-nemo` all support Ollama's
tool-calling format. If you use a different one, set it in `config.py` or via
the `OLLAMA_CHAT_MODEL` environment variable.

Make sure the server is running:

```bash
ollama serve
```

(On macOS/Windows the desktop app usually runs this for you already.)

## 2. Set up the Python project in VS Code

```bash
cd customer_support_agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open the folder in VS Code (`code .`) and select the `.venv` interpreter.

## 3. Run it

```bash
python app.py
```

Gradio will print a local URL (usually `http://127.0.0.1:7860`) — open it in
your browser.

## 4. Try the required multi-tool example

In the **Chat** tab, type:

> My headphones arrived broken, order ORD-20458. Can I get a refund?

Expected flow (visible in the "Tool Activity" panel on the right):
1. `get_order_status("ORD-20458")` — confirms the order and delivery date
2. `check_refund_eligibility("ORD-20458", "arrived broken")` — confirms eligibility
3. `create_support_ticket(...)` — opens a REFUND ticket
4. The model writes a professional final reply referencing the ticket ID

You can check the **Tickets** tab afterward to see it stored.

## 5. Try the RAG knowledge base

Go to the **Knowledge Base (RAG)** tab and click **Build / Rebuild Index**
first (this embeds the `.md` files in `data/policies/` — only needs to be
done once, or whenever you edit the policy docs). Then ask something like:

> Can I return my headphones after 40 days if they still work fine?

The answer is generated only from retrieved policy text, and the sources /
similarity scores are shown underneath so you can see what was retrieved.

The main Chat agent also has access to this same `search_company_policy`
tool, so if you ask a policy question directly in the Chat tab (e.g. *"What's
your warranty on the NovaBeat Studio headphones?"*), it will call it too.

## Sample IDs for testing

**Orders:** `ORD-20458`, `ORD-20501` (in transit), `ORD-19875`, `ORD-20602`
**Products:** `PRD-1001`, `PRD-1002` (out of stock), `PRD-1003`, `PRD-1004`
**Customers:** `CUST-3391`, `CUST-2210`, `CUST-4477`

## Notes / things you can extend

- `data/tickets.json` is a flat file for simplicity — swap in SQLite easily
  if you want persistence across a real deployment.
- The classify/summarize functions run as separate lightweight LLM calls and
  are shown in the "Request Classification" panel next to the chat.
- If Ollama isn't running, tool calls and chat will raise a clear
  `OllamaError` explaining what to check, instead of failing silently.
- To add more policy knowledge, just drop more `.md` files into
  `data/policies/` and click "Build / Rebuild Index" again.
