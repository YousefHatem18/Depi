"""
Gradio customer support chatbot for NovaTech Audio.

Run with:
    python app.py

Requires a local Ollama server running with the chat + embedding models pulled
(see config.py / README.md).
"""

import gradio as gr

import agent
import tools
import rag
import config


# ---------------------------------------------------------------------------
# Chat tab callbacks
# ---------------------------------------------------------------------------

def respond(user_message, chat_history, customer_id):
    """Main chat callback. chat_history is a list of {"role", "content"} dicts
    (Gradio 'messages' format), which we also reuse as the agent's history."""
    if not user_message or not user_message.strip():
        return chat_history, "", "", ""

    category = agent.classify_request(user_message)
    summary = agent.summarize_message(user_message)

    # Build plain-role history (user/assistant only) for the agent call
    plain_history = [
        {"role": m["role"], "content": m["content"]}
        for m in chat_history
        if m["role"] in ("user", "assistant")
    ]

    result = agent.handle_message(user_message, plain_history, customer_id=customer_id.strip() or None)

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": result["reply"]},
    ]

    tool_log_lines = []
    for tc in result["tool_calls"]:
        tool_log_lines.append(f"🔧 **{tc['name']}**({tc['args']}) → `{tc['result']}`")
    tool_log_text = "\n\n".join(tool_log_lines) if tool_log_lines else "_No tools were called for this message._"

    detected_text = f"**Category:** {category}\n\n**Summary:** {summary}"

    return chat_history, "", detected_text, tool_log_text


def clear_chat():
    return [], "", ""


# ---------------------------------------------------------------------------
# Tickets tab callback
# ---------------------------------------------------------------------------

def refresh_tickets():
    ticket_list = tools.list_tickets()
    if not ticket_list:
        return "_No tickets created yet._"
    rows = ["| Ticket ID | Customer | Category | Priority | Status | Issue |",
            "|---|---|---|---|---|---|"]
    for t in reversed(ticket_list):
        rows.append(
            f"| {t['ticket_id']} | {t['customer_id']} | {t['category']} | "
            f"{t['priority']} | {t['status']} | {t['issue'][:60]} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Knowledge base tab callbacks
# ---------------------------------------------------------------------------

def build_kb():
    try:
        n = rag.build_index()
        return f"✅ Indexed {n} chunks from company policy documents."
    except Exception as e:
        return f"❌ Failed to build index: {e}"


def query_kb(question):
    if not question or not question.strip():
        return "Enter a question first.", ""
    try:
        result = rag.generate_answer(question)
    except Exception as e:
        return f"❌ {e}", ""
    sources_text = "\n\n".join(
        f"**{r['source']}** (score={r['score']:.3f})\n> {r['text'][:300]}..."
        for r in result["sources"]
    )
    return result["answer"], sources_text


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title=f"{config.COMPANY_NAME} Support") as demo:
    gr.Markdown(f"# 🎧 {config.COMPANY_NAME} — AI Customer Support")

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot( height=480, label="Support Chat")
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="e.g. My headphones arrived broken, order ORD-20458, can I get a refund?",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    customer_id_box = gr.Textbox(
                        label="Customer ID (optional, e.g. CUST-3391)",
                        placeholder="CUST-3391",
                        scale=3,
                    )
                    clear_btn = gr.Button("Clear conversation", scale=1)

            with gr.Column(scale=2):
                gr.Markdown("### 🏷️ Request Classification")
                detected_panel = gr.Markdown("_Send a message to see its category and summary._")
                gr.Markdown("### 🔧 Tool Activity")
                tool_log_panel = gr.Markdown("_No tools called yet._")

        send_btn.click(
            respond,
            inputs=[msg, chatbot, customer_id_box],
            outputs=[chatbot, msg, detected_panel, tool_log_panel],
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot, customer_id_box],
            outputs=[chatbot, msg, detected_panel, tool_log_panel],
        )
        clear_btn.click(clear_chat, outputs=[chatbot, detected_panel, tool_log_panel])

    with gr.Tab("Tickets"):
        gr.Markdown("### 🎫 Support Tickets")
        refresh_btn = gr.Button("Refresh")
        tickets_panel = gr.Markdown("_Click refresh to load tickets._")
        refresh_btn.click(refresh_tickets, outputs=tickets_panel)

    with gr.Tab("Knowledge Base (RAG)"):
        gr.Markdown(
            "### 📚 Company Policy Knowledge Base\n"
            "Build the vector index from the policy documents in `data/policies/`, "
            "then ask a policy question directly (bypassing the chat agent) to test retrieval."
        )
        build_btn = gr.Button("Build / Rebuild Index")
        build_status = gr.Markdown("")
        build_btn.click(build_kb, outputs=build_status)

        kb_question = gr.Textbox(label="Ask a policy question", placeholder="Can I return my headphones after 40 days?")
        kb_ask_btn = gr.Button("Ask", variant="primary")
        kb_answer = gr.Markdown(label="Answer")
        kb_sources = gr.Markdown(label="Retrieved sources")
        kb_ask_btn.click(query_kb, inputs=kb_question, outputs=[kb_answer, kb_sources])

    with gr.Tab("Demo Data"):
        gr.Markdown(
            "### Sample IDs you can try in the Chat tab\n\n"
            "**Orders:**\n"
            "- `ORD-20458` — CUST-3391 — NovaBeat Wireless Headphones — Delivered\n"
            "- `ORD-20501` — CUST-3391 — NovaBoom Portable Speaker — In Transit\n"
            "- `ORD-19875` — CUST-2210 — NovaBuds Pro Earbuds (x2) — Delivered\n"
            "- `ORD-20602` — CUST-4477 — NovaBeat Studio Headphones — Delivered\n\n"
            "**Products:**\n"
            "- `PRD-1001` — NovaBeat Wireless Headphones ($129.99)\n"
            "- `PRD-1002` — NovaBuds Pro Earbuds ($89.99, out of stock)\n"
            "- `PRD-1003` — NovaBoom Portable Speaker ($59.99)\n"
            "- `PRD-1004` — NovaBeat Studio Headphones ($199.99)\n\n"
            "**Try this required multi-tool example:**\n"
            "> \"My headphones arrived broken, order ORD-20458. Can I get a refund?\"\n\n"
            "Expected flow: get_order_status → check_refund_eligibility → create_support_ticket → response."
        )

if __name__ == "__main__":
    demo.launch()
