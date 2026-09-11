"""
Agent orchestration layer.

- classify_request(): categorize a customer message
- summarize_message(): one/two-sentence summary of a customer message
- handle_message(): the main multi-tool loop. Sends the conversation + tool
  schemas to Ollama, executes any tool calls the model makes (order lookup,
  refund check, ticket creation, policy search, ...), feeds results back,
  and returns a final professional support response.
"""

import json

import config
import tools
from ollama_client import chat, OllamaError

MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = f"""You are a professional, empathetic customer support agent for {config.COMPANY_NAME}, \
a company that sells wireless headphones, earbuds, and speakers.

You have access to tools to look up orders, products, stock, refund eligibility, company policy, \
and to open support tickets. Use them whenever you need real data instead of guessing.

Guidelines:
- If the customer mentions an order problem (broken, missing, refund, delayed), look up the order \
first with get_order_status before anything else.
- For any refund request, call check_refund_eligibility before promising or denying a refund.
- For any question about policy, refund rules, warranty, or shipping rules, call \
search_company_policy and base your answer on the retrieved text rather than assuming.
- If the issue can't be resolved immediately (e.g. broken item, escalation, unresolved complaint), \
create a support ticket with create_support_ticket using an appropriate category \
({', '.join(config.SUPPORT_CATEGORIES)}) so a human can follow up. Mention the ticket ID to the customer.
- Keep the final reply concise (3-6 sentences), warm, and professional. Don't mention internal \
tool names or that you "called a function" - just use the information naturally.
- If you don't have a customer_id or order_id you need, ask the customer for it instead of inventing one.
"""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_request(message: str) -> str:
    """Classify a customer message into one of config.SUPPORT_CATEGORIES."""
    prompt = (
        "Classify the following customer support message into exactly one of these categories: "
        f"{', '.join(config.SUPPORT_CATEGORIES)}.\n"
        "Respond with ONLY the category word, nothing else.\n\n"
        f"Message: \"{message}\""
    )
    try:
        reply = chat([{"role": "user", "content": prompt}], temperature=0.0)
    except OllamaError:
        return "OTHER"

    text = reply.get("content", "").strip().upper()
    for category in config.SUPPORT_CATEGORIES:
        if category in text:
            return category
    return "OTHER"


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def summarize_message(message: str) -> str:
    """Produce a 1-2 sentence summary of a customer message."""
    prompt = (
        "Summarize the following customer support message in 1-2 short sentences, "
        "capturing the key issue and any order/product details mentioned. "
        "Respond with ONLY the summary.\n\n"
        f"Message: \"{message}\""
    )
    try:
        reply = chat([{"role": "user", "content": prompt}], temperature=0.2)
    except OllamaError as e:
        return f"(summary unavailable: {e})"
    return reply.get("content", "").strip()


# ---------------------------------------------------------------------------
# Main multi-tool agent loop
# ---------------------------------------------------------------------------

def _execute_tool_call(tool_call):
    """Run the python function behind a single tool call and return its result."""
    fn_name = tool_call["function"]["name"]
    raw_args = tool_call["function"].get("arguments", {})
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
    else:
        args = raw_args or {}

    fn = tools.TOOL_FUNCTIONS.get(fn_name)
    if not fn:
        return {"error": f"Unknown tool '{fn_name}'"}

    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"Bad arguments for {fn_name}: {e}"}


def handle_message(user_message: str, history: list, customer_id: str = None):
    """
    Run the full multi-tool agent loop for one user turn.

    history: list of {"role": "user"|"assistant", "content": str} from prior turns
             (Gradio-style plain history, no tool messages needed across turns).
    customer_id: optional known customer ID to hint the model with, since the
                 chat UI doesn't otherwise authenticate the user.

    Returns: dict with keys "reply", "tool_calls" (list of {name, args, result})
    """
    system = SYSTEM_PROMPT
    if customer_id:
        system += f"\nThe current customer's ID is {customer_id}. Use it for ticket creation if needed."

    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": user_message}
    ]

    tool_call_log = []

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            assistant_msg = chat(messages, tools=tools.TOOL_SCHEMAS)
        except OllamaError as e:
            return {"reply": f"Sorry, I'm having trouble reaching the support system: {e}", "tool_calls": tool_call_log}

        messages.append(assistant_msg)
        tool_calls = assistant_msg.get("tool_calls")

        if not tool_calls:
            return {"reply": assistant_msg.get("content", "").strip(), "tool_calls": tool_call_log}

        for tc in tool_calls:
            result = _execute_tool_call(tc)
            tool_call_log.append({
                "name": tc["function"]["name"],
                "args": tc["function"].get("arguments", {}),
                "result": result,
            })
            messages.append({
                "role": "tool",
                "content": json.dumps(result, default=str),
            })

    # Safety net if the model loops without converging
    return {
        "reply": "I've gathered the relevant details but need a moment longer - could you rephrase "
                 "or simplify your request?",
        "tool_calls": tool_call_log,
    }
