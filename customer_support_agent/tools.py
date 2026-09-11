"""
The five (+1) callable tools the agent can use, plus the JSON schemas
that describe them to the LLM for function calling.

Tools:
    get_order_status(order_id)
    get_product_info(product_id)
    check_stock(product_id)
    create_support_ticket(customer_id, issue, category, order_id, priority)
    check_refund_eligibility(order_id, reason)
    search_company_policy(query)   <- RAG tool, wired in from rag.py
"""

import json
import os
import uuid
from datetime import datetime, date

import config

# rag is imported lazily inside search_company_policy so that the rest of the
# tools (and the Gradio app) still work even before the KB index is built.


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _find(items, key, value):
    for item in items:
        if item.get(key) == value:
            return item
    return None


# ---------------------------------------------------------------------------
# Tool 1: Order Status
# ---------------------------------------------------------------------------

def get_order_status(order_id: str) -> dict:
    """Look up an order and return its status, dates, and amount."""
    orders = _load_json(config.ORDERS_FILE)
    order = _find(orders, "order_id", order_id)
    if not order:
        return {"found": False, "error": f"No order found with ID {order_id}."}
    return {"found": True, **order}


# ---------------------------------------------------------------------------
# Tool 2: Product Info
# ---------------------------------------------------------------------------

def get_product_info(product_id: str) -> dict:
    """Look up a product's name, category, price, description, and warranty."""
    products = _load_json(config.PRODUCTS_FILE)
    product = _find(products, "product_id", product_id)
    if not product:
        return {"found": False, "error": f"No product found with ID {product_id}."}
    return {"found": True, **product}


# ---------------------------------------------------------------------------
# Tool 3: Stock Check
# ---------------------------------------------------------------------------

def check_stock(product_id: str) -> dict:
    """Check how many units of a product are currently in stock."""
    products = _load_json(config.PRODUCTS_FILE)
    product = _find(products, "product_id", product_id)
    if not product:
        return {"found": False, "error": f"No product found with ID {product_id}."}
    stock = product.get("stock", 0)
    return {
        "found": True,
        "product_id": product_id,
        "name": product["name"],
        "in_stock": stock > 0,
        "quantity_available": stock,
    }


# ---------------------------------------------------------------------------
# Tool 4: Create Support Ticket
# ---------------------------------------------------------------------------

def create_support_ticket(
    customer_id: str,
    issue: str,
    category: str = "OTHER",
    order_id: str = None,
    priority: str = "normal",
) -> dict:
    """Create and persist a new support ticket. Returns the created ticket."""
    if category not in config.SUPPORT_CATEGORIES:
        category = "OTHER"

    tickets = _load_json(config.TICKETS_FILE)
    ticket = {
        "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
        "customer_id": customer_id,
        "order_id": order_id,
        "category": category,
        "issue": issue,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    tickets.append(ticket)
    _save_json(config.TICKETS_FILE, tickets)
    return {"created": True, **ticket}


def list_tickets() -> list:
    """Return all stored tickets (used by the Gradio 'Tickets' tab)."""
    return _load_json(config.TICKETS_FILE)


# ---------------------------------------------------------------------------
# Tool 5: Refund Eligibility
# ---------------------------------------------------------------------------

def check_refund_eligibility(order_id: str, reason: str = "defective") -> dict:
    """
    Determine whether an order is eligible for a refund, based on company policy:
      - Not yet delivered -> not eligible yet.
      - Delivered <= 30 days ago -> eligible (standard window).
      - Delivered <= 60 days ago AND reason is defective/damaged -> eligible.
      - Delivered > 60 days ago -> not eligible for a refund; warranty may apply instead.
    """
    orders = _load_json(config.ORDERS_FILE)
    order = _find(orders, "order_id", order_id)
    if not order:
        return {"found": False, "error": f"No order found with ID {order_id}."}

    if order["status"] != "Delivered" or not order.get("delivery_date"):
        return {
            "found": True,
            "order_id": order_id,
            "eligible": False,
            "reason": "Order has not been delivered yet, so a refund cannot be processed. "
                      "If it's an item lost in transit, that follows the shipping policy instead.",
        }

    delivered = datetime.strptime(order["delivery_date"], "%Y-%m-%d").date()
    days_since = (date.today() - delivered).days
    reason_lower = (reason or "").lower()
    is_defect_related = any(
        kw in reason_lower for kw in ["defect", "broken", "damage", "not working", "doesn't work", "faulty"]
    )

    if days_since <= 30:
        return {
            "found": True,
            "order_id": order_id,
            "eligible": True,
            "days_since_delivery": days_since,
            "reason": "Within the standard 30-day refund window.",
        }

    if days_since <= 60 and is_defect_related:
        return {
            "found": True,
            "order_id": order_id,
            "eligible": True,
            "days_since_delivery": days_since,
            "reason": "Outside the 30-day standard window, but within 60 days and reported as "
                      "defective/damaged, so it is still eligible for a refund or replacement.",
        }

    return {
        "found": True,
        "order_id": order_id,
        "eligible": False,
        "days_since_delivery": days_since,
        "reason": "Outside the 60-day refund window. This may still be covered under the "
                  "product warranty for a free repair/replacement rather than a refund.",
    }


# ---------------------------------------------------------------------------
# Tool 6: Company Policy Search (RAG)
# ---------------------------------------------------------------------------

def search_company_policy(query: str) -> dict:
    """Search the company's policy knowledge base and return relevant excerpts."""
    import rag  # lazy import to avoid a hard dependency at module load time

    try:
        results = rag.search(query, top_k=config.RAG_TOP_K)
    except Exception as e:
        return {"found": False, "error": str(e)}

    if not results:
        return {"found": False, "error": "No relevant policy content found."}

    return {
        "found": True,
        "excerpts": [
            {"source": r["source"], "text": r["text"], "score": round(r["score"], 3)}
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Tool schemas for Ollama function calling (OpenAI-style JSON schema)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the delivery/order status, dates, and amount for a given order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID, e.g. ORD-20458"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Get product details (name, price, description, warranty) for a product ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "The product ID, e.g. PRD-1001"},
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check how many units of a product are currently available in stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "The product ID, e.g. PRD-1002"},
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_refund_eligibility",
            "description": "Determine whether an order is eligible for a refund based on company policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID, e.g. ORD-20458"},
                    "reason": {
                        "type": "string",
                        "description": "Why the customer wants a refund, e.g. 'arrived broken', 'changed my mind'",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Open a new support ticket for the customer. Use this after you have enough "
                            "information about the issue (e.g. after checking order/refund status).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The customer's ID, e.g. CUST-3391"},
                    "issue": {"type": "string", "description": "A concise summary of the customer's issue."},
                    "category": {
                        "type": "string",
                        "enum": config.SUPPORT_CATEGORIES,
                        "description": "The support category for this ticket.",
                    },
                    "order_id": {"type": "string", "description": "Related order ID, if any."},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "description": "Ticket priority. Use 'high' for escalations per the support guidelines.",
                    },
                },
                "required": ["customer_id", "issue", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_company_policy",
            "description": "Search NovaTech Audio's refund, shipping, warranty, and support policy "
                            "documents for relevant information. Always use this before answering any "
                            "question about policy, refund rules, warranty coverage, or shipping rules, "
                            "so your answer matches the official company policy rather than a guess.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The policy question to search for."},
                },
                "required": ["query"],
            },
        },
    },
]

# Maps tool name -> python callable, used by the agent to execute tool calls.
TOOL_FUNCTIONS = {
    "get_order_status": get_order_status,
    "get_product_info": get_product_info,
    "check_stock": check_stock,
    "create_support_ticket": create_support_ticket,
    "check_refund_eligibility": check_refund_eligibility,
    "search_company_policy": search_company_policy,
}
