import json
import re
import gradio as gr

from openai import OpenAI

from tools.support_tools import (
    check_refund_eligibility,
    check_stock,
    create_support_ticket,
    get_order_status,
    get_product_info,
)


# =========================================================
# Ollama Local Model
# =========================================================

model = "qwen3:4b"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


# =========================================================
# Tools Definition
# =========================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a support ticket for a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string"
                    },
                    "issue": {
                        "type": "string"
                    },
                },
                "required": [
                    "customer_id",
                    "issue"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "check_refund_eligibility",
            "description": "Check whether an order is eligible for a refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string"
                    }
                },
                "required": [
                    "order_id"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check the current stock of a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string"
                    }
                },
                "required": [
                    "product_id"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Get product name, price, description, stock, and warranty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string"
                    }
                },
                "required": [
                    "product_id"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the status and details of an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string"
                    }
                },
                "required": [
                    "order_id"
                ],
            },
        },
    },
]


# =========================================================
# Extractor Agent Prompt
# =========================================================

Extractor_prompt = """
You are an Extractor agent.

Your job is to extract important information from the user's message
and give it to the next agent.

You must understand English, Arabic, or any other language.

Extract information related to:

- customers
- orders
- products
- policies
- tickets
- refunds
- stock

Examples:

User:
"where is my order ORD1001"

Output:
{
    "entity": "order",
    "order_id": "ORD1001",
    "request": "status"
}

User:
"what products do you have?"

Output:
{
    "entity": "product",
    "request": "list"
}

User:
"عايز اعرف سعر السماعات"

Output:
{
    "entity": "product",
    "product_name": "السماعات",
    "request": "information"
}

Your output MUST be valid JSON.
Do not write anything outside the JSON.
"""


# =========================================================
# Tool Agent Prompt
# =========================================================

tool_prompt = """
You are a tool-use agent for a customer support system.

Your job is to analyze the JSON input received from the extractor
and use the appropriate available tool.

Rules:

1. Analyze the entity, request, and available information.

2. Choose the most appropriate tool.

3. Use the values from the JSON as arguments.

4. If required information is missing, DO NOT invent it.

5. Do not answer the user directly.

6. Do not explain your decision.

7. Always use a tool when the requested information can be obtained
   using one.

Examples:

Input:
{
    "entity": "order",
    "order_id": "ORD1001",
    "request": "status"
}

Use:
get_order_status(order_id="ORD1001")


Input:
{
    "entity": "order",
    "order_id": "ORD1001",
    "request": "refund"
}

Use:
check_refund_eligibility(order_id="ORD1001")


Input:
{
    "entity": "product",
    "product_id": "P1001",
    "request": "information"
}

Use:
get_product_info(product_id="P1001")


Input:
{
    "entity": "product",
    "product_id": "P1001",
    "request": "stock"
}

Use:
check_stock(product_id="P1001")
"""


# =========================================================
# Response Agent Prompt
# =========================================================

Response_prompt = """
You are the final response agent for a customer support system.

Your job is to answer the user's original request using the tool result.

Rules:

1. Answer naturally and conversationally.

2. Answer only what the user asked for.

3. Do not return JSON.

4. Do not mention agents, tools, function calls,
   or internal processing.

5. Do not invent information.

6. Use only information provided by the tool result.

7. Respond in the same language as the user.

8. If the tool result does not contain enough information,
   clearly say that the information is unavailable.

9. Keep the response concise and useful.

Example:

User:
"where is my order ORD1001"

Tool result:
{
    "order_id": "ORD1001",
    "status": "Delivered",
    "delivered_date": "2026-09-06"
}

Answer:
"Your order ORD1001 was delivered on September 6, 2026."

Do not output JSON.
"""


# =========================================================
# Available Python Tools
# =========================================================

AVAILABLE_TOOLS = {
    "create_support_ticket": create_support_ticket,
    "check_refund_eligibility": check_refund_eligibility,
    "check_stock": check_stock,
    "get_order_status": get_order_status,
    "get_product_info": get_product_info,
}


def normalize_extracted_data(extracted_data, user_message=""):
    if not isinstance(extracted_data, dict):
        extracted_data = {}

    normalized = {}
    for key, value in extracted_data.items():
        normalized[str(key)] = value

    key_fixes = {
        "order_id_id": "order_id",
        "product_id_id": "product_id",
        "customer_id_id": "customer_id",
        "product_name_name": "product_name",
    }

    for bad_key, good_key in key_fixes.items():
        if bad_key in normalized and good_key not in normalized:
            normalized[good_key] = normalized.pop(bad_key)

    text = (user_message or "").strip()
    lower_text = text.lower()

    if not normalized.get("entity"):
        if "order" in lower_text:
            normalized["entity"] = "order"
        elif "product" in lower_text:
            normalized["entity"] = "product"
        elif "ticket" in lower_text or "support" in lower_text:
            normalized["entity"] = "ticket"
        elif "customer" in lower_text:
            normalized["entity"] = "customer"

    if not normalized.get("request"):
        if "refund" in lower_text:
            normalized["request"] = "refund"
        elif "stock" in lower_text or "available" in lower_text:
            normalized["request"] = "stock"
        elif "status" in lower_text or "where is my order" in lower_text or "track" in lower_text:
            normalized["request"] = "status"
        elif "price" in lower_text or "cost" in lower_text or "warranty" in lower_text or "info" in lower_text:
            normalized["request"] = "information"
        elif "ticket" in lower_text or "support" in lower_text:
            normalized["request"] = "ticket"

    for field_name, pattern in {
        "order_id": r"ORD\d+",
        "product_id": r"P\d+",
        "customer_id": r"C\d+",
    }.items():
        if not normalized.get(field_name):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                normalized[field_name] = match.group(0).upper()

    if not normalized.get("product_name") and "product" in lower_text:
        product_match = re.search(r"(?:product|item|device|headphones|speaker|earbuds|phone|watch|laptop|mouse|keyboard)[\s\w\-]*", text, flags=re.IGNORECASE)
        if product_match:
            normalized["product_name"] = product_match.group(0).strip()

    return normalized


def resolve_tool_call(extracted_data, user_message=""):
    data = normalize_extracted_data(extracted_data, user_message)
    entity = (data.get("entity") or "").lower()
    request = (data.get("request") or "").lower()
    order_id = data.get("order_id")
    product_id = data.get("product_id")
    customer_id = data.get("customer_id")

    if entity == "order" or order_id:
        if request in {"status", "track", "details", "info", "information"}:
            return {
                "function_name": "get_order_status",
                "arguments": {"order_id": order_id}
            }
        if request in {"refund", "eligible", "refund_check"}:
            return {
                "function_name": "check_refund_eligibility",
                "arguments": {"order_id": order_id}
            }

    if entity == "product" or product_id:
        if request in {"stock", "availability"}:
            return {
                "function_name": "check_stock",
                "arguments": {"product_id": product_id}
            }
        if request in {"information", "info", "details", "price", "warranty", "description"}:
            return {
                "function_name": "get_product_info",
                "arguments": {"product_id": product_id}
            }

    if entity == "ticket" or request == "ticket":
        if customer_id:
            issue = user_message.strip() if user_message else "Customer support request"
            return {
                "function_name": "create_support_ticket",
                "arguments": {"customer_id": customer_id, "issue": issue}
            }

    if request == "list":
        return {
            "function_name": "get_product_info",
            "arguments": {"product_id": product_id or "P1001"}
        }

    return None


# =========================================================
# Agent 1: Extractor
# =========================================================

def Extractor_agent(user_message):

    messages = [
        {
            "role": "system",
            "content": Extractor_prompt
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={
            "type": "json_object"
        }
    )

    content = response.choices[0].message.content

    if not content:
        return {}

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return normalize_extracted_data(parsed, user_message)
    except json.JSONDecodeError:
        pass

    text = content.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return normalize_extracted_data(parsed, user_message)
    except json.JSONDecodeError:
        pass

    return normalize_extracted_data({}, user_message)


# =========================================================
# Agent 2: Tool Agent
# =========================================================

def Tool_agent(extracted_data, user_message=""):

    messages = [
        {
            "role": "system",
            "content": tool_prompt
        },
        {
            "role": "user",
            "content": json.dumps(
                normalize_extracted_data(extracted_data, user_message),
                ensure_ascii=False
            )
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        tools=TOOLS,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # Fallback when local model refuses to call a tool
    if not message.tool_calls:
        tool_call = resolve_tool_call(extracted_data, user_message)
        if not tool_call:
            return {
                "success": False,
                "message": "No tool was selected."
            }

        function_name = tool_call["function_name"]
        arguments = tool_call["arguments"]

        function = AVAILABLE_TOOLS.get(function_name)
        if not function:
            return {
                "success": False,
                "message": f"Tool {function_name} not found."
            }

        result = function(**arguments)
        return {
            "success": True,
            "tool_name": function_name,
            "arguments": arguments,
            "result": result
        }

    # Get first tool call
    tool_call = message.tool_calls[0]

    function_name = tool_call.function.name

    arguments = json.loads(
        tool_call.function.arguments
    )

    print("Tool:", function_name)
    print("Arguments:", arguments)

    # Find actual Python function
    function = AVAILABLE_TOOLS.get(function_name)

    if not function:
        return {
            "success": False,
            "message": f"Tool {function_name} not found."
        }

    # Execute Python function
    result = function(**arguments)

    print("Tool Result:", result)

    return {
        "success": True,
        "tool_name": function_name,
        "arguments": arguments,
        "result": result
    }


# =========================================================
# Agent 3: Response Agent
# =========================================================

def Response_agent(
    user_message,
    extracted_data,
    tool_result
):

    messages = [
        {
            "role": "system",
            "content": Response_prompt
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "original_message": user_message,
                    "extracted_data": extracted_data,
                    "tool_result": tool_result
                },
                ensure_ascii=False
            )
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content


# =========================================================
# Main Chat Function
# =========================================================

def chat_with_ai(user_message, history):

    try:

        # =========================================
        # Agent 1: Extractor
        # =========================================

        extracted_data = Extractor_agent(
            user_message
        )

        print(
            "\n================ EXTRACTOR ================"
        )

        print(
            json.dumps(
                extracted_data,
                indent=2,
                ensure_ascii=False
            )
        )


        # =========================================
        # Agent 2: Tool Agent
        # =========================================

        tool_result = Tool_agent(
            extracted_data,
            user_message
        )

        print(
            "\n================ TOOL RESULT ================"
        )

        print(
            json.dumps(
                tool_result,
                indent=2,
                ensure_ascii=False
            )
        )


        # =========================================
        # Agent 3: Response Agent
        # =========================================

        final_response = Response_agent(
            user_message,
            extracted_data,
            tool_result
        )

        print(
            "\n================ FINAL RESPONSE ================"
        )

        print(final_response)


        # =========================================
        # Update Gradio History
        # =========================================

        history.append({
            "role": "user",
            "content": user_message
        })

        history.append({
            "role": "assistant",
            "content": final_response
        })

        return history, ""


    except Exception as e:

        print(
            "\n================ ERROR ================"
        )

        print(e)

        history.append({
            "role": "user",
            "content": user_message
        })

        history.append({
            "role": "assistant",
            "content": "Sorry, something went wrong. Please try again."
        })

        return history, ""


# =========================================================
# Gradio UI
# =========================================================

with gr.Blocks(
    title="AI Customer Support"
) as demo:

    gr.Markdown(
        """
        # 🤖 AI Customer Support

        ### Ask me anything about your orders, products, refunds, or support.
        """
    )

    chatbot = gr.Chatbot(
        label="Customer Support",
        height=500,
        layout="bubble"
    )

    with gr.Row():

        message = gr.Textbox(
            placeholder="Type your message...",
            label="Message",
            scale=5
        )

        send = gr.Button(
            "Send",
            variant="primary",
            scale=1
        )

    send.click(
        fn=chat_with_ai,
        inputs=[
            message,
            chatbot
        ],
        outputs=[
            chatbot,
            message
        ]
    )

    message.submit(
        fn=chat_with_ai,
        inputs=[
            message,
            chatbot
        ],
        outputs=[
            chatbot,
            message
        ]
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    demo.launch()