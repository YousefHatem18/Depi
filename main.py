import json
import os
import gradio as gr
import json

from dotenv import load_dotenv
from openai import OpenAI

from tools.support_tools import (
    check_refund_eligibility,
    check_stock,
    create_support_ticket,
    get_order_status,
    get_product_info,
)


load_dotenv(".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from the .env file.")

model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a support ticket for a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "issue": {"type": "string"},
                },
                "required": ["customer_id", "issue"],
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
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
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
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
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
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
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
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]

Extractor_prompt = """
you are an Extractor agent your job is extract important Words and information to give them to next agent 
extract every language English , arabic or any langauge
information like 
- where is my order ORD1001
- what is your products
or anything about three fields (customers, orders, polices, productions, tickets)
your output must be in json format        
        """

tool_prompt = """
You are a tool-use agent for a customer support system.

Your job is to analyze the JSON input received from the extractor agent
and use the appropriate available tool to fulfill the user's request.

Rules:
1. Analyze the "entity", "request", and available data in the JSON.
2. Choose the most appropriate tool based on the user's intent.
3. Use the values from the JSON as arguments for the selected tool.
4. If the required information for a tool is missing, do not invent it.
5. Do not answer the user directly.
6. Do not explain your decision.
7. Always use a tool when the requested information can be obtained using one.

Examples:

If the input is:
{
    "entity": "order",
    "order_id": "ORD1001",
    "request": "status"
}

Use:
get_order_status(order_id="ORD1001")

If the input is:
{
    "entity": "product",
    "product_name": "AirPods Pro",
    "request": "information"
}

Use:
get_product_info(product_name="AirPods Pro")

If the input is:
{
    "entity": "product",
    "product_name": "AirPods Pro",
    "request": "stock"
}

Use:
check_stock(product_name="AirPods Pro")

If the input is:
{
    "entity": "order",
    "order_id": "ORD1001",
    "request": "refund"
}

Use:
check_refund_eligibility(order_id="ORD1001")
"""
Response_prompt = """
You are the final response agent for a customer support system.

Your job is to answer the user's original request using the tool result.

Rules:

1. Answer naturally and conversationally.
2. Answer only what the user asked for.
3. Do not return JSON.
4. Do not mention agents, tools, function calls, or internal processing.
5. Do not invent information.
6. Use only the information provided by the tool result.
7. Respond in the same language as the user.
8. If the tool result does not contain enough information, clearly say that the information is unavailable.
9. Keep the response concise and useful.

Examples:

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


AVAILABLE_TOOLS = {
    "create_support_ticket": create_support_ticket,
    "check_refund_eligibility": check_refund_eligibility,
    "check_stock": check_stock,
    "get_order_status": get_order_status,
    "get_product_info": get_product_info,
}


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
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def Tool_agent(extracted_data):

    messages = [
        {
            "role": "system",
            "content": tool_prompt
        },
        {
            "role": "user",
            "content": json.dumps(extracted_data, ensure_ascii=False)
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

    if not message.tool_calls:
        return {
            "success": False,
            "message": "No tool was selected."
        }

    tool_call = message.tool_calls[0]

    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    print("Tool:", function_name)
    print("Arguments:", arguments)

    # Get actual Python function
    function = AVAILABLE_TOOLS.get(function_name)

    if not function:
        return {
            "success": False,
            "message": f"Tool {function_name} not found."
        }

    # Execute tool
    result = function(**arguments)

    print("Tool Result:", result)

    return {
        "tool_name": function_name,
        "arguments": arguments,
        "result": result
    }
    
    
    
    
def Response_agent(user_message, extracted_data, tool_result):

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
import gradio as gr
import json


def chat_with_ai(user_message, history):

    try:
        # =========================
        # Agent 1: Extractor
        # =========================
        extracted_data = Extractor_agent(user_message)

        print("\n================ EXTRACTOR ================")
        print(json.dumps(extracted_data, indent=2, ensure_ascii=False))

        # =========================
        # Agent 2: Tool Agent
        # =========================
        tool_result = Tool_agent(extracted_data)

        print("\n================ TOOL RESULT ================")
        print(json.dumps(tool_result, indent=2, ensure_ascii=False))

        # =========================
        # Agent 3: Response Agent
        # =========================
        final_response = Response_agent(
            user_message,
            extracted_data,
            tool_result
        )

        print("\n================ FINAL RESPONSE ================")
        print(final_response)

        # Gradio يعرض آخر response فقط
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

        print("\n================ ERROR ================")
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


# =========================
# Gradio UI
# =========================

with gr.Blocks(title="AI Customer Support") as demo:

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
        inputs=[message, chatbot],
        outputs=[chatbot, message]
    )

    message.submit(
        fn=chat_with_ai,
        inputs=[message, chatbot],
        outputs=[chatbot, message]
    )


if __name__ == "__main__":
    demo.launch()