import json
import os
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(filename, data):
    path = os.path.join(DATA_DIR, filename)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# =========================================================
# TOOL 1 - Get Order Status
# =========================================================

def get_order_status(order_id):

    orders = load_json("orders.json")

    order = orders.get(order_id)

    if not order:
        return {
            "success": False,
            "error": f"Order {order_id} was not found."
        }

    return {
        "success": True,
        "order_id": order_id,
        "customer_id": order["customer_id"],
        "product_id": order["product_id"],
        "quantity": order["quantity"],
        "status": order["status"],
        "order_date": order["order_date"],
        "estimated_delivery": order["estimated_delivery"],
        "delivered_date": order["delivered_date"],
        "payment_status": order["payment_status"],
        "total_price": order["total_price"],
        "currency": order["currency"]
    }


# =========================================================
# TOOL 2 - Get Product Info
# =========================================================

def get_product_info(product_id):

    products = load_json("products.json")

    product = products.get(product_id)

    if not product:
        return {
            "success": False,
            "error": f"Product {product_id} was not found."
        }

    return {
        "success": True,
        "product_id": product_id,
        **product
    }


# =========================================================
# TOOL 3 - Check Stock
# =========================================================

def check_stock(product_id):

    products = load_json("products.json")

    product = products.get(product_id)

    if not product:
        return {
            "success": False,
            "error": f"Product {product_id} was not found."
        }

    stock = product["stock"]

    return {
        "success": True,
        "product_id": product_id,
        "product_name": product["name"],
        "stock": stock,
        "available": stock > 0
    }


# =========================================================
# TOOL 4 - Check Refund Eligibility
# =========================================================

def check_refund_eligibility(order_id):

    orders = load_json("orders.json")

    order = orders.get(order_id)

    if not order:
        return {
            "success": False,
            "eligible": False,
            "error": f"Order {order_id} was not found."
        }

    # Already refunded
    if order["payment_status"] == "Refunded":

        return {
            "success": True,
            "eligible": False,
            "reason": "This order has already been refunded."
        }

    # Cancelled order
    if (
        order["status"] == "Cancelled"
        and order["payment_status"] == "Paid"
    ):

        return {
            "success": True,
            "eligible": True,
            "reason": "Cancelled paid orders are eligible for a refund."
        }

    # Delivered order
    if (
        order["status"] == "Delivered"
        and order["delivered_date"]
    ):

        delivered_date = datetime.strptime(
            order["delivered_date"],
            "%Y-%m-%d"
        ).date()

        today = datetime.now().date()

        days_since_delivery = (
            today - delivered_date
        ).days

        if days_since_delivery <= 14:

            return {
                "success": True,
                "eligible": True,
                "reason": "The order is within the 14-day refund period.",
                "days_since_delivery": days_since_delivery
            }

        return {
            "success": True,
            "eligible": False,
            "reason": "The 14-day refund period has expired.",
            "days_since_delivery": days_since_delivery
        }

    # Not delivered
    return {
        "success": True,
        "eligible": False,
        "reason": "The order has not been delivered yet."
    }


# =========================================================
# TOOL 5 - Create Support Ticket
# =========================================================

def create_support_ticket(customer_id, issue):

    customers = load_json("customers.json")

    if customer_id not in customers:

        return {
            "success": False,
            "error": f"Customer {customer_id} was not found."
        }

    if not issue or not issue.strip():

        return {
            "success": False,
            "error": "Issue cannot be empty."
        }

    tickets = load_json("tickets.json")

    ticket_number = len(tickets) + 1

    ticket_id = f"T{ticket_number:04d}"

    ticket = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "issue": issue,
        "status": "Open",
        "priority": "Normal",
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        )
    }

    tickets.append(ticket)

    save_json("tickets.json", tickets)

    return {
        "success": True,
        "ticket": ticket
    }


# =========================================================
# MANUAL TEST
# =========================================================

if __name__ == "__main__":

    print("\n===== TOOL 1 =====")
    print(get_order_status("ORD1001"))

    print("\n===== TOOL 2 =====")
    print(get_product_info("P1001"))

    print("\n===== TOOL 3 =====")
    print(check_stock("P1001"))

    print("\n===== TOOL 4 =====")
    print(check_refund_eligibility("ORD1001"))

    print("\n===== TOOL 5 =====")
    print(
        create_support_ticket(
            "C001",
            "My headphones arrived broken."
        )
    )