"""Order service.

Handles placing and listing food orders. Orders are structured, price-critical
data, so they live in MongoDB (never ChromaDB).

Order document shape::

    {
      "_id": ObjectId,
      "user_id": "user id as string",
      "items": [
        {"food_id": "...", "name": "Chicken Momo", "price": 250, "quantity": 2, "subtotal": 500}
      ],
      "total": 500,
      "status": "placed",
      "status_history": [{"status": "placed", "at": datetime}],
      "estimated_minutes": 20,
      "created_at": datetime
    }
"""

import re
from datetime import timezone
from typing import Any, Dict, List, Optional

from database import mongo
from services import food_service
from utils.date_utils import utcnow
from utils.id_utils import stringify_doc, to_object_id

# The order lifecycle, in the sequence the kitchen moves through it.
ORDER_STATUSES = ["placed", "confirmed", "preparing", "ready", "delivered"]
TERMINAL_STATUSES = {"delivered", "cancelled"}
# A user may cancel only before the kitchen starts cooking.
CANCELLABLE_STATUSES = {"placed", "confirmed"}

BASE_PREP_MINUTES = 15
EXTRA_MINUTES_PER_ITEM = 5
MAX_PREP_MINUTES = 45


def find_food_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Find a single available food by (fuzzy, case-insensitive) name.

    Tries an exact match first, then a "contains" match so "momo" finds
    "Chicken Momo" when that is the only momo dish, etc.
    """
    if not name or not str(name).strip():
        return None
    term = str(name).strip()

    exact = mongo.foods().find_one(
        {"name": re.compile(f"^{re.escape(term)}$", re.IGNORECASE), "available": True}
    )
    if exact:
        return stringify_doc(exact)

    matches = list(
        mongo.foods().find(
            {"name": re.compile(re.escape(term), re.IGNORECASE), "available": True}
        )
    )
    if matches:
        # Prefer the shortest name (closest to the search term).
        matches.sort(key=lambda f: len(f.get("name", "")))
        return stringify_doc(matches[0])
    return None


def resolve_items(
    raw_items: List[Dict[str, Any]],
    last_recommended: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Turn loosely-described order items into concrete order lines.

    ``raw_items`` come from the LLM/fallback as ``[{"name": ..., "quantity": n}]``.
    Each is matched to a real food (falling back to the last recommended list),
    and priced. Unmatched items are dropped.
    """
    last_recommended = last_recommended or []
    resolved: List[Dict[str, Any]] = []

    for item in raw_items or []:
        name = str(item.get("name", "")).strip()
        try:
            quantity = max(1, int(item.get("quantity", 1)))
        except (TypeError, ValueError):
            quantity = 1
        if not name:
            continue

        food = _match_in_recommended(name, last_recommended) or find_food_by_name(name)
        if not food:
            continue

        price = food_service.effective_price(food)
        resolved.append(
            {
                "food_id": food.get("id") or food.get("_id"),
                "name": food.get("name"),
                "price": price,
                "quantity": quantity,
                "subtotal": round(price * quantity, 2),
            }
        )
    return resolved


def _match_in_recommended(
    name: str, last_recommended: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Find a dish the bot just recommended by (partial) name."""
    term = name.lower()
    for food in last_recommended:
        fname = str(food.get("name", "")).lower()
        if term == fname or term in fname or fname in term:
            return food
    return None


def place_order(
    user_id: str, items: List[Dict[str, Any]], note: str = ""
) -> Dict[str, Any]:
    """Persist an order and return the stored document (with id).

    ``note`` is an optional customer message (address, table number, "less
    spicy please", ...) shown to the kitchen.
    """
    if not to_object_id(user_id):
        return {"success": False, "error": "Invalid user id."}
    if not items:
        return {"success": False, "error": "No valid items to order."}

    now = utcnow()
    total = round(sum(i.get("subtotal", 0) for i in items), 2)
    doc = {
        "user_id": user_id,
        "items": items,
        "total": total,
        "note": str(note or "").strip()[:300],
        "status": "placed",
        "status_history": [{"status": "placed", "at": now}],
        "estimated_minutes": estimate_prep_minutes(items),
        "created_at": now,
    }
    result = mongo.orders().insert_one(doc)
    doc["_id"] = result.inserted_id
    return {"success": True, "order": stringify_doc(doc)}


def estimate_prep_minutes(items: List[Dict[str, Any]]) -> int:
    """Rough kitchen prep estimate: a base time plus a little per extra item."""
    total_qty = sum(max(1, int(i.get("quantity", 1) or 1)) for i in items)
    estimate = BASE_PREP_MINUTES + EXTRA_MINUTES_PER_ITEM * max(0, total_qty - 1)
    return min(estimate, MAX_PREP_MINUTES)


def items_from_ids(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build priced order lines from ``[{"food_id", "quantity"}]`` pairs.

    Unknown or unavailable foods are skipped. Used when the waiter agent selects
    order items directly by menu id.
    """
    items: List[Dict[str, Any]] = []
    for pair in pairs or []:
        food = food_service.get_food(str(pair.get("food_id", "")))
        if not food or not food.get("available", True):
            continue
        try:
            quantity = max(1, int(pair.get("quantity", 1)))
        except (TypeError, ValueError):
            quantity = 1
        price = food_service.effective_price(food)
        items.append(
            {
                "food_id": food.get("id") or food.get("_id"),
                "name": food.get("name"),
                "price": price,
                "quantity": quantity,
                "subtotal": round(price * quantity, 2),
            }
        )
    return items


def order_one(
    user_id: str, food_id: str, quantity: int = 1, note: str = ""
) -> Dict[str, Any]:
    """Convenience helper used by the 'Order' button on a food card."""
    food = food_service.get_food(food_id)
    if not food:
        return {"success": False, "error": "Food not found."}
    price = food_service.effective_price(food)
    quantity = max(1, int(quantity))
    item = {
        "food_id": food_id,
        "name": food.get("name"),
        "price": price,
        "quantity": quantity,
        "subtotal": round(price * quantity, 2),
    }
    return place_order(user_id, [item], note=note)


def list_user_orders(user_id: str) -> List[Dict[str, Any]]:
    return [
        stringify_doc(o)
        for o in mongo.orders().find({"user_id": user_id}).sort("created_at", -1)
    ]


def cancel_order(user_id: str, order_id: str) -> Dict[str, Any]:
    """Cancel a user's own order while the kitchen has not started cooking.

    Marks the order ``cancelled`` (kept for history) rather than deleting it.
    """
    oid = to_object_id(order_id)
    if not oid:
        return {"success": False, "error": "Invalid order id."}

    order = mongo.orders().find_one({"_id": oid})
    if not order or order.get("user_id") != user_id:
        return {"success": False, "error": "Order not found."}
    if order.get("status") not in CANCELLABLE_STATUSES:
        return {"success": False, "error": "This order can no longer be cancelled."}

    now = utcnow()
    mongo.orders().update_one(
        {"_id": oid},
        {
            "$set": {"status": "cancelled", "updated_at": now},
            "$push": {"status_history": {"status": "cancelled", "at": now}},
        },
    )
    return {"success": True}


def delete_order(user_id: str, order_id: str) -> Dict[str, Any]:
    """Permanently delete one of the user's own finished orders.

    Only ``delivered`` or ``cancelled`` orders can be removed: an order the
    kitchen is still working on must be cancelled first, so a customer can
    never make a live ticket disappear on the restaurant.
    """
    oid = to_object_id(order_id)
    if not oid:
        return {"success": False, "error": "Invalid order id."}

    order = mongo.orders().find_one({"_id": oid})
    if not order or order.get("user_id") != user_id:
        return {"success": False, "error": "Order not found."}
    if order.get("status") not in TERMINAL_STATUSES:
        return {
            "success": False,
            "error": "Only delivered or cancelled orders can be deleted. "
                     "Cancel this order first.",
        }

    result = mongo.orders().delete_one({"_id": oid, "user_id": user_id})
    if not result.deleted_count:
        return {"success": False, "error": "Order not found."}
    return {"success": True}


def delete_finished_orders(user_id: str) -> int:
    """Delete every finished (delivered/cancelled) order for a user.

    Returns the number of orders removed. Active orders are left alone.
    """
    try:
        result = mongo.orders().delete_many(
            {"user_id": user_id, "status": {"$in": list(TERMINAL_STATUSES)}}
        )
        return int(result.deleted_count)
    except Exception:
        return 0


def count_finished_orders(user_id: str) -> int:
    """How many of the user's orders are finished (and so deletable)."""
    try:
        return mongo.orders().count_documents(
            {"user_id": user_id, "status": {"$in": list(TERMINAL_STATUSES)}}
        )
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Order lifecycle (admin) and tracking helpers
# ---------------------------------------------------------------------------
def update_status(order_id: str, new_status: str) -> Dict[str, Any]:
    """Move an order to a new lifecycle status (admin action)."""
    oid = to_object_id(order_id)
    if not oid:
        return {"success": False, "error": "Invalid order id."}
    if new_status not in ORDER_STATUSES and new_status != "cancelled":
        return {"success": False, "error": f"Invalid status '{new_status}'."}

    order = mongo.orders().find_one({"_id": oid})
    if not order:
        return {"success": False, "error": "Order not found."}
    current = order.get("status", "placed")
    if current in TERMINAL_STATUSES:
        return {"success": False, "error": f"Order is already {current}."}
    if new_status == current:
        return {"success": False, "error": f"Order is already {current}."}

    now = utcnow()
    mongo.orders().update_one(
        {"_id": oid},
        {
            "$set": {"status": new_status, "updated_at": now},
            "$push": {"status_history": {"status": new_status, "at": now}},
        },
    )
    return {"success": True, "status": new_status}


def next_status(current: str) -> Optional[str]:
    """The next step in the lifecycle, or None when the order is done."""
    try:
        idx = ORDER_STATUSES.index(current)
    except ValueError:
        return None
    if idx + 1 < len(ORDER_STATUSES):
        return ORDER_STATUSES[idx + 1]
    return None


def status_progress(status: str) -> float:
    """Progress through the lifecycle as 0.0-1.0 (for the tracking bar)."""
    try:
        idx = ORDER_STATUSES.index(status)
    except ValueError:
        return 0.0
    return (idx + 1) / len(ORDER_STATUSES)


def eta_minutes_left(order: Dict[str, Any]) -> Optional[int]:
    """Minutes until the order should be ready, or None when not applicable."""
    if order.get("status") in TERMINAL_STATUSES or order.get("status") == "ready":
        return None
    estimate = order.get("estimated_minutes")
    created = order.get("created_at")
    if not estimate or created is None or isinstance(created, str):
        return None
    # PyMongo returns naive UTC datetimes; make it comparable with utcnow().
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    elapsed = (utcnow() - created).total_seconds() / 60.0
    return max(0, round(float(estimate) - elapsed))


def list_all_orders(status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """All orders (newest first) for the admin view, optionally by status."""
    query = {"status": status} if status else {}
    return [
        stringify_doc(o)
        for o in mongo.orders().find(query).sort("created_at", -1).limit(limit)
    ]


def count_active_orders() -> int:
    """Orders the kitchen still has to act on."""
    return mongo.orders().count_documents(
        {"status": {"$in": ["placed", "confirmed", "preparing", "ready"]}}
    )


def admin_delete_order(order_id: str) -> Dict[str, Any]:
    """Delete a finished order from the kitchen board (admin action).

    Same rule as the customer-facing delete: an order still moving through the
    kitchen must be cancelled first, so nobody can drop a live ticket by
    accident. Useful for clearing demo and test orders.
    """
    oid = to_object_id(order_id)
    if not oid:
        return {"success": False, "error": "Invalid order id."}

    order = mongo.orders().find_one({"_id": oid})
    if not order:
        return {"success": False, "error": "Order not found."}
    if order.get("status") not in TERMINAL_STATUSES:
        return {
            "success": False,
            "error": "Cancel this order before deleting it.",
        }

    mongo.orders().delete_one({"_id": oid})
    return {"success": True}


def reorder(user_id: str, order_id: str) -> Dict[str, Any]:
    """Place a fresh order with the same items as a previous one.

    Items are re-resolved against the current menu so today's prices,
    discounts, and availability apply. Foods that no longer exist are skipped.
    """
    oid = to_object_id(order_id)
    if not oid:
        return {"success": False, "error": "Invalid order id."}
    previous = mongo.orders().find_one({"_id": oid})
    if not previous or previous.get("user_id") != user_id:
        return {"success": False, "error": "Order not found."}

    pairs = [
        {"food_id": i.get("food_id"), "quantity": i.get("quantity", 1)}
        for i in previous.get("items", [])
    ]
    items = items_from_ids(pairs)
    if not items:
        return {
            "success": False,
            "error": "None of the items in that order are available right now.",
        }
    return place_order(user_id, items)



def order_history_signals(user_id: str) -> Dict[str, Any]:
    """Summarise what this user has actually ordered, for recommendation boosts.

    Returns ``{"food_counts": {food_id: times_ordered}, "cuisine_counts":
    {cuisine_lower: times_ordered}}`` built from non-cancelled orders. This is
    the behavioural half of the hybrid recommender: stated preferences come
    from the profile/memory, revealed preferences come from real orders.
    """
    food_counts: Dict[str, int] = {}
    try:
        cursor = mongo.orders().find(
            {"user_id": user_id, "status": {"$ne": "cancelled"}},
            {"items.food_id": 1, "items.quantity": 1},
        )
        for order in cursor:
            for item in order.get("items", []):
                fid = str(item.get("food_id", ""))
                if fid:
                    food_counts[fid] = food_counts.get(fid, 0) + 1
    except Exception:
        return {"food_counts": {}, "cuisine_counts": {}}

    cuisine_counts: Dict[str, int] = {}
    if food_counts:
        for food in food_service.get_foods_by_ids(list(food_counts.keys())):
            cuisine = str(food.get("cuisine", "")).strip().lower()
            if cuisine:
                fid = food.get("id") or food.get("_id")
                cuisine_counts[cuisine] = (
                    cuisine_counts.get(cuisine, 0) + food_counts.get(fid, 0)
                )
    return {"food_counts": food_counts, "cuisine_counts": cuisine_counts}
