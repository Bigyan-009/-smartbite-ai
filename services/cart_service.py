"""Shopping-cart logic.

The cart itself lives in Streamlit's ``session_state`` (it is per-browser, not
per-user data, so it never touches MongoDB). Keeping the rules in here as pure
functions over a plain list means they can be unit tested without a browser.

A cart line looks like::

    {"food_id": "...", "name": "Chicken Momo", "price": 212, "quantity": 2}

``price`` is only a display snapshot. At checkout the items are re-priced from
MongoDB via ``order_service.items_from_ids`` so today's discounts and
availability always win.
"""

from typing import Any, Dict, List

MAX_QUANTITY = 20

Cart = List[Dict[str, Any]]


def _clamp(quantity: Any) -> int:
    """Coerce a quantity to a sane 1..MAX_QUANTITY integer."""
    try:
        value = int(quantity)
    except (TypeError, ValueError):
        value = 1
    return max(1, min(value, MAX_QUANTITY))


def find_line(cart: Cart, food_id: str) -> Dict[str, Any] | None:
    """The cart line for a food, or None when it is not in the cart."""
    for line in cart or []:
        if str(line.get("food_id")) == str(food_id):
            return line
    return None


def add_item(cart: Cart, food: Dict[str, Any], quantity: int = 1, price: float | None = None) -> Cart:
    """Add a dish, merging with the existing line when it is already in the cart."""
    cart = list(cart or [])
    food_id = str(food.get("id") or food.get("_id") or "")
    if not food_id:
        return cart

    quantity = _clamp(quantity)
    existing = find_line(cart, food_id)
    if existing:
        existing["quantity"] = _clamp(existing.get("quantity", 1) + quantity)
        return cart

    cart.append(
        {
            "food_id": food_id,
            "name": food.get("name", "Unknown dish"),
            "price": float(food.get("price", 0) if price is None else price),
            "quantity": quantity,
        }
    )
    return cart


def set_quantity(cart: Cart, food_id: str, quantity: int) -> Cart:
    """Change a line's quantity; a quantity of 0 or less removes the line."""
    try:
        requested = int(quantity)
    except (TypeError, ValueError):
        return list(cart or [])
    if requested <= 0:
        return remove_item(cart, food_id)

    cart = list(cart or [])
    line = find_line(cart, food_id)
    if line:
        line["quantity"] = _clamp(requested)
    return cart


def remove_item(cart: Cart, food_id: str) -> Cart:
    return [l for l in (cart or []) if str(l.get("food_id")) != str(food_id)]


def clear(cart: Cart = None) -> Cart:
    """Return an empty cart (the argument is accepted for symmetry)."""
    return []


def count_items(cart: Cart) -> int:
    """Total number of dishes, counting quantities."""
    return sum(_clamp(l.get("quantity", 1)) for l in (cart or []))


def cart_total(cart: Cart) -> float:
    """Indicative total from the snapshot prices (checkout re-prices for real)."""
    total = sum(
        float(l.get("price", 0)) * _clamp(l.get("quantity", 1)) for l in (cart or [])
    )
    return round(total, 2)


def to_pairs(cart: Cart) -> List[Dict[str, Any]]:
    """``[{"food_id", "quantity"}]`` for ``order_service.items_from_ids``."""
    return [
        {"food_id": l.get("food_id"), "quantity": _clamp(l.get("quantity", 1))}
        for l in (cart or [])
    ]
