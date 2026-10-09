"""Browsable menu page — for users who want to scroll and filter, not chat."""

import streamlit as st

from auth import auth_service
from services import cart_service, food_service, order_service
from ui import components
from ui.components import pretty, render_food_card


def render() -> None:
    """The browsable menu, with search and filters."""
    user = auth_service.get_current_user()
    st.title("📖 Menu")
    st.caption("Browse every dish — filter, favourite, rate, or order directly.")

    _cart_panel(user["id"])

    foods = food_service.list_foods(include_unavailable=False)
    if not foods:
        st.info("The menu is empty right now — please check back soon.")
        return

    filtered = _apply_filters(foods)
    st.caption(f"{len(filtered)} of {len(foods)} dishes")
    if not filtered:
        st.info("No dishes match those filters — try widening them.")
        return

    # Two-column grid of the shared food cards.
    columns = st.columns(2)
    for i, food in enumerate(filtered):
        with columns[i % 2]:
            render_food_card(food, user["id"], key=f"menu_{i}_{food.get('id')}")


def _cart_panel(user_id: str) -> None:
    """The cart: adjust quantities, add a note, and order everything at once."""
    cart = components.get_cart()
    count = cart_service.count_items(cart)
    if not count:
        return

    total = int(cart_service.cart_total(cart))
    with st.expander(f"🛒 Your cart — {count} item(s) · Rs. {total}", expanded=True):
        for line in list(cart):
            food_id = line["food_id"]
            row = st.columns([4, 1, 1])
            with row[0]:
                st.write(
                    f"**{line['name']}** — Rs. {int(line['price'])} each "
                    f"= Rs. {int(line['price'] * line['quantity'])}"
                )
            with row[1]:
                quantity = st.number_input(
                    "Qty",
                    min_value=0,
                    max_value=cart_service.MAX_QUANTITY,
                    value=int(line["quantity"]),
                    key=f"cart_qty_{food_id}",
                    label_visibility="collapsed",
                )
                if quantity != int(line["quantity"]):
                    components.set_cart(
                        cart_service.set_quantity(
                            components.get_cart(), food_id, quantity
                        )
                    )
                    st.rerun()
            with row[2]:
                if st.button("🗑", key=f"cart_rm_{food_id}", help="Remove"):
                    components.set_cart(
                        cart_service.remove_item(components.get_cart(), food_id)
                    )
                    st.rerun()

        note = st.text_input(
            "Note (optional)",
            key="cart_note",
            placeholder="e.g. table 4, home delivery to..., less spicy",
        )
        actions = st.columns([2, 1, 2])
        with actions[0]:
            if st.button(f"✅ Place order — Rs. {total}", type="primary",
                         width="stretch", key="cart_checkout"):
                _checkout(user_id, note)
        with actions[1]:
            if st.button("Clear", width="stretch", key="cart_clear"):
                components.set_cart(cart_service.clear())
                st.rerun()


def _checkout(user_id: str, note: str) -> None:
    """Re-price the cart from MongoDB, then place it as one order."""
    pairs = cart_service.to_pairs(components.get_cart())
    items = order_service.items_from_ids(pairs)
    if not items:
        st.error("None of the dishes in your cart are available right now.")
        return
    if len(items) < len(pairs):
        st.warning(
            f"{len(pairs) - len(items)} dish(es) are no longer available and "
            "were left out of this order."
        )

    result = order_service.place_order(user_id, items, note=note)
    if not result.get("success"):
        st.error(result.get("error", "Could not place the order. Please try again."))
        return

    components.set_cart(cart_service.clear())
    total = int(result["order"].get("total", 0))
    st.session_state["_order_toast"] = (
        f'Order placed — Rs. {total}. Track it under "My Orders"!'
    )
    st.rerun()


def _apply_filters(foods: list) -> list:
    """Search box + cuisine/diet/spice/price filters above the grid."""
    cuisines = sorted({str(f.get("cuisine", "")) for f in foods if f.get("cuisine")})
    diets = sorted({str(f.get("diet_type", "")) for f in foods if f.get("diet_type")})
    spices = sorted({str(f.get("spice_level", "")) for f in foods if f.get("spice_level")})
    max_price = max(int(f.get("price", 0)) for f in foods)

    row = st.columns([2, 1, 1, 1, 2])
    with row[0]:
        search = st.text_input("Search", placeholder="momo, chicken, salad...").strip().lower()
    with row[1]:
        cuisine = st.selectbox("Cuisine", ["All"] + cuisines)
    with row[2]:
        diet = st.selectbox("Diet", ["All"] + diets, format_func=pretty)
    with row[3]:
        spice = st.selectbox("Spice", ["All"] + spices, format_func=pretty)
    with row[4]:
        price_limit = st.slider("Max price (Rs.)", 0, max_price, max_price, step=50)

    def matches(food: dict) -> bool:
        if cuisine != "All" and food.get("cuisine") != cuisine:
            return False
        if diet != "All" and food.get("diet_type") != diet:
            return False
        if spice != "All" and food.get("spice_level") != spice:
            return False
        if int(food.get("price", 0)) > price_limit:
            return False
        if search:
            blob = " ".join(
                [str(food.get("name", "")), str(food.get("description", "")),
                 str(food.get("cuisine", ""))]
                + [str(i) for i in food.get("ingredients", []) or []]
                + [str(t) for t in food.get("tags", []) or []]
            ).lower()
            return search in blob
        return True

    return [f for f in foods if matches(f)]
