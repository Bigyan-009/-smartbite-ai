"""Reusable UI pieces shared by the chatbot and menu pages."""

import streamlit as st

from services import cart_service, food_service, order_service


def _nutrition_caption(food: dict) -> str:
    """Calories/protein suffix for a food card, empty when not recorded."""
    calories = food.get("calories")
    protein = food.get("protein")
    parts = []
    if calories is not None:
        parts.append(f"🔥 {int(float(calories))} kcal")
    if protein is not None:
        parts.append(f"💪 {int(float(protein))}g protein")
    return (" · " + " · ".join(parts)) if parts else ""


def render_food_card(food: dict, user_id: str, key: str) -> None:
    """One food card: price/special info, favourite, order-with-confirm, rating.

    ``key`` must be unique per rendered card (it namespaces the widgets).
    """
    food_id = food.get("id") or food.get("_id")
    price = int(food.get("price", 0))
    pay_price = int(food_service.effective_price(food))

    with st.container(border=True):
        # A photo (when we have one) gets its own column on the left.
        photo = food_service.image_path(food)
        if photo:
            image_col, detail_col, action_col = st.columns([1, 2.4, 1])
            with image_col:
                st.image(photo, width="stretch")
        else:
            detail_col, action_col = st.columns([3, 1])
        with detail_col:
            if food.get("is_special") and pay_price < price:
                discount = int(food.get("discount_percent", 0) or 0)
                st.markdown(
                    f"**{food.get('name')}** — ~~Rs. {price}~~ **Rs. {pay_price}** "
                    f"· 🌟 Today's Special (-{discount}%)"
                )
            elif food.get("is_special"):
                st.markdown(f"**{food.get('name')}** — Rs. {price} · 🌟 Today's Special")
            else:
                st.markdown(f"**{food.get('name')}** — Rs. {price}")
            st.caption(
                f"{food.get('cuisine', '')} · {pretty(food.get('diet_type'))} · "
                f"{pretty(food.get('spice_level'))} · ⭐ {food.get('rating_avg', 0)}"
                f"{_nutrition_caption(food)}"
            )
            if food.get("recommendation_reason"):
                st.write(f"💡 {food['recommendation_reason']}")
            if food.get("recommendation_score") is not None:
                st.caption(f"Recommendation score: {food['recommendation_score']}")
        with action_col:
            fav = food_service.is_favourite(user_id, food_id)
            label = "★ Saved" if fav else "☆ Favourite"
            if st.button(label, key=f"fav_{key}", width="stretch"):
                if fav:
                    food_service.remove_favourite(user_id, food_id)
                else:
                    food_service.add_favourite(user_id, food_id)
                st.rerun()
            if st.button("🛒 Order", key=f"order_{key}", width="stretch"):
                _order_dialog(food, user_id, pay_price)
            if st.button("➕ Cart", key=f"cart_{key}", width="stretch",
                         help="Add to your cart and order several dishes together"):
                add_to_cart(food, pay_price)
                st.rerun()

        _star_rating(food, user_id, key)


# ---------------------------------------------------------------------------
# Cart (session-scoped; the rules live in services/cart_service.py)
# ---------------------------------------------------------------------------
def get_cart() -> list:
    """The current browser session's cart."""
    if "cart" not in st.session_state:
        st.session_state["cart"] = []
    return st.session_state["cart"]


def set_cart(cart: list) -> None:
    """Replace the cart held in this browser session."""
    st.session_state["cart"] = cart


def add_to_cart(food: dict, price: float) -> None:
    """Add one of a dish to the cart and queue the confirmation toast."""
    set_cart(cart_service.add_item(get_cart(), food, quantity=1, price=price))
    st.session_state["_cart_toast"] = f"Added {food.get('name')} to your cart."


def show_pending_cart_toast() -> None:
    """Show the 'added to cart' toast after the rerun that followed the click."""
    message = st.session_state.pop("_cart_toast", None)
    if message:
        st.toast(message, icon="🛒")


def render_sidebar_cart(on_view) -> None:
    """Compact cart summary for the sidebar, with a link to the full cart."""
    cart = get_cart()
    count = cart_service.count_items(cart)
    st.divider()
    st.markdown("#### 🛒 Cart")
    if not count:
        st.caption("Empty — add dishes from the menu or a recommendation.")
        return
    st.caption(f"{count} item(s) · Rs. {int(cart_service.cart_total(cart))}")
    if st.button("View cart & checkout", width="stretch", key="sidebar_cart"):
        on_view()


def show_pending_order_toast() -> None:
    """Show the 'order placed' toast left behind by the order dialog.

    The dialog closes itself with ``st.rerun()``, which happens before a toast
    could be seen — so it stashes the message in session state and the next
    render (this call, made from ``app.py``) displays it.
    """
    message = st.session_state.pop("_order_toast", None)
    if message:
        st.toast(message, icon="🎉")


@st.dialog("Confirm your order")
def _order_dialog(food: dict, user_id: str, pay_price: int) -> None:
    """Modal confirmation: quantity + optional note, closes itself on success."""
    food_id = food.get("id") or food.get("_id")
    st.markdown(f"**{food.get('name')}** — Rs. {pay_price} each")

    quantity = st.number_input(
        "Quantity", min_value=1, max_value=10, value=1, key=f"dlg_qty_{food_id}"
    )
    note = st.text_input(
        "Note (optional)",
        key=f"dlg_note_{food_id}",
        placeholder="e.g. table 4, home delivery to..., less spicy",
    )
    total = int(pay_price * quantity)

    if st.button(
        f"✅ Place order — Rs. {total}",
        key=f"dlg_confirm_{food_id}",
        width="stretch",
        type="primary",
    ):
        result = order_service.order_one(user_id, food_id, quantity, note=note)
        if result.get("success"):
            st.session_state["_order_toast"] = (
                f"Order placed: {quantity} × {food.get('name')} — Rs. {total}. "
                'Track it under "My Orders"!'
            )
            st.rerun()  # closes the dialog; app.py shows the toast
        else:
            st.error(result.get("error", "Could not place the order. Please try again."))


def _star_rating(food: dict, user_id: str, key: str) -> None:
    """Native star widget; the caption shows the user's saved rating."""
    food_id = food.get("id") or food.get("_id")
    current = int(food_service.get_user_rating(user_id, food_id) or 0)
    cols = st.columns([1, 3])
    with cols[0]:
        st.caption(f"Your rating: {current}★" if current else "Rate this dish:")
    with cols[1]:
        stars = st.feedback("stars", key=f"rate_{key}")
    if stars is not None and (stars + 1) != current:
        result = food_service.rate_food(user_id, food_id, stars + 1)
        if result.get("success"):
            st.toast(f"Saved your {stars + 1}★ rating for {food.get('name')}")


def pretty(value) -> str:
    """'non_veg' -> 'Non Veg' for captions."""
    return str(value).replace("_", " ").title() if value else ""
