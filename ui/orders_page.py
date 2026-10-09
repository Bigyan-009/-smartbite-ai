"""User orders page - shows the orders a user has placed, with live tracking."""

import streamlit as st

from auth import auth_service
from services import order_service
from utils.date_utils import format_display

STATUS_ICONS = {
    "placed": "🕒",
    "confirmed": "✅",
    "preparing": "👨‍🍳",
    "ready": "🛎️",
    "delivered": "🚚",
    "cancelled": "✖️",
}

STATUS_CAPTIONS = {
    "placed": "We've received your order.",
    "confirmed": "The restaurant confirmed your order.",
    "preparing": "The kitchen is cooking it now.",
    "ready": "Your order is ready for pickup/delivery!",
    "delivered": "Delivered. Enjoy your meal! 😋",
    "cancelled": "This order was cancelled.",
}


def render() -> None:
    """The user's own orders, with live status updates."""
    user = auth_service.get_current_user()
    st.title("🛒 My Orders")
    st.caption("Everything you've ordered through SmartBite AI.")
    _orders_live(user["id"])


@st.fragment(run_every=10)
def _orders_live(user_id: str) -> None:
    """Order list that refreshes itself every few seconds (live tracking)."""
    orders = order_service.list_user_orders(user_id)
    if not orders:
        st.info(
            "You haven't placed any orders yet. Ask the chatbot to order something, "
            "or use the 🛒 Order button on a recommendation!"
        )
        return

    active = [o for o in orders if o.get("status") not in ("delivered", "cancelled")]
    if active:
        st.caption(f"🔄 {len(active)} active order(s) — updates automatically.")

    _clear_finished_control(user_id, orders)

    for order in orders:
        _render_order(order, user_id)


def _clear_finished_control(user_id: str, orders: list) -> None:
    """Bulk-delete every finished order, behind an explicit confirm step."""
    finished = sum(
        1 for o in orders if o.get("status") in order_service.TERMINAL_STATUSES
    )
    if not finished:
        return

    if st.session_state.get("confirm_clear_orders"):
        st.warning(
            f"Permanently delete {finished} finished order(s)? This cannot be undone."
        )
        cols = st.columns([1, 1, 3])
        with cols[0]:
            if st.button("Yes, delete", type="primary", width="stretch",
                         key="confirm_clear_yes"):
                removed = order_service.delete_finished_orders(user_id)
                st.session_state["confirm_clear_orders"] = False
                st.toast(f"Deleted {removed} order(s). 🧹")
                st.rerun()
        with cols[1]:
            if st.button("Keep them", width="stretch",
                         key="confirm_clear_no"):
                st.session_state["confirm_clear_orders"] = False
                st.rerun()
        return

    cols = st.columns([1, 3])
    with cols[0]:
        if st.button(f"🧹 Clear history ({finished})", width="stretch",
                     key="clear_finished_orders",
                     help="Delete all your delivered and cancelled orders"):
            st.session_state["confirm_clear_orders"] = True
            st.rerun()


def _render_order(order: dict, user_id: str) -> None:
    """One of the user's own orders, with its live status and ETA."""
    order_id = order.get("id") or order.get("_id")
    placed = format_display(order.get("created_at"))
    status = str(order.get("status", "placed"))

    with st.container(border=True):
        header = st.columns([3, 1])
        with header[0]:
            st.markdown(f"**Order** · {placed}")
            icon = STATUS_ICONS.get(status, "⚪")
            st.caption(f"Status: {icon} {status.title()} — {STATUS_CAPTIONS.get(status, '')}")
        with header[1]:
            st.markdown(f"### Rs. {int(order.get('total', 0))}")

        _render_tracker(order, status)

        for item in order.get("items", []):
            st.write(
                f"- {item.get('quantity')} × {item.get('name')} "
                f"(Rs. {int(item.get('price', 0))} each) "
                f"= Rs. {int(item.get('subtotal', 0))}"
            )
        if order.get("note"):
            st.caption(f"📝 Note: {order['note']}")

        _render_actions(order_id, status, user_id)


def _render_actions(order_id: str, status: str, user_id: str) -> None:
    """Cancel / reorder / delete buttons for a single order."""
    # Delete asks for confirmation first. The pending choice lives in
    # session_state so it survives the 10-second live refresh of this fragment.
    if st.session_state.get("confirm_delete_order") == order_id:
        st.warning("Permanently delete this order from your history?")
        confirm = st.columns([1, 1, 3])
        with confirm[0]:
            if st.button("Yes, delete", type="primary", width="stretch",
                         key=f"confirm_del_yes_{order_id}"):
                res = order_service.delete_order(user_id, order_id)
                st.session_state["confirm_delete_order"] = None
                st.toast(
                    "Order deleted. 🗑️" if res.get("success")
                    else res.get("error", "Could not delete order.")
                )
                st.rerun()
        with confirm[1]:
            if st.button("Keep it", width="stretch",
                         key=f"confirm_del_no_{order_id}"):
                st.session_state["confirm_delete_order"] = None
                st.rerun()
        return

    actions = st.columns([1, 1, 1, 2])
    with actions[0]:
        if status in order_service.CANCELLABLE_STATUSES:
            if st.button("✖ Cancel order", key=f"cancel_{order_id}",
                         width="stretch"):
                res = order_service.cancel_order(user_id, order_id)
                if res.get("success"):
                    st.toast("Order cancelled.")
                    st.rerun()
                else:
                    st.toast(res.get("error", "Could not cancel order."))
    with actions[1]:
        if status in order_service.TERMINAL_STATUSES:
            if st.button("🔁 Order again", key=f"reorder_{order_id}",
                         width="stretch"):
                res = order_service.reorder(user_id, order_id)
                if res.get("success"):
                    st.toast("Order placed again! 🎉")
                    st.rerun()
                else:
                    st.toast(res.get("error", "Could not repeat this order."))
    with actions[2]:
        # Only finished orders can be removed — a live kitchen ticket must be
        # cancelled first.
        if status in order_service.TERMINAL_STATUSES:
            if st.button("🗑 Delete", key=f"delete_{order_id}",
                         width="stretch",
                         help="Remove this order from your history"):
                st.session_state["confirm_delete_order"] = order_id
                st.rerun()


def _render_tracker(order: dict, status: str) -> None:
    """Visual progress through the order lifecycle."""
    if status == "cancelled":
        st.progress(0.0, text="✖️ Cancelled")
        return

    steps = order_service.ORDER_STATUSES
    labels = []
    reached = True
    for step in steps:
        icon = STATUS_ICONS.get(step, "")
        if reached:
            labels.append(f"{icon} **{step.title()}**")
        else:
            labels.append(f"{icon} {step.title()}")
        if step == status:
            reached = False

    st.progress(
        order_service.status_progress(status),
        text=" → ".join(labels),
    )

    eta = order_service.eta_minutes_left(order)
    if eta is not None:
        if eta > 0:
            st.caption(f"⏱️ Estimated ready in about {eta} min")
        else:
            st.caption("⏱️ Should be ready any moment now!")
