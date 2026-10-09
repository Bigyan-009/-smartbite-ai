"""Admin order management page - track and advance orders through the kitchen."""

import streamlit as st

from services import order_service, user_service
from utils.date_utils import format_display

STATUS_ICONS = {
    "placed": "🕒",
    "confirmed": "✅",
    "preparing": "👨‍🍳",
    "ready": "🛎️",
    "delivered": "🚚",
    "cancelled": "✖️",
}


def render() -> None:
    """Admin screen for moving orders through their lifecycle."""
    st.title("🧾 Manage Orders")
    st.caption("Advance orders through the kitchen: placed → confirmed → preparing → ready → delivered.")

    statuses = ["all", "active"] + order_service.ORDER_STATUSES + ["cancelled"]
    choice = st.selectbox("Show", statuses, index=1, format_func=lambda s: s.title())
    _orders_live(choice)


@st.fragment(run_every=10)
def _orders_live(choice: str) -> None:
    """Kitchen order board that refreshes itself as new orders come in."""
    if choice == "all":
        orders = order_service.list_all_orders()
    elif choice == "active":
        orders = [
            o
            for o in order_service.list_all_orders()
            if o.get("status") not in ("delivered", "cancelled")
        ]
    else:
        orders = order_service.list_all_orders(status=choice)

    st.caption(
        f"{len(orders)} order(s) · {order_service.count_active_orders()} active "
        "in total · updates automatically"
    )
    if not orders:
        st.info("No orders here yet.")
        return

    user_names = {u["id"]: u.get("name", "Unknown") for u in user_service.list_users()}

    for order in orders:
        _render_order(order, user_names)


def _delete_button(order_id: str) -> None:
    """Arm the delete confirmation for a finished order."""
    if st.button("🗑 Delete", key=f"admin_delete_{order_id}",
                 width="stretch",
                 help="Remove this finished order from the board"):
        st.session_state["confirm_admin_delete"] = order_id
        st.rerun()


def _render_order(order: dict, user_names: dict) -> None:
    """One order card with the buttons that move it to its next status."""
    order_id = order.get("id") or order.get("_id")
    status = str(order.get("status", "placed"))
    placed = format_display(order.get("created_at"))
    customer = user_names.get(order.get("user_id"), "Unknown user")

    with st.container(border=True):
        cols = st.columns([3, 1, 1])
        with cols[0]:
            icon = STATUS_ICONS.get(status, "⚪")
            st.markdown(
                f"**{customer}** · {placed} · "
                f"{icon} {status.title()}"
            )
            summary = ", ".join(
                f"{i.get('quantity')}× {i.get('name')}" for i in order.get("items", [])
            )
            st.caption(summary or "(no items)")
            if order.get("note"):
                st.caption(f"📝 Customer note: {order['note']}")
        with cols[1]:
            st.markdown(f"**Rs. {int(order.get('total', 0))}**")
            eta = order_service.eta_minutes_left(order)
            if eta is not None:
                st.caption(f"⏱️ ~{eta} min left")
        with cols[2]:
            nxt = order_service.next_status(status)
            if status not in order_service.TERMINAL_STATUSES and nxt:
                arrow = f"{STATUS_ICONS.get(nxt, '')} {nxt.title()}"
                if st.button(f"→ {arrow}", key=f"advance_{order_id}", width="stretch"):
                    res = order_service.update_status(order_id, nxt)
                    if res.get("success"):
                        st.toast(f"Order moved to {nxt.title()}.")
                        st.rerun()
                    else:
                        st.toast(res.get("error", "Could not update order."))
            if status not in order_service.TERMINAL_STATUSES:
                if st.button("✖ Cancel", key=f"admin_cancel_{order_id}", width="stretch"):
                    res = order_service.update_status(order_id, "cancelled")
                    if res.get("success"):
                        st.toast("Order cancelled.")
                        st.rerun()
                    else:
                        st.toast(res.get("error", "Could not cancel order."))
            else:
                _delete_button(order_id)

        if st.session_state.get("confirm_admin_delete") == order_id:
            st.warning("Delete this finished order permanently?")
            confirm = st.columns([1, 1, 3])
            with confirm[0]:
                if st.button("Yes, delete", type="primary", width="stretch",
                             key=f"admin_del_yes_{order_id}"):
                    res = order_service.admin_delete_order(order_id)
                    st.session_state["confirm_admin_delete"] = None
                    st.toast(
                        "Order deleted. 🗑️" if res.get("success")
                        else res.get("error", "Could not delete order.")
                    )
                    st.rerun()
            with confirm[1]:
                if st.button("Keep it", width="stretch",
                             key=f"admin_del_no_{order_id}"):
                    st.session_state["confirm_admin_delete"] = None
                    st.rerun()

        history = order.get("status_history") or []
        if len(history) > 1:
            with st.expander("Status history"):
                for entry in history:
                    at = format_display(entry.get("at"))
                    st.caption(
                        f"{STATUS_ICONS.get(entry.get('status'), '')} "
                        f"{str(entry.get('status', '')).title()} — {at}"
                    )
