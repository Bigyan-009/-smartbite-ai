"""User profile & preferences page."""

import streamlit as st

from auth import auth_service
from config import settings
from services import memory_service, user_service


def render() -> None:
    """Profile screen: preferences, AI memories and password change."""
    user = auth_service.get_current_user()
    st.title("👤 My Profile")

    fresh = user_service.get_user_by_id(user["id"]) or user
    prefs = fresh.get("preferences", {}) or {}

    st.write(f"**Name:** {fresh.get('name')}")
    st.write(f"**Email:** {fresh.get('email')}")

    st.subheader("Food preferences")
    with st.form("prefs_form"):
        cuisines = st.multiselect(
            "Preferred cuisines",
            ["Nepali", "Indian", "Chinese", "Italian", "Fast Food"],
            default=prefs.get("preferred_cuisines", []) or [],
        )
        col1, col2 = st.columns(2)
        with col1:
            diet = st.selectbox(
                "Diet type",
                ["", *settings.DIET_TYPES],
                index=_index(["", *settings.DIET_TYPES], prefs.get("diet_type")),
                format_func=_pretty,
            )
            budget = st.number_input(
                "Usual budget (Rs.)",
                min_value=0,
                max_value=100000,
                value=int(prefs.get("usual_budget") or 0),
                step=50,
            )
        with col2:
            spice = st.selectbox(
                "Spice level",
                ["", *settings.SPICE_LEVELS],
                index=_index(["", *settings.SPICE_LEVELS], prefs.get("spice_level")),
                format_func=_pretty,
            )
        disliked = st.text_input(
            "Disliked ingredients (comma separated)",
            value=", ".join(prefs.get("disliked_ingredients", []) or []),
        )
        allergies = st.text_input(
            "Allergies (comma separated)",
            value=", ".join(prefs.get("allergies", []) or []),
        )
        submitted = st.form_submit_button("Save preferences")

    if submitted:
        new_prefs = {
            "preferred_cuisines": cuisines,
            "diet_type": diet or None,
            "spice_level": spice or None,
            "usual_budget": budget or None,
            "disliked_ingredients": _split(disliked),
            "allergies": _split(allergies),
        }
        if user_service.update_preferences(user["id"], new_prefs):
            # Refresh session copy so other pages see the change.
            updated = user_service.get_user_by_id(user["id"])
            if updated:
                auth_service.start_session(updated)
            st.success("Preferences updated.")
        else:
            st.error("Could not update preferences.")

    _memories_section(user["id"])
    _password_section(user["id"])


def _password_section(user_id: str) -> None:
    """Let a signed-in user change their password."""
    st.divider()
    st.subheader("🔒 Change password")
    with st.form("change_password_form"):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Update password")

    if submitted:
        if new != confirm:
            st.error("The two new passwords do not match.")
            return
        result = auth_service.change_password(user_id, current, new)
        if result["success"]:
            st.success("Password updated.")
        else:
            st.error(result["error"])


def _memories_section(user_id: str) -> None:
    """List what the chatbot has learned, with a way to forget any of it."""
    st.subheader("🧠 What the AI remembers about you")
    memories = memory_service.list_memories(user_id)
    if not memories:
        st.caption("No memories yet. Tell the chatbot your preferences to build these.")
        return

    st.caption(
        "These shape your recommendations. Delete anything that is wrong or "
        "out of date — the chatbot will stop using it immediately."
    )
    for mem in memories:
        row = st.columns([6, 1])
        with row[0]:
            st.write(f"- {mem.get('document')}")
        with row[1]:
            if st.button("🗑", key=f"forget_{mem['id']}", help="Forget this"):
                if memory_service.delete_memory(user_id, mem["id"]):
                    st.toast("Forgotten. 🧠")
                else:
                    st.toast("Could not delete that memory.")
                st.rerun()

    st.divider()
    if st.session_state.get("confirm_clear_memories"):
        st.warning(f"Forget all {len(memories)} memories? This cannot be undone.")
        cols = st.columns([1, 1, 3])
        with cols[0]:
            if st.button("Yes, forget all", type="primary", key="clear_mem_yes",
                         width="stretch"):
                removed = memory_service.clear_memories(user_id)
                st.session_state["confirm_clear_memories"] = False
                st.toast(f"Forgot {removed} memory/memories.")
                st.rerun()
        with cols[1]:
            if st.button("Keep them", key="clear_mem_no", width="stretch"):
                st.session_state["confirm_clear_memories"] = False
                st.rerun()
    else:
        cols = st.columns([1, 3])
        with cols[0]:
            if st.button("🧹 Forget everything", key="clear_mem",
                         width="stretch"):
                st.session_state["confirm_clear_memories"] = True
                st.rerun()


def _split(text: str):
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def _pretty(value: str) -> str:
    return value.replace("_", " ").title() if value else "—"


def _index(options, value):
    """Index of a value in a select box's options, defaulting to the first."""
    try:
        return options.index(value)
    except (ValueError, TypeError):
        return 0
