"""Registration page."""

import streamlit as st

from auth import auth_service
from config import settings


def render() -> None:
    """The account-creation screen."""
    st.title(f"🍽️ {settings.APP_NAME}")
    st.subheader("Create your account")

    with st.form("register_form"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        col1, col2 = st.columns(2)
        with col1:
            password = st.text_input("Password", type="password")
        with col2:
            confirm = st.text_input("Confirm password", type="password")

        st.markdown("**Optional preferences** (helps the AI recommend better)")
        col3, col4 = st.columns(2)
        with col3:
            cuisines = st.multiselect(
                "Preferred cuisines",
                ["Nepali", "Indian", "Chinese", "Italian", "Fast Food"],
            )
            diet_type = st.selectbox(
                "Diet type", ["", *settings.DIET_TYPES], format_func=_pretty
            )
        with col4:
            spice_level = st.selectbox(
                "Spice level", ["", *settings.SPICE_LEVELS], format_func=_pretty
            )
            usual_budget = st.number_input(
                "Usual budget (Rs.)", min_value=0, max_value=100000, value=0, step=50
            )

        submitted = st.form_submit_button("Register", width="stretch")

    if submitted:
        if password != confirm:
            st.error("Passwords do not match.")
            return
        preferences = {
            "preferred_cuisines": cuisines,
            "diet_type": diet_type or None,
            "spice_level": spice_level or None,
            "usual_budget": usual_budget or None,
            "disliked_ingredients": [],
            "allergies": [],
        }
        try:
            result = auth_service.register_user(name, email, password, preferences)
        except Exception as exc:
            st.error(f"Registration failed: {exc}")
            return
        if result.get("success"):
            st.success("Account created! Please log in.")
            st.session_state["page"] = "login"
            st.rerun()
        else:
            st.error(result.get("error", "Registration failed."))

    st.divider()
    if st.button("Back to login", width="stretch"):
        st.session_state["page"] = "login"
        st.rerun()


def _pretty(value: str) -> str:
    return value.replace("_", " ").title() if value else "—"
