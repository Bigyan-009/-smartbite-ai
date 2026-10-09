"""Login page."""

import streamlit as st

from auth import auth_service
from config import settings


def render() -> None:
    """The login screen."""
    st.title(f"🍽️ {settings.APP_NAME}")
    st.caption("AI-powered restaurant food recommendation chatbot")
    st.subheader("Login")

    with st.form("login_form"):
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", width="stretch")

    if submitted:
        try:
            user = auth_service.login_user(email, password)
        except Exception as exc:
            st.error(f"Login failed: {exc}")
            return
        if user:
            auth_service.start_session(user)
            st.session_state["page"] = (
                "admin_dashboard" if user.get("role") == "admin" else "chatbot"
            )
            st.success(f"Welcome back, {user.get('name')}!")
            st.rerun()
        else:
            st.error("Invalid email or password, or the account is inactive.")

    if st.button("Forgot your password?", width="stretch"):
        st.session_state["page"] = "forgot_password"
        st.rerun()

    st.divider()
    st.write("New here?")
    if st.button("Create an account", width="stretch"):
        st.session_state["page"] = "register"
        st.rerun()

    with st.expander("Demo admin credentials"):
        st.code(f"Email: {settings.ADMIN_EMAIL}\nPassword: {settings.ADMIN_PASSWORD}")
