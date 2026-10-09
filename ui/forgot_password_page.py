"""Forgot-password page: request a code by email, then set a new password."""

import streamlit as st

from auth import auth_service
from config import settings
from services import email_service, password_reset_service


def render() -> None:
    """Forgot-password screen: request a code, then set a new password."""
    st.title(f"🍽️ {settings.APP_NAME}")
    st.subheader("Forgot your password?")

    if not email_service.is_configured():
        if settings.APP_ENV == "development":
            st.info(
                "Email sending is not configured, so reset codes are shown on "
                "screen instead (development mode). Add SMTP_USER and "
                "SMTP_PASSWORD to your .env to send real emails.",
                icon="✉️",
            )
        else:
            st.error("Password reset is unavailable: email is not configured.")

    step = st.session_state.get("reset_step", "request")

    if step == "request":
        _render_request_step()
    else:
        _render_confirm_step()

    st.divider()
    if st.button("← Back to login", width="stretch"):
        _clear_reset_state()
        st.session_state["page"] = "login"
        st.rerun()


def _render_request_step() -> None:
    """Step 1: ask for the email address and send a reset code to it."""
    st.write("Enter the email you registered with and we'll send you a reset code.")
    with st.form("reset_request_form"):
        email = st.text_input("Email", placeholder="you@example.com")
        submitted = st.form_submit_button("Send reset code", width="stretch")

    if submitted:
        with st.spinner("Sending code..."):
            result = password_reset_service.request_code(email)
        if not result["success"]:
            st.error(result["error"])
            return
        st.session_state["reset_email"] = email.strip().lower()
        st.session_state["reset_step"] = "confirm"
        st.session_state["reset_notice"] = result["message"]
        st.session_state["reset_debug_code"] = result.get("debug_code")
        st.rerun()


def _render_confirm_step() -> None:
    """Step 2: take the emailed code and the new password."""
    email = st.session_state.get("reset_email", "")
    notice = st.session_state.get("reset_notice")
    if notice:
        st.success(notice)
    debug_code = st.session_state.get("reset_debug_code")
    if debug_code:
        st.warning(f"Development mode - your reset code is: **{debug_code}**")

    st.caption(f"Resetting the password for **{email}**")
    with st.form("reset_confirm_form"):
        code = st.text_input("6-digit code from the email", max_chars=6)
        password = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Change password", width="stretch")

    if submitted:
        if password != confirm:
            st.error("The two passwords do not match.")
            return
        result = auth_service.reset_password(email, code, password)
        if not result["success"]:
            st.error(result["error"])
            return
        _clear_reset_state()
        st.session_state["page"] = "login"
        st.success("Password changed. You can log in with your new password now.")
        st.rerun()

    if st.button("Send a new code", width="stretch"):
        st.session_state["reset_step"] = "request"
        st.session_state.pop("reset_debug_code", None)
        st.rerun()


def _clear_reset_state() -> None:
    """Forget the half-finished reset held in session state."""
    for key in ("reset_step", "reset_email", "reset_notice", "reset_debug_code"):
        st.session_state.pop(key, None)
