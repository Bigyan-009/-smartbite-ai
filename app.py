"""SmartBite AI - Streamlit entry point and page router.

Uses simple in-app routing driven by ``st.session_state["page"]`` so the whole
system runs as a single Streamlit app with role-based navigation.
"""

import streamlit as st

from auth import auth_service
from config import settings
from database import mongo
from ui import (
    admin_dashboard,
    analytics_page,
    chatbot_page,
    components,
    favourites_page,
    forgot_password_page,
    logs_page,
    login_page,
    manage_foods_page,
    manage_orders_page,
    menu_page,
    orders_page,
    profile_page,
    register_page,
)

st.set_page_config(page_title=settings.APP_NAME, page_icon="🍽️", layout="wide")

# Public pages (no login required).
PUBLIC_PAGES = {
    "login": login_page.render,
    "register": register_page.render,
    "forgot_password": forgot_password_page.render,
}

# User pages.
USER_PAGES = {
    "chatbot": chatbot_page.render,
    "menu": menu_page.render,
    "orders": orders_page.render,
    "profile": profile_page.render,
    "favourites": favourites_page.render,
}

# Admin pages.
ADMIN_PAGES = {
    "admin_dashboard": admin_dashboard.render,
    "manage_foods": manage_foods_page.render,
    "manage_orders": manage_orders_page.render,
    "logs": logs_page.render,
    "analytics": analytics_page.render,
}


def _check_database() -> bool:
    """Warn (once) if MongoDB is unreachable so the user gets a clear message."""
    if not mongo.ping():
        st.error(
            "⚠️ Cannot connect to MongoDB. Please start MongoDB "
            f"({settings.MONGODB_URI}) or set MONGODB_URI in your .env file, "
            "then reload this page."
        )
        return False
    return True


def _sidebar(user: dict) -> None:
    """Navigation sidebar, showing only the pages this role may open."""
    with st.sidebar:
        st.markdown(f"### 🍽️ {settings.APP_NAME}")
        st.write(f"Signed in as **{user.get('name')}**")
        st.caption(f"Role: {user.get('role')}")
        st.divider()

        if user.get("role") == "admin":
            _nav_button("📊 Dashboard", "admin_dashboard")
            _nav_button("🍲 Manage Foods", "manage_foods")
            _nav_button("🧾 Manage Orders", "manage_orders")
            _nav_button("📜 Logs", "logs")
            _nav_button("📈 Analytics", "analytics")
        else:
            _nav_button("💬 Chatbot", "chatbot")
            _nav_button("📖 Menu", "menu")
            _nav_button("🛒 My Orders", "orders")
            _nav_button("⭐ Favourites", "favourites")
            _nav_button("👤 Profile", "profile")
            components.render_sidebar_cart(lambda: _go_to("menu"))

        st.divider()
        if st.button("🚪 Logout", width="stretch"):
            auth_service.logout_user()
            st.session_state["page"] = "login"
            st.rerun()


def _nav_button(label: str, page: str) -> None:
    """One sidebar navigation button."""
    if st.button(label, width="stretch"):
        _go_to(page)


def _go_to(page: str) -> None:
    """Switch to another page and redraw."""
    st.session_state["page"] = page
    st.rerun()


def main() -> None:
    """Route the browser to the right page for the current user and role."""
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if not _check_database():
        return

    # Restore a logged-in user from the URL token after a page refresh.
    auth_service.restore_session()

    page = st.session_state["page"]
    user = auth_service.get_current_user()

    # Not logged in -> only public pages are reachable.
    if not auth_service.require_login():
        if page not in PUBLIC_PAGES:
            page = "login"
            st.session_state["page"] = "login"
        PUBLIC_PAGES[page]()
        return

    # Logged in -> render sidebar and route to the right page set.
    _sidebar(user)
    components.show_pending_order_toast()
    components.show_pending_cart_toast()
    is_admin = auth_service.require_admin()

    if page in ADMIN_PAGES:
        # Role guard: admin pages are unreachable for a normal user, even if
        # they set the page name by hand.
        if not auth_service.require_admin():
            st.error("You do not have permission to view this page.")
            return
        ADMIN_PAGES[page]()
    elif page in USER_PAGES:
        USER_PAGES[page]()
    else:
        # Default landing page depends on role.
        default = "admin_dashboard" if is_admin else "chatbot"
        st.session_state["page"] = default
        (ADMIN_PAGES if is_admin else USER_PAGES)[default]()


if __name__ == "__main__":
    main()
