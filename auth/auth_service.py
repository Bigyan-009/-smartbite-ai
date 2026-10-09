"""Authentication service.

Handles password hashing, registration, login, and Streamlit session helpers.
Passwords are hashed with bcrypt and never stored in plain text.
"""

from typing import Optional

import bcrypt

from database import mongo
from services import session_service, user_service
from utils import validators
from utils.date_utils import utcnow
from utils.id_utils import stringify_doc

# bcrypt only hashes the first 72 bytes of a password.
_MAX_BCRYPT_BYTES = 72


def hash_password(password: str) -> str:
    """Return a bcrypt hash for the given plain-text password."""
    pw_bytes = password.encode("utf-8")[:_MAX_BCRYPT_BYTES]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Return True when ``password`` matches the stored bcrypt hash."""
    if not password or not password_hash:
        return False
    try:
        pw_bytes = password.encode("utf-8")[:_MAX_BCRYPT_BYTES]
        return bcrypt.checkpw(pw_bytes, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def register_user(
    name: str, email: str, password: str, preferences: Optional[dict] = None
) -> dict:
    """Create a new user account.

    Returns a dict with ``success`` and either ``user`` or ``error``.
    """
    name = validators.sanitize_text(name, 120)
    email = validators.normalize_email(email)

    if not name:
        return {"success": False, "error": "Name is required."}
    if not validators.is_valid_email(email):
        return {"success": False, "error": "Please enter a valid email address."}
    if not validators.is_valid_password(password):
        return {"success": False, "error": "Password must be at least 6 characters."}
    if user_service.get_user_by_email(email):
        return {"success": False, "error": "An account with this email already exists."}

    now = utcnow()
    doc = {
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
        "role": "user",
        "is_active": True,
        "preferences": _clean_preferences(preferences or {}),
        "created_at": now,
        "updated_at": now,
        "last_login": None,
    }
    result = mongo.users().insert_one(doc)
    doc["_id"] = result.inserted_id
    return {"success": True, "user": _public_user(doc)}


def login_user(email: str, password: str) -> Optional[dict]:
    """Validate credentials and return the public user dict, or None."""
    email = validators.normalize_email(email)
    user = mongo.users().find_one({"email": email})
    if not user:
        return None
    if not user.get("is_active", True):
        return None
    if not verify_password(password, user.get("password_hash", "")):
        return None

    mongo.users().update_one(
        {"_id": user["_id"]}, {"$set": {"last_login": utcnow()}}
    )
    return _public_user(user)


def reset_password(email: str, code: str, new_password: str) -> dict:
    """Set a new password after checking the emailed reset code.

    Every existing login session for that account is dropped, so anyone holding
    an old session token is logged out by the reset.
    """
    from services import password_reset_service

    email = validators.normalize_email(email)
    if not validators.is_valid_password(new_password):
        return {"success": False, "error": "Password must be at least 6 characters."}

    result = password_reset_service.consume_code(email, code)
    if not result.get("valid"):
        return {"success": False, "error": result.get("error", "Invalid code.")}

    user = mongo.users().find_one({"email": email})
    if not user:
        return {"success": False, "error": "Account not found."}

    mongo.users().update_one(
        {"_id": user["_id"]},
        {"$set": {"password_hash": hash_password(new_password), "updated_at": utcnow()}},
    )
    # A password change must invalidate every existing session.
    try:
        mongo.sessions().delete_many({"user_id": str(user["_id"])})
    except Exception:  # pragma: no cover - infra dependent
        pass
    return {"success": True, "error": None}


def change_password(user_id: str, current_password: str, new_password: str) -> dict:
    """Change a signed-in user's password after confirming the current one."""
    from utils.id_utils import to_object_id

    oid = to_object_id(user_id)
    if not oid:
        return {"success": False, "error": "Invalid user."}
    user = mongo.users().find_one({"_id": oid})
    if not user or not verify_password(current_password, user.get("password_hash", "")):
        return {"success": False, "error": "Your current password is not correct."}
    if not validators.is_valid_password(new_password):
        return {"success": False, "error": "Password must be at least 6 characters."}

    mongo.users().update_one(
        {"_id": oid},
        {"$set": {"password_hash": hash_password(new_password), "updated_at": utcnow()}},
    )
    return {"success": True, "error": None}


# ---------------------------------------------------------------------------
# Streamlit session helpers
# ---------------------------------------------------------------------------
def _session():
    """Streamlit's per-browser session state (imported lazily so services stay UI-free)."""
    import streamlit as st

    return st.session_state


def start_session(user: dict) -> None:
    """Mark the given user as logged in for the current Streamlit session.

    Also creates a persistent token (stored in MongoDB and the browser URL) so a
    page refresh keeps the user logged in.
    """
    import streamlit as st

    state = _session()
    state["authenticated"] = True
    state["user"] = user
    state["role"] = user.get("role", "user")

    try:
        token = session_service.create_session(user.get("id"))
        state["sid"] = token
        st.query_params["sid"] = token
    except Exception:
        # Persistence is best-effort; login still works for this session.
        pass


def get_current_user() -> Optional[dict]:
    """Return the logged-in user dict, or None."""
    state = _session()
    if state.get("authenticated"):
        return state.get("user")
    return None


def sync_session_url() -> None:
    """Put the session token back in the URL if it went missing.

    The token lives in the query string so a page refresh can restore the
    login. That makes the URL part of the browser's history: pressing Back
    lands on the pre-login URL (no ``sid``), and refreshing there would drop
    the user at the login page even though their session is still valid.

    Re-writing the token whenever we are logged in but the URL has lost it
    keeps Back-then-refresh working. Note the token is only as private as the
    address bar - see the security note in the README.
    """
    import streamlit as st

    state = _session()
    token = state.get("sid")
    if not state.get("authenticated") or not token:
        return
    try:
        if st.query_params.get("sid") != token:
            st.query_params["sid"] = token
    except Exception:
        # Query params are best-effort; never break a page over them.
        pass


def restore_session() -> Optional[dict]:
    """Restore a logged-in user from the URL token after a page refresh."""
    import streamlit as st

    state = _session()
    if state.get("authenticated"):
        sync_session_url()
        return state.get("user")

    token = st.query_params.get("sid")
    if not token:
        return None
    try:
        user_id = session_service.get_user_id(token)
        if not user_id:
            return None
        user = user_service.get_user_by_id(user_id)
        if not user or not user.get("is_active", True):
            return None
    except Exception:
        return None

    state["authenticated"] = True
    state["user"] = user
    state["role"] = user.get("role", "user")
    state["sid"] = token
    return user


def logout_user() -> None:
    """Clear the current session (local and persistent)."""
    import streamlit as st

    state = _session()
    token = state.get("sid")
    if token:
        try:
            session_service.delete_session(token)
        except Exception:
            pass
    try:
        if "sid" in st.query_params:
            del st.query_params["sid"]
    except Exception:
        pass
    for key in ("authenticated", "user", "role", "sid", "chat", "session_id", "chat_loaded"):
        if key in state:
            del state[key]


def require_login() -> bool:
    """Return True when a user is logged in."""
    return get_current_user() is not None


def require_admin() -> bool:
    """Return True when the logged-in user is an admin."""
    user = get_current_user()
    return bool(user) and user.get("role") == "admin"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _clean_preferences(prefs: dict) -> dict:
    """Normalise the optional preference block collected at registration."""
    return {
        "preferred_cuisines": prefs.get("preferred_cuisines", []) or [],
        "diet_type": validators.validate_diet_type(prefs.get("diet_type")),
        "spice_level": validators.validate_spice_level(prefs.get("spice_level")),
        "usual_budget": prefs.get("usual_budget"),
        "disliked_ingredients": prefs.get("disliked_ingredients", []) or [],
        "allergies": prefs.get("allergies", []) or [],
    }


def _public_user(user: dict) -> dict:
    """Return a session-safe user dict without the password hash."""
    public = stringify_doc(user) or {}
    public.pop("password_hash", None)
    return public
