"""Persistent login sessions.

A random token is stored in MongoDB and kept in the browser URL (``?sid=``) so a
page refresh can restore the logged-in user instead of dropping them at the login
screen. Tokens expire after ``SESSION_TTL_DAYS``.
"""

import secrets
from datetime import timedelta
from typing import Optional

from config import settings
from database import mongo
from utils.date_utils import utcnow


def create_session(user_id: str) -> str:
    """Create a new persistent session for a user and return its token."""
    token = secrets.token_urlsafe(32)
    now = utcnow()
    mongo.sessions().insert_one(
        {
            "token": token,
            "user_id": user_id,
            "created_at": now,
            "expires_at": now + timedelta(days=settings.SESSION_TTL_DAYS),
        }
    )
    return token


def get_user_id(token: str) -> Optional[str]:
    """Return the user id for a valid, unexpired token, else None."""
    if not token:
        return None
    doc = mongo.sessions().find_one({"token": token})
    if not doc:
        return None
    expires = doc.get("expires_at")
    if expires and _aware(expires) < utcnow():
        mongo.sessions().delete_one({"token": token})
        return None
    return doc.get("user_id")


def delete_session(token: str) -> None:
    """Remove a session token (used on logout)."""
    if token:
        mongo.sessions().delete_one({"token": token})


def _aware(dt):
    """Ensure a datetime is timezone-aware for safe comparison."""
    if dt.tzinfo is None:
        from datetime import timezone

        return dt.replace(tzinfo=timezone.utc)
    return dt
