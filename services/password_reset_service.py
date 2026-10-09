"""Password-reset codes.

A six-digit code is emailed to the account owner and stored in MongoDB as a
SHA-256 hash, so a leaked database dump never reveals a usable code. Codes are
single-use, expire after ``RESET_CODE_TTL_MINUTES``, and are thrown away after
``RESET_MAX_ATTEMPTS`` wrong guesses.

Requests deliberately return the same result whether or not the email belongs
to a real account, so this page cannot be used to discover who is registered.
"""

import hashlib
import secrets
from datetime import timedelta

from config import settings
from database import mongo
from services import email_service, user_service
from utils import validators
from utils.date_utils import utcnow

# Wait this long between code requests for the same address.
RESEND_COOLDOWN_SECONDS = 60


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _generate_code() -> str:
    """A cryptographically random six-digit code."""
    return f"{secrets.randbelow(1_000_000):06d}"


def request_code(email: str) -> dict:
    """Create and email a reset code for ``email``.

    Returns ``{"success", "message", "error", "debug_code"}``. ``debug_code`` is
    only ever populated when email is not configured AND the app is running in
    development, so the demo still works offline.
    """
    email = validators.normalize_email(email)
    generic = (
        "If that email belongs to an account, a reset code is on its way. "
        "Check your inbox (and spam folder)."
    )
    if not validators.is_valid_email(email):
        return {"success": False, "message": None,
                "error": "Please enter a valid email address.", "debug_code": None}

    user = user_service.get_user_by_email(email)
    if not user or not user.get("is_active", True):
        # Same answer as the success path: never confirm who has an account.
        return {"success": True, "message": generic, "error": None, "debug_code": None}

    existing = mongo.password_resets().find_one({"email": email})
    if existing and _seconds_since(existing.get("created_at")) < RESEND_COOLDOWN_SECONDS:
        wait = RESEND_COOLDOWN_SECONDS - int(_seconds_since(existing.get("created_at")))
        return {"success": False, "message": None,
                "error": f"A code was just sent. Please wait {wait} seconds "
                         f"before asking for another.",
                "debug_code": None}

    code = _generate_code()
    now = utcnow()
    mongo.password_resets().update_one(
        {"email": email},
        {"$set": {
            "email": email,
            "user_id": str(user.get("id") or user.get("_id")),
            "code_hash": _hash_code(code),
            "attempts": 0,
            "created_at": now,
            "expires_at": now + timedelta(minutes=settings.RESET_CODE_TTL_MINUTES),
        }},
        upsert=True,
    )

    if not email_service.is_configured():
        # Offline / demo mode: show the code in the UI instead of emailing it.
        if settings.APP_ENV == "development":
            return {"success": True,
                    "message": "Email is not configured on this machine, so the "
                               "code is shown below (development mode only).",
                    "error": None, "debug_code": code}
        return {"success": False, "message": None,
                "error": "Email is not configured on this server.",
                "debug_code": None}

    sent = email_service.send_reset_code(
        email, user.get("name", "there"), code, settings.RESET_CODE_TTL_MINUTES
    )
    if not sent["success"]:
        mongo.password_resets().delete_one({"email": email})
        return {"success": False, "message": None,
                "error": sent["error"], "debug_code": None}
    return {"success": True, "message": generic, "error": None, "debug_code": None}


def verify_code(email: str, code: str) -> dict:
    """Check a code without consuming it. Returns ``{"valid", "error"}``."""
    email = validators.normalize_email(email)
    record = mongo.password_resets().find_one({"email": email})
    if not record:
        return {"valid": False, "error": "No reset was requested for that email."}

    if _expired(record):
        mongo.password_resets().delete_one({"email": email})
        return {"valid": False, "error": "That code has expired. Please request a new one."}

    if record.get("attempts", 0) >= settings.RESET_MAX_ATTEMPTS:
        mongo.password_resets().delete_one({"email": email})
        return {"valid": False,
                "error": "Too many incorrect attempts. Please request a new code."}

    if _hash_code((code or "").strip()) != record.get("code_hash"):
        mongo.password_resets().update_one({"email": email}, {"$inc": {"attempts": 1}})
        left = settings.RESET_MAX_ATTEMPTS - record.get("attempts", 0) - 1
        suffix = f" {left} attempt(s) left." if left > 0 else ""
        return {"valid": False, "error": f"That code is not correct.{suffix}"}

    return {"valid": True, "error": None}


def consume_code(email: str, code: str) -> dict:
    """Verify a code and delete it so it can never be reused."""
    result = verify_code(email, code)
    if result["valid"]:
        record = mongo.password_resets().find_one_and_delete(
            {"email": validators.normalize_email(email)}
        )
        result["user_id"] = (record or {}).get("user_id")
    return result


def clear(email: str) -> None:
    """Drop any pending reset for an address (used after a successful change)."""
    mongo.password_resets().delete_one({"email": validators.normalize_email(email)})


def _expired(record: dict) -> bool:
    """True when a stored reset code has passed its expiry time."""
    expires = record.get("expires_at")
    return bool(expires) and _aware(expires) < utcnow()


def _seconds_since(moment) -> float:
    """Seconds elapsed since a stored timestamp (used for the resend cooldown)."""
    if not moment:
        return RESEND_COOLDOWN_SECONDS
    return (utcnow() - _aware(moment)).total_seconds()


def _aware(dt):
    """Ensure a datetime read back from MongoDB is timezone-aware."""
    if dt.tzinfo is None:
        from datetime import timezone

        return dt.replace(tzinfo=timezone.utc)
    return dt
