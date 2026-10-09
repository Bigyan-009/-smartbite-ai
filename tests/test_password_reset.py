"""Tests for the forgot-password flow (codes, expiry, and the reset itself)."""

import os
import sys
from datetime import timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import auth_service
from config import settings
from database import mongo
from services import password_reset_service
from utils.date_utils import utcnow

pytestmark = pytest.mark.skipif(not mongo.ping(), reason="MongoDB is not available")

EMAIL = "pytest_reset_user@example.com"
PASSWORD = "originalpass123"


@pytest.fixture(autouse=True)
def mail(monkeypatch):
    """Capture outgoing reset codes instead of sending real email.

    Every test then runs the production path (email configured) regardless of
    whether SMTP credentials happen to be set in .env on this machine.
    """
    from services import email_service

    sent = []

    def _capture(to, name, code, minutes):
        sent.append({"to": to, "name": name, "code": code, "minutes": minutes})
        return {"success": True, "error": None}

    monkeypatch.setattr(email_service, "is_configured", lambda: True)
    monkeypatch.setattr(email_service, "send_reset_code", _capture)
    return sent


@pytest.fixture
def account():
    mongo.users().delete_many({"email": EMAIL})
    mongo.password_resets().delete_many({"email": EMAIL})
    result = auth_service.register_user("PyTest Reset", EMAIL, PASSWORD)
    assert result["success"], result.get("error")
    yield result["user"]
    mongo.users().delete_many({"email": EMAIL})
    mongo.password_resets().delete_many({"email": EMAIL})


def _issue_code(mail):
    """Request a code and read it back out of the captured email."""
    password_reset_service.clear(EMAIL)
    result = password_reset_service.request_code(EMAIL)
    assert result["success"], result.get("error")
    assert result["debug_code"] is None, "a configured mailer must not print the code"
    assert mail, "no email was sent"
    return mail[-1]["code"]


# ---------------------------------------------------------------------------
# Requesting a code
# ---------------------------------------------------------------------------
def test_unknown_email_gives_the_same_answer_as_a_real_one(account):
    """No user enumeration: with email working, both replies are identical.

    (In development mode without SMTP the code is printed on screen instead,
    so this guarantee only applies to the real, email-sending path.)
    """
    real = password_reset_service.request_code(EMAIL)
    password_reset_service.clear(EMAIL)
    fake = password_reset_service.request_code("definitely_not_a_user@example.com")

    assert real["success"] is fake["success"] is True
    assert real["message"] == fake["message"]
    assert real["debug_code"] is fake["debug_code"] is None


def test_email_failure_does_not_leave_a_dangling_code(account, monkeypatch):
    """If the mail cannot be sent, the stored code is removed again."""
    from services import email_service

    monkeypatch.setattr(
        email_service, "send_reset_code",
        lambda *args, **kwargs: {"success": False, "error": "SMTP is down"},
    )
    password_reset_service.clear(EMAIL)
    result = password_reset_service.request_code(EMAIL)
    assert not result["success"]
    assert "SMTP is down" in result["error"]
    assert mongo.password_resets().find_one({"email": EMAIL}) is None


def test_invalid_email_is_rejected():
    result = password_reset_service.request_code("not-an-email")
    assert not result["success"]
    assert "valid email" in result["error"]


def test_code_is_never_stored_in_plain_text(account, mail):
    code = _issue_code(mail)
    record = mongo.password_resets().find_one({"email": EMAIL})
    assert record is not None
    assert code not in str(record)
    assert record["code_hash"] != code
    assert len(record["code_hash"]) == 64  # sha256 hex


def test_a_second_request_is_rate_limited(account, mail):
    _issue_code(mail)
    again = password_reset_service.request_code(EMAIL)
    assert not again["success"]
    assert "wait" in again["error"].lower()


# ---------------------------------------------------------------------------
# Verifying a code
# ---------------------------------------------------------------------------
def test_correct_code_verifies(account, mail):
    code = _issue_code(mail)
    assert password_reset_service.verify_code(EMAIL, code)["valid"]
    # The code went to the account owner, with the configured lifetime.
    assert mail[-1]["to"] == EMAIL
    assert mail[-1]["minutes"] == settings.RESET_CODE_TTL_MINUTES


def test_wrong_code_is_rejected_and_counted(account, mail):
    _issue_code(mail)
    assert not password_reset_service.verify_code(EMAIL, "000000")["valid"]
    assert mongo.password_resets().find_one({"email": EMAIL})["attempts"] == 1


def test_code_is_discarded_after_too_many_wrong_attempts(account, mail):
    _issue_code(mail)
    for _ in range(settings.RESET_MAX_ATTEMPTS):
        password_reset_service.verify_code(EMAIL, "000000")
    result = password_reset_service.verify_code(EMAIL, "000000")
    assert not result["valid"]
    assert "request a new code" in result["error"]
    assert mongo.password_resets().find_one({"email": EMAIL}) is None


def test_expired_code_is_rejected(account, mail):
    code = _issue_code(mail)
    mongo.password_resets().update_one(
        {"email": EMAIL}, {"$set": {"expires_at": utcnow() - timedelta(minutes=1)}}
    )
    result = password_reset_service.verify_code(EMAIL, code)
    assert not result["valid"]
    assert "expired" in result["error"]


# ---------------------------------------------------------------------------
# Completing the reset
# ---------------------------------------------------------------------------
def test_reset_changes_the_password(account, mail):
    code = _issue_code(mail)
    result = auth_service.reset_password(EMAIL, code, "brandnewpass456")
    assert result["success"], result.get("error")
    assert auth_service.login_user(EMAIL, "brandnewpass456") is not None
    assert auth_service.login_user(EMAIL, PASSWORD) is None


def test_a_code_cannot_be_used_twice(account, mail):
    code = _issue_code(mail)
    assert auth_service.reset_password(EMAIL, code, "brandnewpass456")["success"]
    second = auth_service.reset_password(EMAIL, code, "yetanotherpass789")
    assert not second["success"]
    assert auth_service.login_user(EMAIL, "yetanotherpass789") is None


def test_reset_rejects_a_short_password(account, mail):
    code = _issue_code(mail)
    result = auth_service.reset_password(EMAIL, code, "abc")
    assert not result["success"]
    assert "6 characters" in result["error"]


def test_reset_logs_out_existing_sessions(account, mail):
    from services import session_service

    token = session_service.create_session(account["id"])
    assert session_service.get_user_id(token) == account["id"]
    code = _issue_code(mail)
    assert auth_service.reset_password(EMAIL, code, "brandnewpass456")["success"]
    assert session_service.get_user_id(token) is None


# ---------------------------------------------------------------------------
# Signed-in password change
# ---------------------------------------------------------------------------
def test_change_password_requires_the_current_one(account):
    wrong = auth_service.change_password(account["id"], "not-the-password", "newpass123")
    assert not wrong["success"]
    ok = auth_service.change_password(account["id"], PASSWORD, "newpass123")
    assert ok["success"], ok.get("error")
    assert auth_service.login_user(EMAIL, "newpass123") is not None
