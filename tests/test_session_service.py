"""Tests for persistent login sessions. Skipped when MongoDB is unavailable."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import mongo
from services import session_service

pytestmark = pytest.mark.skipif(not mongo.ping(), reason="MongoDB is not available")


def test_create_and_restore_session():
    token = session_service.create_session("user-123")
    assert token
    assert session_service.get_user_id(token) == "user-123"

    session_service.delete_session(token)
    assert session_service.get_user_id(token) is None


def test_unknown_token_returns_none():
    assert session_service.get_user_id("does-not-exist") is None
    assert session_service.get_user_id("") is None


def test_expired_session_is_rejected():
    from datetime import timedelta

    from utils.date_utils import utcnow

    token = session_service.create_session("user-expired")
    # Force the session to be already expired.
    mongo.sessions().update_one(
        {"token": token},
        {"$set": {"expires_at": utcnow() - timedelta(days=1)}},
    )
    assert session_service.get_user_id(token) is None
    # Expired token should have been cleaned up.
    assert mongo.sessions().find_one({"token": token}) is None
