"""Tests for chat history: session listing, titles, and per-session delete.

Persistence tests use a throwaway user id so a real user's conversations are
never touched, and are skipped when MongoDB is unavailable.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import mongo
from services import chat_service
from utils.date_utils import utcnow

TEST_USER = "pytest-chat-user"

pytest_db = pytest.mark.skipif(not mongo.ping(), reason="MongoDB is not available")


@pytest.fixture(autouse=True)
def _clean_test_chats():
    yield
    try:
        mongo.chat_history().delete_many({"user_id": TEST_USER})
    except Exception:
        pass


def _turn(session_id, user_message, minutes):
    """A chat_history document, ordered by an explicit created_at offset."""
    from datetime import timedelta

    return {
        "user_id": TEST_USER,
        "session_id": session_id,
        "user_message": user_message,
        "bot_response": "Sure!",
        "recommended_food_ids": [],
        "created_at": utcnow() + timedelta(minutes=minutes),
    }


# ---------------------------------------------------------------------------
# Titles - pure unit tests, no database
# ---------------------------------------------------------------------------
def test_title_skips_greetings():
    title = chat_service._session_title(["hi", "hello", "Recommend spicy momo"])
    assert title == "Recommend spicy momo"


def test_title_falls_back_to_the_greeting_when_thats_all_there_is():
    assert chat_service._session_title(["hi"]) == "hi"


def test_title_handles_an_empty_conversation():
    assert chat_service._session_title([]) == "(empty chat)"
    assert chat_service._session_title(["", "   "]) == "(empty chat)"


def test_title_is_truncated():
    title = chat_service._session_title(["x" * 100])
    assert len(title) == 40
    assert title.endswith("…")


def test_title_collapses_whitespace():
    assert chat_service._session_title(["  spicy    momo  "]) == "spicy momo"


# ---------------------------------------------------------------------------
# Session listing and deletion - require MongoDB
# ---------------------------------------------------------------------------
@pytest_db
def test_list_sessions_groups_and_orders_conversations():
    mongo.chat_history().insert_many(
        [
            _turn("session-old", "hi", 0),
            _turn("session-old", "Recommend veg momo", 1),
            _turn("session-new", "I want biryani", 5),
        ]
    )

    sessions = chat_service.list_sessions(TEST_USER)
    assert len(sessions) == 2

    # Most recently used conversation comes first.
    assert sessions[0]["session_id"] == "session-new"
    assert sessions[0]["message_count"] == 1
    assert sessions[0]["title"] == "I want biryani"

    # The greeting is skipped in favour of the real request.
    assert sessions[1]["session_id"] == "session-old"
    assert sessions[1]["message_count"] == 2
    assert sessions[1]["title"] == "Recommend veg momo"


@pytest_db
def test_list_sessions_is_empty_for_a_user_with_no_chats():
    assert chat_service.list_sessions("pytest-nobody") == []


@pytest_db
def test_delete_session_removes_only_that_conversation():
    mongo.chat_history().insert_many(
        [
            _turn("session-a", "Recommend momo", 0),
            _turn("session-a", "something spicy", 1),
            _turn("session-b", "I want biryani", 2),
        ]
    )

    removed = chat_service.delete_session(TEST_USER, "session-a")
    assert removed == 2

    remaining = chat_service.list_sessions(TEST_USER)
    assert [s["session_id"] for s in remaining] == ["session-b"]


@pytest_db
def test_cannot_delete_another_users_session():
    mongo.chat_history().insert_many([_turn("session-a", "Recommend momo", 0)])

    assert chat_service.delete_session("someone-else", "session-a") == 0
    assert len(chat_service.list_sessions(TEST_USER)) == 1


@pytest_db
def test_delete_session_with_no_id_is_a_no_op():
    mongo.chat_history().insert_many([_turn("session-a", "Recommend momo", 0)])
    assert chat_service.delete_session(TEST_USER, "") == 0
    assert len(chat_service.list_sessions(TEST_USER)) == 1


@pytest_db
def test_load_recent_messages_is_scoped_to_one_session():
    mongo.chat_history().insert_many(
        [
            _turn("session-a", "Recommend momo", 0),
            _turn("session-b", "I want biryani", 1),
        ]
    )

    messages = chat_service.load_recent_messages(TEST_USER, session_id="session-a")
    contents = [m["content"] for m in messages]
    assert "Recommend momo" in contents
    assert "I want biryani" not in contents
