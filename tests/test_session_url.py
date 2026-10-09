"""Tests for persistent login via the URL session token.

The token is kept in the query string so a refresh can restore the login. That
makes the URL part of browser history, so these tests pin down the behaviour
around Back/Forward: being logged in must survive losing the token from the
URL, and being logged out must never put it back.

They drive the real app through Streamlit's AppTest harness, so they need
MongoDB and are skipped without it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

from database import mongo
from services import session_service, user_service

APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
)

pytestmark = pytest.mark.skipif(not mongo.ping(), reason="MongoDB is not available")


def _url_token(at):
    """The ``sid`` currently in the URL (AppTest returns values as lists)."""
    value = dict(at.query_params).get("sid")
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _state(at, key):
    try:
        return at.session_state[key]
    except Exception:
        return None


@pytest.fixture
def user():
    users = [u for u in user_service.list_users() if u.get("role") != "admin"]
    if not users:
        pytest.skip("No non-admin user seeded")
    return users[0]


@pytest.fixture
def token(user):
    value = session_service.create_session(user["id"])
    yield value
    session_service.delete_session(value)


def _logged_in_app(user, token, page="menu"):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["authenticated"] = True
    at.session_state["user"] = user
    at.session_state["role"] = user.get("role", "user")
    at.session_state["sid"] = token
    at.session_state["page"] = page
    return at


def test_url_token_is_restored_when_history_drops_it(user, token):
    # Pressing Back lands on the pre-login URL, which has no sid. The next
    # rerun must put it back so a refresh from there stays logged in.
    at = _logged_in_app(user, token)
    at.run()
    assert not at.exception
    assert _url_token(at) == token


def test_refresh_with_the_token_restores_the_login(user, token):
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["sid"] = token
    at.run()
    assert not at.exception
    assert _state(at, "authenticated") is True
    assert (_state(at, "user") or {}).get("id") == user["id"]


def test_no_token_means_the_login_page_and_a_clean_url():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert _state(at, "authenticated") in (None, False)
    assert _url_token(at) is None


def test_a_stale_token_is_not_re_added_when_logged_out(token):
    # Session state may still carry a sid after logout clears the flag; the
    # app must not resurrect it into the URL.
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["sid"] = token
    at.run()
    assert not at.exception
    assert _url_token(at) is None


def test_an_expired_or_unknown_token_does_not_log_anyone_in():
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["sid"] = "definitely-not-a-real-token"
    at.run()
    assert not at.exception
    assert _state(at, "authenticated") in (None, False)
