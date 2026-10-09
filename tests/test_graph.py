"""Integration tests for the LangGraph chatbot workflow.

Skipped when MongoDB is unavailable. They exercise the full graph end-to-end
using whatever seed data is present, without needing an LLM API key (the graph
falls back to rule-based understanding offline).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import mongo
from services import user_service

pytestmark = pytest.mark.skipif(
    not mongo.ping(), reason="MongoDB is not available"
)


def _any_user_id():
    users = user_service.list_users()
    # Prefer a user with no allergies: the allergy safety filter (correctly)
    # strips allergen dishes, which would skew these generic behavior tests.
    for user in users:
        if not (user.get("preferences") or {}).get("allergies"):
            return user["id"]
    return users[0]["id"] if users else None


TEST_SESSION = "test-session"


@pytest.fixture(autouse=True)
def _clean_test_session_history():
    """Keep graph tests out of a real user's chat history.

    The graph saves every turn to chat_history, so without this the seeded
    demo user's saved conversations slowly fill up with test queries - which
    are then visible in the chat history sidebar.
    """
    yield
    try:
        mongo.chat_history().delete_many({"session_id": TEST_SESSION})
    except Exception:
        pass


def test_graph_returns_bot_response():
    from agents import graph

    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")
    result = graph.run_chat(user_id, "test-session", "Recommend spicy food under Rs. 500")
    assert isinstance(result.get("bot_response"), str)
    assert result["bot_response"]


def test_graph_saves_chat_history():
    from agents import graph

    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")
    before = mongo.chat_history().count_documents({"user_id": user_id})
    graph.run_chat(user_id, "test-session", "I want vegetarian dinner")
    after = mongo.chat_history().count_documents({"user_id": user_id})
    # One turn saves (at least) one chat-history record for this user.
    assert after >= before + 1


def test_graph_recommends_within_budget():
    from agents import graph

    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")
    result = graph.run_chat(user_id, "test-session", "spicy food under Rs. 500")
    foods = result.get("ranked_foods", [])
    assert result.get("intent") in ("recommend", "list")
    assert foods, "expected at least one recommendation"
    # The waiter may stretch ~20% for a near-budget pick, but not wildly over.
    assert all(float(f.get("price", 0)) <= 600 for f in foods)


def test_graph_lists_all_chicken_items():
    from agents import graph

    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")
    result = graph.run_chat(user_id, "test-session", "show me all items with chicken")
    foods = result.get("ranked_foods", [])
    assert foods, "expected chicken dishes to be listed"

    def has_chicken(food):
        blob = (food.get("name", "") + " "
                + " ".join(food.get("ingredients", []) or [])).lower()
        return "chicken" in blob

    assert all(has_chicken(f) for f in foods)


def test_graph_declines_off_topic():
    from agents import graph

    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")
    result = graph.run_chat(user_id, "test-session", "write me a python script to sort a list")
    # It must not actually help with the off-topic task; a reply is always returned.
    assert isinstance(result.get("bot_response"), str) and result["bot_response"]
    assert "def " not in result["bot_response"]  # did not produce code


def test_graph_handles_impossible_request_gracefully():
    from agents import graph

    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")
    result = graph.run_chat(
        user_id, "test-session", "Recommend food under Rs. 1 with unicorn meat"
    )
    # Even with no match, a helpful response string is always returned.
    assert isinstance(result.get("bot_response"), str)
    assert result["bot_response"]


def test_graph_greeting_replies_without_ordering():
    from agents import graph
    from services import order_service

    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")
    before = len(order_service.list_user_orders(user_id))
    result = graph.run_chat(user_id, "test-session", "hi there!")
    # A greeting must produce a friendly reply and must never place an order.
    assert result.get("bot_response")
    assert result.get("intent") != "order"
    assert len(order_service.list_user_orders(user_id)) == before


def test_graph_routes_order_and_places_it():
    from agents import graph
    from services import food_service, order_service

    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")
    foods = food_service.list_foods(include_unavailable=False)
    if not foods:
        pytest.skip("No foods seeded")

    target = foods[0]
    before = len(order_service.list_user_orders(user_id))
    result = graph.run_chat(
        user_id,
        "test-session",
        f"order the first one",
        last_recommended=[target],
    )
    assert result.get("intent") == "order"
    after = order_service.list_user_orders(user_id)
    assert len(after) == before + 1
    assert after[0]["items"][0]["name"] == target["name"]

    # Clean up the order created by this test.
    from utils.id_utils import to_object_id

    mongo.orders().delete_one({"_id": to_object_id(after[0].get("id") or after[0].get("_id"))})
