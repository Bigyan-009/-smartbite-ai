"""Unit tests for the offline (no-LLM) chatbot logic in agents/nodes.py.

These cover the rule-based fallbacks that keep the bot working without an LLM
key: intent routing, conversational replies, and order-item extraction. All are
pure functions - no MongoDB, ChromaDB, or LLM needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import nodes


# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------
def test_fallback_intent_detects_order():
    assert nodes._fallback_intent("order 2 chicken momo") == "order"
    assert nodes._fallback_intent("I'll take the first one") == "order"
    assert nodes._fallback_intent("can you buy me a veg burger") == "order"


def test_fallback_intent_detects_menu():
    assert nodes._fallback_intent("show me the menu") == "menu"
    assert nodes._fallback_intent("what do you have?") == "menu"
    assert nodes._fallback_intent("what's available") == "menu"


def test_waiter_menu_guardrail_overrides_llm_refusal(monkeypatch):
    """Even if the LLM refuses a 'show full menu' ask, the app shows the menu."""
    foods = [
        {"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5},
        {"id": "2", "name": "Veg Burger", "price": 180, "rating_avg": 4.0},
        {"id": "3", "name": "Spicy Ramen", "price": 420, "rating_avg": 4.2},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    # Simulate the failure we saw live: the model apologises and only offers a
    # couple of dishes instead of using the "menu" intent.
    refusal = {
        "reply": "I'm sorry, but I can't provide the full menu in a list format.",
        "intent": "recommend",
        "dishes": ["Chicken Momo"],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: refusal)

    state = {"user_id": "u", "user_message": "show full menu on list"}
    result = nodes.waiter_agent(state)

    assert result["intent"] == "menu"
    assert len(result["selected_food_ids"]) == len(foods)
    assert "sorry" not in result["bot_response"].lower()


def test_waiter_menu_guardrail_fills_partial_menu(monkeypatch):
    """LLM picks intent 'menu' but lists only some dishes -> all are shown."""
    foods = [
        {"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5},
        {"id": "2", "name": "Veg Burger", "price": 180, "rating_avg": 4.0},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    partial = {
        "reply": "Here's our menu:",
        "intent": "menu",
        "dishes": ["Chicken Momo"],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: partial)

    result = nodes.waiter_agent({"user_id": "u", "user_message": "menu please"})
    assert result["intent"] == "menu"
    assert len(result["selected_food_ids"]) == len(foods)


def test_waiter_never_denies_a_dish_on_the_menu(monkeypatch):
    """Live bug: user typed 'Paneer Butter Masala' (on the menu) and the LLM
    replied 'we don't have that'. The guardrail must show the real dish."""
    foods = [
        {"id": "1", "name": "Paneer Butter Masala", "price": 350, "rating_avg": 4.4},
        {"id": "2", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    denial = {
        "reply": "I'm sorry, but we don't have Paneer Butter Masala on our menu. "
                 "How about the Chicken Momo for Rs. 250?",
        "intent": "recommend",
        "dishes": ["Chicken Momo"],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: denial)

    result = nodes.waiter_agent({"user_id": "u", "user_message": "Paneer Butter Masala"})
    assert "1" in result["selected_food_ids"]
    assert "we do have Paneer Butter Masala" in result["bot_response"]


def test_waiter_dish_fragment_matches_menu(monkeypatch):
    """A fragment like 'paneer' must surface every matching paneer dish."""
    foods = [
        {"id": "1", "name": "Paneer Butter Masala", "price": 350, "rating_avg": 4.4},
        {"id": "2", "name": "Paneer Curry", "price": 350, "rating_avg": 4.0},
        {"id": "3", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    shrug = {"reply": "Hmm, tell me more about what you feel like eating!",
             "intent": "chat", "dishes": [], "order_items": [], "preferences": {}}
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: shrug)

    result = nodes.waiter_agent({"user_id": "u", "user_message": "paneer"})
    assert set(result["selected_food_ids"]) >= {"1", "2"}
    assert "3" not in result["selected_food_ids"]


def test_waiter_never_denies_an_ingredient_we_serve(monkeypatch):
    """Live bug: 'something with mushroom?' was denied although Mushroom Pizza
    is on the menu (a stored 'dislikes mushrooms' memory misled the LLM)."""
    foods = [
        {"id": "1", "name": "Mushroom Pizza", "price": 400, "rating_avg": 4.1,
         "ingredients": ["mushroom", "cheese", "flour"], "tags": ["pizza"]},
        {"id": "2", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5,
         "ingredients": ["chicken", "flour"], "tags": ["dumpling"]},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    denial = {
        "reply": "Unfortunately, we don't have any mushroom dishes on the menu. "
                 "How about our Chicken Momo (Rs.250)?",
        "intent": "recommend",
        "dishes": ["Chicken Momo"],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: denial)

    result = nodes.waiter_agent({"user_id": "u", "user_message": "something with mushroom?"})
    assert "1" in result["selected_food_ids"]
    assert "we do have Mushroom Pizza" in result["bot_response"]


def test_list_filter_drops_lookalikes_without_the_ingredient(monkeypatch):
    """Live bug: 'all items with chicken' listed Buff Momo (no chicken). The
    deterministic filter must drop look-alike dishes from ingredient lists."""
    foods = [
        {"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5,
         "ingredients": ["chicken", "flour"], "tags": []},
        {"id": "2", "name": "Buff Momo", "price": 220, "rating_avg": 4.3,
         "ingredients": ["buff", "flour"], "tags": []},
        {"id": "3", "name": "Chicken Sekuwa", "price": 350, "rating_avg": 4.6,
         "ingredients": ["chicken", "spices"], "tags": []},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    padded = {
        "reply": "Here are our chicken dishes:",
        "intent": "list",
        "dishes": ["Chicken Momo", "Buff Momo", "Chicken Sekuwa"],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: padded)

    result = nodes.waiter_agent(
        {"user_id": "u", "user_message": "show me all items with chicken"}
    )
    assert result["selected_food_ids"] == ["1", "3"]


def test_list_filter_leaves_non_ingredient_lists_alone():
    """'everything under 300' is a price filter — no ingredient tokens, so the
    LLM's chosen list must pass through unchanged."""
    foods = [
        {"id": "1", "name": "Chicken Momo", "ingredients": ["chicken"], "tags": []},
        {"id": "2", "name": "Veg Momo", "ingredients": ["cabbage"], "tags": []},
    ]
    kept = nodes._filter_list_by_requested_ingredient(
        "everything under 300", foods, ["1", "2"]
    )
    assert kept == ["1", "2"]


def test_outside_budget_ask_with_nothing_above_gets_honest_reply(monkeypatch):
    """Live bug: 'foods outside my budget' (Rs. 1000) was answered with a list
    of dishes all UNDER Rs. 1000. Prices must be verified numerically."""
    foods = [
        {"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5,
         "tags": [], "ingredients": []},
        {"id": "2", "name": "Mushroom Pizza", "price": 550, "rating_avg": 4.1,
         "tags": [], "ingredients": []},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    wrong = {
        "reply": "Dishes outside your budget: Chicken Momo (Rs. 250), "
                 "Mushroom Pizza (Rs. 550).",
        "intent": "list",
        "dishes": ["Chicken Momo", "Mushroom Pizza"],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: wrong)

    result = nodes.waiter_agent(
        {
            "user_id": "u",
            "user_message": "what are the foods outside my budget",
            "user_profile": {"preferences": {"usual_budget": 1000}},
        }
    )
    assert result["selected_food_ids"] == []
    assert "nothing on our menu costs more than Rs. 1000" in result["bot_response"]


def test_price_comparison_drops_items_over_the_limit():
    foods = [
        {"id": "1", "name": "Veg Momo", "price": 180},
        {"id": "2", "name": "Mushroom Pizza", "price": 550},
    ]
    kept, reply = nodes._enforce_price_comparison(
        "everything under 300", foods, ["1", "2"], None
    )
    assert kept == ["1"]
    assert "under Rs. 300" in reply
    # Correct picks are left untouched (reply stays the LLM's own).
    kept, reply = nodes._enforce_price_comparison(
        "everything under 300", foods, ["1"], None
    )
    assert kept == ["1"] and reply is None
    # No comparison in the message -> no changes.
    kept, reply = nodes._enforce_price_comparison(
        "something spicy", foods, ["1", "2"], None
    )
    assert kept == ["1", "2"] and reply is None


def _allergy_foods():
    return [
        {"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5,
         "ingredients": ["chicken", "flour"], "allergens": ["gluten"], "tags": []},
        {"id": "2", "name": "Veg Momo", "price": 180, "rating_avg": 4.2,
         "ingredients": ["cabbage", "flour"], "allergens": ["gluten"], "tags": []},
    ]


def test_allergy_filter_removes_allergen_recommendations(monkeypatch):
    """Live bug: profile said 'allergic to chicken' but chat suggested chicken.
    The allergy exclusion must be deterministic, not trusted to the LLM."""
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: _allergy_foods())
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    picks = {
        "reply": "I recommend the Chicken Momo and Veg Momo!",
        "intent": "recommend",
        "dishes": ["Chicken Momo", "Veg Momo"],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: picks)

    result = nodes.waiter_agent(
        {
            "user_id": "u",
            "user_message": "recommend me momo",
            "user_profile": {"preferences": {"allergies": ["chicken"]}},
        }
    )
    assert result["selected_food_ids"] == ["2"]
    assert "allergy" in result["bot_response"].lower()


def test_allergy_filter_blocks_chat_orders(monkeypatch):
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: _allergy_foods())
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    order = {
        "reply": "One Chicken Momo coming up!",
        "intent": "order",
        "dishes": [],
        "order_items": [{"dish": "Chicken Momo", "quantity": 1}],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: order)

    result = nodes.waiter_agent(
        {
            "user_id": "u",
            "user_message": "order 1 chicken momo",
            "user_profile": {"preferences": {"allergies": ["chicken"]}},
        }
    )
    assert result["pending_order_items"] == []
    assert "allergic" in result["bot_response"].lower()


def test_allergy_override_when_ordering_for_someone_else(monkeypatch):
    """Live bug: 'chicken items, I have my brother with me' still had every
    chicken dish stripped. Explicit for-someone-else wording lifts the filter."""
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: _allergy_foods())
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    picks = {
        "reply": "For your brother, our Chicken Momo is a great pick!",
        "intent": "recommend",
        "dishes": ["Chicken Momo"],
        "order_items": [],
        "allergy_override": True,
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: picks)

    result = nodes.waiter_agent(
        {
            "user_id": "u",
            "user_message": "i want a chicken item for my brother",
            "user_profile": {"preferences": {"allergies": ["chicken"]}},
        }
    )
    assert result["selected_food_ids"] == ["1"]
    assert "left out" not in result["bot_response"]


def test_allergies_extracted_from_memories():
    prefs = nodes._remembered_preferences(
        {}, ["User is allergic to peanuts.", "User likes spicy food."]
    )
    assert prefs["allergies"] == ["peanuts"]


def test_cooling_ask_swaps_hot_dishes_for_cold_ones(monkeypatch):
    """Live bug: 'anything cool?' was answered with steamed momo and hot soup
    ('perfect for cooling down'). Dish temperature is a fact — hot picks must
    be replaced with dishes actually served cold."""
    foods = [
        {"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5,
         "category": "Snack", "tags": ["dumpling", "steamed"], "ingredients": []},
        {"id": "2", "name": "Veg Soup", "price": 180, "rating_avg": 4.0,
         "category": "Soup", "tags": ["soup", "warm"], "ingredients": []},
        {"id": "3", "name": "Fresh Fruit Salad", "price": 220, "rating_avg": 4.4,
         "category": "Salad", "tags": ["cold", "fresh", "cooling"], "ingredients": []},
        {"id": "4", "name": "Mango Lassi", "price": 150, "rating_avg": 4.6,
         "category": "Drink", "tags": ["drink", "cold"], "ingredients": []},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    hot_picks = {
        "reply": "The Veg Soup and Chicken Momo are perfect for cooling down!",
        "intent": "recommend",
        "dishes": ["Veg Soup", "Chicken Momo"],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: hot_picks)

    result = nodes.waiter_agent(
        {"user_id": "u", "user_message": "anything good for hot weather anything cool?"}
    )
    assert set(result["selected_food_ids"]) == {"3", "4"}
    assert "Momo" not in result["bot_response"]
    assert "Soup" not in result["bot_response"]


def test_cooling_guard_leaves_good_cold_picks_alone():
    foods = [
        {"id": "1", "name": "Mango Lassi", "rating_avg": 4.6,
         "category": "Drink", "tags": ["drink", "cold"]},
        {"id": "2", "name": "Chicken Momo", "rating_avg": 4.5,
         "category": "Snack", "tags": ["steamed"]},
    ]
    ids, replaced = nodes._ensure_cooling_dishes("something cold please", foods, ["1"])
    assert ids == ["1"] and replaced is False
    # Non-cooling messages pass through untouched.
    ids, replaced = nodes._ensure_cooling_dishes("something spicy", foods, ["2"])
    assert ids == ["2"] and replaced is False


def test_keyword_matcher_ignores_negated_ingredients():
    name_map = {
        "mushroom pizza": {"id": "1", "name": "Mushroom Pizza",
                           "ingredients": ["mushroom", "cheese"], "tags": []},
        "veg momo": {"id": "2", "name": "Veg Momo",
                     "ingredients": ["cabbage", "carrot"], "tags": []},
    }
    # "without mushroom" negates the ingredient -> mushroom pizza NOT matched.
    ids = nodes._menu_dishes_by_keyword("something veg without mushroom", name_map)
    assert "1" not in ids
    # A positive mention still matches.
    ids = nodes._menu_dishes_by_keyword("something with mushroom please", name_map)
    assert ids == ["1"]


def test_waiter_ingredient_dodge_is_corrected(monkeypatch):
    """Live bug #2: model didn't deny mushroom but dodged it ('how about Veg
    Momo instead?') because of a remembered dislike. Today's ask must win."""
    foods = [
        {"id": "1", "name": "Mushroom Pizza", "price": 400, "rating_avg": 4.1,
         "ingredients": ["mushroom", "cheese", "flour"], "tags": ["pizza"]},
        {"id": "2", "name": "Veg Momo", "price": 180, "rating_avg": 4.0,
         "ingredients": ["cabbage", "carrot", "flour"], "tags": ["dumpling"]},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    dodge = {
        "reply": "I remember you don't usually like mushrooms — how about our "
                 "Veg Momo instead? Only Rs. 180!",
        "intent": "recommend",
        "dishes": ["Veg Momo"],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: dodge)

    result = nodes.waiter_agent({"user_id": "u", "user_message": "something with mushroom?"})
    assert "1" in result["selected_food_ids"]  # Mushroom Pizza added
    assert "Mushroom Pizza" in result["bot_response"]


def test_waiter_satisfied_ingredient_ask_is_untouched(monkeypatch):
    """When the model already picked a dish with the asked ingredient, the
    dodge guard must not add anything."""
    foods = [
        {"id": "1", "name": "Mushroom Pizza", "price": 400, "rating_avg": 4.1,
         "ingredients": ["mushroom", "cheese"], "tags": []},
        {"id": "2", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5,
         "ingredients": ["chicken"], "tags": []},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    good = {"reply": "Our Mushroom Pizza (Rs. 400) is exactly that!",
            "intent": "recommend", "dishes": ["Mushroom Pizza"],
            "order_items": [], "preferences": {}}
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: good)

    result = nodes.waiter_agent({"user_id": "u", "user_message": "something with mushroom?"})
    assert result["selected_food_ids"] == ["1"]
    assert "And since you asked" not in result["bot_response"]


def test_waiter_clarifying_question_is_not_overridden(monkeypatch):
    """A vague ask answered with a clarifying question (no denial) must be
    left alone — keyword matching only rescues actual denials."""
    foods = [
        {"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5,
         "ingredients": ["chicken"], "tags": [], "meal_type": ["dinner"]},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    clarify = {"reply": "Sure — what are you in the mood for tonight, something "
                        "light or something filling?",
               "intent": "chat", "dishes": [], "order_items": [], "preferences": {}}
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: clarify)

    result = nodes.waiter_agent({"user_id": "u", "user_message": "something for dinner"})
    assert result["intent"] == "chat"
    assert result["selected_food_ids"] == []


def test_waiter_legitimate_denial_is_untouched(monkeypatch):
    """Denying a dish we truly don't serve (sushi) must NOT be overridden."""
    foods = [{"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5}]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    denial = {"reply": "Sorry, we don't have sushi — can I tempt you with momo instead?",
              "intent": "chat", "dishes": [], "order_items": [], "preferences": {}}
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: denial)

    result = nodes.waiter_agent({"user_id": "u", "user_message": "do you have sushi?"})
    assert result["selected_food_ids"] == []
    assert "sushi" in result["bot_response"]


def test_fallback_intent_detects_order_status():
    """Live bug: asking about orders must never place a new order."""
    assert nodes._fallback_intent("what is my order status?") == "order_status"
    assert nodes._fallback_intent("show my previous orders") == "order_status"
    assert nodes._fallback_intent("where is my order?") == "order_status"
    assert nodes._fallback_intent("track my order") == "order_status"
    assert nodes._fallback_intent("order history please") == "order_status"
    assert nodes._fallback_intent("what did i order before?") == "order_status"


def test_fallback_intent_detects_cancel_order():
    assert nodes._fallback_intent("cancel my order") == "cancel_order"
    assert nodes._fallback_intent("please cancel that order") == "cancel_order"
    assert nodes._fallback_intent("delete my last order") == "cancel_order"


def test_fallback_intent_ordering_still_works():
    """New-order phrasings must still route to 'order', not status."""
    assert nodes._fallback_intent("order 2 chicken momo") == "order"
    assert nodes._fallback_intent("add veg burger to my order") == "order"
    assert nodes._fallback_intent("order the same again") == "order"


def test_fallback_intent_reorder_with_modification_is_order():
    """Live bug: 'order my last order but in veg' was hijacked to order_status
    by the 'my last order' pattern. Imperative reorders are new orders."""
    assert nodes._fallback_intent("order my last order but in veg") == "order"
    assert nodes._fallback_intent("reorder my last order") == "order"
    assert nodes._fallback_intent("get me my usual") == "order"
    # ...but QUESTIONS about the last order are still status checks.
    assert nodes._fallback_intent("show my last order") == "order_status"
    assert nodes._fallback_intent("what was my previous order?") == "order_status"


def test_waiter_reorder_with_modification_places_order(monkeypatch):
    """The guardrail must not hijack an LLM-classified reorder-with-changes."""
    foods = [
        {"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5},
        {"id": "2", "name": "Veg Momo", "price": 180, "rating_avg": 4.2},
    ]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    good = {
        "reply": "Switching your usual momo to veg — great choice!",
        "intent": "order",
        "dishes": [],
        "order_items": [{"dish": "Veg Momo", "quantity": 1}],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: good)

    result = nodes.waiter_agent(
        {"user_id": "u", "user_message": "order my last order but in veg"}
    )
    assert result["intent"] == "order"
    assert result["pending_order_items"] == [{"food_id": "2", "quantity": 1}]


def test_waiter_llm_intent_is_final_for_order_management(monkeypatch):
    """With an LLM, the waiter's intent decision routes directly — no regex
    override — and its natural lead-in reply is preserved."""
    foods = [{"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5}]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    chosen = {
        "reply": "Of course — cancelling that for you.",
        "intent": "cancel_order",
        "dishes": [],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: chosen)

    result = nodes.waiter_agent(
        {"user_id": "u", "user_message": "actually I changed my mind, don't send it"}
    )
    assert result["intent"] == "cancel_order"
    assert result["bot_response"] == "Of course — cancelling that for you."
    assert result["pending_order_items"] == []


def test_offline_waiter_routes_order_management(monkeypatch):
    foods = [{"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5}]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: False)

    result = nodes.waiter_agent({"user_id": "u", "user_message": "show my orders"})
    assert result["intent"] == "order_status"
    assert result["pending_order_items"] == []

    result = nodes.waiter_agent({"user_id": "u", "user_message": "cancel my order"})
    assert result["intent"] == "cancel_order"
    assert result["pending_order_items"] == []


def test_waiter_indirect_status_question_keeps_llm_lead(monkeypatch):
    """Indirect ask ('did my food get made yet?') that the LLM itself maps to
    order_status: intent honored and the natural lead-in reply is kept."""
    foods = [{"id": "1", "name": "Chicken Momo", "price": 250, "rating_avg": 4.5}]
    monkeypatch.setattr(nodes.food_service, "list_foods", lambda **kw: list(foods))
    monkeypatch.setattr(nodes.llm_provider, "is_available", lambda: True)
    good = {
        "reply": "Let me check on that for you! 🙂",
        "intent": "order_status",
        "dishes": [],
        "order_items": [],
        "preferences": {},
    }
    monkeypatch.setattr(nodes.llm_provider, "generate_json", lambda *a, **kw: good)

    result = nodes.waiter_agent(
        {"user_id": "u", "user_message": "did my food get made yet?"}
    )
    assert result["intent"] == "order_status"
    assert result["bot_response"] == "Let me check on that for you! 🙂"
    assert result["pending_order_items"] == []


def test_show_orders_node_combines_lead_with_summary(monkeypatch):
    orders = [
        {"id": "o1", "status": "preparing",
         "items": [{"name": "Chicken Momo", "quantity": 1}], "total": 250},
    ]
    monkeypatch.setattr(
        nodes.order_service, "list_user_orders", lambda uid: list(orders)
    )
    result = nodes.show_orders_node(
        {"user_id": "u", "user_message": "x", "bot_response": "Let me check! 🙂"}
    )
    assert result["bot_response"].startswith("Let me check! 🙂")
    assert "Chicken Momo" in result["bot_response"]
    assert "Preparing" in result["bot_response"]


def test_orders_for_prompt_shows_status_and_cancellability():
    orders = [
        {"status": "placed",
         "items": [{"name": "Chicken Momo", "quantity": 2}], "total": 500},
        {"status": "preparing",
         "items": [{"name": "Veg Burger", "quantity": 1}], "total": 180},
    ]
    text = nodes._orders_for_prompt(orders)
    assert "PLACED" in text and "PREPARING" in text
    assert "can still be cancelled" in text
    assert "can no longer be cancelled" in text
    assert nodes._orders_for_prompt([]) == "(no orders yet)"


# ---------------------------------------------------------------------------
# Order summary / cancel target helpers
# ---------------------------------------------------------------------------
def test_orders_summary_lists_status_items_and_total():
    orders = [
        {
            "id": "o1",
            "status": "preparing",
            "items": [{"name": "Chicken Momo", "quantity": 2}],
            "total": 500,
        },
        {
            "id": "o2",
            "status": "delivered",
            "items": [{"name": "Veg Burger", "quantity": 1}],
            "total": 180,
        },
    ]
    text = nodes._orders_summary(orders)
    assert "Preparing" in text
    assert "Delivered" in text
    assert "2× Chicken Momo" in text
    assert "500" in text and "180" in text


def test_pick_order_to_cancel_matches_dish_name():
    cancellable = [
        {"id": "new", "status": "placed",
         "items": [{"name": "Veg Burger", "quantity": 1}]},
        {"id": "old", "status": "placed",
         "items": [{"name": "Chicken Momo", "quantity": 2}]},
    ]
    picked = nodes._pick_order_to_cancel("cancel the chicken momo order", cancellable)
    assert picked["id"] == "old"
    # No dish named -> most recent cancellable order.
    picked = nodes._pick_order_to_cancel("cancel my order", cancellable)
    assert picked["id"] == "new"


def test_fallback_intent_detects_chat():
    assert nodes._fallback_intent("hi") == "chat"
    assert nodes._fallback_intent("hello there") == "chat"
    assert nodes._fallback_intent("thank you") == "chat"
    assert nodes._fallback_intent("what can you do") == "chat"


def test_fallback_intent_defaults_to_recommend_for_food():
    assert nodes._fallback_intent("recommend spicy food under 500") == "recommend"
    assert nodes._fallback_intent("I want vegetarian dinner") == "recommend"
    assert nodes._fallback_intent("something cheap and not oily") == "recommend"


# ---------------------------------------------------------------------------
# Weather context suggestions
# ---------------------------------------------------------------------------
def test_hot_weather_suggests_cooling_food_even_in_monsoon():
    """Live bug: monsoon season forced 'soup/steamed' hints even on a hot,
    clear day. Live weather must win over the season fallback."""
    from services.context_service import _suggest_tags

    hot_clear = {"temp_c": 32.0, "condition": "clear",
                 "is_cold": False, "is_hot": True, "is_rainy": False}
    tags = _suggest_tags("afternoon", "monsoon", hot_clear)
    assert "salad" in tags and "light" in tags
    assert "soup" not in tags and "steamed" not in tags


def test_season_fallback_used_only_without_weather():
    from services.context_service import _suggest_tags

    tags = _suggest_tags("afternoon", "monsoon", None)
    assert "soup" in tags  # no live weather -> monsoon implies rainy comfort food


# ---------------------------------------------------------------------------
# Conversational replies
# ---------------------------------------------------------------------------
def test_fallback_chat_replies_are_friendly_strings():
    assert "welcome" in nodes._fallback_chat("thanks a lot").lower()
    assert nodes._fallback_chat("bye").strip() != ""
    assert "recommend" in nodes._fallback_chat("what can you do").lower()
    # A plain greeting still returns a helpful, non-empty message.
    assert nodes._fallback_chat("hi").strip() != ""


# ---------------------------------------------------------------------------
# Order-item extraction
# ---------------------------------------------------------------------------
def _last():
    return [
        {"id": "a", "name": "Chicken Momo", "price": 250},
        {"id": "b", "name": "Veg Burger", "price": 180},
    ]


def test_fallback_order_items_ordinal_reference():
    items = nodes._fallback_order_items("order the first one", _last())
    assert items == [{"name": "Chicken Momo", "quantity": 1}]


def test_fallback_order_items_by_name_with_quantity():
    items = nodes._fallback_order_items("order 2 veg burger", _last())
    assert items == [{"name": "Veg Burger", "quantity": 2}]


def test_fallback_order_items_none_when_unclear():
    assert nodes._fallback_order_items("order something tasty", _last()) == []


def test_qty_near_reads_leading_number():
    assert nodes._qty_near("i want 3 chicken momo", "chicken momo") == 3
    assert nodes._qty_near("i want chicken momo", "chicken momo") == 1


# ---------------------------------------------------------------------------
# Order confirmation formatting
# ---------------------------------------------------------------------------
def test_order_confirmation_lists_items_and_total():
    order = {
        "items": [
            {"name": "Chicken Momo", "price": 250, "quantity": 2, "subtotal": 500}
        ],
        "total": 500,
    }
    text = nodes._order_confirmation(order)
    assert "Chicken Momo" in text
    assert "500" in text
    assert "Order placed" in text


# ---------------------------------------------------------------------------
# Nutrition goals (offline understanding)
# ---------------------------------------------------------------------------
def test_fallback_detects_light_request():
    for message in ("something light please", "I want low calorie food",
                    "give me something healthy", "nothing too oily"):
        prefs = nodes._fallback_extract(message, {})
        assert prefs["nutrition_goal"] == "light", message


def test_fallback_detects_high_protein_request():
    for message in ("high protein food", "I'm going to the gym",
                    "something for muscle"):
        prefs = nodes._fallback_extract(message, {})
        assert prefs["nutrition_goal"] == "high_protein", message


def test_fallback_detects_hearty_request():
    prefs = nodes._fallback_extract("I'm starving, something filling", {})
    assert prefs["nutrition_goal"] == "hearty"


def test_fallback_leaves_nutrition_goal_unset_for_plain_requests():
    prefs = nodes._fallback_extract("recommend spicy momo under 300", {})
    assert prefs["nutrition_goal"] is None


# ---------------------------------------------------------------------------
# Nutrition / ratings in the waiter prompt
# ---------------------------------------------------------------------------
def test_menu_line_carries_nutrition():
    assert nodes._nutrition_for_prompt({"calories": 450, "protein": 18}) == (
        "450kcal 18g protein"
    )
    assert nodes._nutrition_for_prompt({}) == "nutrition n/a"


def test_rated_dishes_are_split_into_loved_and_disliked():
    foods = [{"id": "1", "name": "Momo"}, {"id": "2", "name": "Pizza"},
             {"id": "3", "name": "Ramen"}]
    text = nodes._rated_for_prompt({"1": 5, "2": 1, "3": 3}, foods)
    assert "LOVED: Momo (5/5)" in text
    assert "DISLIKED" in text and "Pizza (1/5)" in text
    assert "Ramen" not in text  # 3 stars is neutral


def test_rated_prompt_handles_no_ratings():
    assert "not rated anything" in nodes._rated_for_prompt({}, [])


# ---------------------------------------------------------------------------
# Favourites (the bot must never invent a saved dish)
# ---------------------------------------------------------------------------
def test_fallback_routes_favourite_questions():
    for message in ("what are my favourites?", "show my saved dishes",
                    "which food did I star?", "my favorite items"):
        assert nodes._fallback_intent(message) == "favourites", message


def test_favourites_prompt_line():
    assert nodes._favourites_for_prompt([{"name": "Veg Momo"}]) == "Veg Momo"
    assert "none saved" in nodes._favourites_for_prompt([])


def test_show_favourites_lists_only_saved_dishes():
    state = {"favourites": [{"id": "1", "name": "Vanilla Ice Cream", "price": 180}],
             "bot_response": "Here you go!"}
    result = nodes.show_favourites_node(state)
    assert [f["name"] for f in result["ranked_foods"]] == ["Vanilla Ice Cream"]
    assert "saved favourite" in result["bot_response"]


def test_show_favourites_handles_an_empty_list():
    result = nodes.show_favourites_node({"favourites": [], "bot_response": "Sure!"})
    assert result["ranked_foods"] == []
    assert "haven't saved any favourites" in result["bot_response"]


def test_invented_favourites_are_stripped_from_the_reply():
    """The waiter must not pass off other dishes as the customer's favourites."""
    favourites = [{"name": "Vanilla Ice Cream"}]
    invented = "Your favourites are Veg Thakali Set and Vanilla Ice Cream."
    assert nodes._safe_favourites_lead(invented, favourites) == ""
    # A lead-in that names nothing, or only a real favourite, is kept.
    assert nodes._safe_favourites_lead("Here you go!", favourites) == "Here you go!"
    assert nodes._safe_favourites_lead(
        "You love Vanilla Ice Cream!", favourites
    ) == "You love Vanilla Ice Cream!"
