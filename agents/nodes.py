"""LangGraph node implementations.

Every node accepts the shared :class:`ChatState` and returns a partial dict of
updates. All external calls (LLM, MongoDB, ChromaDB) are wrapped so a failure
degrades gracefully instead of crashing the chat turn.

The chatbot is driven by one reasoning agent (``waiter_agent``); the other
nodes load context before it and act on its decision after it. When no LLM is
configured, ``_offline_waiter`` and the ``_fallback_*`` helpers provide
rule-based understanding so the whole system still works offline.
"""

import re
import time
from typing import Any, Dict, List, Optional

from agents import prompts
from agents.state import ChatState
from config import settings
from database import chroma
from recommendation import hybrid_ranker
from services import (
    context_service,
    food_service,
    llm_provider,
    memory_service,
    order_service,
    user_service,
)
from utils.date_utils import format_display, utcnow

DEFAULT_COUNT = 3
VALID_INTENTS = {
    "menu", "recommend", "list", "order", "order_status", "cancel_order",
    "favourites", "chat", "decline",
}


# ---------------------------------------------------------------------------
# Node 1: load_user_profile
# ---------------------------------------------------------------------------
def load_user_profile(state: ChatState) -> Dict[str, Any]:
    """Graph step 1: load everything the system already knows about the user.

    Pulls their account and stated preferences, the dishes they have
    actually ordered before (revealed preferences), their recent orders
    and their own star ratings - all of which feed the ranker and the
    waiter prompt further down the graph.
    """
    try:
        profile = user_service.get_user_by_id(state["user_id"]) or {}
    except Exception as exc:  # pragma: no cover - infra dependent
        return {"user_profile": {}, "errors": _err(state, f"profile: {exc}")}
    # Revealed preferences: what this user has actually ordered before feeds
    # the hybrid ranker alongside stated preferences and memories.
    history = order_service.order_history_signals(state["user_id"])
    # Recent orders give the waiter real context for indirect questions like
    # "is my food coming?" or "forget that order".
    try:
        recent = order_service.list_user_orders(state["user_id"])[:3]
    except Exception:
        recent = []
    # The user's own ratings: dishes they scored badly must stop coming back,
    # dishes they loved should surface more often.
    try:
        ratings = food_service.user_ratings(state["user_id"])
    except Exception:
        ratings = {}
    # The dishes this user has starred, so the waiter can answer "what are my
    # favourites?" from real data instead of guessing.
    try:
        favourites = food_service.list_favourites(state["user_id"])
    except Exception:
        favourites = []
    return {
        "user_profile": profile,
        "order_history": history,
        "recent_orders": recent,
        "user_ratings": ratings,
        "favourites": favourites,
    }


# ---------------------------------------------------------------------------
# Node 1b: load_context (weather / season / time)
# ---------------------------------------------------------------------------
def load_context(state: ChatState) -> Dict[str, Any]:
    """Load ambient context so recommendations fit the weather, season & time."""
    try:
        context = context_service.get_context()
    except Exception as exc:  # pragma: no cover - network/infra dependent
        return {"context": {}, "errors": _err(state, f"context: {exc}")}
    return {"context": context}


# ===========================================================================
# The waiter agent - the single reasoning brain of the chatbot.
#
# It is handed the full available menu and behaves like a real waiter: it decides
# what the customer wants (recommend / list / menu / order / chat / decline),
# writes a natural reply, and picks which menu items to show or order. Because it
# reasons over the real menu (including ingredients), open questions like "all
# items with chicken" or "cheapest veg option" work with no hardcoded filters.
# ===========================================================================
def waiter_agent(state: ChatState) -> Dict[str, Any]:
    """Graph step 4: the single reasoning brain of the chatbot.

    Hands the whole available menu to the LLM and asks it to behave like a
    waiter: work out what the customer wants (recommend / list / menu /
    order / order_status / cancel_order / chat / decline), write a natural
    reply, and name the dishes involved. Dish names are then resolved back
    to real ids, so the bot can never invent something off-menu. With no
    LLM configured it falls back to ``_offline_waiter``.
    """
    try:
        foods = food_service.list_foods(include_unavailable=False)
    except Exception as exc:
        return {
            "intent": "chat",
            "bot_response": "Sorry, our kitchen system is momentarily unavailable. "
            "Please try again in a moment.",
            "selected_food_ids": [],
            "pending_order_items": [],
            "errors": _err(state, f"waiter: {exc}"),
        }

    # Map dish name -> food doc so we can resolve the model's name-based picks to
    # real ids (names are meaningful, so the LLM chooses them reliably; opaque ids
    # it tends to mangle).
    name_map = {str(f.get("name", "")).strip().lower(): f for f in foods}

    if not llm_provider.is_available():
        return _offline_waiter(state, foods)

    remembered_prefs = _remembered_preferences(
        state.get("user_profile", {}), state.get("retrieved_memories", [])
    )
    prompt = prompts.WAITER_PROMPT.format(
        menu=_menu_for_prompt(foods),
        remembered=_format_remembered(remembered_prefs),
        ordered_before=_ordered_before_for_prompt(state.get("order_history", {}), foods),
        rated=_rated_for_prompt(state.get("user_ratings", {}), foods),
        favourites=_favourites_for_prompt(state.get("favourites", [])),
        recent_orders=_orders_for_prompt(state.get("recent_orders", [])),
        memories=state.get("retrieved_memories", []) or "(none yet)",
        context=_format_context(state.get("context", {})),
        history=_format_history(state.get("history", [])),
        last_recommended=_format_recommended(state.get("last_recommended", [])),
        user_message=state["user_message"],
    )
    parsed = llm_provider.generate_json(prompt, system=prompts.WAITER_SYSTEM)
    if not isinstance(parsed, dict):
        return _offline_waiter(state, foods)

    intent = str(parsed.get("intent", "chat")).strip().lower()
    if intent not in VALID_INTENTS:
        intent = "chat"
    reply = str(parsed.get("reply", "")).strip() or "How can I help you with our menu?"

    # The waiter's intent decision is final. Order-management intents route to
    # dedicated nodes that append accurate MongoDB data after the waiter's
    # natural lead-in reply — the LLM never invents order facts.
    if intent in ("order_status", "cancel_order", "favourites"):
        return {
            "intent": intent,
            "bot_response": reply,
            "selected_food_ids": [],
            "pending_order_items": [],
            "extracted_preferences": _preferences_from_waiter(parsed.get("preferences")),
        }

    # Resolve dish names -> real food ids (anti-hallucination + fuzzy match).
    food_ids = _resolve_dish_ids(parsed.get("dishes"), name_map)

    # An explicit "show me the menu" must always show the whole menu — the LLM
    # occasionally refuses (trying to keep replies short). Only clear full-menu
    # wording triggers this, so ordinary conversation is never hijacked.
    if _FULL_MENU_RE.search(state["user_message"].lower()) and intent in (
        "chat", "recommend", "decline"
    ):
        intent = "menu"
        reply = "Here's our full menu 🍽️ — tap ☆ to save or 🛒 to order any dish:"
    if intent == "menu" and len(food_ids) < len(foods):
        food_ids = [f.get("id") or f.get("_id") for f in foods]

    # Guardrail: never deny food we actually serve. If the customer named a
    # dish (or, when the LLM claims we have nothing, asked for an ingredient
    # like "something with mushroom"), show the real matching dishes.
    if intent != "order":
        denied = bool(reply) and _DENIAL_RE.search(reply.lower())
        mentioned = _menu_dishes_mentioned(state["user_message"], name_map)
        if denied and not mentioned:
            mentioned = _menu_dishes_by_keyword(state["user_message"], name_map)
        missing = [fid for fid in mentioned if fid not in food_ids]
        if missing and (denied or intent in ("chat", "decline")):
            if intent in ("chat", "decline", "recommend"):
                intent = "list"
            food_ids = list(dict.fromkeys(mentioned + food_ids))
            if denied:
                by_id = {str(f.get("id") or f.get("_id")): f for f in foods}
                names = [by_id[fid].get("name") for fid in mentioned if fid in by_id]
                listed = ", ".join(n for n in names[:4] if n)
                reply = f"Good news — we do have {listed}! Here you go:"
        elif intent == "recommend" and food_ids:
            # Dodge guard: the customer positively asked for an ingredient
            # (e.g. "something with mushroom?") but — usually because of a
            # remembered dislike — none of the chosen dishes contain it.
            # Today's request wins: add the best matching dish.
            added, note = _ensure_requested_ingredient(
                state["user_message"], foods, food_ids
            )
            if added:
                food_ids = added + food_ids
                if note:
                    reply = f"{reply}\n\n{note}" if reply else note

    # An ingredient list must contain only dishes with that ingredient.
    if intent == "list":
        food_ids = _filter_list_by_requested_ingredient(
            state["user_message"], foods, food_ids
        )

    # Price-comparison asks are verified numerically against real prices.
    if intent in ("recommend", "list") and food_ids:
        food_ids, price_reply = _enforce_price_comparison(
            state["user_message"], foods, food_ids, remembered_prefs.get("budget")
        )
        if price_reply:
            reply = price_reply

    # A cool-down ask must get dishes that are actually served cold.
    if intent in ("recommend", "list") and food_ids:
        food_ids, replaced = _ensure_cooling_dishes(
            state["user_message"], foods, food_ids
        )
        if replaced:
            reply = (
                "Perfect for cooling down 🧊 — here are our chilled, "
                "refreshing picks:"
            )

    # SAFETY: never recommend a dish containing the user's allergens, no
    # matter what the LLM chose. Sole exception: the customer explicitly says
    # it's for someone else (or insists after the warning) — signalled by the
    # LLM's allergy_override flag or clear wording in the message itself.
    allergies = _lower(remembered_prefs.get("allergies"))
    if allergies and (
        bool(parsed.get("allergy_override"))
        or _ALLERGY_OVERRIDE_RE.search(state["user_message"].lower())
    ):
        allergies = set()
    if allergies and intent in ("recommend", "list") and food_ids:
        food_ids, removed = _remove_allergen_dishes(foods, food_ids, allergies)
        if removed:
            note = (
                "⚠️ I've left out " + ", ".join(removed[:4])
                + " because of your " + ", ".join(sorted(allergies)) + " allergy."
            )
            reply = f"{reply}\n\n{note}" if reply else note

    order_items = []
    for oi in parsed.get("order_items") or []:
        if not isinstance(oi, dict):
            continue
        food = _match_dish(oi.get("dish"), name_map)
        if food:
            order_items.append(
                {"food_id": food.get("id") or food.get("_id"),
                 "quantity": oi.get("quantity", 1)}
            )

    # SAFETY: refuse to order allergen dishes through chat.
    if allergies and order_items:
        safe_ids, removed = _remove_allergen_dishes(
            foods, [oi["food_id"] for oi in order_items], allergies
        )
        if removed:
            order_items = [oi for oi in order_items if oi["food_id"] in set(safe_ids)]
            warning = (
                "⚠️ I didn't order " + ", ".join(removed[:4]) + " — it contains "
                + ", ".join(sorted(allergies))
                + ", which you're allergic to. If you're absolutely sure, you can "
                "still order it yourself from the Menu page."
            )
            reply = f"{reply}\n\n{warning}" if reply else warning

    return {
        "intent": intent,
        "bot_response": reply,
        "selected_food_ids": food_ids,
        "pending_order_items": order_items,
        "extracted_preferences": _preferences_from_waiter(parsed.get("preferences")),
    }


# ---------------------------------------------------------------------------
# place_order (only when the waiter decided the customer is ordering)
# ---------------------------------------------------------------------------
def place_order_node(state: ChatState) -> Dict[str, Any]:
    """Place the order the waiter agreed to, and confirm it with real figures.

    The LLM never invents prices or totals: it only chooses dishes and
    quantities, and this node prices them from MongoDB.
    """
    items = order_service.items_from_ids(state.get("pending_order_items", []))
    if not items:
        return {
            "bot_response": state.get("bot_response")
            or "Which dish would you like to order? For example, \"order 2 Chicken Momo\".",
            "ranked_foods": [],
            "order": {},
        }

    result = order_service.place_order(state["user_id"], items)
    if not result.get("success"):
        return {
            "bot_response": "Sorry, I couldn't place that order just now. Please try again.",
            "ranked_foods": [],
            "order": {},
        }

    order = result["order"]
    reply = state.get("bot_response", "").strip()
    confirmation = _order_confirmation(order)
    combined = f"{reply}\n\n{confirmation}" if reply else confirmation
    return {"bot_response": combined, "ranked_foods": [], "order": order}


# ---------------------------------------------------------------------------
# show_orders (intent: order_status — status / tracking / order history)
# ---------------------------------------------------------------------------
def show_orders_node(state: ChatState) -> Dict[str, Any]:
    """Answer questions about existing orders: status, tracking, history."""
    try:
        orders = order_service.list_user_orders(state["user_id"])
    except Exception as exc:
        return {
            "bot_response": "Sorry, I couldn't look up your orders just now. "
            "Please try again in a moment.",
            "ranked_foods": [],
            "errors": _err(state, f"orders: {exc}"),
        }

    if not orders:
        return {
            "bot_response": "You haven't placed any orders yet. "
            "Tell me what you're craving and I'll help you order something! 😊",
            "ranked_foods": [],
        }

    # Keep the waiter's natural lead-in ("Let me check on that!") when present.
    lead = state.get("bot_response", "").strip()
    summary = _orders_summary(orders)
    combined = f"{lead}\n\n{summary}" if lead else summary
    return {"bot_response": combined, "ranked_foods": []}


# ---------------------------------------------------------------------------
# cancel_order (intent: cancel_order — cancel/remove an order via chat)
# ---------------------------------------------------------------------------
def cancel_order_node(state: ChatState) -> Dict[str, Any]:
    """Cancel one of the user's orders directly from the chat."""
    try:
        orders = order_service.list_user_orders(state["user_id"])
    except Exception as exc:
        return {
            "bot_response": "Sorry, I couldn't reach your orders just now. "
            "Please try again in a moment.",
            "ranked_foods": [],
            "errors": _err(state, f"cancel: {exc}"),
        }

    cancellable = [
        o for o in orders if o.get("status") in order_service.CANCELLABLE_STATUSES
    ]
    if not cancellable:
        active = [
            o for o in orders if o.get("status") not in order_service.TERMINAL_STATUSES
        ]
        if active:
            reply = (
                "Your order is already being prepared, so it can't be cancelled "
                "anymore. 👨‍🍳 Say \"show my orders\" to track it."
            )
        elif orders:
            reply = (
                "You have no active orders to cancel — your previous orders are "
                "already delivered or cancelled."
            )
        else:
            reply = "You don't have any orders to cancel yet."
        return {"bot_response": reply, "ranked_foods": []}

    target = _pick_order_to_cancel(state["user_message"], cancellable)
    order_id = target.get("id") or target.get("_id")
    result = order_service.cancel_order(state["user_id"], order_id)
    if not result.get("success"):
        return {
            "bot_response": "Sorry, I couldn't cancel that order: "
            f"{result.get('error', 'unknown error')}",
            "ranked_foods": [],
        }

    items = ", ".join(
        f"{i.get('quantity')}× {i.get('name')}" for i in target.get("items", [])
    )
    reply = (
        f"✅ Done — I've cancelled your order of {items} "
        f"(Rs. {int(target.get('total', 0))})."
    )
    if len(cancellable) > 1:
        reply += ' You still have other active orders — say "show my orders" to see them.'
    lead = state.get("bot_response", "").strip()
    if lead:
        reply = f"{lead}\n\n{reply}"
    return {"bot_response": reply, "ranked_foods": []}


# ---------------------------------------------------------------------------
# present_foods (resolve the waiter's chosen ids into cards)
# ---------------------------------------------------------------------------
def show_favourites_node(state: ChatState) -> Dict[str, Any]:
    """List the dishes this user has actually starred.

    Like the order nodes, the facts come from MongoDB rather than the LLM: the
    waiter only writes the friendly lead-in, so the bot can never invent a
    favourite the customer never saved.
    """
    favourites = state.get("favourites") or []
    if not favourites:
        return {
            "bot_response": "You haven't saved any favourites yet. Tap the ☆ "
            "button on any dish I show you and it'll be waiting here next time!",
            "ranked_foods": [],
        }

    count = len(favourites)
    header = (
        "Here's your saved favourite:" if count == 1
        else f"Here are your {count} saved favourites:"
    )
    # Fact guard: the waiter sometimes lists dishes from earlier in the
    # conversation as if they were saved. Any lead-in naming a dish that
    # is NOT actually a favourite is dropped, so only real data is shown.
    lead = _safe_favourites_lead(state.get("bot_response", ""), favourites)
    combined = f"{lead}\n\n{header}" if lead else header
    # The cards render underneath, so the reply stays short.
    return {"bot_response": combined, "ranked_foods": favourites}


def present_foods_node(state: ChatState) -> Dict[str, Any]:
    """Turn the waiter's chosen dish ids into the cards shown under the reply.

    For a recommendation this is where the hybrid ranking algorithm runs, so
    every card carries a score and a human-readable reason. For a plain menu
    or filtered list the dishes stay in the order the waiter picked.
    """
    ids = state.get("selected_food_ids", [])
    if not ids:
        return {"ranked_foods": []}

    foods = food_service.get_foods_by_ids(ids)
    if not foods:
        return {"ranked_foods": []}

    # For actual recommendations, run the hybrid ranking algorithm so each card
    # carries a score + reason (and ChromaDB similarity still contributes).
    if state.get("intent") == "recommend":
        try:
            candidates = chroma.search_foods(state["user_message"], n_results=10)
            similarity = {c["food_id"]: c.get("similarity", 0.0) for c in candidates}
        except Exception:
            candidates, similarity = [], {}
        for food in foods:
            food["similarity"] = similarity.get(food.get("id", food.get("_id")), 0.0)
        ranked = hybrid_ranker.rank_foods(
            foods,
            extracted_preferences=state.get("extracted_preferences", {}),
            user_profile=state.get("user_profile", {}),
            retrieved_memories=state.get("retrieved_memories", []),
            context=state.get("context", {}),
            order_history=state.get("order_history", {}),
            user_ratings=state.get("user_ratings", {}),
        )
        return {"ranked_foods": ranked, "semantic_candidates": candidates}

    # Menu / list: show the items in the order the waiter chose.
    return {"ranked_foods": foods}


# ---------------------------------------------------------------------------
# Node 2: retrieve_user_memory
# ---------------------------------------------------------------------------
def retrieve_user_memory(state: ChatState) -> Dict[str, Any]:
    """Graph step 3: fetch the most relevant long-term memories for this message.

    Always scoped to the current ``user_id`` - one user must never see
    another user's remembered preferences.
    """
    try:
        memories = memory_service.retrieve_memories(
            state["user_id"], state["user_message"], n_results=5
        )
    except Exception as exc:
        return {"retrieved_memories": [], "errors": _err(state, f"memory: {exc}")}
    return {"retrieved_memories": memories}


# ---------------------------------------------------------------------------
# Node 8: save_chat_history
# ---------------------------------------------------------------------------
def save_chat_history(state: ChatState) -> Dict[str, Any]:
    """Save the finished turn: the chat message pair plus a recommendation log.

    The recommendation log is what the admin analytics and the report's
    result analysis are built from.
    """
    from database import mongo

    ranked = state.get("ranked_foods", [])
    prefs = state.get("extracted_preferences", {})
    food_ids = [f.get("id", f.get("_id")) for f in ranked]

    try:
        order = state.get("order") or {}
        mongo.chat_history().insert_one(
            {
                "user_id": state["user_id"],
                "session_id": state["session_id"],
                "user_message": state["user_message"],
                "bot_response": state.get("bot_response", ""),
                "intent": state.get("intent", "recommend"),
                "extracted_preferences": prefs,
                "recommended_food_ids": food_ids,
                "order_id": order.get("id") or order.get("_id"),
                "created_at": utcnow(),
            }
        )
        if ranked and state.get("intent") == "recommend":
            mongo.recommendation_logs().insert_one(
                {
                    "user_id": state["user_id"],
                    "query": state["user_message"],
                    "retrieved_memory": state.get("retrieved_memories", []),
                    "candidate_foods": [
                        c["food_id"] for c in state.get("semantic_candidates", [])
                    ],
                    "final_recommendations": [
                        {
                            "food_id": f.get("id", f.get("_id")),
                            "name": f.get("name"),
                            "score": f.get("recommendation_score", 0),
                            "reason": f.get("recommendation_reason", ""),
                        }
                        for f in ranked
                    ],
                    "created_at": utcnow(),
                    # How long this answer took end to end, in milliseconds.
                    # Averaged on the analytics page for the report.
                    "response_ms": _elapsed_ms(state),
                }
            )
            food_service.increment_recommendation_count(food_ids)
    except Exception as exc:
        return {"errors": _err(state, f"save: {exc}")}
    return {}


# ---------------------------------------------------------------------------
# Node 9: update_user_memory_agent
# ---------------------------------------------------------------------------
def update_user_memory_agent(state: ChatState) -> Dict[str, Any]:
    """Final graph step: decide whether this turn taught us anything durable.

    "I don't like mushroom" is worth remembering; "hello" is not. Anything
    worth keeping is written to the ChromaDB ``user_memories`` collection
    and influences every future recommendation.
    """
    message = state["user_message"]
    memories: List[str] = []

    if llm_provider.is_available():
        prompt = prompts.MEMORY_UPDATE_PROMPT.format(
            user_message=message, bot_response=state.get("bot_response", "")
        )
        parsed = llm_provider.generate_json(prompt)
        if isinstance(parsed, dict) and parsed.get("should_save"):
            memories = [m for m in parsed.get("memories", []) if m]
    else:
        memories = _fallback_memories(message)

    if memories:
        try:
            memory_service.save_memories(state["user_id"], memories)
        except Exception as exc:
            return {"memory_updates": memories, "errors": _err(state, f"memsave: {exc}")}
    return {"memory_updates": memories}


# ===========================================================================
# Helpers
# ===========================================================================
def _err(state: ChatState, message: str) -> List[str]:
    return list(state.get("errors", [])) + [message]


def _filter_with_relaxation(
    foods: List[Dict[str, Any]],
    explicit: Dict[str, Any],
    remembered: Dict[str, Any],
    min_count: int = DEFAULT_COUNT,
):
    """Filter foods like a waiter who defaults to your tastes but honors requests.

    Two tiers of constraints are layered:

    * **Hard** (never dropped): availability, allergies, and disliked ingredients
      — unless the user explicitly asked for that very dish.
    * **Preference** constraints, each tagged *explicit* (what the user asked for
      now) or *remembered* (soft defaults from profile/memory). If the result set
      is smaller than ``min_count``, remembered constraints are dropped first
      (lowest priority first) so an explicit request is never left empty. The
      remembered constraints we set aside are returned so the reply can mention
      them ("a bit above your usual budget, but...").

    Returns ``(filtered_foods, relaxations)``.
    """
    # Safety exclusions (always applied).
    dislikes = _lower({*_as_list(explicit.get("disliked_ingredients")),
                       *_as_list(remembered.get("disliked_ingredients"))})
    allergies = _lower({*_as_list(explicit.get("allergies")),
                        *_as_list(remembered.get("allergies"))})
    requested = _norm(explicit.get("specific_dish"))

    def passes_hard(food: Dict[str, Any]) -> bool:
        if not food.get("available", True):
            return False
        ingredients = _lower(food.get("ingredients", []))
        allergens = _lower(food.get("allergens", []))
        if allergies and (allergens & allergies or ingredients & allergies):
            return False
        if dislikes and (ingredients & dislikes):
            # Allow only if the user explicitly asked for this exact dish.
            name = str(food.get("name", "")).lower()
            if not (requested and requested in name):
                return False
        return True

    # Build the ordered preference constraints (lower priority = dropped first).
    constraints = _build_constraints(explicit, remembered)

    base = [f for f in foods if passes_hard(f)]

    active = list(constraints)  # already sorted by ascending priority
    result = _evaluate(base, active)
    relaxations: List[str] = []

    # Drop remembered constraints (only) until we have enough options.
    while len(result) < min_count and active and active[0]["tier"] == "remembered":
        dropped = active.pop(0)
        if dropped.get("label"):
            relaxations.append(dropped["label"])
        result = _evaluate(base, active)

    # Last resort for an explicit budget: allow a 20% stretch before giving up.
    if len(result) < min_count and explicit.get("budget"):
        widened = [
            c for c in active if c.get("key") != "explicit_budget"
        ]
        stretched = _evaluate(base, widened)
        stretched = [
            f
            for f in stretched
            if _num(f.get("price")) is None
            or _num(f["price"]) <= _num(explicit["budget"]) * 1.2
        ]
        if len(stretched) > len(result):
            result = stretched

    return result, relaxations


def _build_constraints(
    explicit: Dict[str, Any], remembered: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Create the ordered list of relaxable preference constraints."""
    cons: List[Dict[str, Any]] = []

    # --- Remembered (soft) defaults: only when the user did not state them. ---
    if not explicit.get("spice_level") and remembered.get("spice_level"):
        spice = _norm(remembered["spice_level"])
        cons.append(_con(10, "remembered", None, "rem_spice",
                         lambda f: _norm(f.get("spice_level")) == spice))
    if not explicit.get("cuisine") and remembered.get("preferred_cuisines"):
        cset = _lower(remembered["preferred_cuisines"])
        cons.append(_con(20, "remembered", None, "rem_cuisine",
                         lambda f: _norm(f.get("cuisine")) in cset))
    if not explicit.get("budget") and remembered.get("budget"):
        rb = _num(remembered["budget"])
        cons.append(_con(40, "remembered", f"your usual Rs. {int(rb)} budget",
                         "rem_budget",
                         lambda f, rb=rb: _num(f.get("price")) is None
                         or _num(f["price"]) <= rb))
    if not explicit.get("diet_type") and remembered.get("diet_type"):
        rd = _norm(remembered["diet_type"])
        cons.append(_con(50, "remembered", f"your usual {rd.replace('_', ' ')} choice",
                         "rem_diet",
                         lambda f, rd=rd: _norm(f.get("diet_type")) == rd))

    # --- Explicit request: high priority, only dropped as a last resort. ---
    if explicit.get("spice_level"):
        spice = _norm(explicit["spice_level"])
        cons.append(_con(60, "explicit", None, "exp_spice",
                         lambda f: _norm(f.get("spice_level")) == spice))
    if explicit.get("cuisine"):
        cuisine = _norm(explicit["cuisine"])
        cons.append(_con(70, "explicit", None, "exp_cuisine",
                         lambda f: _norm(f.get("cuisine")) == cuisine))
    if explicit.get("meal_type"):
        meal = _norm(explicit["meal_type"])
        cons.append(_con(80, "explicit", None, "exp_meal",
                         lambda f: meal in _lower(f.get("meal_type", []))))
    if explicit.get("specific_dish"):
        dish = _norm(explicit["specific_dish"])
        cons.append(_con(90, "explicit", None, "exp_dish",
                         lambda f: dish in _food_text(f)))
    if explicit.get("budget"):
        eb = _num(explicit["budget"])
        cons.append(_con(95, "explicit", None, "explicit_budget",
                         lambda f, eb=eb: _num(f.get("price")) is None
                         or _num(f["price"]) <= eb))
    if explicit.get("diet_type"):
        diet = _norm(explicit["diet_type"])
        cons.append(_con(100, "explicit", None, "exp_diet",
                         lambda f: _norm(f.get("diet_type")) == diet))

    cons.sort(key=lambda c: c["priority"])
    return cons


def _con(priority, tier, label, key, test):
    return {"priority": priority, "tier": tier, "label": label, "key": key, "test": test}


def _evaluate(foods: List[Dict[str, Any]], active: List[Dict[str, Any]]):
    return [f for f in foods if all(c["test"](f) for c in active)]


def _elapsed_ms(state: ChatState) -> int:
    """Milliseconds since this chat turn began, or 0 if it was never timed."""
    started = state.get("started_at")
    if not started:
        return 0
    return int((time.perf_counter() - float(started)) * 1000)


def _food_text(food: Dict[str, Any]) -> str:
    """All of a dish's searchable text as one lower-case blob, for keyword tests."""
    parts = [str(food.get("name", "")), str(food.get("description", ""))]
    parts += [str(t) for t in food.get("tags", []) or []]
    return " ".join(parts).lower()


def _normalize_explicit(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Structured preferences the user stated in THIS message (explicit only)."""
    diet = parsed.get("diet_type")
    spice = parsed.get("spice_level")
    return {
        "budget": _num(parsed.get("budget")),
        "cuisine": parsed.get("cuisine"),
        "diet_type": diet if diet in settings.DIET_TYPES else None,
        "spice_level": spice if spice in settings.SPICE_LEVELS else None,
        "meal_type": parsed.get("meal_type"),
        "mood": parsed.get("mood"),
        "specific_dish": parsed.get("specific_dish"),
        "disliked_ingredients": _as_list(parsed.get("disliked_ingredients")),
        "allergies": _as_list(parsed.get("allergies")),
        "requested_count": int(parsed.get("requested_count") or DEFAULT_COUNT),
        "needs_clarification": bool(parsed.get("needs_clarification", False)),
        "clarification_question": parsed.get("clarification_question"),
    }


def _remembered_preferences(
    profile: Dict[str, Any], memories: List[str]
) -> Dict[str, Any]:
    """Soft defaults from the stored profile + long-term memory (the 'regular')."""
    block = (profile or {}).get("preferences", {}) or {}
    diet = block.get("diet_type")
    spice = block.get("spice_level")
    dislikes = _as_list(block.get("disliked_ingredients")) + _dislikes_from_memories(
        memories or []
    )
    allergies = _as_list(block.get("allergies")) + _allergies_from_memories(
        memories or []
    )
    budget = _num(block.get("usual_budget")) or _budget_from_memories(memories or [])
    return {
        "budget": budget,
        "diet_type": diet if diet in settings.DIET_TYPES else None,
        "spice_level": spice if spice in settings.SPICE_LEVELS else None,
        "preferred_cuisines": _as_list(block.get("preferred_cuisines")),
        "disliked_ingredients": list(dict.fromkeys(dislikes)),
        "allergies": list(dict.fromkeys(allergies)),
    }


_ALLERGY_RE = re.compile(r"allergic\s+to\s+([a-z]+)")

# Explicit "it's not for me" wording that lifts the allergy filter for a turn.
_ALLERGY_OVERRIDE_RE = re.compile(
    r"\bfor\s+(?:my|our|a|the)\s+(?:brother|sister|friend|mom|mum|dad|mother|father|"
    r"wife|husband|son|daughter|kids?|family|guest|colleague|him|her|them|someone)\b|"
    r"\b(?:not|isn'?t)\s+for\s+me\b|\bsomeone\s+else\b|\bsomebody\s+else\b|"
    r"\bi(?:'m|\s+am)\s+sure\b|\border\s+it\s+anyway\b|\bstill\s+want\s+it\b"
)


def _allergies_from_memories(memories: List[str]) -> List[str]:
    """Extract allergens from stored memory texts ('User is allergic to X.')."""
    result: List[str] = []
    for mem in memories or []:
        match = _ALLERGY_RE.search(str(mem).lower())
        if match and match.group(1) not in result:
            result.append(match.group(1))
    return result


def _remove_allergen_dishes(
    foods: List[Dict[str, Any]], food_ids: List[str], allergies
) -> tuple:
    """SAFETY: drop every dish containing one of the user's allergens.

    This is deterministic on purpose — the allergy rule must hold even when
    the LLM ignores it. Matches allergens against each dish's name,
    ingredients, and allergens list. Returns ``(safe_ids, removed_names)``.
    """
    allergies = _lower(allergies)
    if not allergies or not food_ids:
        return food_ids, []
    by_id = {str(f.get("id") or f.get("_id")): f for f in foods}
    safe: List[str] = []
    removed: List[str] = []
    for fid in food_ids:
        food = by_id.get(fid)
        if not food:
            continue
        blob = " ".join(
            [str(food.get("name", ""))]
            + [str(i) for i in food.get("ingredients", []) or []]
            + [str(a) for a in food.get("allergens", []) or []]
        ).lower()
        if any(a in blob or a.rstrip("s") in blob for a in allergies):
            removed.append(str(food.get("name")))
        else:
            safe.append(fid)
    return safe, removed


def _lower(values) -> set:
    """Normalise a value or list into a set of lower-case strings."""
    if not values:
        return set()
    if isinstance(values, str):
        values = [values]
    return {str(v).strip().lower() for v in values if str(v).strip()}


# --- Offline fallbacks -----------------------------------------------------
def _detect_diet(text: str):
    """Detect diet type from free text, checking specific terms first.

    Order matters: ``non-veg`` / meat words must win before the generic
    ``veg`` substring (otherwise "non-veg" would wrongly match "veg").
    """
    if re.search(r"non[\s-]?veg|chicken|buff|mutton|fish|\bmeat\b", text):
        return "non_veg"
    if "vegan" in text:
        return "vegan"
    if re.search(r"\beggs?\b", text):
        return "egg"
    if re.search(r"veg", text):  # vegetarian / veg
        return "veg"
    return None


_SPICE_KEYWORDS = {
    "very spicy": "very_spicy",
    "extra spicy": "very_spicy",
    "spicy": "spicy",
    "medium": "medium",
    "mild": "mild",
    "not oily": "mild",
    "light": "mild",
}

_KNOWN_DISHES = (
    "pizza", "momo", "burger", "biryani", "chowmein", "noodle", "ramen",
    "fried rice", "thakali", "dal bhat", "sekuwa", "paratha", "dosa", "samosa",
    "salad", "soup", "sandwich", "thukpa",
)

_DISLIKE_RE = re.compile(
    r"(?:don'?t\s+like|do\s+not\s+like|dislikes?|hates?|without|not\s+a\s+fan\s+of)"
    r"\s+([a-z]+)"
)
_STOP_TOKENS = {"it", "that", "this", "the", "a", "an", "any", "some", "food", "to"}


def _extract_dislike(text: str):
    """Pull a single disliked-ingredient word from free text, or None."""
    match = _DISLIKE_RE.search(text)
    if match:
        token = match.group(1).strip()
        if token and token not in _STOP_TOKENS:
            return token
    return None


def _dislikes_from_memories(memories: List[str]) -> List[str]:
    """Extract disliked ingredients from previously-stored memory texts."""
    result: List[str] = []
    for mem in memories or []:
        token = _extract_dislike(str(mem).lower())
        if token and token not in result:
            result.append(token)
    return result


def _budget_from_memories(memories: List[str]):
    """Pull a remembered 'usually under Rs. X' budget from memory texts."""
    for mem in memories or []:
        match = re.search(
            r"(?:under|below|less than|around|about)\s*(?:rs\.?\s*)?(\d{2,5})",
            str(mem).lower(),
        )
        if match:
            return float(match.group(1))
    return None


def _fallback_extract(message: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based preference extraction used when no LLM is configured."""
    text = message.lower()
    prefs: Dict[str, Any] = {
        "budget": None,
        "cuisine": None,
        "diet_type": None,
        "spice_level": None,
        "meal_type": None,
        "mood": None,
        "specific_dish": None,
        "nutrition_goal": None,
        "disliked_ingredients": [],
        "allergies": [],
        "requested_count": DEFAULT_COUNT,
        "needs_clarification": False,
        "clarification_question": None,
    }

    # Specific dish the user named (e.g. "pizza", "momo").
    for dish in _KNOWN_DISHES:
        if dish in text:
            prefs["specific_dish"] = dish
            break

    # Budget: "Rs. 500", "under 500", "below 300".
    money = re.search(r"(?:rs\.?\s*|under\s*|below\s*|less than\s*)(\d{2,5})", text)
    if money:
        prefs["budget"] = float(money.group(1))
    elif any(w in text for w in ("cheap", "low budget", "affordable")):
        prefs["budget"] = (profile.get("preferences") or {}).get("usual_budget")

    prefs["diet_type"] = _detect_diet(text)
    for key, value in _SPICE_KEYWORDS.items():
        if key in text:
            prefs["spice_level"] = value
            break

    for cuisine in ("nepali", "indian", "chinese", "italian", "fast food"):
        if cuisine in text:
            prefs["cuisine"] = cuisine.title()
            break
    for meal in ("breakfast", "lunch", "dinner", "snack"):
        if meal in text:
            prefs["meal_type"] = meal
            break

    disliked = _extract_dislike(text)
    if disliked:
        prefs["disliked_ingredients"] = [disliked]

    prefs["nutrition_goal"] = _detect_nutrition_goal(text)

    count = re.search(r"(\d+)\s+(?:foods?|items?|options?|recommendations?)", text)
    if count:
        prefs["requested_count"] = int(count.group(1))
    return prefs


_LIGHT_WORDS = ("light", "lighter", "low calorie", "low-calorie", "healthy",
                "not heavy", "diet")
# "not oily", "too oily", "less oil", "without oil" - all mean "keep it light".
_LESS_OIL_RE = re.compile(r"\b(?:not|no|less|without|too|avoid)\s+\w*\s*oil")
_PROTEIN_WORDS = ("protein", "gym", "workout", "muscle", "post-workout")
_HEARTY_WORDS = ("heavy", "filling", "hearty", "starving", "very hungry", "big meal")


def _detect_nutrition_goal(text: str) -> Optional[str]:
    """Spot a light / high-protein / hearty ask without an LLM."""
    if any(word in text for word in _PROTEIN_WORDS):
        return "high_protein"
    if any(word in text for word in _LIGHT_WORDS) or _LESS_OIL_RE.search(text):
        return "light"
    if any(word in text for word in _HEARTY_WORDS):
        return "hearty"
    return None


def _fallback_memories(message: str) -> List[str]:
    """Detect durable preferences without an LLM."""
    text = message.lower()
    memories: List[str] = []

    disliked = _extract_dislike(text)
    if disliked:
        memories.append(f"User dislikes {disliked}.")

    if ("i am" in text or "i'm" in text) and _detect_diet(text):
        memories.append(f"User diet preference: {_detect_diet(text)}.")

    if ("usually" in text or "always" in text or "budget" in text) and not memories:
        memories.append(f"User note: {message.strip()}")
    elif "remember" in text and not memories:
        memories.append(f"User note: {message.strip()}")

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for mem in memories:
        if mem not in seen:
            seen.add(mem)
            unique.append(mem)
    return unique


def _format_history(history: List[Dict[str, str]], max_turns: int = 6) -> str:
    """Render recent conversation turns for prompt context."""
    if not history:
        return "(no previous messages)"
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no previous messages)"


def _format_context(context: Dict[str, Any]) -> str:
    """Render the weather/season/time context for prompts."""
    if not context:
        return "(unknown)"
    weather = context.get("weather")
    parts = [
        f"Right now it is {context.get('summary', 'unknown')}",
        f"in {context.get('city', 'the city')}",
        f"(local time of day: {context.get('time_of_day')}, season: {context.get('season')}).",
    ]
    if weather:
        parts.append(
            f"Temperature {weather.get('temp_c')}°C, {weather.get('condition')}."
        )
    parts.append(
        "Dishes suiting this (gentle hint only): "
        + ", ".join(context.get("suggest_tags", []) or ["anything"]) + "."
    )
    return " ".join(parts)


def _menu_for_prompt(foods: List[Dict[str, Any]]) -> str:
    """Render the full menu compactly for the waiter agent (keyed by dish name)."""
    lines = []
    for f in foods:
        ingredients = ", ".join(str(i) for i in f.get("ingredients", []) or [])
        tags = ", ".join(str(t) for t in f.get("tags", []) or [])
        price = int(f.get("price", 0))
        pay = int(food_service.effective_price(f))
        if f.get("is_special") and pay < price:
            price_text = (
                f"Rs.{pay} TODAY'S SPECIAL -{int(f.get('discount_percent', 0))}% "
                f"(was Rs.{price})"
            )
        elif f.get("is_special"):
            price_text = f"Rs.{price} TODAY'S SPECIAL"
        else:
            price_text = f"Rs.{price}"
        lines.append(
            f"{f.get('name')} | {f.get('cuisine')} | {f.get('diet_type')} | "
            f"{f.get('spice_level')} | {price_text} | "
            f"rating {f.get('rating_avg', 0)} | {_nutrition_for_prompt(f)} | "
            f"{ingredients} | {tags}"
        )
    return "\n".join(lines)


def _nutrition_for_prompt(food: Dict[str, Any]) -> str:
    """Calories/protein for one menu line, so the waiter can answer health asks."""
    calories = food.get("calories")
    protein = food.get("protein")
    if calories is None and protein is None:
        return "nutrition n/a"
    parts = []
    if calories is not None:
        parts.append(f"{int(float(calories))}kcal")
    if protein is not None:
        parts.append(f"{int(float(protein))}g protein")
    return " ".join(parts)


def _safe_favourites_lead(lead: str, favourites: List[Dict[str, Any]]) -> str:
    """Drop the waiter's lead-in if it names a dish that is not a favourite.

    The saved list is a fact owned by MongoDB. Rather than trust the model to
    repeat it correctly, any lead-in mentioning a menu dish that is not on that
    list is thrown away, leaving only the accurate header and the real cards.
    """
    lead = (lead or "").strip()
    if not lead:
        return ""
    saved = {str(f.get("name", "")).lower() for f in favourites}
    lowered = lead.lower()
    try:
        menu_names = [
            str(f.get("name", ""))
            for f in food_service.list_foods(include_unavailable=True)
        ]
    except Exception:
        return lead
    for name in menu_names:
        if name and name.lower() in lowered and name.lower() not in saved:
            return ""
    return lead


def _favourites_for_prompt(favourites: List[Dict[str, Any]]) -> str:
    """The user's starred dishes, for the waiter prompt."""
    if not favourites:
        return "(none saved yet)"
    return ", ".join(str(f.get("name")) for f in favourites if f.get("name"))


def _rated_for_prompt(
    user_ratings: Dict[str, int], foods: List[Dict[str, Any]]
) -> str:
    """Dishes this user personally rated, split into loved and disliked.

    The ranker already scores on these, but the waiter needs them too so its
    written reply never talks up a dish the customer gave one star.
    """
    if not user_ratings:
        return "(they have not rated anything yet)"
    by_id = {str(f.get("id") or f.get("_id")): f for f in foods}
    loved, disliked = [], []
    for food_id, stars in user_ratings.items():
        food = by_id.get(str(food_id))
        if not food:
            continue
        try:
            stars = int(stars)
        except (TypeError, ValueError):
            continue
        if stars >= 4:
            loved.append(f"{food.get('name')} ({stars}/5)")
        elif stars <= 2:
            disliked.append(f"{food.get('name')} ({stars}/5)")
    parts = []
    if loved:
        parts.append("LOVED: " + ", ".join(loved))
    if disliked:
        parts.append("DISLIKED (do not suggest unless they ask): " + ", ".join(disliked))
    return " | ".join(parts) if parts else "(nothing strongly rated yet)"


def _ordered_before_for_prompt(
    order_history: Dict[str, Any], foods: List[Dict[str, Any]], limit: int = 5
) -> str:
    """Names of the dishes this user has ordered most, for the waiter prompt."""
    counts = (order_history or {}).get("food_counts") or {}
    if not counts:
        return "(no previous orders)"
    by_id = {str(f.get("id") or f.get("_id")): f for f in foods}
    named = [
        (by_id[fid].get("name"), count)
        for fid, count in counts.items()
        if fid in by_id
    ]
    if not named:
        return "(no previous orders)"
    named.sort(key=lambda pair: pair[1], reverse=True)
    return ", ".join(f"{name} (x{count})" for name, count in named[:limit])


_DENIAL_RE = re.compile(
    r"(?:don'?t|do\s+not|doesn'?t)\s+have|not\s+(?:on|in)\s+(?:our|the)\s+menu|"
    r"not\s+available|no\s+longer\s+(?:serve|have|offer)|we\s+don'?t\s+serve"
)

# Words the customer explicitly negated ("without mushroom", "no onion",
# "don't like paneer") — these must NOT count as requests for that ingredient.
_NEGATED_TOKEN_RE = re.compile(
    r"(?:without|no|not|don'?t\s+(?:like|want|eat)|do\s+not\s+(?:like|want|eat)|"
    r"dislikes?|hates?|allergic\s+to|avoid|except)\s+([a-z]+)"
)


def _menu_dishes_mentioned(message: str, name_map: Dict[str, Any]) -> List[str]:
    """Ids of menu dishes whose NAME the message refers to.

    Matches when a dish name appears in the message ("do you have paneer butter
    masala?") or the whole message is a fragment of a dish name ("paneer").
    """
    msg = str(message or "").strip().lower()
    if not msg:
        return []
    ids: List[str] = []
    for name, food in name_map.items():
        if name in msg or (len(msg) >= 4 and msg in name):
            fid = food.get("id") or food.get("_id")
            if fid and fid not in ids:
                ids.append(fid)
    return ids


def _menu_dishes_by_keyword(message: str, name_map: Dict[str, Any]) -> List[str]:
    """Ids of dishes matching meaningful words in the message (negation-aware).

    Catches ingredient asks like "something with mushroom?" — matching each
    non-negated word against every dish's name/ingredients/tags. Only used
    when the LLM claims we have nothing, so broad matches are safe: showing
    every chicken dish for a denied "chicken" ask is the desired behaviour.
    """
    msg = str(message or "").strip().lower()
    if not msg:
        return []
    negated = set(_NEGATED_TOKEN_RE.findall(msg))
    negated |= {f"{t}s" for t in negated} | {t.rstrip("s") for t in negated}
    tokens = {
        t
        for t in re.findall(r"[a-z]{4,}", msg)
        if t not in _QUERY_STOPWORDS and t not in negated
    }
    if not tokens:
        return []
    ids: List[str] = []
    for food in name_map.values():
        haystack = " ".join(
            [str(food.get("name", "")), str(food.get("cuisine", "")),
             " ".join(str(i) for i in food.get("ingredients", []) or []),
             " ".join(str(t) for t in food.get("tags", []) or [])]
        ).lower()
        if any(t in haystack or t.rstrip("s") in haystack for t in tokens):
            fid = food.get("id") or food.get("_id")
            if fid and fid not in ids:
                ids.append(fid)
    return ids


# Price-comparison asks ("under 300", "outside my budget"). The LLM is bad at
# arithmetic, so its picks are verified numerically against real prices.
_PRICE_ABOVE_RE = re.compile(
    r"\b(?:above|over|more\s+than|pricier\s+than|costlier\s+than|"
    r"expensive\s+than|beyond|outside|exceed(?:s|ing)?)\b"
)
_PRICE_BELOW_RE = re.compile(
    r"\b(?:under|below|less\s+than|cheaper\s+than|within|inside|at\s+most|up\s+to)\b"
)
_PRICE_NUM_RE = re.compile(r"(?:rs\.?\s*)?(\d{2,5})\b")


def _price_comparison_from_message(message: str, budget) -> tuple | None:
    """Detect '(above|below) <limit>' asks; 'my budget' resolves to the limit."""
    text = str(message or "").lower()
    above = _PRICE_ABOVE_RE.search(text)
    below = _PRICE_BELOW_RE.search(text)
    if not above and not below:
        return None
    number = _PRICE_NUM_RE.search(text)
    if number:
        limit = float(number.group(1))
    elif budget and "budget" in text:
        limit = float(budget)
    else:
        return None
    return ("above" if above else "below", limit)


def _enforce_price_comparison(
    message: str, foods: List[Dict[str, Any]], chosen_ids: List[str], budget
):
    """Numerically verify the LLM's picks for a price-comparison ask.

    Live bug: "foods outside my budget" (Rs. 1000) was answered with a list of
    dishes all UNDER Rs. 1000. Dishes failing the comparison are dropped; when
    everything is dropped, an honest templated reply replaces the wrong one.
    Returns ``(food_ids, replacement_reply_or_None)``.
    """
    comparison = _price_comparison_from_message(message, budget)
    if not comparison or not chosen_ids:
        return chosen_ids, None
    op, limit = comparison

    by_id = {str(f.get("id") or f.get("_id")): f for f in foods}

    def qualifies(fid: str) -> bool:
        food = by_id.get(fid)
        if not food:
            return False
        price = food_service.effective_price(food)
        return price > limit if op == "above" else price <= limit

    kept = [fid for fid in chosen_ids if qualifies(fid)]
    if len(kept) == len(chosen_ids):
        return chosen_ids, None  # the LLM got it right; keep its reply

    if not kept:
        if op == "above":
            reply = (
                f"Good news — nothing on our menu costs more than "
                f"Rs. {int(limit)}! Everything fits. 😊"
            )
        else:
            reply = f"Hmm, nothing on our menu is under Rs. {int(limit)} right now."
        return [], reply

    direction = "above" if op == "above" else "under"
    return kept, f"Here are the dishes {direction} Rs. {int(limit)}:"


# A request to cool down / beat the heat. Kept narrow so slang like a bare
# "cool, thanks" (intent: chat, no dishes) is never affected.
_COOLING_ASK_RE = re.compile(
    r"\bcool\s+(?:me|my|us|down|off)\b|\bcool(?:ing)?\s+(?:food|dish|drink|item|option)|"
    r"\bsomething\s+(?:cool|cold|chilled|refreshing)\b|\banything\s+(?:cool|cold|chilled)\b|"
    r"\bcold\s+(?:drink|dish|item|food)s?\b|\brefreshing\b|\bbeat\s+the\s+heat\b|"
    r"\b(?:too|so|very)\s+hot\b|\bhot\s+(?:weather|day|afternoon|summer)\b|\bsweat(?:y|ing)?\b"
)


def _is_cooling_dish(food: Dict[str, Any]) -> bool:
    """True for dishes actually served cold (by tags/category), never hot ones."""
    tags = _lower(food.get("tags", []))
    if tags & {"cold", "cooling", "chilled", "refreshing", "ice cream"}:
        return True
    category = str(food.get("category", "")).strip().lower()
    return category in ("drink", "dessert") or "salad" in tags


def _ensure_cooling_dishes(
    message: str, foods: List[Dict[str, Any]], chosen_ids: List[str]
):
    """When the customer asks for cooling food, serve only cold dishes.

    Dish temperature is a menu fact the LLM repeatedly gets wrong (offering
    steamed momo "to cool down"). If the ask matches and any chosen dish is
    hot, keep the cold picks and top up with the best-rated cold dishes.
    Returns ``(food_ids, replaced)`` — ``replaced`` means the reply text can
    no longer be trusted (it praised hot dishes) and must be rewritten.
    """
    if not _COOLING_ASK_RE.search(str(message or "").lower()):
        return chosen_ids, False
    cold = [f for f in foods if _is_cooling_dish(f)]
    if not cold:
        return chosen_ids, False  # nothing cold on the menu; leave it be

    cold_ids = {str(f.get("id") or f.get("_id")) for f in cold}
    kept = [fid for fid in chosen_ids if fid in cold_ids]
    if kept and len(kept) == len(chosen_ids):
        return chosen_ids, False  # all picks are already cold

    cold.sort(key=lambda f: -(float(f.get("rating_avg", 0) or 0)))
    for food in cold:
        fid = str(food.get("id") or food.get("_id"))
        if fid not in kept:
            kept.append(fid)
        if len(kept) >= DEFAULT_COUNT:
            break
    return kept, True


def _ingredient_blob(food: Dict[str, Any]) -> str:
    """A dish's name + ingredients as one lowercase string for matching."""
    return " ".join(
        [str(food.get("name", ""))]
        + [str(i) for i in food.get("ingredients", []) or []]
    ).lower()


def _filter_list_by_requested_ingredient(
    message: str, foods: List[Dict[str, Any]], chosen_ids: List[str]
) -> List[str]:
    """For a 'list' intent that asks for an ingredient, drop look-alike dishes.

    The LLM occasionally pads an ingredient list with similar dishes that lack
    the ingredient (e.g. Buff Momo in an "all items with chicken" list). When
    the message names a real menu ingredient, keep only dishes containing it.
    """
    msg = str(message or "").strip().lower()
    if not msg or not chosen_ids:
        return chosen_ids
    negated = set(_NEGATED_TOKEN_RE.findall(msg))
    negated |= {f"{t}s" for t in negated} | {t.rstrip("s") for t in negated}
    tokens = {
        t
        for t in re.findall(r"[a-z]{4,}", msg)
        if t not in _QUERY_STOPWORDS and t not in negated
    }
    all_blob = " ".join(_ingredient_blob(f) for f in foods)
    wanted = {t for t in tokens if t in all_blob or t.rstrip("s") in all_blob}
    if not wanted:
        return chosen_ids  # not an ingredient ask (e.g. "everything under 300")

    by_id = {str(f.get("id") or f.get("_id")): f for f in foods}

    def has_wanted(fid: str) -> bool:
        food = by_id.get(fid)
        if not food:
            return False
        blob = _ingredient_blob(food)
        return any(t in blob or t.rstrip("s") in blob for t in wanted)

    kept = [fid for fid in chosen_ids if has_wanted(fid)]
    return kept or chosen_ids  # never empty the list on a mismatch


def _ensure_requested_ingredient(
    message: str, foods: List[Dict[str, Any]], chosen_ids: List[str]
):
    """If the message positively asks for a real menu ingredient that none of
    the chosen dishes contain, return (ids_to_add, note_for_reply).

    Returns ([], None) when the request is satisfied (or no ingredient asked).
    """
    msg = str(message or "").strip().lower()
    if not msg:
        return [], None
    negated = set(_NEGATED_TOKEN_RE.findall(msg))
    negated |= {f"{t}s" for t in negated} | {t.rstrip("s") for t in negated}
    tokens = {
        t
        for t in re.findall(r"[a-z]{4,}", msg)
        if t not in _QUERY_STOPWORDS and t not in negated
    }
    if not tokens:
        return [], None

    # Keep only tokens that are actual ingredients somewhere on the menu.
    all_blob = " ".join(_ingredient_blob(f) for f in foods)
    wanted = {t for t in tokens if t in all_blob or t.rstrip("s") in all_blob}
    if not wanted:
        return [], None

    def _has_wanted(food) -> bool:
        blob = _ingredient_blob(food)
        return any(t in blob or t.rstrip("s") in blob for t in wanted)

    by_id = {str(f.get("id") or f.get("_id")): f for f in foods}
    if any(_has_wanted(by_id[fid]) for fid in chosen_ids if fid in by_id):
        return [], None  # request already satisfied

    matches = [f for f in foods if _has_wanted(f)]
    if not matches:
        return [], None
    matches.sort(key=lambda f: -(float(f.get("rating_avg", 0) or 0)))
    best = matches[0]
    note = (
        f"And since you asked — our {best.get('name')} "
        f"(Rs. {int(food_service.effective_price(best))}) fits that perfectly:"
    )
    return [best.get("id") or best.get("_id")], note


def _match_dish(name, name_map: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a dish name from the model to a real food doc (exact, then fuzzy)."""
    if not name:
        return None
    key = str(name).strip().lower()
    if key in name_map:
        return name_map[key]
    # Fuzzy: unique containment match (e.g. "momo" -> the only momo, "veg burger").
    contains = [food for n, food in name_map.items() if key in n or n in key]
    if len(contains) == 1:
        return contains[0]
    return None


def _resolve_dish_ids(dishes, name_map: Dict[str, Any]) -> List[str]:
    """Resolve a list of dish names to a de-duplicated list of real food ids."""
    ids: List[str] = []
    for name in dishes or []:
        food = _match_dish(name, name_map)
        if food:
            fid = food.get("id") or food.get("_id")
            if fid and fid not in ids:
                ids.append(fid)
    return ids


def _format_remembered(remembered: Dict[str, Any]) -> str:
    """Human-readable summary of a regular's soft default preferences."""
    if not remembered:
        return "(none on file)"
    bits = []
    if remembered.get("budget"):
        bits.append(f"usually spends around Rs. {int(remembered['budget'])}")
    if remembered.get("diet_type"):
        bits.append(f"{str(remembered['diet_type']).replace('_', ' ')}")
    if remembered.get("spice_level"):
        bits.append(f"likes {remembered['spice_level']} spice")
    if remembered.get("preferred_cuisines"):
        bits.append("enjoys " + ", ".join(remembered["preferred_cuisines"]))
    if remembered.get("disliked_ingredients"):
        bits.append("dislikes " + ", ".join(remembered["disliked_ingredients"]))
    if remembered.get("allergies"):
        bits.append("allergic to " + ", ".join(remembered["allergies"]))
    return "; ".join(bits) if bits else "(none on file)"


def _preferences_from_waiter(prefs) -> Dict[str, Any]:
    """Normalise the waiter's explicit-preferences object for ranking/logging."""
    prefs = prefs if isinstance(prefs, dict) else {}
    diet = prefs.get("diet_type")
    spice = prefs.get("spice_level")
    return {
        "budget": _num(prefs.get("budget")),
        "cuisine": prefs.get("cuisine"),
        "diet_type": diet if diet in settings.DIET_TYPES else None,
        "spice_level": spice if spice in settings.SPICE_LEVELS else None,
        "nutrition_goal": hybrid_ranker.normalize_nutrition_goal(
            prefs.get("nutrition_goal")
        ),
        "requested_count": DEFAULT_COUNT,
    }


def _offline_waiter(state: ChatState, foods: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rule-based waiter used when no LLM is configured (degraded but functional)."""
    message = state["user_message"]
    text = message.lower()
    intent = _fallback_intent(message)

    if intent == "menu":
        return {
            "intent": "menu",
            "bot_response": "Here's our menu — tap the buttons to save or order any dish:",
            "selected_food_ids": [f.get("id") for f in foods],
            "pending_order_items": [],
        }

    if intent in ("order_status", "cancel_order"):
        # The dedicated graph nodes build the reply from real order data.
        return {
            "intent": intent,
            "bot_response": "",
            "selected_food_ids": [],
            "pending_order_items": [],
        }

    if intent == "order":
        raw = _fallback_order_items(message, state.get("last_recommended", []))
        resolved = order_service.resolve_items(raw, state.get("last_recommended", []))
        return {
            "intent": "order",
            "bot_response": "",
            "selected_food_ids": [],
            "pending_order_items": [
                {"food_id": it["food_id"], "quantity": it["quantity"]} for it in resolved
            ],
        }

    if intent == "chat":
        return {
            "intent": "chat",
            "bot_response": _fallback_chat(message),
            "selected_food_ids": [],
            "pending_order_items": [],
        }

    # recommend / list: filter the menu by explicit + remembered preferences.
    explicit = _normalize_explicit(_fallback_extract(message, state.get("user_profile", {})))
    remembered = _remembered_preferences(
        state.get("user_profile", {}), state.get("retrieved_memories", [])
    )
    # "all items with X" style: keyword match across name/ingredients/tags.
    keyword_hits = _offline_keyword_matches(foods, text)
    if keyword_hits and re.search(r"\b(all|every|list|any|with|contain|options?)\b", text):
        return {
            "intent": "list",
            "bot_response": "Here's what I found on our menu:",
            "selected_food_ids": [f.get("id") for f in keyword_hits],
            "pending_order_items": [],
            "extracted_preferences": explicit,
        }

    filtered, _relax = _filter_with_relaxation(foods, explicit, remembered)
    return {
        "intent": "recommend",
        "bot_response": "Here are a few options I think you'll enjoy:",
        "selected_food_ids": [f.get("id") for f in filtered[: DEFAULT_COUNT]],
        "pending_order_items": [],
        "extracted_preferences": explicit,
    }


def _offline_keyword_matches(
    foods: List[Dict[str, Any]], text: str
) -> List[Dict[str, Any]]:
    """Find menu items whose name/ingredients/tags contain a word from the query."""
    words = {w for w in re.findall(r"[a-z]{3,}", text)} - _QUERY_STOPWORDS
    hits = []
    for f in foods:
        blob = " ".join(
            [str(f.get("name", "")), str(f.get("cuisine", "")),
             " ".join(str(i) for i in f.get("ingredients", []) or []),
             " ".join(str(t) for t in f.get("tags", []) or [])]
        ).lower()
        if any(w in blob for w in words):
            hits.append(f)
    return hits


_QUERY_STOPWORDS = {
    "all", "every", "list", "any", "with", "contain", "contains", "option", "options",
    "item", "items", "food", "foods", "dish", "dishes", "show", "the", "and", "for",
    "have", "want", "give", "some", "please", "can", "you", "your", "what", "which",
    "that", "this", "get", "would", "like", "under", "over", "around",
}


def _format_recommended(foods: List[Dict[str, Any]]) -> str:
    """Render the last recommended foods so the router/order agent can reference them."""
    if not foods:
        return "(none)"
    lines = []
    for i, f in enumerate(foods, 1):
        lines.append(f"{i}. {f.get('name')} (Rs. {int(f.get('price', 0))})")
    return "\n".join(lines)


_GREETING_RE = re.compile(
    r"\b(hi|hii+|hey+|hello|yo|namaste|namaskar|good\s+morning|good\s+evening|"
    r"good\s+afternoon|thanks|thank\s+you|thankyou|thx|bye|goodbye|ok|okay|cool|"
    r"nice|great|who\s+are\s+you|what\s+can\s+you\s+do|help)\b"
)
_ORDER_RE = re.compile(
    r"\b(order|buy|purchase|i'?ll\s+take|get\s+me|add\s+.*\s+to\s+.*order|"
    r"place\s+.*order|i\s+want\s+to\s+order)\b"
)
# Offline-mode routing only (no LLM configured): questions ABOUT existing
# orders must never place a new order. With an LLM, the waiter decides.
_ORDER_STATUS_RE = re.compile(
    r"(?:\b(?:status|track|where|history|previous|past|earlier|show|view|check|"
    r"see|list)\b[^.?!]{0,40}\borders?\b)"
    r"|(?:\borders?\b[^.?!]{0,40}\b(?:status|history|update|updates|going|ready|"
    r"so\s+far|placed)\b)"
    r"|\bwhat\s+did\s+i\s+order\b|\bhave\s+i\s+ordered\b"
    r"|\bmy\s+(?:last|latest|recent|previous|past|active|current)\s+orders?\b"
)
_CANCEL_ORDER_RE = re.compile(
    r"\b(?:cancel|remove|delete|scrap)\b[^.?!]{0,40}\borders?\b"
    r"|\borders?\b[^.?!]{0,40}\bcancel"
)
# "same again" / "reorder" style messages ARE new orders, not status questions.
# Includes imperative forms like "order my last order (but veg)" / "get me my usual".
_REORDER_RE = re.compile(
    r"\b(?:re-?order|order\s+(?:it\s+|that\s+)?again|same\s+again|"
    r"repeat\s+(?:my|the|that)|order\s+the\s+same|"
    r"(?:order|buy|get|place|give)\s+(?:me\s+)?(?:my|the)\s+"
    r"(?:last|latest|previous|usual|same|regular)\b)"
)
# Favourites questions - offline routing only; with an LLM the waiter decides.
_FAVOURITES_RE = re.compile(
    r"\b(?:favourites?|favorites?|saved|bookmarked?|starred)"
    # "did I star" is spelled out so "5 star rating" does not match.
    r"|\b(?:did i|i)\s+star\b"
)
# Broad menu detection — offline routing only (no LLM to interpret nuance).
_MENU_RE = re.compile(
    r"\b(menu|what\s+do\s+you\s+have|what'?s\s+available|what\s+can\s+i\s+(?:eat|order|get)|"
    r"show\s+me\s+(?:the\s+)?(?:food|dishes|menu)|list\s+(?:the\s+)?(?:food|dishes|menu)|"
    r"full\s+menu|all\s+(?:the\s+)?(?:food|dishes|items))\b"
)
# Strict full-menu detection — the only wording allowed to override the LLM's
# intent, so casual questions mid-conversation never dump the whole menu.
# "all items WITH chicken" is a filtered list, not a menu ask, hence the
# negative lookahead after "all ... items".
_FULL_MENU_RE = re.compile(
    r"\b(?:full|whole|entire|complete)\s+menu\b|\bmenu\b|"
    r"\ball\s+(?:the\s+)?(?:foods?|dishes|items)\b"
    r"(?!\s+(?:with|without|that|which|containing|under|below|over|above|for|in)\b)|"
    r"\beverything\s+you\s+(?:have|serve|offer)\b"
)


def _fallback_intent(message: str) -> str:
    """Rule-based intent routing used when no LLM is configured."""
    text = message.lower().strip()
    if _MENU_RE.search(text):
        return "menu"
    # Order-management questions must win over the plain "order" keyword,
    # otherwise "what's my order status?" would place a new order.
    if _CANCEL_ORDER_RE.search(text):
        return "cancel_order"
    if _FAVOURITES_RE.search(text):
        return "favourites"
    if _ORDER_STATUS_RE.search(text) and not _REORDER_RE.search(text):
        return "order_status"
    if _ORDER_RE.search(text):
        return "order"
    # A short message that's only a greeting/thanks -> chat.
    if _GREETING_RE.search(text) and len(text.split()) <= 4:
        return "chat"
    # Question with no food-request signal -> chat.
    food_signal = re.search(
        r"\b(recommend|suggest|spicy|veg|vegan|cuisine|dinner|lunch|breakfast|snack|"
        r"cheap|budget|rs\.?|under|hungry|eat|food|dish|momo|pizza|burger|rice|"
        r"biryani|chowmein)\b",
        text,
    )
    if not food_signal:
        return "chat"
    return "recommend"


def _fallback_chat(message: str) -> str:
    """A friendly reply for small talk when no LLM is configured."""
    text = message.lower()
    if re.search(r"\b(thanks|thank you|thx)\b", text):
        return "You're welcome! 😊 Would you like me to recommend or order something?"
    if re.search(r"\b(bye|goodbye)\b", text):
        return "Goodbye! Come back hungry. 🍽️"
    if re.search(r"who\s+are\s+you|what\s+can\s+you\s+do|help", text):
        return (
            "I'm SmartBite AI — I can recommend dishes based on your taste and budget, "
            "place orders, check your order status, and cancel an order for you. "
            "What are you in the mood for?"
        )
    return (
        "Hi there! 👋 I can suggest great dishes and take your order. "
        "Tell me what you feel like eating — for example, \"spicy chicken under Rs. 500\"."
    )


def _fallback_order_items(
    message: str, last_recommended: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Detect order items without an LLM."""
    text = message.lower()
    items: List[Dict[str, Any]] = []

    # Ordinal references: "the first one", "second".
    ordinals = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
    for word, idx in ordinals.items():
        if word in text and idx < len(last_recommended):
            items.append({"name": last_recommended[idx].get("name"), "quantity": 1})

    # Match any recommended food mentioned by name.
    for food in last_recommended:
        name = str(food.get("name", "")).lower()
        if name and name in text:
            qty = _qty_near(text, name)
            items.append({"name": food.get("name"), "quantity": qty})

    # De-duplicate by name.
    seen = set()
    unique = []
    for it in items:
        key = str(it.get("name", "")).lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(it)
    return unique


def _qty_near(text: str, name: str) -> int:
    """Find a quantity written just before a food name, defaulting to 1."""
    match = re.search(r"(\d+)\s+" + re.escape(name), text)
    if match:
        try:
            return max(1, int(match.group(1)))
        except ValueError:
            return 1
    return 1


def _orders_for_prompt(orders: List[Dict[str, Any]], limit: int = 3) -> str:
    """Render the user's recent orders for the waiter prompt (Nepal time)."""
    if not orders:
        return "(no orders yet)"
    lines = []
    for order in orders[:limit]:
        status = str(order.get("status", "placed"))
        items = ", ".join(
            f"{i.get('quantity')}x {i.get('name')}" for i in order.get("items", [])
        )
        bits = [f"{status.upper()} ({format_display(order.get('created_at'))})",
                f"{items} — Rs.{int(order.get('total', 0))}"]
        if status in order_service.CANCELLABLE_STATUSES:
            bits.append("can still be cancelled")
        else:
            bits.append("can no longer be cancelled")
        eta = order_service.eta_minutes_left(order)
        if eta is not None and eta > 0:
            bits.append(f"~{eta} min until ready")
        lines.append("- " + " | ".join(bits))
    return "\n".join(lines)


_STATUS_ICONS = {
    "placed": "🕒",
    "confirmed": "✅",
    "preparing": "👨‍🍳",
    "ready": "🛎️",
    "delivered": "🚚",
    "cancelled": "✖️",
}


def _orders_summary(orders: List[Dict[str, Any]], limit: int = 5) -> str:
    """Chat-friendly summary of the user's most recent orders (Nepal time)."""
    shown = orders[:limit]
    if len(orders) > limit:
        header = f"Here are your {len(shown)} most recent orders (of {len(orders)}):"
    elif len(shown) == 1:
        header = "Here's your order:"
    else:
        header = "Here are your orders:"

    lines = [header, ""]
    for order in shown:
        status = str(order.get("status", "placed"))
        icon = _STATUS_ICONS.get(status, "⚪")
        when = format_display(order.get("created_at"))
        items = ", ".join(
            f"{i.get('quantity')}× {i.get('name')}" for i in order.get("items", [])
        )
        lines.append(
            f"- {icon} **{status.title()}** · {when} · {items} — "
            f"Rs. {int(order.get('total', 0))}"
        )
        eta = order_service.eta_minutes_left(order)
        if eta is not None and eta > 0:
            lines.append(f"  ⏱️ Estimated ready in about {eta} min")

    if any(o.get("status") in order_service.CANCELLABLE_STATUSES for o in shown):
        lines.append("")
        lines.append(
            'You can say "cancel my order" to cancel one the kitchen hasn\'t '
            "started yet, or manage everything under \"My Orders\"."
        )
    return "\n".join(lines)


def _pick_order_to_cancel(
    message: str, cancellable: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Choose which cancellable order the user means.

    If the message names a dish from one of the orders, cancel that order;
    otherwise cancel the most recent one (the list is newest first).
    """
    text = str(message or "").lower()
    for order in cancellable:
        for item in order.get("items", []):
            name = str(item.get("name", "")).strip().lower()
            if name and name in text:
                return order
    return cancellable[0]


def _order_confirmation(order: Dict[str, Any]) -> str:
    """Format a placed order into the receipt-style message shown in the chat."""
    lines = ["✅ Order placed! Here's your summary:", ""]
    for item in order.get("items", []):
        lines.append(
            f"- {item.get('quantity')} × {item.get('name')} "
            f"(Rs. {int(item.get('price', 0))} each) = Rs. {int(item.get('subtotal', 0))}"
        )
    lines.append("")
    lines.append(f"**Total: Rs. {int(order.get('total', 0))}**")
    lines.append("You can view it anytime under \"My Orders\".")
    return "\n".join(lines)


def _num(value):
    """Best-effort float conversion; returns None when the value is not a number."""
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm(value):
    """Lower-case, trimmed text, or None when empty."""
    if value is None:
        return None
    return str(value).strip().lower() or None


def _as_list(value) -> List[str]:
    """Coerce a string or list into a clean list of strings."""
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(v).strip() for v in value if str(v).strip()]
