"""LangGraph state definition for the recommendation chatbot."""

import time

from typing import Any, Dict, List, TypedDict


class ChatState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes.

    ``total=False`` keeps every field optional so nodes can populate the
    state incrementally as the graph runs.
    """

    user_id: str
    session_id: str
    user_message: str
    # Monotonic clock reading taken when the turn started, so the graph can
    # record how long the whole answer took (see save_chat_history).
    started_at: float
    history: List[Dict[str, str]]
    last_recommended: List[Dict[str, Any]]
    intent: str
    context: Dict[str, Any]
    selected_food_ids: List[str]
    pending_order_items: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    order_history: Dict[str, Any]
    user_ratings: Dict[str, int]
    recent_orders: List[Dict[str, Any]]
    favourites: List[Dict[str, Any]]
    retrieved_memories: List[str]
    extracted_preferences: Dict[str, Any]
    semantic_candidates: List[Dict[str, Any]]
    ranked_foods: List[Dict[str, Any]]
    order: Dict[str, Any]
    bot_response: str
    memory_updates: List[str]
    errors: List[str]


def initial_state(
    user_id: str,
    session_id: str,
    user_message: str,
    history: List[Dict[str, str]] | None = None,
    last_recommended: List[Dict[str, Any]] | None = None,
) -> ChatState:
    """Build a fresh state for one chatbot turn."""
    return ChatState(
        started_at=time.perf_counter(),
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        history=history or [],
        last_recommended=last_recommended or [],
        intent="recommend",
        context={},
        selected_food_ids=[],
        pending_order_items=[],
        user_profile={},
        order_history={},
        user_ratings={},
        recent_orders=[],
        favourites=[],
        retrieved_memories=[],
        extracted_preferences={},
        semantic_candidates=[],
        ranked_foods=[],
        order={},
        bot_response="",
        memory_updates=[],
        errors=[],
    )
