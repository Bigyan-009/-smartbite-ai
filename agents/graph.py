"""LangGraph workflow assembly and public entry point.

The chatbot is driven by a single reasoning agent (``waiter_agent``) that is handed
the full menu and behaves like a real waiter — deciding what the customer wants,
writing a natural reply, staying strictly on-topic, and choosing which dishes to
show or order. Supporting nodes load context/memory before it and act on its
decision after it:

    load_user_profile -> load_context -> retrieve_user_memory -> waiter_agent
        -> (order?)         place_order_node   -> save_chat_history
        -> (order_status?)  show_orders_node   -> save_chat_history
        -> (cancel_order?)  cancel_order_node  -> save_chat_history
        -> (else)           present_foods_node -> save_chat_history
        -> update_user_memory_agent -> END

The hybrid ranking algorithm still runs inside ``present_foods_node`` for genuine
recommendations, and ChromaDB is used for user-memory retrieval and food similarity.
"""

from functools import lru_cache
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from agents import nodes
from agents.state import ChatState, initial_state


def _route_after_waiter(state: ChatState) -> str:
    """Route by intent: place / track / cancel orders, else present food cards."""
    intent = state.get("intent")
    if intent == "order" and state.get("pending_order_items"):
        return "place_order_node"
    if intent == "order_status":
        return "show_orders_node"
    if intent == "cancel_order":
        return "cancel_order_node"
    if intent == "favourites":
        return "show_favourites_node"
    return "present_foods_node"


@lru_cache(maxsize=1)
def build_graph():
    """Compile the LangGraph workflow once and cache it."""
    graph = StateGraph(ChatState)

    graph.add_node("load_user_profile", nodes.load_user_profile)
    graph.add_node("load_context", nodes.load_context)
    graph.add_node("retrieve_user_memory", nodes.retrieve_user_memory)
    graph.add_node("waiter_agent", nodes.waiter_agent)
    graph.add_node("place_order_node", nodes.place_order_node)
    graph.add_node("show_orders_node", nodes.show_orders_node)
    graph.add_node("cancel_order_node", nodes.cancel_order_node)
    graph.add_node("show_favourites_node", nodes.show_favourites_node)
    graph.add_node("present_foods_node", nodes.present_foods_node)
    graph.add_node("save_chat_history", nodes.save_chat_history)
    graph.add_node("update_user_memory_agent", nodes.update_user_memory_agent)

    graph.add_edge(START, "load_user_profile")
    graph.add_edge("load_user_profile", "load_context")
    graph.add_edge("load_context", "retrieve_user_memory")
    graph.add_edge("retrieve_user_memory", "waiter_agent")

    graph.add_conditional_edges(
        "waiter_agent",
        _route_after_waiter,
        {
            "place_order_node": "place_order_node",
            "show_orders_node": "show_orders_node",
            "cancel_order_node": "cancel_order_node",
            "show_favourites_node": "show_favourites_node",
            "present_foods_node": "present_foods_node",
        },
    )

    graph.add_edge("place_order_node", "save_chat_history")
    graph.add_edge("show_orders_node", "save_chat_history")
    graph.add_edge("cancel_order_node", "save_chat_history")
    graph.add_edge("show_favourites_node", "save_chat_history")
    graph.add_edge("present_foods_node", "save_chat_history")
    graph.add_edge("save_chat_history", "update_user_memory_agent")
    graph.add_edge("update_user_memory_agent", END)

    return graph.compile()


def run_chat(
    user_id: str,
    session_id: str,
    user_message: str,
    history: List[Dict[str, str]] | None = None,
    last_recommended: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Run one full chatbot turn and return the final state as a dict.

    ``history`` is the recent conversation and ``last_recommended`` is the food
    list from the previous assistant turn, giving the agent conversational memory
    (so it can resolve "order the first one", "make it cheaper", etc.).
    """
    app = build_graph()
    state = initial_state(
        user_id, session_id, user_message, history=history, last_recommended=last_recommended
    )
    result = app.invoke(state)
    return dict(result)
