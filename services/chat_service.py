"""Chat history persistence helpers.

Chat turns are already saved to MongoDB by the LangGraph ``save_chat_history``
node. These helpers reload a user's recent conversation so the chatbot page can
restore it after a page refresh ("continue where you left off").
"""

from typing import Any, Dict, List

from database import mongo
from services import food_service


def load_recent_messages(
    user_id: str, session_id: str = "", limit: int = 20
) -> List[Dict[str, Any]]:
    """Rebuild the chat UI message list from the user's saved history.

    When ``session_id`` is given, only that conversation is restored — so a
    "New chat" stays new across page refreshes instead of resurrecting older
    sessions. Returns ``{"role", "content", "foods"}`` items in chronological
    order; recommended foods are reloaded so their cards work again.
    """
    query: Dict[str, Any] = {"user_id": user_id}
    if session_id:
        query["session_id"] = session_id
    try:
        docs = list(
            mongo.chat_history().find(query).sort("created_at", -1).limit(limit)
        )
    except Exception:
        return []
    docs.reverse()  # oldest first for display

    messages: List[Dict[str, Any]] = []
    for doc in docs:
        user_msg = doc.get("user_message")
        if user_msg:
            messages.append({"role": "user", "content": user_msg, "foods": []})

        bot_msg = doc.get("bot_response")
        if bot_msg:
            food_ids = doc.get("recommended_food_ids") or []
            foods = food_service.get_foods_by_ids([f for f in food_ids if f])
            messages.append(
                {"role": "assistant", "content": bot_msg, "foods": foods}
            )
    return messages


def delete_history(user_id: str) -> int:
    """Permanently delete ALL of a user's chat history. Returns rows removed.

    Recommendation logs (admin analytics) and long-term preference memories
    in ChromaDB are intentionally left untouched.
    """
    try:
        result = mongo.chat_history().delete_many({"user_id": user_id})
        return int(result.deleted_count)
    except Exception:
        return 0


def latest_session_id(user_id: str) -> str:
    """Return the user's most recent chat session id, or '' if none."""
    try:
        doc = mongo.chat_history().find_one(
            {"user_id": user_id}, sort=[("created_at", -1)]
        )
    except Exception:
        return ""
    return (doc or {}).get("session_id", "") or ""


def list_sessions(user_id: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Summarise a user's past conversations, most recently used first.

    Each entry is ``{"session_id", "title", "message_count", "last_at"}`` where
    ``title`` is the first thing the user said in that conversation - enough to
    recognise it in the history list.
    """
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"created_at": 1}},
        {
            "$group": {
                "_id": "$session_id",
                "messages": {"$push": "$user_message"},
                "last_at": {"$last": "$created_at"},
                "message_count": {"$sum": 1},
            }
        },
        {"$sort": {"last_at": -1}},
        {"$limit": int(limit)},
    ]
    try:
        rows = list(mongo.chat_history().aggregate(pipeline))
    except Exception:
        return []

    sessions: List[Dict[str, Any]] = []
    for row in rows:
        session_id = row.get("_id") or ""
        if not session_id:
            continue
        sessions.append(
            {
                "session_id": session_id,
                "title": _session_title(row.get("messages") or []),
                "message_count": int(row.get("message_count", 0)),
                "last_at": row.get("last_at"),
            }
        )
    return sessions


# Openers that say nothing about what the conversation was about.
_GREETINGS = {
    "hi", "hii", "hey", "hello", "yo", "hi there", "hello there",
    "good morning", "good evening", "good afternoon", "namaste", "thanks",
    "thank you", "ok", "okay",
}


def _session_title(messages, max_length: int = 40) -> str:
    """Short, human-readable label for a conversation.

    Uses the first message that actually says something - people usually open
    with "hi", which would otherwise make every conversation look identical.
    """
    cleaned = [" ".join(str(m or "").split()) for m in (messages or [])]
    cleaned = [m for m in cleaned if m]
    if not cleaned:
        return "(empty chat)"

    meaningful = [
        m for m in cleaned if m.lower().strip("!?. ") not in _GREETINGS and len(m) > 3
    ]
    text = (meaningful or cleaned)[0]
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def delete_session(user_id: str, session_id: str) -> int:
    """Delete a single conversation. Returns how many turns were removed.

    Scoped by ``user_id`` as well as ``session_id`` so one user can never
    delete another user's conversation.
    """
    if not session_id:
        return 0
    try:
        result = mongo.chat_history().delete_many(
            {"user_id": user_id, "session_id": session_id}
        )
        return int(result.deleted_count)
    except Exception:
        return 0
