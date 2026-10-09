"""Admin recommendation logs and chat history summary page.

Two tabs over the two audit trails the chatbot writes:

* ``recommendation_logs`` - one row per recommendation, with the query, the
  memories used, the dishes returned, each score and each reason. This is the
  evidence behind the result-analysis figures.
* ``chat_history``        - every user/bot exchange, so a conversation can be
  read back in full.

Both are read newest-first and capped, and every row names the user who caused
it (the collections store a ``user_id``, which is resolved to a name here).
"""

import pandas as pd
import streamlit as st

from database import mongo
from utils.date_utils import format_display

# How many rows each tab loads. Both collections grow on every chat turn.
PAGE_SIZE = 100


def render() -> None:
    """Admin view of recommendation logs and chat history."""
    st.title("📜 Recommendation Logs")
    st.caption(
        "Every recommendation and conversation the chatbot has produced. "
        "This is the raw evidence behind the Analytics page."
    )

    tab_recs, tab_chats = st.tabs(["Recommendation logs", "Chat history"])

    with tab_recs:
        _render_recommendation_logs()
    with tab_chats:
        _render_chat_history()


def _user_names() -> dict:
    """Map ``user_id`` -> display name, so logs show a person, not an id."""
    return {
        str(user["_id"]): user.get("name") or user.get("email") or "Unknown"
        for user in mongo.users().find({}, {"name": 1, "email": 1})
    }


def _render_recommendation_logs() -> None:
    """Table of what the recommender returned for each query."""
    total = mongo.recommendation_logs().count_documents({})
    logs = list(
        mongo.recommendation_logs().find().sort("created_at", -1).limit(PAGE_SIZE)
    )
    if not logs:
        st.info(
            "No recommendation logs yet. Ask the chatbot for a recommendation "
            "and the first one will appear here."
        )
        return

    names = _user_names()
    st.caption(f"Showing the latest {len(logs)} of {total} recommendation events")

    for log in logs:
        with st.container(border=True):
            head = st.columns([3, 1])
            with head[0]:
                st.markdown(f"**Query:** {log.get('query', '')}")
                who = names.get(str(log.get("user_id")), "Unknown user")
                st.caption(f"👤 {who} · {format_display(log.get('created_at'))}")
            with head[1]:
                # Recorded from the graph; older logs predate this field.
                if log.get("response_ms"):
                    st.metric("Answered in", f"{log['response_ms'] / 1000:.1f}s")

            recs = log.get("final_recommendations", [])
            if recs:
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Food": r.get("name"),
                            "Score": r.get("score"),
                            "Reason": r.get("reason"),
                        }
                        for r in recs
                    ]),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("No foods matched this query.")

            memories = log.get("retrieved_memory", [])
            if memories:
                st.caption("🧠 Memories used: " + "; ".join(memories))
            candidates = log.get("candidate_foods", [])
            if candidates:
                st.caption(
                    f"🔍 ChromaDB returned {len(candidates)} semantic candidates, "
                    f"ranked down to {len(recs)}"
                )


def _render_chat_history() -> None:
    """Table of every saved user/bot exchange."""
    total = mongo.chat_history().count_documents({})
    chats = list(mongo.chat_history().find().sort("created_at", -1).limit(PAGE_SIZE))
    if not chats:
        st.info("No chat history yet.")
        return

    names = _user_names()
    st.caption(f"Showing the latest {len(chats)} of {total} messages")
    table = [
        {
            "Time": format_display(c.get("created_at")),
            "User": names.get(str(c.get("user_id")), "Unknown"),
            "User message": c.get("user_message"),
            "Bot response": (c.get("bot_response") or "")[:120],
            "Dishes shown": len(c.get("recommended_food_ids", [])),
        }
        for c in chats
    ]
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
