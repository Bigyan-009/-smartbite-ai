"""User chatbot page - the heart of SmartBite AI."""

import streamlit as st

from agents import graph
from auth import auth_service
from services import chat_service, context_service, llm_provider
from ui.components import render_food_card
from utils.date_utils import format_display
from utils.id_utils import new_session_id


def render() -> None:
    """The chatbot screen - the heart of SmartBite AI."""
    user = auth_service.get_current_user()
    header = st.columns([3, 1, 1])
    with header[0]:
        st.title("🍽️ SmartBite AI")
        st.caption("Chat with me — ask for recommendations, or just say hi!")
    with header[1]:
        if st.button("🆕 New chat", width="stretch",
                     help="Start a fresh conversation (old chats stay saved)"):
            _start_new_chat()
            st.rerun()
    with header[2]:
        with st.popover("🗑️ Delete chat", width="stretch"):
            st.caption(
                "Permanently delete your entire chat history? "
                "This cannot be undone. (Your saved food preferences are kept.)"
            )
            if st.button("Yes, delete everything", type="primary",
                         key="confirm_delete_chat", width="stretch"):
                deleted = chat_service.delete_history(user["id"])
                _start_new_chat()
                st.toast(f"Deleted {deleted} chat message(s). Fresh start! 🧹")
                st.rerun()

    _show_context_banner()

    if not llm_provider.is_available():
        st.info(
            "Running in offline mode (no LLM API key set). The bot still chats, "
            "recommends, and takes orders using rule-based logic.",
            icon="🧠",
        )

    _ensure_session(user["id"])
    _history_sidebar(user["id"])

    # Replay the conversation so far.
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for food in msg.get("foods", []):
                render_food_card(food, user["id"], key=f"chat_{id(food)}")

    prompt = st.chat_input("Say hi, ask for food, or order something...")
    if prompt:
        _handle_message(prompt, user)


def _start_new_chat() -> None:
    """Reset the on-screen conversation and start a fresh session."""
    st.session_state["chat"] = []
    st.session_state["session_id"] = new_session_id()
    st.session_state["chat_loaded"] = True


def _open_session(user_id: str, session_id: str) -> None:
    """Reopen a saved conversation in the chat window."""
    st.session_state["session_id"] = session_id
    st.session_state["chat"] = chat_service.load_recent_messages(
        user_id, session_id=session_id
    )
    st.session_state["chat_loaded"] = True


def _history_sidebar(user_id: str) -> None:
    """Sidebar list of past conversations: click to reopen, 🗑 to delete one."""
    sessions = chat_service.list_sessions(user_id)
    current = st.session_state.get("session_id")

    with st.sidebar:
        st.divider()
        st.markdown("#### 💬 Chat history")
        if not sessions:
            st.caption("No saved conversations yet.")
            return

        for session in sessions:
            sid = session["session_id"]
            is_current = sid == current
            row = st.columns([5, 1])
            with row[0]:
                label = ("▶ " if is_current else "") + session["title"]
                if st.button(
                    label,
                    key=f"open_session_{sid}",
                    width="stretch",
                    disabled=is_current,
                    help=f"{session['message_count']} message(s) · "
                         f"{format_display(session['last_at'])}",
                ):
                    _open_session(user_id, sid)
                    st.rerun()
            with row[1]:
                if st.button("🗑", key=f"del_session_{sid}",
                             help="Delete this conversation"):
                    removed = chat_service.delete_session(user_id, sid)
                    # Deleting the open conversation leaves nothing to show.
                    if is_current:
                        _start_new_chat()
                    st.toast(f"Deleted conversation ({removed} message(s)).")
                    st.rerun()


def _ensure_session(user_id: str) -> None:
    """Restore the previous conversation once per browser session."""
    # Restore the previous conversation once per browser session so a page
    # refresh continues where the user left off. Only the LATEST session is
    # restored, so "New chat" stays new after a refresh.
    if not st.session_state.get("chat_loaded"):
        prev = chat_service.latest_session_id(user_id)
        st.session_state["chat"] = (
            chat_service.load_recent_messages(user_id, session_id=prev) if prev else []
        )
        st.session_state["session_id"] = prev or new_session_id()
        st.session_state["chat_loaded"] = True

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = new_session_id()
    if "chat" not in st.session_state:
        st.session_state["chat"] = []


def _show_context_banner() -> None:
    """Small banner showing the weather/time context the bot is using."""
    try:
        ctx = context_service.get_context()
    except Exception:
        return
    weather = ctx.get("weather")
    icon = "🌤️"
    if weather:
        cond = weather.get("condition")
        icon = {"rain": "🌧️", "snow": "❄️", "cloudy": "☁️", "clear": "☀️"}.get(cond, "🌤️")
    bits = [f"{ctx.get('time_of_day', '').title()}", ctx.get("season", "").title()]
    if weather:
        bits.append(f"{weather.get('temp_c')}°C {weather.get('condition')}")
    st.caption(f"{icon} {ctx.get('city', '')} · " + " · ".join(b for b in bits if b))


def _recent_history(limit: int = 8) -> list:
    """Return the last few turns as {role, content} for conversation memory."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["chat"][-limit:]
    ]


def _last_recommended_foods() -> list:
    """Foods from the most recent assistant message that had recommendations."""
    for msg in reversed(st.session_state["chat"]):
        if msg["role"] == "assistant" and msg.get("foods"):
            return msg["foods"]
    return []


def _handle_message(prompt: str, user: dict) -> None:
    """Run one chat turn through the LangGraph workflow and draw the answer."""
    history = _recent_history()
    last_recommended = _last_recommended_foods()

    st.session_state["chat"].append({"role": "user", "content": prompt, "foods": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = graph.run_chat(
                    user_id=user["id"],
                    session_id=st.session_state["session_id"],
                    user_message=prompt,
                    history=history,
                    last_recommended=last_recommended,
                )
            except Exception as exc:
                error_msg = f"Sorry, something went wrong: {exc}"
                st.error(error_msg)
                st.session_state["chat"].append(
                    {"role": "assistant", "content": error_msg, "foods": []}
                )
                return

        response = result.get("bot_response", "I'm not sure how to help with that.")
        foods = result.get("ranked_foods", [])
        st.markdown(response)
        for food in foods:
            render_food_card(food, user["id"], key=f"chat_{id(food)}")

    st.session_state["chat"].append(
        {"role": "assistant", "content": response, "foods": foods}
    )


