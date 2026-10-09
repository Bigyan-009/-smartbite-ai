"""User favourites page."""

import streamlit as st

from auth import auth_service
from services import food_service


def render() -> None:
    """The dishes this user has starred."""
    user = auth_service.get_current_user()
    st.title("⭐ My Favourites")

    favourites = food_service.list_favourites(user["id"])
    if not favourites:
        st.info("You have no favourites yet. Save foods from the chatbot page!")
        return

    for food in favourites:
        food_id = food.get("id") or food.get("_id")
        with st.container(border=True):
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**{food.get('name')}** — Rs. {int(food.get('price', 0))}")
                st.caption(
                    f"{food.get('cuisine', '')} · "
                    f"{_pretty(food.get('diet_type'))} · "
                    f"{_pretty(food.get('spice_level'))} · ⭐ {food.get('rating_avg', 0)}"
                )
                if food.get("description"):
                    st.write(food["description"])
            with cols[1]:
                if st.button("Remove", key=f"rmfav_{food_id}"):
                    food_service.remove_favourite(user["id"], food_id)
                    st.rerun()


def _pretty(value) -> str:
    return str(value).replace("_", " ").title() if value else ""
