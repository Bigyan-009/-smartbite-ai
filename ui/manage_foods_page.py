"""Admin food management page (add / edit / deactivate / delete / reindex)."""

import streamlit as st

from config import settings
from database import chroma
from services import food_service


def render() -> None:
    """Admin CRUD screen for the food catalogue."""
    st.title("🍲 Manage Foods")

    tab_list, tab_add = st.tabs(["Food catalogue", "Add new food"])

    with tab_add:
        _render_add_form()

    with tab_list:
        _render_catalogue()


def _render_add_form(existing: dict = None) -> None:
    """The add/edit food form; pass an existing food to edit it."""
    editing = existing is not None
    prefix = "edit" if editing else "add"
    st.subheader("Edit food" if editing else "Add a new food item")

    with st.form(f"{prefix}_food_form"):
        name = st.text_input("Name", value=existing.get("name", "") if editing else "")
        col1, col2 = st.columns(2)
        with col1:
            category = st.text_input(
                "Category", value=existing.get("category", "") if editing else ""
            )
            cuisine = st.text_input(
                "Cuisine", value=existing.get("cuisine", "") if editing else ""
            )
            diet = st.selectbox(
                "Diet type",
                settings.DIET_TYPES,
                index=_idx(settings.DIET_TYPES, existing.get("diet_type") if editing else None),
                format_func=_pretty,
            )
        with col2:
            spice = st.selectbox(
                "Spice level",
                settings.SPICE_LEVELS,
                index=_idx(settings.SPICE_LEVELS, existing.get("spice_level") if editing else None),
                format_func=_pretty,
            )
            price = st.number_input(
                "Price (Rs.)",
                min_value=0,
                max_value=100000,
                value=int(existing.get("price", 0)) if editing else 0,
                step=10,
            )
            available = st.checkbox(
                "Available", value=existing.get("available", True) if editing else True
            )
            is_special = st.checkbox(
                "🌟 Today's special",
                value=existing.get("is_special", False) if editing else False,
                help="Specials get a discount and are promoted by the chatbot.",
            )
            discount = st.number_input(
                "Discount % (when special)",
                min_value=0,
                max_value=90,
                value=int(existing.get("discount_percent", 0)) if editing else 0,
                step=5,
            )

        current_photo = food_service.image_path(existing) if editing else None
        if current_photo:
            st.image(current_photo, width=180, caption="Current photo")
        photo = st.file_uploader(
            "Photo (jpg / png)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Shown on the food cards. Leave empty to keep the current photo.",
        )

        nut1, nut2 = st.columns(2)
        with nut1:
            calories = st.number_input(
                "Calories (kcal)",
                min_value=0,
                max_value=5000,
                value=int(existing.get("calories") or 0) if editing else 0,
                step=10,
                help="Used for 'something light' / 'filling' recommendations. "
                "Leave 0 if unknown.",
            )
        with nut2:
            protein = st.number_input(
                "Protein (g)",
                min_value=0,
                max_value=500,
                value=int(existing.get("protein") or 0) if editing else 0,
                step=1,
                help="Used for high-protein recommendations. Leave 0 if unknown.",
            )

        meal = st.text_input(
            "Meal types (comma separated)",
            value=", ".join(existing.get("meal_type", [])) if editing else "lunch, dinner",
        )
        ingredients = st.text_input(
            "Ingredients (comma separated)",
            value=", ".join(existing.get("ingredients", [])) if editing else "",
        )
        allergens = st.text_input(
            "Allergens (comma separated)",
            value=", ".join(existing.get("allergens", [])) if editing else "",
        )
        tags = st.text_input(
            "Tags (comma separated)",
            value=", ".join(existing.get("tags", [])) if editing else "",
        )
        description = st.text_area(
            "Description", value=existing.get("description", "") if editing else ""
        )

        submitted = st.form_submit_button("Save food" if editing else "Add food")

    if submitted:
        data = {
            "name": name,
            "category": category,
            "cuisine": cuisine,
            "diet_type": diet,
            "spice_level": spice,
            "price": price,
            "available": available,
            "is_special": is_special,
            "discount_percent": discount,
            # A new upload replaces the photo; otherwise the current one stays.
            "image": food_service.save_image(photo, name)
            or (existing.get("image") if editing else None),
            # 0 means "not recorded" rather than a genuine zero-calorie dish.
            "calories": calories or None,
            "protein": protein or None,
            "meal_type": meal,
            "ingredients": ingredients,
            "allergens": allergens,
            "tags": tags,
            "description": description,
        }
        if editing:
            result = food_service.update_food(existing["id"], data)
        else:
            result = food_service.add_food(data)

        if result.get("success"):
            st.success(f"'{name}' saved to MongoDB and indexed in ChromaDB.")
            if editing:
                st.session_state.pop("editing_food", None)
            st.rerun()
        else:
            st.error(result.get("error", "Could not save food."))


def _render_catalogue() -> None:
    """The list of foods with edit, availability and delete controls."""
    # If an item is being edited, show the edit form at the top.
    editing_id = st.session_state.get("editing_food")
    if editing_id:
        food = food_service.get_food(editing_id)
        if food:
            _render_add_form(existing=food)
            if st.button("Cancel edit"):
                st.session_state.pop("editing_food", None)
                st.rerun()
            st.divider()

    foods = food_service.list_foods(include_unavailable=True)

    # Vector-index health: MongoDB is the source of truth, ChromaDB is a copy
    # kept in sync on every write. If the two counts drift apart, a rebuild
    # puts them back in step.
    stats = chroma.index_stats()
    col_a, col_b = st.columns([3, 1])
    with col_a:
        in_sync = stats["foods"] == len(foods)
        st.caption(
            f"{len(foods)} foods in MongoDB · {stats['foods']} indexed in ChromaDB "
            + ("✅ in sync" if in_sync else "⚠️ out of sync — rebuild the index")
        )
    with col_b:
        if st.button("🔄 Rebuild ChromaDB index", width="stretch"):
            count = food_service.rebuild_index()
            st.success(f"Re-indexed {count} foods into ChromaDB.")
            st.rerun()

    for food in foods:
        food_id = food["id"]
        with st.container(border=True):
            cols = st.columns([4, 1, 1, 1])
            with cols[0]:
                status = "" if food.get("available", True) else " · 🚫 inactive"
                if food.get("is_special"):
                    now_price = food_service.effective_price(food)
                    discount = int(food.get("discount_percent", 0) or 0)
                    special = (
                        f" · 🌟 special (-{discount}%: Rs. {int(now_price)})"
                        if discount
                        else " · 🌟 special"
                    )
                else:
                    special = ""
                st.markdown(
                    f"**{food.get('name')}** — Rs. {int(food.get('price', 0))}"
                    f"{special}{status}"
                )
                st.caption(
                    f"{food.get('cuisine', '')} · {_pretty(food.get('diet_type'))} · "
                    f"{_pretty(food.get('spice_level'))} · ⭐ {food.get('rating_avg', 0)}"
                )
            with cols[1]:
                if st.button("Edit", key=f"edit_{food_id}"):
                    st.session_state["editing_food"] = food_id
                    st.rerun()
            with cols[2]:
                toggle = "Deactivate" if food.get("available", True) else "Activate"
                if st.button(toggle, key=f"toggle_{food_id}"):
                    food_service.set_availability(
                        food_id, not food.get("available", True)
                    )
                    st.rerun()
            with cols[3]:
                if st.button("Delete", key=f"del_{food_id}"):
                    food_service.delete_food(food_id)
                    st.rerun()


def _pretty(value) -> str:
    return str(value).replace("_", " ").title() if value else ""


def _idx(options, value):
    """Index of a value in a select box's options, defaulting to the first."""
    try:
        return list(options).index(value)
    except (ValueError, TypeError):
        return 0
