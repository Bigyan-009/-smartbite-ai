"""Integration tests for the food service.

These require a running MongoDB and a working ChromaDB store. They are skipped
automatically when MongoDB is unreachable so the rest of the suite still runs.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import mongo
from services import food_service

pytestmark = pytest.mark.skipif(
    not mongo.ping(), reason="MongoDB is not available"
)


@pytest.fixture
def sample_food():
    result = food_service.add_food(
        {
            "name": "PyTest Momo",
            "cuisine": "Nepali",
            "diet_type": "non_veg",
            "spice_level": "spicy",
            "price": 199,
            "meal_type": "lunch, dinner",
            "ingredients": "chicken, flour",
            "tags": "test, momo",
            "description": "A test momo used by the automated test suite.",
        }
    )
    assert result["success"], result.get("error")
    food = result["food"]
    yield food
    # Cleanup.
    food_service.delete_food(food["id"])


def test_add_food_creates_document(sample_food):
    fetched = food_service.get_food(sample_food["id"])
    assert fetched is not None
    assert fetched["name"] == "PyTest Momo"
    assert fetched["price"] == 199


def test_update_food_changes_fields(sample_food):
    result = food_service.update_food(
        sample_food["id"], {**sample_food, "price": 250, "spice_level": "mild"}
    )
    assert result["success"]
    fetched = food_service.get_food(sample_food["id"])
    assert fetched["price"] == 250
    assert fetched["spice_level"] == "mild"


def test_deactivate_food(sample_food):
    assert food_service.set_availability(sample_food["id"], False)
    fetched = food_service.get_food(sample_food["id"])
    assert fetched["available"] is False


def test_delete_food_removes_document(sample_food):
    assert food_service.delete_food(sample_food["id"])
    assert food_service.get_food(sample_food["id"]) is None
    # Fixture cleanup calling delete again on a missing id is safe.


# ---------------------------------------------------------------------------
# Nutrition data
# ---------------------------------------------------------------------------
def test_nutrition_is_saved_and_kept_on_update():
    result = food_service.add_food(
        {
            "name": "PyTest Protein Bowl",
            "cuisine": "Continental",
            "diet_type": "veg",
            "spice_level": "mild",
            "price": 320,
            "calories": 410,
            "protein": 28,
            "description": "A test bowl with nutrition figures.",
        }
    )
    assert result["success"], result.get("error")
    food_id = result["food"]["id"]
    try:
        saved = food_service.get_food(food_id)
        assert saved["calories"] == 410
        assert saved["protein"] == 28

        # Editing the dish must not silently wipe its nutrition figures.
        food_service.update_food(
            food_id, {**saved, "price": 350, "calories": 400, "protein": 30}
        )
        updated = food_service.get_food(food_id)
        assert updated["calories"] == 400
        assert updated["protein"] == 30
    finally:
        food_service.delete_food(food_id)


def test_nutrition_is_optional_and_validated():
    assert food_service._as_nutrition("") is None
    assert food_service._as_nutrition(None) is None
    assert food_service._as_nutrition("250") == 250.0
    assert food_service._as_nutrition(-5) is None

    error = food_service._validate_food(
        {"name": "Bad", "price": 100, "calories": "many"}
    )
    assert error and "Calories" in error


def test_nutrition_appears_in_the_indexed_document():
    document = food_service.build_chroma_document(
        {"name": "Veg Soup", "calories": 120, "protein": 4}
    )
    assert "120 kcal" in document
    assert "light" in document
    assert "4g protein" in document


def test_user_ratings_returns_a_food_id_to_stars_map(sample_food):
    user_id = "6a44c5ce648425c442b3c2fc"  # any valid ObjectId string
    food_service.rate_food(user_id, sample_food["id"], 5)
    try:
        ratings = food_service.user_ratings(user_id)
        assert ratings.get(sample_food["id"]) == 5
    finally:
        mongo.ratings().delete_many({"user_id": user_id, "food_id": sample_food["id"]})


def test_user_ratings_handles_invalid_user_id():
    assert food_service.user_ratings("not-an-object-id") == {}


# ---------------------------------------------------------------------------
# Food photos
# ---------------------------------------------------------------------------
def test_image_filename_is_sanitised():
    assert food_service._as_image_name("chicken-momo.jpg") == "chicken-momo.jpg"
    assert food_service._as_image_name("sub/dir/pic.png") == "pic.png"
    # Path traversal and non-image files must never be accepted.
    assert food_service._as_image_name("../../.env") is None
    assert food_service._as_image_name("C:/Windows/system32/evil.exe") is None
    assert food_service._as_image_name("") is None
    assert food_service._as_image_name(None) is None


def test_image_path_only_returns_files_that_exist():
    assert food_service.image_path({"image": "does-not-exist.jpg"}) is None
    assert food_service.image_path({}) is None
    real = food_service.image_path({"image": "chicken-momo.jpg"})
    assert real is not None and real.endswith("chicken-momo.jpg")


def test_every_seeded_dish_has_a_bundled_photo():
    """The demo must never show a card with a missing image."""
    from database import seed_data

    missing = [name for (name, *_) in seed_data.FOODS
               if seed_data._image_for(name) is None]
    assert missing == [], f"dishes without a photo: {missing}"


def test_image_survives_an_edit(sample_food):
    food_service.update_food(sample_food["id"], {**sample_food,
                                                 "image": "chicken-momo.jpg"})
    assert food_service.get_food(sample_food["id"])["image"] == "chicken-momo.jpg"


def test_image_falls_back_to_the_dish_name():
    """A cached food document with no image field still finds its photo."""
    assert food_service.image_path({"name": "Chicken Momo"}) is not None
    assert food_service.image_path({"name": "No Such Dish"}) is None


# ---------------------------------------------------------------------------
# Vector index (ChromaDB)
# ---------------------------------------------------------------------------
def test_semantic_search_skips_unavailable_dishes(sample_food):
    """An unavailable dish must not use up one of the candidate slots."""
    from database import chroma

    food_id = sample_food["id"]
    food_service.set_availability(food_id, True)
    hits = chroma.search_foods("PyTest Momo spicy chicken momo", n_results=20)
    assert any(h["food_id"] == food_id for h in hits), "the dish should be findable"

    food_service.set_availability(food_id, False)
    hits = chroma.search_foods("PyTest Momo spicy chicken momo", n_results=20)
    assert all(h["food_id"] != food_id for h in hits), "unavailable dish still returned"

    # ...but it is still there when the filter is switched off.
    hits = chroma.search_foods(
        "PyTest Momo spicy chicken momo", n_results=20, available_only=False
    )
    assert any(h["food_id"] == food_id for h in hits)


def test_rebuild_index_matches_the_food_count():
    """After a rebuild, MongoDB and ChromaDB must hold the same number of foods."""
    from database import chroma

    indexed = food_service.rebuild_index()
    assert indexed == mongo.foods().count_documents({})
    assert chroma.index_stats()["foods"] == indexed


def test_bulk_upsert_returns_the_number_written():
    from database import chroma

    assert chroma.upsert_foods([]) == 0
