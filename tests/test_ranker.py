"""Unit tests for the hybrid recommendation ranker."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation import hybrid_ranker


def _food(**overrides):
    base = {
        "id": "1",
        "name": "Test Food",
        "cuisine": "Nepali",
        "diet_type": "non_veg",
        "spice_level": "medium",
        "price": 200,
        "rating_avg": 3.0,
        "recommendation_count": 0,
        "ingredients": [],
        "available": True,
        "similarity": 0.5,
    }
    base.update(overrides)
    return base


def test_food_under_budget_scores_higher():
    cheap = _food(id="cheap", price=200)
    pricey = _food(id="pricey", price=900)
    ranked = hybrid_ranker.rank_foods([pricey, cheap], {"budget": 300})
    assert ranked[0]["id"] == "cheap"
    assert ranked[0]["recommendation_score"] > ranked[1]["recommendation_score"]


def test_matching_diet_scores_higher():
    veg = _food(id="veg", diet_type="veg")
    nonveg = _food(id="nonveg", diet_type="non_veg")
    ranked = hybrid_ranker.rank_foods([nonveg, veg], {"diet_type": "veg"})
    assert ranked[0]["id"] == "veg"


def test_disliked_ingredient_is_penalised():
    with_mushroom = _food(id="mush", ingredients=["mushroom", "cheese"])
    without = _food(id="clean", ingredients=["chicken"])
    ranked = hybrid_ranker.rank_foods(
        [with_mushroom, without], {"disliked_ingredients": ["mushroom"]}
    )
    assert ranked[0]["id"] == "clean"
    mushroom_result = next(f for f in ranked if f["id"] == "mush")
    assert mushroom_result["recommendation_score"] < 0


def test_unavailable_food_is_heavily_penalised():
    available = _food(id="on", available=True)
    unavailable = _food(id="off", available=False)
    ranked = hybrid_ranker.rank_foods([unavailable, available], {})
    assert ranked[0]["id"] == "on"
    off = next(f for f in ranked if f["id"] == "off")
    assert off["recommendation_score"] <= -90


def test_every_food_gets_a_reason():
    ranked = hybrid_ranker.rank_foods([_food()], {"budget": 500, "cuisine": "Nepali"})
    assert ranked[0]["recommendation_reason"]
    assert "recommendation_score" in ranked[0]


def test_context_boosts_matching_food():
    soup = _food(id="soup", name="Hot Noodle Soup", tags=["soup", "hot"])
    ice = _food(id="ice", name="Ice Cream Sundae", tags=["cold", "dessert"])
    context = {
        "suggest_tags": ["soup", "hot", "spicy", "steamed"],
        "summary": "a cold rainy evening",
        "meal_type": "dinner",
    }
    ranked = hybrid_ranker.rank_foods([ice, soup], {}, context=context)
    assert ranked[0]["id"] == "soup"
    assert "rainy evening" in ranked[0]["recommendation_reason"]


def test_meal_time_context_boosts_matching_meal():
    breakfast = _food(id="bfast", name="Aloo Paratha", meal_type=["breakfast"])
    dinner = _food(id="dnr", name="Heavy Thali", meal_type=["dinner"])
    context = {"suggest_tags": [], "meal_type": "breakfast", "summary": "a morning"}
    ranked = hybrid_ranker.rank_foods([dinner, breakfast], {}, context=context)
    assert ranked[0]["id"] == "bfast"


def test_no_context_still_ranks():
    # Backwards compatibility: omitting context must not break ranking.
    ranked = hybrid_ranker.rank_foods([_food()], {"budget": 500})
    assert ranked and "recommendation_score" in ranked[0]


# ---------------------------------------------------------------------------
# Order-history ("revealed preference") boosts
# ---------------------------------------------------------------------------
def test_previously_ordered_food_ranks_higher():
    ordered = _food(id="fav", name="Chicken Momo")
    other = _food(id="new", name="Veg Burger")
    history = {"food_counts": {"fav": 3}, "cuisine_counts": {"nepali": 3}}
    ranked = hybrid_ranker.rank_foods([other, ordered], {}, order_history=history)
    assert ranked[0]["id"] == "fav"
    assert "ordered before" in ranked[0]["recommendation_reason"]


def test_reorder_boost_is_capped():
    heavy = _food(id="heavy")
    ranked = hybrid_ranker.rank_foods(
        [heavy], {}, order_history={"food_counts": {"heavy": 100}, "cuisine_counts": {}}
    )
    baseline = hybrid_ranker.rank_foods([_food(id="heavy")], {})
    boost = ranked[0]["recommendation_score"] - baseline[0]["recommendation_score"]
    assert boost <= hybrid_ranker.REORDER_CAP + 0.01


def test_often_ordered_cuisine_gets_small_boost():
    nepali = _food(id="nep", cuisine="Nepali")
    italian = _food(id="ita", cuisine="Italian")
    history = {"food_counts": {"other-id": 2}, "cuisine_counts": {"nepali": 2}}
    ranked = hybrid_ranker.rank_foods([italian, nepali], {}, order_history=history)
    assert ranked[0]["id"] == "nep"


def test_no_order_history_still_ranks():
    ranked = hybrid_ranker.rank_foods([_food()], {}, order_history=None)
    assert ranked and "recommendation_score" in ranked[0]


# ---------------------------------------------------------------------------
# Today's special
# ---------------------------------------------------------------------------
def test_special_food_gets_boost_and_reason():
    special = _food(id="sp", is_special=True, discount_percent=20)
    plain = _food(id="pl")
    ranked = hybrid_ranker.rank_foods([plain, special], {})
    assert ranked[0]["id"] == "sp"
    assert "special" in ranked[0]["recommendation_reason"]


def test_discounted_price_counts_for_budget():
    # Rs. 500 dish at 30% off = Rs. 350, inside a Rs. 400 budget.
    special = _food(id="sp", price=500, is_special=True, discount_percent=30)
    ranked = hybrid_ranker.rank_foods([special], {"budget": 400})
    assert "within your Rs. 400 budget" in ranked[0]["recommendation_reason"]


# ---------------------------------------------------------------------------
# Nutrition-aware ranking
# ---------------------------------------------------------------------------
def test_light_request_prefers_low_calorie_food():
    salad = _food(id="salad", calories=180, protein=4)
    biryani = _food(id="biryani", calories=720, protein=32)
    ranked = hybrid_ranker.rank_foods([biryani, salad], {"nutrition_goal": "light"})
    assert ranked[0]["id"] == "salad"
    assert "180 kcal" in ranked[0]["recommendation_reason"]


def test_high_protein_request_prefers_protein_rich_food():
    salad = _food(id="salad", calories=180, protein=4)
    sekuwa = _food(id="sekuwa", calories=520, protein=30)
    ranked = hybrid_ranker.rank_foods(
        [salad, sekuwa], {"nutrition_goal": "high_protein"}
    )
    assert ranked[0]["id"] == "sekuwa"
    assert "30g of protein" in ranked[0]["recommendation_reason"]


def test_hearty_request_prefers_filling_food():
    soup = _food(id="soup", calories=120, protein=4)
    thakali = _food(id="thakali", calories=780, protein=35)
    ranked = hybrid_ranker.rank_foods([soup, thakali], {"nutrition_goal": "hearty"})
    assert ranked[0]["id"] == "thakali"


def test_nutrition_goal_synonyms_are_understood():
    for phrase in ("low calorie", "healthy", "diet"):
        assert hybrid_ranker.normalize_nutrition_goal(phrase) == "light"
    for phrase in ("gym", "protein", "high-protein"):
        assert hybrid_ranker.normalize_nutrition_goal(phrase) == "high_protein"
    assert hybrid_ranker.normalize_nutrition_goal("filling") == "hearty"
    assert hybrid_ranker.normalize_nutrition_goal("nonsense") is None


def test_food_without_nutrition_data_is_not_punished():
    """A dish with no calories recorded must not be pushed below one that has."""
    unknown = _food(id="unknown")
    heavy = _food(id="heavy", calories=900)
    ranked = hybrid_ranker.rank_foods([heavy, unknown], {"nutrition_goal": "light"})
    assert ranked[0]["id"] == "unknown"


# ---------------------------------------------------------------------------
# The user's own ratings
# ---------------------------------------------------------------------------
def test_user_rated_food_badly_is_pushed_down():
    hated = _food(id="hated", rating_avg=4.8)
    neutral = _food(id="neutral", rating_avg=3.0)
    ranked = hybrid_ranker.rank_foods(
        [hated, neutral], {}, user_ratings={"hated": 1}
    )
    assert ranked[0]["id"] == "neutral"


def test_user_rated_food_highly_is_pulled_up():
    loved = _food(id="loved")
    other = _food(id="other")
    ranked = hybrid_ranker.rank_foods([other, loved], {}, user_ratings={"loved": 5})
    assert ranked[0]["id"] == "loved"
    assert "rated 5/5" in ranked[0]["recommendation_reason"]


def test_personal_rating_beats_menu_average():
    """A dish the whole restaurant loves but this user hates must not win."""
    popular = _food(id="popular", rating_avg=5.0, recommendation_count=100)
    plain = _food(id="plain", rating_avg=2.0)
    ranked = hybrid_ranker.rank_foods([popular, plain], {}, user_ratings={"popular": 1})
    assert ranked[0]["id"] == "plain"


def test_three_star_rating_is_neutral():
    rated = _food(id="rated")
    unrated = _food(id="unrated")
    with_three = hybrid_ranker.rank_foods([rated], {}, user_ratings={"rated": 3})
    baseline = hybrid_ranker.rank_foods([unrated], {})
    assert with_three[0]["recommendation_score"] == baseline[0]["recommendation_score"]


def test_missing_or_invalid_ratings_are_ignored():
    food = _food(id="a")
    baseline = hybrid_ranker.rank_foods([food], {})[0]["recommendation_score"]
    for bad in ({}, {"a": None}, {"a": "abc"}, {"a": 9}, None):
        ranked = hybrid_ranker.rank_foods([food], {}, user_ratings=bad)
        assert ranked[0]["recommendation_score"] == baseline
