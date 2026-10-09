"""Hybrid recommendation ranking.

Combines ChromaDB vector similarity with structured signals (cuisine, budget,
diet, spice, rating, popularity, nutrition), long-term memory matches, the
user's stored profile, their real order history (revealed preferences) and the
ratings they personally gave. Disliked ingredients, dishes the user rated badly
and unavailable foods are penalised. Every ranked food carries a human-readable
reason.
"""

from typing import Any, Dict, List, Optional

# Scoring weights (kept as named constants so they are easy to explain in a
# viva / report).
VECTOR_WEIGHT = 5.0
CUISINE_SCORE = 3.0
BUDGET_SCORE = 2.0
BUDGET_NEAR_SCORE = 0.5
DIET_SCORE = 3.0
SPICE_SCORE = 2.0
MEMORY_SCORE = 2.0
PROFILE_CUISINE_SCORE = 1.0
POPULARITY_CAP = 2.0
DISLIKE_PENALTY = 10.0
UNAVAILABLE_PENALTY = 100.0
# Weather/season/time-aware bonuses.
CONTEXT_TAG_SCORE = 1.5
CONTEXT_CAP = 3.0
MEAL_TIME_SCORE = 1.0
# An explicitly requested dish should outrank ambient (weather/time) nudges.
EXPLICIT_DISH_SCORE = 6.0
# Order-history ("revealed preference") bonuses: dishes and cuisines the user
# has actually ordered before get a nudge, scaled by how often, then capped so
# past behaviour never drowns out what the user is asking for right now.
REORDER_SCORE_PER_ORDER = 1.0
REORDER_CAP = 2.5
ORDER_CUISINE_SCORE = 1.0
# Today's special: a small nudge so discounted dishes surface naturally.
SPECIAL_SCORE = 1.5
# Nutrition-aware scoring. The customer can ask for something light, something
# filling, or something high in protein; foods carry ``calories``/``protein``.
LIGHT_CALORIE_MAX = 400.0
HEARTY_CALORIE_MIN = 600.0
HIGH_PROTEIN_MIN = 20.0
NUTRITION_SCORE = 2.5
NUTRITION_NEAR_SCORE = 1.0
NUTRITION_MISMATCH_PENALTY = 2.0
# Personal ratings ("this user's own stars"), which matter far more than the
# menu-wide average: a dish they disliked is pushed down, one they loved is
# pulled up. 3 stars is neutral.
PERSONAL_RATING_SCORES = {5: 3.0, 4: 1.5, 3: 0.0, 2: -4.0, 1: -8.0}


def rank_foods(
    foods: List[Dict[str, Any]],
    extracted_preferences: Dict[str, Any],
    user_profile: Optional[Dict[str, Any]] = None,
    retrieved_memories: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    order_history: Optional[Dict[str, Any]] = None,
    user_ratings: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Return the foods sorted by descending recommendation score.

    Each returned food gains two extra keys:
      * ``recommendation_score``  - the numeric final score
      * ``recommendation_reason`` - a friendly explanation string
    Also mirrored to ``score``/``reason`` for logging convenience.

    ``order_history`` is ``order_service.order_history_signals()`` output:
    ``{"food_counts": {...}, "cuisine_counts": {...}}``.

    ``user_ratings`` is ``food_service.user_ratings()`` output: a
    ``{food_id: stars}`` map of the ratings this user gave themselves.
    """
    prefs = extracted_preferences or {}
    profile = user_profile or {}
    memories = retrieved_memories or []
    context = context or {}
    history = order_history or {}
    ratings_given = {str(k): v for k, v in (user_ratings or {}).items()}
    ordered_food_counts = history.get("food_counts") or {}
    ordered_cuisines = _lower_set(list((history.get("cuisine_counts") or {}).keys()))
    prefs_block = profile.get("preferences") or {}

    context_tags = _lower_set(context.get("suggest_tags", []))
    context_meal = _norm(context.get("meal_type"))
    context_summary = context.get("summary")

    budget = _to_number(prefs.get("budget")) or _to_number(
        prefs_block.get("usual_budget")
    )
    cuisine = _norm(prefs.get("cuisine"))
    diet = _norm(prefs.get("diet_type"))
    spice = _norm(prefs.get("spice_level"))
    specific_dish = _norm(prefs.get("specific_dish"))
    nutrition_goal = normalize_nutrition_goal(prefs.get("nutrition_goal"))
    disliked = _lower_set(prefs.get("disliked_ingredients", []))
    disliked |= _lower_set(prefs_block.get("disliked_ingredients", []))
    memory_text = " ".join(memories).lower()

    ranked: List[Dict[str, Any]] = []
    for food in foods:
        food = dict(food)  # never mutate the caller's document
        food_id = str(food.get("id") or food.get("_id") or "")
        score = 0.0
        reasons: List[str] = []

        similarity = float(food.get("similarity", 0.0) or 0.0)
        score += similarity * VECTOR_WEIGHT
        if similarity >= 0.5:
            reasons.append("closely matches what you asked for")

        # Explicitly requested dish wins over ambient nudges.
        if specific_dish and specific_dish in _food_haystack(food):
            score += EXPLICIT_DISH_SCORE
            reasons.append(f"is the {specific_dish} you asked for")

        food_cuisine = _norm(food.get("cuisine"))
        if cuisine and food_cuisine == cuisine:
            score += CUISINE_SCORE
            reasons.append(f"is {food.get('cuisine')} cuisine")

        # Budget checks use the price the customer actually pays today
        # (specials may be discounted).
        price = _effective_price(food)
        if budget:
            if price <= budget:
                score += BUDGET_SCORE
                reasons.append(f"is within your Rs. {int(budget)} budget")
            elif price <= budget * 1.2:
                score += BUDGET_NEAR_SCORE
                reasons.append("is just slightly above your budget")

        food_diet = _norm(food.get("diet_type"))
        if diet and food_diet == diet:
            score += DIET_SCORE
            reasons.append(f"is {food.get('diet_type')}")

        food_spice = _norm(food.get("spice_level"))
        if spice and food_spice == spice:
            score += SPICE_SCORE
            reasons.append(f"matches your {food.get('spice_level')} spice preference")

        # Nutrition goal: light / hearty / high-protein, using the dish's
        # calorie and protein figures.
        if nutrition_goal:
            score += _nutrition_score(food, nutrition_goal, reasons)

        rating = _to_number(food.get("rating_avg")) or 0
        score += rating
        if rating >= 4.0:
            reasons.append(f"is highly rated ({rating})")

        # This user's OWN stars outweigh the menu-wide average: a dish they
        # rated 1-2 is pushed down hard, one they rated 4-5 is pulled up.
        own_stars = _stars(ratings_given.get(food_id))
        if own_stars:
            score += PERSONAL_RATING_SCORES.get(own_stars, 0.0)
            if own_stars >= 4:
                reasons.append(f"is one you rated {own_stars}/5")

        # Profile cuisine preference (from stored account preferences).
        preferred_cuisines = _lower_set(prefs_block.get("preferred_cuisines", []))
        if preferred_cuisines and food_cuisine in preferred_cuisines:
            score += PROFILE_CUISINE_SCORE

        # Long-term memory match: name/cuisine/tags appear in remembered text.
        if memory_text and _matches_memory(food, memory_text):
            score += MEMORY_SCORE
            reasons.append("fits what you usually like")

        # Revealed preference: the user has actually ordered this before.
        times_ordered = int(ordered_food_counts.get(food_id, 0) or 0)
        if times_ordered:
            score += min(times_ordered * REORDER_SCORE_PER_ORDER, REORDER_CAP)
            reasons.append("is something you've ordered before")
        elif ordered_cuisines and food_cuisine in ordered_cuisines:
            score += ORDER_CUISINE_SCORE
            reasons.append(f"is {food.get('cuisine')} — a cuisine you often order")

        rec_count = _to_number(food.get("recommendation_count")) or 0
        score += min(rec_count / 10.0, POPULARITY_CAP)

        # Today's special gets a small boost (and a great reason to mention).
        if food.get("is_special"):
            score += SPECIAL_SCORE
            discount = _to_number(food.get("discount_percent")) or 0
            if discount > 0:
                reasons.append(f"is today's special at {int(discount)}% off")
            else:
                reasons.append("is today's special")

        # Weather/season/time context: reward foods that suit the moment.
        if context_tags:
            haystack = _food_haystack(food)
            hits = {t for t in context_tags if t in haystack}
            if hits:
                score += min(len(hits) * CONTEXT_TAG_SCORE, CONTEXT_CAP)
                if context_summary:
                    reasons.append(f"is perfect for {context_summary}")
        if context_meal and context_meal in _meal_types(food):
            score += MEAL_TIME_SCORE
            reasons.append(f"suits {context_meal}")

        # Penalties.
        ingredients = _lower_set(food.get("ingredients", []))
        if disliked and (ingredients & disliked):
            score -= DISLIKE_PENALTY
        if not food.get("available", True):
            score -= UNAVAILABLE_PENALTY

        food["recommendation_score"] = round(score, 2)
        food["score"] = food["recommendation_score"]
        food["recommendation_reason"] = _build_reason(reasons)
        food["reason"] = food["recommendation_reason"]
        ranked.append(food)

    ranked.sort(key=lambda f: f["recommendation_score"], reverse=True)
    return ranked


# Words a customer may use for each nutrition goal, mapped to the canonical
# goal name the ranker scores on.
_NUTRITION_GOALS = {
    "light": "light",
    "lighter": "light",
    "low_calorie": "light",
    "low-calorie": "light",
    "healthy": "light",
    "diet": "light",
    "high_protein": "high_protein",
    "high-protein": "high_protein",
    "protein": "high_protein",
    "gym": "high_protein",
    "hearty": "hearty",
    "heavy": "hearty",
    "filling": "hearty",
}


def normalize_nutrition_goal(value) -> Optional[str]:
    """Normalise a requested nutrition goal to light / high_protein / hearty."""
    goal = _norm(value)
    if not goal:
        return None
    return _NUTRITION_GOALS.get(goal.replace(" ", "_"))


def _nutrition_score(
    food: Dict[str, Any], goal: str, reasons: List[str]
) -> float:
    """Score one food against the customer's nutrition goal.

    Appends a human-readable reason when the dish clearly fits. Foods with no
    nutrition data recorded simply score 0 here - they are never punished for
    missing information.
    """
    calories = _to_number(food.get("calories"))
    protein = _to_number(food.get("protein"))

    if goal == "light":
        if calories is None:
            return 0.0
        if calories <= LIGHT_CALORIE_MAX:
            reasons.append(f"is light at {int(calories)} kcal")
            return NUTRITION_SCORE
        if calories <= LIGHT_CALORIE_MAX * 1.25:
            return NUTRITION_NEAR_SCORE
        if calories >= HEARTY_CALORIE_MIN:
            return -NUTRITION_MISMATCH_PENALTY
        return 0.0

    if goal == "hearty":
        if calories is None:
            return 0.0
        if calories >= HEARTY_CALORIE_MIN:
            reasons.append(f"is filling at {int(calories)} kcal")
            return NUTRITION_SCORE
        if calories >= HEARTY_CALORIE_MIN * 0.75:
            return NUTRITION_NEAR_SCORE
        if calories <= LIGHT_CALORIE_MAX:
            return -NUTRITION_MISMATCH_PENALTY
        return 0.0

    if goal == "high_protein":
        if protein is None:
            return 0.0
        if protein >= HIGH_PROTEIN_MIN:
            reasons.append(f"packs {int(protein)}g of protein")
            return NUTRITION_SCORE
        if protein >= HIGH_PROTEIN_MIN * 0.6:
            return NUTRITION_NEAR_SCORE
        return -NUTRITION_MISMATCH_PENALTY

    return 0.0


def _stars(value) -> Optional[int]:
    """Coerce a stored rating to an int in 1..5, or None."""
    number = _to_number(value)
    if number is None:
        return None
    stars = int(round(number))
    return stars if 1 <= stars <= 5 else None


def _effective_price(food: Dict[str, Any]) -> float:
    """Price after any today's-special discount (mirrors food_service)."""
    price = _to_number(food.get("price")) or 0.0
    if food.get("is_special"):
        discount = _to_number(food.get("discount_percent")) or 0.0
        if 0 < discount <= 90:
            return round(price * (1 - discount / 100.0), 2)
    return price


def _build_reason(reasons: List[str]) -> str:
    """Join the bonuses that fired into one readable sentence."""
    if not reasons:
        return "It is a good general match for your request."
    if len(reasons) == 1:
        return f"It {reasons[0]}."
    head = ", ".join(reasons[:-1])
    return f"It {head} and {reasons[-1]}."


def _food_haystack(food: Dict[str, Any]) -> str:
    """Lower-cased text blob of a food's searchable attributes."""
    parts = [
        str(food.get("name", "")),
        str(food.get("category", "")),
        str(food.get("description", "")),
    ]
    parts += [str(t) for t in food.get("tags", []) or []]
    parts += [str(m) for m in _meal_types(food)]
    return " ".join(parts).lower()


def _meal_types(food: Dict[str, Any]) -> List[str]:
    """A food's meal types, always as a list."""
    meal = food.get("meal_type", [])
    if isinstance(meal, str):
        return [meal]
    return list(meal or [])


def _matches_memory(food: Dict[str, Any], memory_text: str) -> bool:
    """True when the dish's name, cuisine or tags appear in a remembered preference."""
    tokens = [str(food.get("name", "")), str(food.get("cuisine", ""))]
    tokens += [str(t) for t in food.get("tags", []) or []]
    return any(tok and tok.lower() in memory_text for tok in tokens)


def _norm(value) -> Optional[str]:
    """Lower-case, trimmed text, or None when empty."""
    if value is None:
        return None
    return str(value).strip().lower() or None


def _lower_set(values) -> set:
    """Normalise a value or list into a set of lower-case strings."""
    if not values:
        return set()
    if isinstance(values, str):
        values = [values]
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _to_number(value):
    """Best-effort float conversion; returns None when the value is not a number."""
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
