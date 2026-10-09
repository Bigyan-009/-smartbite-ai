"""Analytics for the admin dashboard and the report's result analysis.

Every figure here is calculated by MongoDB itself, using **aggregation
pipelines**, rather than by loading whole collections into Python and looping.
A pipeline is a list of stages the database runs in order, for example::

    [{"$match": ...},   # keep only the documents we care about
     {"$group": ...},   # collapse them into totals or averages
     {"$sort":  ...},   # order the result
     {"$limit": ...}]   # and take the top few

This matters for more than tidiness: the database only sends back the finished
numbers, so the work stays where the data and the indexes are, and the app
keeps working as ``recommendation_logs`` grows into the thousands.

Plain ``find().sort().limit()`` is still used where nothing is actually being
aggregated - asking for "the five highest-rated dishes" is a sort, not a
calculation, and an index answers it directly.
"""

from typing import Any, Dict, List

from database import mongo


# ---------------------------------------------------------------------------
# Dashboard metric cards
# ---------------------------------------------------------------------------
def overview_metrics() -> Dict[str, Any]:
    """High-level counters shown as metric cards on the admin dashboard."""
    # One pass over `foods` produces the food count, the most-recommended dish
    # and the number of dishes that have been rated.
    food_stats = _first(mongo.foods().aggregate([
        {"$group": {
            "_id": None,
            "total_foods": {"$sum": 1},
            "rated_food_count": {
                "$sum": {"$cond": [{"$gt": ["$rating_avg", 0]}, 1, 0]}
            },
        }},
    ])) or {}

    most_recommended = _first(mongo.foods().aggregate([
        {"$match": {"recommendation_count": {"$gt": 0}}},
        {"$sort": {"recommendation_count": -1}},
        {"$limit": 1},
        {"$project": {"_id": 0, "name": 1}},
    ])) or {}

    popular_cuisine = _first(mongo.foods().aggregate([
        {"$group": {"_id": "$cuisine", "count": {"$sum": 1}}},
        {"$sort": {"count": -1, "_id": 1}},
        {"$limit": 1},
    ])) or {}

    # The average of every star rating ever given.
    rating_stats = _first(mongo.ratings().aggregate([
        {"$group": {"_id": None, "average": {"$avg": "$rating"}}},
    ])) or {}

    return {
        "total_users": mongo.users().count_documents({}),
        "total_foods": food_stats.get("total_foods", 0),
        "total_recommendations": mongo.recommendation_logs().count_documents({}),
        "average_rating": _round(rating_stats.get("average"), 2),
        "most_recommended_food": most_recommended.get("name", "N/A"),
        "popular_cuisine": popular_cuisine.get("_id") or "N/A",
        "rated_food_count": food_stats.get("rated_food_count", 0),
    }


# ---------------------------------------------------------------------------
# Dashboard charts
# ---------------------------------------------------------------------------
def top_recommended_foods(limit: int = 5) -> List[Dict[str, Any]]:
    """Foods with the highest recommendation counts.

    A sort plus a limit, not an aggregation - the ``recommendation_count``
    value is already stored on each food, so an index answers this directly.
    """
    cursor = (
        mongo.foods()
        .find({"recommendation_count": {"$gt": 0}},
              {"name": 1, "recommendation_count": 1})
        .sort("recommendation_count", -1)
        .limit(limit)
    )
    return [
        {"name": f["name"], "count": f.get("recommendation_count", 0)}
        for f in cursor
    ]


def highest_rated_foods(limit: int = 5) -> List[Dict[str, Any]]:
    """Best-rated foods that have at least one rating."""
    cursor = (
        mongo.foods()
        .find({"rating_count": {"$gt": 0}},
              {"name": 1, "rating_avg": 1, "rating_count": 1})
        .sort("rating_avg", -1)
        .limit(limit)
    )
    return [
        {
            "name": f["name"],
            "rating_avg": f.get("rating_avg", 0),
            "rating_count": f.get("rating_count", 0),
        }
        for f in cursor
    ]


def food_count_by_cuisine() -> List[Dict[str, Any]]:
    """How many dishes the menu has per cuisine (dashboard chart)."""
    rows = mongo.foods().aggregate([
        {"$group": {"_id": {"$ifNull": ["$cuisine", "Unknown"]},
                    "count": {"$sum": 1}}},
        {"$sort": {"count": -1, "_id": 1}},
    ])
    return [{"cuisine": row["_id"], "count": row["count"]} for row in rows]


def ratings_distribution() -> List[Dict[str, Any]]:
    """How many 1..5 star ratings have been given (dashboard chart)."""
    counts = {
        row["_id"]: row["count"]
        for row in mongo.ratings().aggregate([
            {"$group": {"_id": "$rating", "count": {"$sum": 1}}},
        ])
    }
    # Always return all five bars, even the ones nobody has used.
    return [{"rating": star, "count": counts.get(star, 0)} for star in range(1, 6)]


def user_diet_preferences() -> List[Dict[str, Any]]:
    """How many users chose each diet type (dashboard chart)."""
    rows = mongo.users().aggregate([
        {"$match": {"preferences.diet_type": {"$nin": [None, ""]}}},
        {"$group": {"_id": "$preferences.diet_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1, "_id": 1}},
    ])
    return [{"diet_type": row["_id"], "count": row["count"]} for row in rows]


# ---------------------------------------------------------------------------
# Result analysis (for the report)
# ---------------------------------------------------------------------------
def result_analysis() -> Dict[str, Any]:
    """Metrics for the report's result-analysis section.

    The whole calculation happens inside MongoDB. ``$unwind`` opens up the
    ``final_recommendations`` array so every individual recommendation becomes
    its own row, which is what makes the average score possible in one pass.
    """
    # Query totals and the success rate.
    totals = _first(mongo.recommendation_logs().aggregate([
        {"$group": {
            "_id": None,
            "total_queries": {"$sum": 1},
            "successful": {
                "$sum": {
                    "$cond": [
                        {"$gt": [{"$size": {"$ifNull": ["$final_recommendations", []]}}, 0]},
                        1, 0,
                    ]
                }
            },
        }},
    ])) or {}
    total_queries = totals.get("total_queries", 0)
    successful = totals.get("successful", 0)

    # The average score across every recommendation inside every log.
    score_stats = _first(mongo.recommendation_logs().aggregate([
        {"$unwind": "$final_recommendations"},
        {"$group": {
            "_id": None,
            "average_score": {"$avg": {"$ifNull": ["$final_recommendations.score", 0]}},
            "recommendations": {"$sum": 1},
        }},
    ])) or {}

    rating_stats = _first(mongo.ratings().aggregate([
        {"$group": {"_id": None, "average": {"$avg": "$rating"}}},
    ])) or {}

    success_rate = (
        round(successful / total_queries * 100, 1) if total_queries else 0.0
    )
    return {
        "total_queries": total_queries,
        "successful_recommendations": successful,
        "success_rate": success_rate,
        "average_recommendation_score": _round(score_stats.get("average_score"), 2),
        "average_user_rating": _round(rating_stats.get("average"), 2),
        "total_recommendations_made": score_stats.get("recommendations", 0),
        "top_recommended": top_recommended_foods(5),
    }


def response_time_stats() -> Dict[str, Any]:
    """Average and slowest chatbot response time, in seconds.

    Reads the ``response_ms`` figure saved with each recommendation log. Logs
    written before that field existed are simply skipped.
    """
    stats = _first(mongo.recommendation_logs().aggregate([
        {"$match": {"response_ms": {"$gt": 0}}},
        {"$group": {
            "_id": None,
            "average_ms": {"$avg": "$response_ms"},
            "slowest_ms": {"$max": "$response_ms"},
            "fastest_ms": {"$min": "$response_ms"},
            "measured": {"$sum": 1},
        }},
    ])) or {}
    return {
        "measured_queries": stats.get("measured", 0),
        "average_seconds": _round((stats.get("average_ms") or 0) / 1000.0, 2),
        "fastest_seconds": _round((stats.get("fastest_ms") or 0) / 1000.0, 2),
        "slowest_seconds": _round((stats.get("slowest_ms") or 0) / 1000.0, 2),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _first(cursor) -> Dict[str, Any]:
    """The first document of an aggregation result, or an empty dict."""
    for document in cursor:
        return document
    return {}


def _round(value, digits: int) -> float:
    """Round a value that may be None (an empty collection averages to None)."""
    return round(float(value), digits) if value else 0.0
