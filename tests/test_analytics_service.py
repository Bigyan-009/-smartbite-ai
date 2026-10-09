"""Tests for the analytics aggregations.

The figures are produced by MongoDB aggregation pipelines, so these tests
insert a known set of documents and check the pipeline arithmetic against
hand-calculated answers.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import mongo
from services import analytics_service
from utils.date_utils import utcnow

pytestmark = pytest.mark.skipif(not mongo.ping(), reason="MongoDB is not available")

MARKER = "pytest_analytics"


@pytest.fixture
def logs():
    """Three recommendation logs: two successful, one that found nothing."""
    docs = [
        {"user_id": MARKER, "query": "spicy", "response_ms": 1000,
         "final_recommendations": [{"food_id": "a", "score": 10.0},
                                   {"food_id": "b", "score": 20.0}],
         "created_at": utcnow()},
        {"user_id": MARKER, "query": "veg", "response_ms": 3000,
         "final_recommendations": [{"food_id": "c", "score": 30.0}],
         "created_at": utcnow()},
        {"user_id": MARKER, "query": "nothing", "response_ms": 2000,
         "final_recommendations": [], "created_at": utcnow()},
    ]
    mongo.recommendation_logs().delete_many({"user_id": MARKER})
    before = analytics_service.result_analysis()
    mongo.recommendation_logs().insert_many(docs)
    yield before
    mongo.recommendation_logs().delete_many({"user_id": MARKER})


def test_query_totals_and_success_rate(logs):
    before = logs
    after = analytics_service.result_analysis()
    assert after["total_queries"] == before["total_queries"] + 3
    assert after["successful_recommendations"] == before["successful_recommendations"] + 2
    # The rate is a percentage of all queries, rounded to one decimal.
    expected = round(
        after["successful_recommendations"] / after["total_queries"] * 100, 1
    )
    assert after["success_rate"] == expected


def test_unwind_counts_every_individual_recommendation(logs):
    before = logs
    after = analytics_service.result_analysis()
    # Three recommendations were added across the two successful logs.
    assert after["total_recommendations_made"] == (
        before.get("total_recommendations_made", 0) + 3
    )


def test_average_score_is_computed_by_the_database():
    """With only our own logs present, the average must be exactly 20.0."""
    mongo.recommendation_logs().delete_many({"user_id": MARKER})
    saved = list(mongo.recommendation_logs().find({}))
    mongo.recommendation_logs().delete_many({})
    try:
        mongo.recommendation_logs().insert_many([
            {"user_id": MARKER, "final_recommendations": [{"score": 10.0}, {"score": 20.0}],
             "created_at": utcnow()},
            {"user_id": MARKER, "final_recommendations": [{"score": 30.0}],
             "created_at": utcnow()},
        ])
        result = analytics_service.result_analysis()
        assert result["average_recommendation_score"] == 20.0   # (10+20+30)/3
        assert result["total_queries"] == 2
        assert result["success_rate"] == 100.0
    finally:
        mongo.recommendation_logs().delete_many({})
        if saved:
            mongo.recommendation_logs().insert_many(saved)


def test_response_time_stats_read_the_recorded_timings(logs):
    stats = analytics_service.response_time_stats()
    assert stats["measured_queries"] >= 3
    assert stats["fastest_seconds"] <= stats["average_seconds"] <= stats["slowest_seconds"]


def test_empty_collections_do_not_crash_the_pipelines():
    """An average over nothing is None in MongoDB - it must come back as 0.0."""
    saved = list(mongo.recommendation_logs().find({}))
    mongo.recommendation_logs().delete_many({})
    try:
        result = analytics_service.result_analysis()
        assert result["total_queries"] == 0
        assert result["success_rate"] == 0.0
        assert result["average_recommendation_score"] == 0.0
        assert analytics_service.response_time_stats()["measured_queries"] == 0
    finally:
        if saved:
            mongo.recommendation_logs().insert_many(saved)


def test_ratings_distribution_always_returns_five_bars():
    rows = analytics_service.ratings_distribution()
    assert [row["rating"] for row in rows] == [1, 2, 3, 4, 5]
    assert all(isinstance(row["count"], int) for row in rows)


def test_cuisine_counts_add_up_to_the_menu_size():
    rows = analytics_service.food_count_by_cuisine()
    assert sum(row["count"] for row in rows) == mongo.foods().count_documents({})
    # Sorted by count, biggest first.
    counts = [row["count"] for row in rows]
    assert counts == sorted(counts, reverse=True)


def test_overview_metrics_have_every_dashboard_field():
    metrics = analytics_service.overview_metrics()
    for field in ("total_users", "total_foods", "total_recommendations",
                  "average_rating", "most_recommended_food", "popular_cuisine",
                  "rated_food_count"):
        assert field in metrics, field
    assert metrics["total_foods"] == mongo.foods().count_documents({})
