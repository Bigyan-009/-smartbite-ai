"""Tests for the database schema layer: migrations, indexes and validators.

These prove the three guarantees the schema module is supposed to give:
the legacy repair actually runs, the indexes real queries need exist, and
MongoDB itself refuses invalid documents.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo.errors import WriteError

from database import mongo, schema

pytestmark = pytest.mark.skipif(not mongo.ping(), reason="MongoDB is not available")


@pytest.fixture(scope="module")
def db():
    handle = mongo.get_db()  # opening the database applies the schema
    return handle


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------
def test_schema_version_is_recorded(db):
    assert schema.current_version(db) == schema.SCHEMA_VERSION


def test_migrations_are_not_reapplied(db):
    """Running again must be a no-op once the version is already recorded."""
    assert schema.run_migrations(db) == []


def test_legacy_food_documents_were_repaired(db):
    """No food may keep the old numeric spice level or the is_veg flag."""
    assert db.foods.count_documents({"spice_level": {"$type": ["int", "double"]}}) == 0
    assert db.foods.count_documents({"is_veg": {"$exists": True}}) == 0
    for field in ("available", "diet_type", "ingredients", "tags", "created_at"):
        assert db.foods.count_documents({field: {"$exists": False}}) == 0, field


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------
def _index_keys(db, collection):
    return [
        tuple(spec["key"]) for name, spec in db[collection].index_information().items()
    ]


def test_recommendation_logs_are_indexed(db):
    """This collection grows on every chat turn, so it must be indexed."""
    keys = _index_keys(db, "recommendation_logs")
    assert (("created_at", -1),) in keys
    assert (("user_id", 1), ("created_at", -1)) in keys


def test_chat_history_has_the_replay_index(db):
    assert (("user_id", 1), ("session_id", 1), ("created_at", 1)) in _index_keys(
        db, "chat_history"
    )


def test_expiring_collections_have_ttl_indexes(db):
    """MongoDB must delete expired sessions and reset codes by itself."""
    for collection in ("sessions", "password_resets"):
        specs = db[collection].index_information().values()
        ttl = [s for s in specs if "expireAfterSeconds" in s]
        assert ttl, f"{collection} has no TTL index"
        assert ttl[0]["key"] == [("expires_at", 1)]


def test_unique_constraints_exist(db):
    users = db.users.index_information()
    assert any(s.get("unique") and s["key"] == [("email", 1)] for s in users.values())
    ratings = db.ratings.index_information()
    assert any(
        s.get("unique") and s["key"] == [("user_id", 1), ("food_id", 1)]
        for s in ratings.values()
    )


# ---------------------------------------------------------------------------
# Validators - the database refusing bad data on its own
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "collection, document, reason",
    [
        ("foods", {"name": "Bad", "price": -500}, "negative price"),
        ("foods", {"name": "Bad", "price": 10, "diet_type": "meaty"}, "unknown diet"),
        ("foods", {"name": "Bad", "price": 10, "spice_level": 3}, "numeric spice"),
        ("foods", {"name": "", "price": 10}, "empty name"),
        ("foods", {"price": 10}, "no name at all"),
        ("ratings", {"user_id": "u", "food_id": "f", "rating": 9}, "rating above 5"),
        ("ratings", {"user_id": "u", "food_id": "f", "rating": 0}, "rating below 1"),
        ("users", {"email": "a@b.com", "password_hash": "plain", "role": "user"},
         "password that is not a hash"),
        ("users", {"email": "a@b.com", "password_hash": "x" * 60, "role": "root"},
         "unknown role"),
    ],
)
def test_database_rejects_invalid_documents(db, collection, document, reason):
    with pytest.raises(WriteError):
        db[collection].insert_one(dict(document))


def test_database_accepts_a_valid_food(db):
    doc = {
        "name": "Schema Test Dish",
        "price": 250.0,
        "diet_type": "veg",
        "spice_level": "mild",
        "available": True,
        "calories": 300.0,
        "protein": 12.0,
    }
    db.foods.insert_one(doc)
    try:
        assert db.foods.find_one({"_id": doc["_id"]}) is not None
    finally:
        db.foods.delete_one({"_id": doc["_id"]})
