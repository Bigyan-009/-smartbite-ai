"""Database schema management: migrations, indexes and validation rules.

This module owns everything about the *shape* of the MongoDB data, kept apart
from ``mongo.py``, which only owns the connection. It runs automatically the
first time the database is opened, and is safe to run repeatedly.

Three jobs, in this order:

1. **Migrations** - forward-only, numbered repairs to existing documents. The
   version already applied is recorded in the ``meta`` collection, so each
   migration runs exactly once no matter how often the app restarts.
2. **Indexes** - including TTL indexes, which let MongoDB delete expired login
   sessions and password-reset codes by itself.
3. **Validators** - JSON Schema rules stored *in the database*, so a bad
   document is rejected even if something writes to MongoDB directly without
   going through this application.
"""

from typing import Any, Dict, List

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.errors import OperationFailure, PyMongoError

from config import settings

# Bump this when a new migration is added below.
SCHEMA_VERSION = 2

_META_ID = "schema"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def apply(db: Database) -> Dict[str, Any]:
    """Bring the database up to date. Returns a summary of what changed."""
    summary: Dict[str, Any] = {"migrations": [], "indexes": 0, "validators": []}
    try:
        summary["migrations"] = run_migrations(db)
        summary["indexes"] = ensure_indexes(db)
        summary["validators"] = ensure_validators(db)
    except PyMongoError:
        # Schema setup is best-effort: a permission-restricted user (common on
        # shared Atlas clusters) must still be able to run the app.
        pass
    return summary


# ---------------------------------------------------------------------------
# 1. Migrations
# ---------------------------------------------------------------------------
def current_version(db: Database) -> int:
    """The schema version already applied to this database."""
    doc = db.meta.find_one({"_id": _META_ID}) or {}
    return int(doc.get("version", 0))


def run_migrations(db: Database) -> List[str]:
    """Apply every migration newer than the recorded version, in order."""
    applied: List[str] = []
    version = current_version(db)
    for number, (name, migrate) in sorted(_MIGRATIONS.items()):
        if number <= version:
            continue
        migrate(db)
        db.meta.update_one(
            {"_id": _META_ID}, {"$set": {"version": number}}, upsert=True
        )
        applied.append(f"{number}: {name}")
    return applied


def _migration_001_normalise_foods(db: Database) -> None:
    """Repair food documents that were written straight into MongoDB.

    A handful of dishes were inserted without going through ``food_service``,
    so they carry a numeric ``spice_level``, an old ``is_veg`` flag instead of
    ``diet_type``, and none of the list fields. This brings them into the one
    shape the rest of the app expects.
    """
    from utils.date_utils import utcnow

    # Numeric spice levels (an old 0-4 scale) become the named levels.
    numeric_spice = {0: "mild", 1: "mild", 2: "medium", 3: "spicy", 4: "very_spicy"}
    for doc in db.foods.find(
        {"spice_level": {"$type": ["int", "double"]}}, {"spice_level": 1}
    ):
        level = numeric_spice.get(int(doc["spice_level"]), "medium")
        db.foods.update_one({"_id": doc["_id"]}, {"$set": {"spice_level": level}})

    # The legacy boolean becomes a proper diet type, then is removed.
    db.foods.update_many(
        {"is_veg": True, "diet_type": {"$exists": False}},
        {"$set": {"diet_type": "veg"}},
    )
    db.foods.update_many(
        {"is_veg": False, "diet_type": {"$exists": False}},
        {"$set": {"diet_type": "non_veg"}},
    )
    db.foods.update_many({}, {"$unset": {"is_veg": ""}})

    # Fill in the fields every food is expected to have.
    defaults = {
        "available": True,
        "ingredients": [],
        "allergens": [],
        "tags": [],
        "meal_type": [],
        "rating_avg": 0.0,
        "rating_count": 0,
        "recommendation_count": 0,
        "is_special": False,
        "discount_percent": 0,
    }
    for field, value in defaults.items():
        db.foods.update_many({field: {"$exists": False}}, {"$set": {field: value}})

    # A missing creation date falls back to the last update (or now).
    for doc in db.foods.find({"created_at": {"$exists": False}}, {"updated_at": 1}):
        db.foods.update_one(
            {"_id": doc["_id"]},
            {"$set": {"created_at": doc.get("updated_at") or utcnow()}},
        )

    # Prices are stored as numbers, never as text.
    for doc in db.foods.find({"price": {"$type": "string"}}, {"price": 1}):
        try:
            price = float(doc["price"])
        except (TypeError, ValueError):
            price = 0.0
        db.foods.update_one({"_id": doc["_id"]}, {"$set": {"price": price}})


def _migration_002_unique_ratings(db: Database) -> None:
    """Make "one rating per user per dish" a database-level guarantee.

    An earlier version created this index without ``unique``, and MongoDB will
    not change an existing index's options in place. So: remove any duplicate
    ratings first (keeping the most recent), then drop the old index. The
    unique version is recreated by ``ensure_indexes`` immediately afterwards.
    """
    duplicates = db.ratings.aggregate([
        {"$group": {
            "_id": {"user_id": "$user_id", "food_id": "$food_id"},
            "ids": {"$push": "$_id"},
            "count": {"$sum": 1},
        }},
        {"$match": {"count": {"$gt": 1}}},
    ])
    for group in duplicates:
        # Keep the newest document, drop the rest.
        keep = sorted(group["ids"])[-1]
        db.ratings.delete_many({"_id": {"$in": [i for i in group["ids"] if i != keep]}})

    for name, spec in db.ratings.index_information().items():
        same_key = spec.get("key") == [("user_id", 1), ("food_id", 1)]
        if same_key and not spec.get("unique"):
            try:
                db.ratings.drop_index(name)
            except PyMongoError:
                pass


_MIGRATIONS = {
    1: ("normalise legacy food documents", _migration_001_normalise_foods),
    2: ("enforce one rating per user per dish", _migration_002_unique_ratings),
}


# ---------------------------------------------------------------------------
# 2. Indexes
# ---------------------------------------------------------------------------
def ensure_indexes(db: Database) -> int:
    """Create every index the app relies on, and return how many were requested.

    ``create_index`` is idempotent, so this is safe to run on every start-up.
    Each entry below exists because a real query needs it - an index nothing
    queries only slows writes down.
    """
    plan = [
        # Login looks users up by email, and two accounts may never share one.
        ("users", [("email", ASCENDING)], {"unique": True}),
        ("foods", [("name", ASCENDING)], {}),
        ("foods", [("cuisine", ASCENDING)], {}),
        # The live menu is "available dishes, cheapest first".
        ("foods", [("available", ASCENDING), ("price", ASCENDING)], {}),
        # Chat replay reads one session's messages in order.
        ("chat_history", [("user_id", ASCENDING), ("session_id", ASCENDING),
                          ("created_at", ASCENDING)], {}),
        ("chat_history", [("user_id", ASCENDING), ("created_at", DESCENDING)], {}),
        # Recommendation logs drive the admin log page and the result analysis.
        ("recommendation_logs", [("created_at", DESCENDING)], {}),
        ("recommendation_logs", [("user_id", ASCENDING),
                                 ("created_at", DESCENDING)], {}),
        # One rating per user per dish.
        ("ratings", [("user_id", ASCENDING), ("food_id", ASCENDING)],
         {"unique": True}),
        ("ratings", [("food_id", ASCENDING)], {}),
        ("favourites", [("user_id", ASCENDING), ("food_id", ASCENDING)],
         {"unique": True}),
        ("orders", [("user_id", ASCENDING), ("created_at", DESCENDING)], {}),
        ("orders", [("status", ASCENDING), ("created_at", DESCENDING)], {}),
        ("sessions", [("token", ASCENDING)], {"unique": True}),
        ("password_resets", [("email", ASCENDING)], {"unique": True}),
    ]
    for collection, keys, options in plan:
        try:
            db[collection].create_index(keys, **options)
        except OperationFailure:
            # An index with the same keys but different options already exists;
            # leave whatever is there rather than fighting it.
            pass

    _ensure_ttl_indexes(db)
    return len(plan) + 2


def _ensure_ttl_indexes(db: Database) -> None:
    """Let MongoDB delete expired sessions and reset codes on its own.

    A TTL index with ``expireAfterSeconds=0`` removes a document as soon as the
    date in its ``expires_at`` field passes, so these two collections cannot
    grow forever. Any older plain index on the same field is dropped first,
    because MongoDB will not change an existing index's options in place.
    """
    for collection in ("sessions", "password_resets"):
        try:
            existing = db[collection].index_information()
        except PyMongoError:
            continue
        for name, spec in existing.items():
            same_key = spec.get("key") == [("expires_at", 1)]
            if same_key and "expireAfterSeconds" not in spec:
                try:
                    db[collection].drop_index(name)
                except PyMongoError:
                    pass
        try:
            db[collection].create_index(
                [("expires_at", ASCENDING)], expireAfterSeconds=0
            )
        except OperationFailure:
            pass


# ---------------------------------------------------------------------------
# 3. Validation rules (enforced by MongoDB itself)
# ---------------------------------------------------------------------------
# "moderate" means the rules apply to new documents and to updates of documents
# that are already valid, so a rule added later can never make an existing row
# unwritable. "error" means a bad write is refused rather than merely logged.
_NUMBER = ["double", "int", "long", "decimal"]

FOOD_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["name", "price"],
        "properties": {
            "name": {"bsonType": "string", "minLength": 1,
                     "description": "every dish must be named"},
            "price": {"bsonType": _NUMBER, "minimum": 0,
                      "description": "price can never be negative"},
            "diet_type": {"enum": list(settings.DIET_TYPES) + [None]},
            "spice_level": {"enum": list(settings.SPICE_LEVELS) + [None]},
            "available": {"bsonType": "bool"},
            "calories": {"bsonType": _NUMBER + ["null"], "minimum": 0},
            "protein": {"bsonType": _NUMBER + ["null"], "minimum": 0},
            "rating_avg": {"bsonType": _NUMBER, "minimum": 0, "maximum": 5},
            "discount_percent": {"bsonType": _NUMBER, "minimum": 0, "maximum": 90},
        },
    }
}

USER_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["email", "password_hash", "role"],
        "properties": {
            "email": {"bsonType": "string", "minLength": 3},
            "password_hash": {
                "bsonType": "string",
                "minLength": 20,
                "description": "a bcrypt hash - never a plain-text password",
            },
            "role": {"enum": list(settings.ROLES)},
            "is_active": {"bsonType": "bool"},
        },
    }
}

RATING_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "food_id", "rating"],
        "properties": {
            "rating": {"bsonType": ["int", "long"], "minimum": 1, "maximum": 5},
            "user_id": {"bsonType": "string"},
            "food_id": {"bsonType": "string"},
        },
    }
}

_VALIDATORS = {
    "foods": FOOD_VALIDATOR,
    "users": USER_VALIDATOR,
    "ratings": RATING_VALIDATOR,
}


def ensure_validators(db: Database) -> List[str]:
    """Attach the JSON Schema rules to their collections."""
    applied = []
    existing = set(db.list_collection_names())
    for name, validator in _VALIDATORS.items():
        try:
            if name in existing:
                db.command({
                    "collMod": name,
                    "validator": validator,
                    "validationLevel": "moderate",
                    "validationAction": "error",
                })
            else:
                db.create_collection(
                    name,
                    validator=validator,
                    validationLevel="moderate",
                    validationAction="error",
                )
            applied.append(name)
        except OperationFailure:
            # Older MongoDB, or a user without collMod rights: the application
            # layer still validates everything, so this is not fatal.
            pass
    return applied
