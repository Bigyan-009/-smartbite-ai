"""MongoDB connection module.

Provides a single lazily-initialised client and convenient accessors for the
collections used by SmartBite AI. Connection errors are surfaced with a clear
message so the UI can show a friendly warning instead of crashing.
"""

from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from config import settings

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


class DatabaseError(Exception):
    """Raised when MongoDB cannot be reached or a query fails."""


def get_client() -> MongoClient:
    """Return a shared MongoClient, creating and verifying it on first use."""
    global _client
    if _client is None:
        try:
            _client = MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                appname="smartbite_ai",
            )
            # Force a round-trip so misconfiguration fails fast and clearly.
            _client.admin.command("ping")
        except PyMongoError as exc:  # pragma: no cover - infra dependent
            _client = None
            raise DatabaseError(
                "Could not connect to MongoDB at "
                f"'{settings.MONGODB_URI}'. Make sure MongoDB is running. "
                f"Original error: {exc}"
            ) from exc
    return _client


def get_db() -> Database:
    """Return the SmartBite database handle."""
    global _db
    if _db is None:
        _db = get_client()[settings.MONGODB_DB_NAME]
        # Migrations, indexes and validation rules live in database/schema.py.
        # They are applied once, the first time the database is opened.
        from database import schema

        schema.apply(_db)
    return _db


# ---- Collection accessors -------------------------------------------------
def users() -> Collection:
    return get_db().users


def foods() -> Collection:
    return get_db().foods


def chat_history() -> Collection:
    return get_db().chat_history


def recommendation_logs() -> Collection:
    return get_db().recommendation_logs


def ratings() -> Collection:
    return get_db().ratings


def favourites() -> Collection:
    return get_db().favourites


def orders() -> Collection:
    return get_db().orders


def sessions() -> Collection:
    return get_db().sessions


def password_resets() -> Collection:
    return get_db().password_resets


def meta() -> Collection:
    """Bookkeeping collection - currently just the applied schema version."""
    return get_db().meta


def ping() -> bool:
    """Lightweight health check used by the UI."""
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False
