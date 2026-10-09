"""Helpers for working with MongoDB ObjectIds and session ids."""

import uuid
from typing import Optional

from bson import ObjectId


def new_session_id() -> str:
    """Generate a fresh chat session id."""
    return str(uuid.uuid4())


def to_object_id(value) -> Optional[ObjectId]:
    """Convert a string to an ObjectId, returning None when invalid."""
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def id_to_str(value) -> str:
    """Convert an ObjectId (or anything) to its string form."""
    if value is None:
        return ""
    return str(value)


def stringify_doc(doc: Optional[dict]) -> Optional[dict]:
    """Return a copy of a Mongo document with its ``_id`` stringified.

    Also exposes the id under the friendlier ``id`` key for UI code.
    """
    if not doc:
        return doc
    result = dict(doc)
    if "_id" in result:
        result["_id"] = str(result["_id"])
        result["id"] = result["_id"]
    return result
