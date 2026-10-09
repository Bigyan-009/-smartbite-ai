"""User-related database operations."""

from typing import List, Optional

from database import mongo
from utils import validators
from utils.date_utils import utcnow
from utils.id_utils import stringify_doc, to_object_id


def get_user_by_email(email: str) -> Optional[dict]:
    """Return a raw user document by email (or None)."""
    email = validators.normalize_email(email)
    return mongo.users().find_one({"email": email})


def get_user_by_id(user_id: str) -> Optional[dict]:
    """Return a public user dict by id (password hash stripped)."""
    oid = to_object_id(user_id)
    if not oid:
        return None
    user = mongo.users().find_one({"_id": oid})
    if not user:
        return None
    user = stringify_doc(user)
    user.pop("password_hash", None)
    return user


def list_users() -> List[dict]:
    """Return all users as public dicts, newest first."""
    users = []
    for user in mongo.users().find().sort("created_at", -1):
        user = stringify_doc(user)
        user.pop("password_hash", None)
        users.append(user)
    return users


def update_preferences(user_id: str, preferences: dict) -> bool:
    """Update the stored preference block for a user."""
    oid = to_object_id(user_id)
    if not oid:
        return False
    clean = {
        "preferred_cuisines": preferences.get("preferred_cuisines", []) or [],
        "diet_type": validators.validate_diet_type(preferences.get("diet_type")),
        "spice_level": validators.validate_spice_level(
            preferences.get("spice_level")
        ),
        "usual_budget": preferences.get("usual_budget"),
        "disliked_ingredients": preferences.get("disliked_ingredients", []) or [],
        "allergies": preferences.get("allergies", []) or [],
    }
    result = mongo.users().update_one(
        {"_id": oid},
        {"$set": {"preferences": clean, "updated_at": utcnow()}},
    )
    return result.modified_count > 0 or result.matched_count > 0


