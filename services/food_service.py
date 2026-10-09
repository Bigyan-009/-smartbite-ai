"""Food catalogue service.

Owns all reads/writes to the MongoDB ``foods`` collection and keeps the
ChromaDB ``food_items`` vector index in sync on every create/update/delete.
"""

import os
import re
from typing import Any, Dict, List, Optional

from config import settings
from database import chroma, mongo
from utils import validators
from utils.date_utils import utcnow
from utils.id_utils import id_to_str, stringify_doc, to_object_id


def build_chroma_document(food: Dict[str, Any]) -> str:
    """Build the natural-language text that gets embedded for semantic search."""
    meal = food.get("meal_type", [])
    if isinstance(meal, list):
        meal = ", ".join(meal)
    ingredients = ", ".join(food.get("ingredients", []) or [])
    tags = ", ".join(food.get("tags", []) or [])
    return (
        f"Food Name: {food.get('name', '')}.\n"
        f"Cuisine: {food.get('cuisine', '')}.\n"
        f"Diet Type: {food.get('diet_type', '')}.\n"
        f"Spice Level: {food.get('spice_level', '')}.\n"
        f"Meal Type: {meal}.\n"
        f"Price: Rs. {food.get('price', 0)}.\n"
        f"Ingredients: {ingredients}.\n"
        f"Nutrition: {_nutrition_text(food)}.\n"
        f"Tags: {tags}.\n"
        f"Description: {food.get('description', '')}"
    )


def _nutrition_text(food: Dict[str, Any]) -> str:
    """Human-readable nutrition summary embedded in the food's Chroma document."""
    calories = _as_nutrition(food.get("calories"))
    protein = _as_nutrition(food.get("protein"))
    parts: List[str] = []
    if calories is not None:
        parts.append(f"{int(calories)} kcal")
        if calories <= 400:
            parts.append("light")
        elif calories >= 600:
            parts.append("filling")
    if protein is not None:
        parts.append(f"{int(protein)}g protein")
        if protein >= 20:
            parts.append("high protein")
    return ", ".join(parts) if parts else "not recorded"


def _chroma_metadata(food: Dict[str, Any], food_id: str) -> Dict[str, Any]:
    return {
        "food_id": food_id,
        "name": food.get("name", ""),
        "cuisine": food.get("cuisine", ""),
        "diet_type": food.get("diet_type", ""),
        "spice_level": food.get("spice_level", ""),
        "price": food.get("price", 0),
        "available": food.get("available", True),
    }


def index_food(food: Dict[str, Any]) -> None:
    """Upsert a single food document into the ChromaDB food index."""
    food_id = id_to_str(food.get("_id") or food.get("id"))
    if not food_id:
        return
    chroma.upsert_food(
        food_id=food_id,
        document=build_chroma_document(food),
        metadata=_chroma_metadata(food, food_id),
    )


def add_food(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a food item in MongoDB and index it in ChromaDB."""
    error = _validate_food(data)
    if error:
        return {"success": False, "error": error}

    now = utcnow()
    doc = _normalize_food(data)
    doc.update(
        {
            "rating_avg": float(data.get("rating_avg", 0.0)),
            "rating_count": int(data.get("rating_count", 0)),
            "recommendation_count": int(data.get("recommendation_count", 0)),
            "created_at": now,
            "updated_at": now,
        }
    )
    result = mongo.foods().insert_one(doc)
    doc["_id"] = result.inserted_id
    index_food(doc)
    return {"success": True, "food": stringify_doc(doc)}


def update_food(food_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing food item and re-index it."""
    oid = to_object_id(food_id)
    if not oid:
        return {"success": False, "error": "Invalid food id."}
    error = _validate_food(data)
    if error:
        return {"success": False, "error": error}

    update = _normalize_food(data)
    update["updated_at"] = utcnow()
    mongo.foods().update_one({"_id": oid}, {"$set": update})
    food = mongo.foods().find_one({"_id": oid})
    if food:
        index_food(food)
    return {"success": True, "food": stringify_doc(food)}


def set_availability(food_id: str, available: bool) -> bool:
    """Activate or deactivate a food item (keeps it in the catalogue)."""
    oid = to_object_id(food_id)
    if not oid:
        return False
    mongo.foods().update_one(
        {"_id": oid}, {"$set": {"available": available, "updated_at": utcnow()}}
    )
    food = mongo.foods().find_one({"_id": oid})
    if food:
        index_food(food)
    return True


def delete_food(food_id: str) -> bool:
    """Permanently delete a food item from MongoDB and ChromaDB."""
    oid = to_object_id(food_id)
    if not oid:
        return False
    mongo.foods().delete_one({"_id": oid})
    chroma.delete_food(food_id)
    return True


def get_food(food_id: str) -> Optional[dict]:
    """One food by id, or None when the id is unknown or malformed."""
    oid = to_object_id(food_id)
    if not oid:
        return None
    return stringify_doc(mongo.foods().find_one({"_id": oid}))


def get_foods_by_ids(food_ids: List[str]) -> List[dict]:
    """Fetch full food documents for a list of string ids (order preserved)."""
    oids = [to_object_id(fid) for fid in food_ids]
    oids = [oid for oid in oids if oid]
    if not oids:
        return []
    found = {
        str(f["_id"]): stringify_doc(f)
        for f in mongo.foods().find({"_id": {"$in": oids}})
    }
    ordered = []
    for fid in food_ids:
        if fid in found:
            ordered.append(found[fid])
    return ordered


def list_foods(include_unavailable: bool = True) -> List[dict]:
    """Every food, newest first; set include_unavailable=False for the live menu."""
    query = {} if include_unavailable else {"available": True}
    return [stringify_doc(f) for f in mongo.foods().find(query).sort("name", 1)]


# ---------------------------------------------------------------------------
# Today's special / discount pricing
# ---------------------------------------------------------------------------
def effective_price(food: Dict[str, Any]) -> float:
    """The price the customer pays today: discounted when it's a special."""
    try:
        price = float(food.get("price", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    if food.get("is_special"):
        try:
            discount = float(food.get("discount_percent", 0) or 0)
        except (TypeError, ValueError):
            discount = 0.0
        if 0 < discount <= 90:
            return round(price * (1 - discount / 100.0), 2)
    return price





def increment_recommendation_count(food_ids: List[str]) -> None:
    """Bump the recommendation counter for analytics/popularity scoring."""
    oids = [to_object_id(fid) for fid in food_ids]
    oids = [oid for oid in oids if oid]
    if not oids:
        return
    mongo.foods().update_many(
        {"_id": {"$in": oids}}, {"$inc": {"recommendation_count": 1}}
    )


def rebuild_index() -> int:
    """Rebuild the whole ChromaDB food index from MongoDB. Returns count.

    Every dish is embedded in one batch rather than one call each, which makes
    a full rebuild a single request instead of one per food.
    """
    chroma.reset_food_collection()
    batch = []
    for food in mongo.foods().find():
        food_id = id_to_str(food.get("_id"))
        if not food_id:
            continue
        batch.append({
            "id": food_id,
            "document": build_chroma_document(food),
            "metadata": _chroma_metadata(food, food_id),
        })
    return chroma.upsert_foods(batch)


# ---------------------------------------------------------------------------
# Validation / normalisation
# ---------------------------------------------------------------------------
def _validate_food(data: Dict[str, Any]) -> Optional[str]:
    """Return an error message for the first invalid field, or None when the food is fine."""
    if not validators.sanitize_text(data.get("name", "")):
        return "Food name is required."
    if not validators.is_valid_price(data.get("price")):
        return "Price must be a non-negative number."
    if data.get("diet_type") and not validators.validate_diet_type(data["diet_type"]):
        return "Invalid diet type."
    if data.get("spice_level") and not validators.validate_spice_level(
        data["spice_level"]
    ):
        return "Invalid spice level."
    try:
        discount = float(data.get("discount_percent", 0) or 0)
    except (TypeError, ValueError):
        return "Discount must be a number."
    if discount < 0 or discount > 90:
        return "Discount must be between 0 and 90 percent."
    if not validators.is_valid_nutrition(data.get("calories")):
        return "Calories must be a non-negative number."
    if not validators.is_valid_nutrition(data.get("protein")):
        return "Protein must be a non-negative number."
    return None


def _normalize_food(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a submitted form into the exact document shape MongoDB stores."""
    meal_type = data.get("meal_type", [])
    if isinstance(meal_type, str):
        meal_type = [m.strip() for m in meal_type.split(",") if m.strip()]
    return {
        "name": validators.sanitize_text(data.get("name", ""), 120),
        "category": validators.sanitize_text(data.get("category", ""), 60),
        "cuisine": validators.sanitize_text(data.get("cuisine", ""), 60),
        "diet_type": data.get("diet_type", "non_veg"),
        "meal_type": meal_type,
        "spice_level": data.get("spice_level", "medium"),
        "price": float(data.get("price", 0)),
        "ingredients": _as_list(data.get("ingredients", [])),
        "allergens": _as_list(data.get("allergens", [])),
        "tags": _as_list(data.get("tags", [])),
        "description": validators.sanitize_text(data.get("description", ""), 1000),
        "available": bool(data.get("available", True)),
        "image": _as_image_name(data.get("image")),
        "calories": _as_nutrition(data.get("calories")),
        "protein": _as_nutrition(data.get("protein")),
        "is_special": bool(data.get("is_special", False)),
        "discount_percent": float(data.get("discount_percent", 0) or 0)
        if data.get("is_special")
        else 0,
    }


def _as_image_name(value) -> Optional[str]:
    """Sanitise a stored photo filename.

    Only a bare filename with an allowed image extension is kept - any
    directory component is stripped so a crafted value can never escape the
    ``assets/foods`` folder.
    """
    if not value:
        return None
    name = os.path.basename(str(value).strip().replace("\\", "/"))
    if not name or name.startswith("."):
        return None
    if not name.lower().endswith(settings.ALLOWED_IMAGE_EXTENSIONS):
        return None
    return name


def image_path(food: Dict[str, Any]) -> Optional[str]:
    """Absolute path to a food's photo, or None when there is no usable file.

    Falls back to the dish-name filename (``Chicken Momo`` ->
    ``chicken-momo.jpg``) so cards rendered from an older cached food document,
    which has no ``image`` field yet, still show their picture.
    """
    name = _as_image_name(food.get("image"))
    if not name and food.get("name"):
        name = _slug(food["name"]) + ".jpg"
    if not name:
        return None
    path = settings.FOOD_IMAGE_DIR / name
    return str(path) if path.is_file() else None


def save_image(uploaded_file, food_name: str) -> Optional[str]:
    """Store an uploaded photo in ``assets/foods`` and return its filename."""
    if uploaded_file is None:
        return None
    extension = os.path.splitext(uploaded_file.name)[1].lower()
    if extension not in settings.ALLOWED_IMAGE_EXTENSIONS:
        return None
    filename = f"{_slug(food_name)}{extension}"
    settings.FOOD_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.FOOD_IMAGE_DIR / filename, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return filename


def _slug(text: str) -> str:
    """Lower-case, hyphenated form of a dish name, used for its photo file."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "food"


def _as_nutrition(value) -> Optional[float]:
    """Coerce a nutrition figure to a float, or None when it is not recorded."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _as_list(value) -> List[str]:
    """Coerce a comma-separated string or list into a clean list of strings."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------
def rate_food(user_id: str, food_id: str, rating: int, comment: str = "") -> Dict[str, Any]:
    """Save (or replace) a user's rating for a food and refresh its average."""
    if not validators.is_valid_rating(rating):
        return {"success": False, "error": "Rating must be an integer from 1 to 5."}
    food_oid = to_object_id(food_id)
    if not food_oid or not to_object_id(user_id):
        return {"success": False, "error": "Invalid user or food id."}

    mongo.ratings().update_one(
        {"user_id": user_id, "food_id": food_id},
        {
            "$set": {
                "user_id": user_id,
                "food_id": food_id,
                "rating": int(rating),
                "comment": validators.sanitize_text(comment, 500),
                "created_at": utcnow(),
            }
        },
        upsert=True,
    )
    _recompute_rating(food_id)
    return {"success": True}


def _recompute_rating(food_id: str) -> None:
    """Recalculate rating_avg and rating_count for a food from its ratings."""
    oid = to_object_id(food_id)
    if not oid:
        return
    docs = list(mongo.ratings().find({"food_id": food_id}))
    count = len(docs)
    avg = round(sum(d.get("rating", 0) for d in docs) / count, 2) if count else 0.0
    mongo.foods().update_one(
        {"_id": oid}, {"$set": {"rating_avg": avg, "rating_count": count}}
    )
    food = mongo.foods().find_one({"_id": oid})
    if food:
        index_food(food)


def get_user_rating(user_id: str, food_id: str) -> Optional[int]:
    """The stars this user gave one food, or None if they never rated it."""
    doc = mongo.ratings().find_one({"user_id": user_id, "food_id": food_id})
    return doc.get("rating") if doc else None


def user_ratings(user_id: str) -> Dict[str, int]:
    """Every rating this user has given, as ``{food_id: stars}``.

    Feeds the hybrid ranker so dishes the user personally disliked stop being
    recommended and the ones they loved surface more often.
    """
    if not to_object_id(user_id):
        return {}
    try:
        docs = mongo.ratings().find({"user_id": user_id}, {"food_id": 1, "rating": 1})
        return {
            str(d["food_id"]): int(d["rating"])
            for d in docs
            if d.get("food_id") and d.get("rating") is not None
        }
    except Exception:  # pragma: no cover - infra dependent
        return {}


# ---------------------------------------------------------------------------
# Favourites
# ---------------------------------------------------------------------------
def add_favourite(user_id: str, food_id: str) -> bool:
    """Star a food for a user (idempotent)."""
    if not to_object_id(user_id) or not to_object_id(food_id):
        return False
    mongo.favourites().update_one(
        {"user_id": user_id, "food_id": food_id},
        {"$set": {"user_id": user_id, "food_id": food_id, "created_at": utcnow()}},
        upsert=True,
    )
    return True


def remove_favourite(user_id: str, food_id: str) -> bool:
    """Un-star a food for a user."""
    mongo.favourites().delete_one({"user_id": user_id, "food_id": food_id})
    return True


def is_favourite(user_id: str, food_id: str) -> bool:
    return mongo.favourites().count_documents(
        {"user_id": user_id, "food_id": food_id}
    ) > 0


def list_favourites(user_id: str) -> List[dict]:
    """Return the full food documents a user has favourited."""
    fav_ids = [
        f["food_id"]
        for f in mongo.favourites().find({"user_id": user_id}).sort("created_at", -1)
    ]
    return get_foods_by_ids(fav_ids)
