"""Input validation helpers used across UI and services."""

import re
from typing import Optional

from config import settings

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    """Return True when the string looks like a valid email address."""
    if not email:
        return False
    return bool(EMAIL_RE.match(email.strip()))


def is_valid_password(password: str) -> bool:
    """Passwords must be at least 6 characters long."""
    return bool(password) and len(password) >= 6


def normalize_email(email: str) -> str:
    """Lowercase and trim an email so lookups are consistent."""
    return (email or "").strip().lower()


def is_valid_rating(rating) -> bool:
    """Ratings must be integers from 1 to 5."""
    try:
        value = int(rating)
    except (TypeError, ValueError):
        return False
    return 1 <= value <= 5


def is_valid_price(price) -> bool:
    """Prices must be non-negative numbers."""
    try:
        value = float(price)
    except (TypeError, ValueError):
        return False
    return value >= 0


def is_valid_nutrition(value) -> bool:
    """Nutrition figures (calories, protein) are optional non-negative numbers."""
    if value is None or value == "":
        return True
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


def validate_diet_type(diet_type: Optional[str]) -> Optional[str]:
    """Return the diet type if valid, otherwise None."""
    if diet_type and diet_type in settings.DIET_TYPES:
        return diet_type
    return None


def validate_spice_level(spice_level: Optional[str]) -> Optional[str]:
    """Return the spice level if valid, otherwise None."""
    if spice_level and spice_level in settings.SPICE_LEVELS:
        return spice_level
    return None


def sanitize_text(text: str, max_length: int = 2000) -> str:
    """Trim whitespace and cap length to protect against oversized inputs."""
    if not text:
        return ""
    cleaned = str(text).strip()
    return cleaned[:max_length]
