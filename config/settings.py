"""Central configuration loader.

Reads all runtime configuration from environment variables (loaded from a
``.env`` file when present). No secret is ever hardcoded here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if it exists. Values already present
# in the real environment take precedence and are never overwritten.
load_dotenv()

# Project root directory (folder that contains this config package's parent).
BASE_DIR = Path(__file__).resolve().parent.parent


def _get(name: str, default: str = "") -> str:
    """Read one environment variable, falling back to a default."""
    value = os.getenv(name)
    return value if value is not None else default


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGODB_URI = _get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = _get("MONGODB_DB_NAME", "smartbite_ai")

# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
LLM_PROVIDER = _get("LLM_PROVIDER", "gemini").lower().strip()
GEMINI_API_KEY = _get("GEMINI_API_KEY", "")
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")

# A placeholder key means "not configured". We treat the sample values from
# .env.example as unset so the app can fall back to offline rule-based logic.
_PLACEHOLDERS = {
    "",
    "your_gemini_api_key_here",
    "your_openai_api_key_here",
    "your_key_here",
}


def llm_is_configured() -> bool:
    """Return True when a usable LLM API key is available for the provider."""
    if LLM_PROVIDER == "gemini":
        return GEMINI_API_KEY not in _PLACEHOLDERS
    if LLM_PROVIDER == "openai":
        return OPENAI_API_KEY not in _PLACEHOLDERS
    return False


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_PATH = _get("CHROMA_PATH", str(BASE_DIR / "chroma_db"))

# ---------------------------------------------------------------------------
# Food photos (bundled locally so the demo works with no internet)
# ---------------------------------------------------------------------------
FOOD_IMAGE_DIR = BASE_DIR / "assets" / "foods"
ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# ChromaDB collection names.
FOOD_COLLECTION = "food_items"
MEMORY_COLLECTION = "user_memories"

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
APP_NAME = _get("APP_NAME", "SmartBite AI")
APP_ENV = _get("APP_ENV", "development")
ADMIN_EMAIL = _get("ADMIN_EMAIL", "admin@smartbite.ai")
ADMIN_PASSWORD = _get("ADMIN_PASSWORD", "admin12345")


def _get_float(name: str, default: float) -> float:
    """Read a numeric environment variable, ignoring malformed values."""
    try:
        return float(_get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    """Read a true/false environment variable."""
    return _get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Location & weather (for weather/season/time-aware recommendations)
# ---------------------------------------------------------------------------
APP_CITY = _get("APP_CITY", "Kathmandu")
APP_LAT = _get_float("APP_LAT", 27.7172)
APP_LON = _get_float("APP_LON", 85.3240)
# Weather uses the free, keyless Open-Meteo API. Disable to skip network calls.
WEATHER_ENABLED = _get_bool("WEATHER_ENABLED", True)

# ---------------------------------------------------------------------------
# Email (password-reset codes, sent over SMTP - Gmail by default)
# ---------------------------------------------------------------------------
SMTP_HOST = _get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(_get("SMTP_PORT", "587"))
SMTP_USER = _get("SMTP_USER", "")
# For Gmail this must be a 16-character App Password, not the account password.
# Google displays it in four spaced groups ("abcd efgh ijkl mnop") but rejects
# it with the spaces still in, so strip whitespace before use.
SMTP_PASSWORD = _get("SMTP_PASSWORD", "").replace(" ", "").strip()
SMTP_USER = SMTP_USER.strip()
SMTP_FROM_NAME = _get("SMTP_FROM_NAME", APP_NAME)

_EMAIL_PLACEHOLDERS = {
    "", "your_gmail_address_here", "your_gmail_app_password_here",
    "you@gmail.com", "your_app_password",
}


def email_is_configured() -> bool:
    """True when SMTP credentials are present, so reset codes can be emailed."""
    return (
        SMTP_USER not in _EMAIL_PLACEHOLDERS
        and SMTP_PASSWORD not in _EMAIL_PLACEHOLDERS
    )


# How long a password-reset code stays valid, and how many wrong tries are
# allowed before it is thrown away.
RESET_CODE_TTL_MINUTES = int(_get("RESET_CODE_TTL_MINUTES", "15"))
RESET_MAX_ATTEMPTS = 5

# ---------------------------------------------------------------------------
# Sessions (persistent login across page refresh)
# ---------------------------------------------------------------------------
SESSION_TTL_DAYS = int(_get("SESSION_TTL_DAYS", "7"))

# ---------------------------------------------------------------------------
# Domain constants (validated across the app)
# ---------------------------------------------------------------------------
ROLES = ("user", "admin")
DIET_TYPES = ("veg", "non_veg", "vegan", "egg")
SPICE_LEVELS = ("mild", "medium", "spicy", "very_spicy")
