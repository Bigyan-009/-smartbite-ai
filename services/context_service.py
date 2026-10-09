"""Ambient context service: time of day, season, and live weather.

Powers weather/season/time-aware recommendations. Weather comes from the free,
keyless Open-Meteo API (fetched with the standard library, short timeout, and a
short-lived cache). Everything degrades gracefully: if the network is down or
weather is disabled, the bot still uses time-of-day and season.
"""

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from config import settings
from utils.date_utils import nepal_now

# Simple in-process cache so we don't hit the weather API on every chat turn.
_cache: Dict[str, Any] = {"weather": None, "fetched_hour": None}

# WMO weather codes (from Open-Meteo) grouped into simple conditions.
_RAIN_CODES = set(range(51, 68)) | {80, 81, 82, 95, 96, 99}
_SNOW_CODES = set(range(71, 78)) | {85, 86}
_CLOUD_CODES = {1, 2, 3, 45, 48}


def get_context() -> Dict[str, Any]:
    """Return the current ambient context used to tailor recommendations."""
    now = nepal_now()
    time_of_day = _time_of_day(now.hour)
    meal_type = _meal_for_time(now.hour)
    season = _season(now.month)
    weather = _get_weather()

    summary = _summary(time_of_day, season, weather)
    suggest_tags = _suggest_tags(time_of_day, season, weather)

    return {
        "time_of_day": time_of_day,
        "meal_type": meal_type,
        "season": season,
        "weather": weather,  # dict or None
        "summary": summary,
        "suggest_tags": suggest_tags,
        "city": settings.APP_CITY,
    }


# ---------------------------------------------------------------------------
# Time / season
# ---------------------------------------------------------------------------
def _time_of_day(hour: int) -> str:
    """Morning / afternoon / evening / night for a given hour."""
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _meal_for_time(hour: int) -> str:
    """The meal that suits a given hour (breakfast, lunch, snack, dinner)."""
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 16:
        return "lunch"
    if 16 <= hour < 18:
        return "snack"
    return "dinner"


def _season(month: int) -> str:
    """Season tuned for the Nepal/South-Asia context (monsoon included)."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "monsoon"
    return "autumn"


# ---------------------------------------------------------------------------
# Weather (Open-Meteo, best effort)
# ---------------------------------------------------------------------------
def _get_weather() -> Optional[Dict[str, Any]]:
    """Live weather from the free Open-Meteo API, or None if it cannot be reached."""
    if not settings.WEATHER_ENABLED:
        return None

    now_hour = nepal_now().strftime("%Y-%m-%d-%H")
    if _cache["fetched_hour"] == now_hour:
        return _cache["weather"]  # cached for the current hour

    weather = _fetch_weather()
    _cache["weather"] = weather
    _cache["fetched_hour"] = now_hour
    return weather


def _fetch_weather() -> Optional[Dict[str, Any]]:
    """Call the Open-Meteo API for the configured city's current weather."""
    params = urllib.parse.urlencode(
        {
            "latitude": settings.APP_LAT,
            "longitude": settings.APP_LON,
            "current": "temperature_2m,weather_code",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    current = data.get("current") or {}
    temp = current.get("temperature_2m")
    code = current.get("weather_code")
    if temp is None:
        return None

    condition = _condition(code)
    return {
        "temp_c": round(float(temp), 1),
        "condition": condition,
        "is_cold": float(temp) <= 15,
        "is_hot": float(temp) >= 28,
        "is_rainy": condition in ("rain", "snow"),
    }


def _condition(code) -> str:
    """Turn an Open-Meteo weather code into a plain word (rain, snow, clear...)."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "clear"
    if code in _RAIN_CODES:
        return "rain"
    if code in _SNOW_CODES:
        return "snow"
    if code in _CLOUD_CODES:
        return "cloudy"
    return "clear"


# ---------------------------------------------------------------------------
# Summaries & food hints
# ---------------------------------------------------------------------------
def _summary(time_of_day: str, season: str, weather: Optional[Dict[str, Any]]) -> str:
    """One human phrase for the moment, e.g. 'a rainy hot afternoon'."""
    parts: List[str] = []
    if weather:
        if weather["is_rainy"]:
            parts.append("rainy")
        if weather["is_cold"]:
            parts.append("cold")
        elif weather["is_hot"]:
            parts.append("hot")
    parts.append(time_of_day)
    descriptor = " ".join(parts)
    if weather:
        return f"a {descriptor} ({weather['temp_c']}°C, {weather['condition']})"
    return f"a {descriptor} in {season}"


def _suggest_tags(
    time_of_day: str, season: str, weather: Optional[Dict[str, Any]]
) -> List[str]:
    """Food descriptors that suit the current context (matched against foods).

    Live weather always wins; the season is only a fallback when the weather
    API is unavailable. (Otherwise monsoon season would suggest hot soup even
    on a genuinely hot, clear day.)
    """
    tags: List[str] = []

    if weather:
        cold = weather["is_cold"]
        hot = weather["is_hot"]
        rainy = weather["is_rainy"]
    else:
        cold = season == "winter"
        hot = season == "spring"
        rainy = season == "monsoon"

    if cold or rainy:
        tags += ["soup", "hot", "spicy", "steamed", "noodle", "ramen", "momo", "thukpa"]
    if hot and not (cold or rainy):
        tags += ["salad", "cold", "light", "juice", "fresh", "ice cream", "drink"]
    if time_of_day == "morning":
        tags += ["breakfast", "light", "tea", "coffee"]
    if time_of_day == "night":
        tags += ["dinner", "filling"]

    # De-duplicate, preserve order.
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
