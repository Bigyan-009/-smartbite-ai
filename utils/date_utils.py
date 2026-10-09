"""Datetime helpers.

Timestamps are stored in UTC (the MongoDB convention) and converted to Nepal
time (UTC+05:45, no daylight saving) whenever they are shown to a person.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

# Nepal has a fixed UTC+05:45 offset and no daylight-saving time.
NEPAL_TZ = timezone(timedelta(hours=5, minutes=45), name="NPT")


def utcnow() -> datetime:
    """Return the current timezone-aware UTC datetime (for storage)."""
    return datetime.now(timezone.utc)


def nepal_now() -> datetime:
    """Return the current timezone-aware Nepal datetime (for display/context)."""
    return datetime.now(NEPAL_TZ)


def to_nepal(dt) -> Optional[datetime]:
    """Convert a stored timestamp to Nepal time.

    Accepts datetimes or ISO strings. Naive datetimes are treated as UTC,
    which is how PyMongo returns stored dates.
    """
    if not dt:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(NEPAL_TZ)


def to_iso(dt: datetime) -> str:
    """Serialize a datetime to an ISO 8601 string."""
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    return dt.isoformat()


def format_display(dt) -> str:
    """Human friendly Nepal-time display, e.g. '2026-07-01 14:30'."""
    local = to_nepal(dt)
    if local is None:
        return dt if isinstance(dt, str) else ""
    return local.strftime("%Y-%m-%d %H:%M")
