"""User memory service (long-term preferences stored in ChromaDB)."""

import re
import uuid
from typing import Dict, List

from database import chroma
from utils.date_utils import to_iso, utcnow


def _normalize(text: str) -> str:
    """Canonical form for duplicate detection: lowercase, no punctuation."""
    return re.sub(r"[^a-z0-9 ]+", "", str(text).lower()).strip()


def save_memory(
    user_id: str,
    text: str,
    memory_type: str = "preference",
    confidence: float = 0.85,
    source: str = "chat",
) -> str:
    """Store a single long-term memory for a user and return its id.

    Near-duplicates (same text ignoring case/punctuation) are skipped so the
    memory store does not fill up with "User likes spicy food." hundreds of
    times — duplicates crowd out diverse memories at retrieval time.
    """
    text = (text or "").strip()
    if not text:
        return ""

    normalized = _normalize(text)
    existing = {_normalize(m.get("document", "")) for m in chroma.get_all_memories(user_id)}
    if normalized in existing:
        return ""

    memory_id = f"mem_{user_id}_{uuid.uuid4().hex[:12]}"
    chroma.add_memory(
        memory_id=memory_id,
        text=text,
        metadata={
            "user_id": user_id,
            "memory_type": memory_type,
            "confidence": float(confidence),
            "source": source,
            "created_at": to_iso(utcnow()),
        },
    )
    return memory_id


def save_memories(user_id: str, memories: List[str], source: str = "chat") -> int:
    """Store multiple memories; returns how many were saved."""
    saved = 0
    for text in memories or []:
        if save_memory(user_id, text, source=source):
            saved += 1
    return saved


def dedupe_memories(user_id: str) -> int:
    """Remove duplicate memories for a user, keeping the oldest of each.

    Returns how many duplicates were deleted. Used by the maintenance script
    and safe to run repeatedly.
    """
    seen: Dict[str, str] = {}
    to_delete: List[str] = []
    memories = chroma.get_all_memories(user_id)
    memories.sort(key=lambda m: (m.get("metadata") or {}).get("created_at", ""))
    for mem in memories:
        key = _normalize(mem.get("document", ""))
        if not key or key in seen:
            to_delete.append(mem["id"])
        else:
            seen[key] = mem["id"]
    chroma.delete_memories(to_delete)
    return len(to_delete)


def retrieve_memories(user_id: str, query: str, n_results: int = 5) -> List[str]:
    """Return relevant memory texts for a user given the current query."""
    results = chroma.search_memories(query, user_id=user_id, n_results=n_results)
    return [r["document"] for r in results if r.get("document")]


def list_memories(user_id: str) -> List[Dict]:
    """Return all stored memories for a user (for the profile page)."""
    return chroma.get_all_memories(user_id)


def delete_memory(user_id: str, memory_id: str) -> bool:
    """Delete one of a user's own memories. Returns True when it was removed.

    Ownership is verified against the user's own memory list first, so a
    crafted id can never delete another user's memory.
    """
    if not memory_id:
        return False
    owned = {m["id"] for m in chroma.get_all_memories(user_id)}
    if memory_id not in owned:
        return False
    chroma.delete_memories([memory_id])
    return True


def clear_memories(user_id: str) -> int:
    """Delete every memory stored for a user. Returns how many were removed."""
    ids = [m["id"] for m in chroma.get_all_memories(user_id)]
    if not ids:
        return 0
    chroma.delete_memories(ids)
    return len(ids)
