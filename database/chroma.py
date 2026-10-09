"""ChromaDB connection module.

Uses a persistent local ChromaDB store with two collections:

* ``food_items``    - semantic search over the food catalogue
* ``user_memories`` - long-term per-user preference memories

The default ChromaDB embedding function (all-MiniLM-L6-v2, run locally) is
used so the system works without any external embedding API key.
"""

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings

_client: Optional["chromadb.ClientAPI"] = None


def get_client():
    """Return a shared persistent ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
    return _client


def get_food_collection():
    """Collection holding embedded food descriptions."""
    return get_client().get_or_create_collection(
        name=settings.FOOD_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_memory_collection():
    """Collection holding per-user preference memories."""
    return get_client().get_or_create_collection(
        name=settings.MEMORY_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Food items
# ---------------------------------------------------------------------------
def upsert_food(food_id: str, document: str, metadata: Dict[str, Any]) -> None:
    """Insert or update a food item in the vector store."""
    collection = get_food_collection()
    clean_meta = _clean_metadata(metadata)
    collection.upsert(ids=[food_id], documents=[document], metadatas=[clean_meta])


def upsert_foods(items: List[Dict[str, Any]]) -> int:
    """Insert or update many foods in a single call.

    ChromaDB embeds every document passed in one batch together, which is far
    faster than a call per dish - rebuilding the whole index becomes one
    request instead of one per food.
    """
    if not items:
        return 0
    collection = get_food_collection()
    collection.upsert(
        ids=[item["id"] for item in items],
        documents=[item["document"] for item in items],
        metadatas=[_clean_metadata(item["metadata"]) for item in items],
    )
    return len(items)


def delete_food(food_id: str) -> None:
    """Remove a food item from the vector store (safe if it does not exist)."""
    try:
        get_food_collection().delete(ids=[food_id])
    except Exception:
        pass


def search_foods(
    query_text: str, n_results: int = 10, available_only: bool = True
) -> List[Dict[str, Any]]:
    """Return the most semantically similar foods to ``query_text``.

    Each result contains ``food_id``, ``document``, ``metadata`` and a
    ``similarity`` score in the 0-1 range (higher is closer).

    ``available_only`` applies a metadata filter inside ChromaDB, so a dish the
    kitchen has switched off never takes up one of the candidate slots. Doing
    it here rather than afterwards means the caller always gets a full set of
    usable candidates.
    """
    collection = get_food_collection()
    total = collection.count()
    if total == 0:
        return []

    n_results = max(1, min(n_results, total))
    where = {"available": True} if available_only else None
    try:
        res = collection.query(
            query_texts=[query_text], n_results=n_results, where=where
        )
    except Exception:
        # A malformed filter must never break the chat: fall back to an
        # unfiltered search and let the ranker drop anything unavailable.
        try:
            res = collection.query(query_texts=[query_text], n_results=n_results)
        except Exception:
            return []
    return _format_query(res)


# ---------------------------------------------------------------------------
# User memories
# ---------------------------------------------------------------------------
def add_memory(memory_id: str, text: str, metadata: Dict[str, Any]) -> None:
    """Store a single user memory."""
    collection = get_memory_collection()
    clean_meta = _clean_metadata(metadata)
    collection.upsert(ids=[memory_id], documents=[text], metadatas=[clean_meta])


def search_memories(
    query_text: str, user_id: str, n_results: int = 5
) -> List[Dict[str, Any]]:
    """Return the most relevant memories for a specific user.

    Memories are always filtered by ``user_id`` so one user can never see
    another user's memories.
    """
    collection = get_memory_collection()
    if collection.count() == 0:
        return []

    n_results = max(1, min(n_results, collection.count()))
    try:
        res = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"user_id": user_id},
        )
    except Exception:
        return []
    return _format_query(res)


def delete_memories(memory_ids: List[str]) -> None:
    """Remove memories by id (safe if some do not exist)."""
    if not memory_ids:
        return
    try:
        get_memory_collection().delete(ids=list(memory_ids))
    except Exception:
        pass


def get_all_memories(user_id: str) -> List[Dict[str, Any]]:
    """Return every stored memory for a user (used on the profile page)."""
    collection = get_memory_collection()
    try:
        res = collection.get(where={"user_id": user_id})
    except Exception:
        return []
    memories = []
    ids = res.get("ids", []) or []
    docs = res.get("documents", []) or []
    metas = res.get("metadatas", []) or []
    for i, mem_id in enumerate(ids):
        memories.append(
            {
                "id": mem_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
            }
        )
    return memories


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------
def index_stats() -> Dict[str, int]:
    """How many documents each collection currently holds."""
    try:
        return {
            "foods": get_food_collection().count(),
            "memories": get_memory_collection().count(),
        }
    except Exception:  # pragma: no cover - infra dependent
        return {"foods": 0, "memories": 0}


def reset_food_collection() -> None:
    """Delete and recreate the food collection (used to rebuild the index)."""
    client = get_client()
    try:
        client.delete_collection(settings.FOOD_COLLECTION)
    except Exception:
        pass
    client.get_or_create_collection(
        name=settings.FOOD_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB metadata values must be str/int/float/bool and non-null."""
    clean: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _format_query(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a chroma query result into a list of dicts with similarity."""
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    results = []
    for i, item_id in enumerate(ids):
        distance = dists[i] if i < len(dists) else 1.0
        # Cosine distance -> similarity (0..1). Clamp to keep it well behaved.
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        meta = metas[i] if i < len(metas) else {}
        results.append(
            {
                "id": item_id,
                "food_id": meta.get("food_id", item_id),
                "document": docs[i] if i < len(docs) else "",
                "metadata": meta,
                "similarity": similarity,
            }
        )
    return results
