"""Tests for user memory hygiene (duplicate prevention and cleanup).

Uses the real local ChromaDB store with a throwaway user id, cleaned up after
each test.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import chroma
from services import memory_service

TEST_USER = "pytest-memory-user"


@pytest.fixture(autouse=True)
def _clean_test_user_memories():
    yield
    ids = [m["id"] for m in chroma.get_all_memories(TEST_USER)]
    chroma.delete_memories(ids)


def test_duplicate_memory_is_skipped():
    first = memory_service.save_memory(TEST_USER, "User likes spicy food.")
    assert first
    # Same text (even with different case/punctuation) is not stored again.
    assert memory_service.save_memory(TEST_USER, "User likes spicy food.") == ""
    assert memory_service.save_memory(TEST_USER, "user likes spicy food") == ""
    assert len(chroma.get_all_memories(TEST_USER)) == 1


def test_distinct_memories_are_kept():
    saved = memory_service.save_memories(
        TEST_USER,
        ["User likes spicy food.", "User dislikes mushroom.", "User likes spicy food."],
    )
    assert saved == 2
    assert len(chroma.get_all_memories(TEST_USER)) == 2


def test_dedupe_memories_cleans_existing_duplicates():
    # Insert duplicates directly (bypassing the save-time guard).
    for i in range(3):
        chroma.add_memory(
            memory_id=f"mem_{TEST_USER}_dup{i}",
            text="User usually prefers food under Rs. 500.",
            metadata={"user_id": TEST_USER, "created_at": f"2026-01-0{i + 1}"},
        )
    removed = memory_service.dedupe_memories(TEST_USER)
    assert removed == 2
    remaining = chroma.get_all_memories(TEST_USER)
    assert len(remaining) == 1
    # The oldest copy is the one kept.
    assert remaining[0]["metadata"]["created_at"] == "2026-01-01"


# ---------------------------------------------------------------------------
# Deleting memories ("forget this")
# ---------------------------------------------------------------------------
OTHER_USER = "pytest-memory-other-user"


@pytest.fixture(autouse=True)
def _clean_other_user_memories():
    yield
    ids = [m["id"] for m in chroma.get_all_memories(OTHER_USER)]
    chroma.delete_memories(ids)


def test_delete_memory_removes_one_entry():
    memory_service.save_memory(TEST_USER, "User dislikes mushroom.")
    memory_service.save_memory(TEST_USER, "User likes spicy food.")
    memories = memory_service.list_memories(TEST_USER)
    assert len(memories) == 2

    target = next(m for m in memories if "mushroom" in m["document"])
    assert memory_service.delete_memory(TEST_USER, target["id"]) is True

    remaining = memory_service.list_memories(TEST_USER)
    assert len(remaining) == 1
    assert "mushroom" not in remaining[0]["document"]


def test_delete_memory_rejects_unknown_id():
    memory_service.save_memory(TEST_USER, "User likes spicy food.")
    assert memory_service.delete_memory(TEST_USER, "mem_does_not_exist") is False
    assert memory_service.delete_memory(TEST_USER, "") is False
    assert len(memory_service.list_memories(TEST_USER)) == 1


def test_cannot_delete_another_users_memory():
    memory_service.save_memory(OTHER_USER, "Other user is vegetarian.")
    victim = memory_service.list_memories(OTHER_USER)[0]

    # A different user supplying a valid id must not be able to delete it.
    assert memory_service.delete_memory(TEST_USER, victim["id"]) is False
    assert len(memory_service.list_memories(OTHER_USER)) == 1


def test_clear_memories_only_affects_that_user():
    memory_service.save_memories(
        TEST_USER, ["User likes spicy food.", "User dislikes mushroom."]
    )
    memory_service.save_memory(OTHER_USER, "Other user is vegetarian.")

    removed = memory_service.clear_memories(TEST_USER)
    assert removed == 2
    assert memory_service.list_memories(TEST_USER) == []
    # The other user's memory survives.
    assert len(memory_service.list_memories(OTHER_USER)) == 1


def test_clear_memories_on_empty_store_is_zero():
    assert memory_service.clear_memories("pytest-nobody") == 0
