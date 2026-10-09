"""Unit tests for today's-special discount pricing (pure, no DB needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import food_service


def test_effective_price_applies_discount():
    food = {"price": 200, "is_special": True, "discount_percent": 25}
    assert food_service.effective_price(food) == 150


def test_effective_price_ignored_when_not_special():
    food = {"price": 200, "is_special": False, "discount_percent": 25}
    assert food_service.effective_price(food) == 200


def test_effective_price_no_discount_field():
    assert food_service.effective_price({"price": 300, "is_special": True}) == 300


def test_effective_price_rejects_absurd_discount():
    # Discounts above 90% are ignored (bad data guard).
    food = {"price": 200, "is_special": True, "discount_percent": 99}
    assert food_service.effective_price(food) == 200


def test_effective_price_handles_bad_values():
    assert food_service.effective_price({"price": None}) == 0
    assert food_service.effective_price(
        {"price": "abc", "is_special": True, "discount_percent": 10}
    ) == 0.0
