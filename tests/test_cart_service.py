"""Tests for the cart rules.

The cart is a plain list, so every one of these is a pure unit test - no
Streamlit session, no database.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import cart_service


def _food(fid="a", name="Chicken Momo", price=250):
    return {"id": fid, "name": name, "price": price}


def test_add_item_creates_a_line():
    cart = cart_service.add_item([], _food())
    assert len(cart) == 1
    assert cart[0]["name"] == "Chicken Momo"
    assert cart[0]["quantity"] == 1
    assert cart[0]["price"] == 250


def test_adding_the_same_dish_merges_quantities():
    cart = cart_service.add_item([], _food(), quantity=2)
    cart = cart_service.add_item(cart, _food(), quantity=3)
    assert len(cart) == 1
    assert cart[0]["quantity"] == 5


def test_add_item_uses_the_effective_price_when_given():
    # Today's Special: the card passes the discounted price explicitly.
    cart = cart_service.add_item([], _food(price=250), price=212)
    assert cart[0]["price"] == 212


def test_add_item_ignores_a_food_without_an_id():
    assert cart_service.add_item([], {"name": "Ghost dish"}) == []


def test_quantities_are_clamped_to_the_maximum():
    cart = cart_service.add_item([], _food(), quantity=999)
    assert cart[0]["quantity"] == cart_service.MAX_QUANTITY

    cart = cart_service.set_quantity(cart, "a", 999)
    assert cart[0]["quantity"] == cart_service.MAX_QUANTITY


def test_setting_quantity_to_zero_removes_the_line():
    cart = cart_service.add_item([], _food())
    assert cart_service.set_quantity(cart, "a", 0) == []


def test_set_quantity_ignores_unknown_food():
    cart = cart_service.add_item([], _food())
    unchanged = cart_service.set_quantity(cart, "does-not-exist", 5)
    assert unchanged[0]["quantity"] == 1


def test_remove_item_only_removes_the_named_dish():
    cart = cart_service.add_item([], _food("a", "Momo"))
    cart = cart_service.add_item(cart, _food("b", "Burger", 180))
    cart = cart_service.remove_item(cart, "a")
    assert [line["food_id"] for line in cart] == ["b"]


def test_count_and_total_use_quantities():
    cart = cart_service.add_item([], _food("a", "Momo", 250), quantity=2)
    cart = cart_service.add_item(cart, _food("b", "Burger", 180), quantity=1)
    assert cart_service.count_items(cart) == 3
    assert cart_service.cart_total(cart) == 680  # 2*250 + 180


def test_empty_cart_totals_are_zero():
    assert cart_service.count_items([]) == 0
    assert cart_service.cart_total([]) == 0
    assert cart_service.to_pairs([]) == []


def test_clear_empties_the_cart():
    cart = cart_service.add_item([], _food())
    assert cart_service.clear(cart) == []


def test_to_pairs_shape_matches_items_from_ids():
    cart = cart_service.add_item([], _food("a"), quantity=2)
    assert cart_service.to_pairs(cart) == [{"food_id": "a", "quantity": 2}]


def test_bad_quantity_values_fall_back_to_one():
    cart = cart_service.add_item([], _food(), quantity="not a number")
    assert cart[0]["quantity"] == 1
