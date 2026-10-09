"""Tests for the order service.

The item-resolution logic is tested as pure units (no DB needed) by supplying a
``last_recommended`` list. Persistence tests (place / list / cancel) are
integration tests, skipped automatically when MongoDB is unavailable.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import mongo
from services import order_service, user_service


def _rec(name, price, fid):
    return {"id": fid, "name": name, "price": price}


# ---------------------------------------------------------------------------
# Pure unit tests - resolve_items against the last recommended foods (no DB)
# ---------------------------------------------------------------------------
def test_resolve_items_prices_quantities():
    last = [_rec("Chicken Momo", 250, "a"), _rec("Veg Burger", 180, "b")]
    resolved = order_service.resolve_items(
        [{"name": "Chicken Momo", "quantity": 2}], last_recommended=last
    )
    assert len(resolved) == 1
    item = resolved[0]
    assert item["name"] == "Chicken Momo"
    assert item["quantity"] == 2
    assert item["price"] == 250
    assert item["subtotal"] == 500


def test_resolve_items_defaults_quantity_to_one():
    last = [_rec("Veg Burger", 180, "b")]
    resolved = order_service.resolve_items(
        [{"name": "veg burger"}], last_recommended=last
    )
    assert resolved[0]["quantity"] == 1
    assert resolved[0]["subtotal"] == 180


def test_resolve_items_partial_name_matches_recommended():
    last = [_rec("Chicken Momo", 250, "a")]
    resolved = order_service.resolve_items(
        [{"name": "momo", "quantity": 3}], last_recommended=last
    )
    assert resolved and resolved[0]["name"] == "Chicken Momo"
    assert resolved[0]["subtotal"] == 750


def test_resolve_items_empty_when_no_match_and_no_db_lookup():
    # Name is absent from last_recommended; with an empty list there is nothing
    # to match and the (missing) food is simply dropped.
    resolved = order_service.resolve_items(
        [{"name": "definitely-not-a-food", "quantity": 1}], last_recommended=[]
    )
    assert resolved == []


# ---------------------------------------------------------------------------
# Pure unit tests - order lifecycle helpers (no DB)
# ---------------------------------------------------------------------------
def test_next_status_walks_the_lifecycle():
    assert order_service.next_status("placed") == "confirmed"
    assert order_service.next_status("confirmed") == "preparing"
    assert order_service.next_status("preparing") == "ready"
    assert order_service.next_status("ready") == "delivered"
    assert order_service.next_status("delivered") is None
    assert order_service.next_status("cancelled") is None


def test_status_progress_increases_along_lifecycle():
    values = [order_service.status_progress(s) for s in order_service.ORDER_STATUSES]
    assert values == sorted(values)
    assert values[-1] == 1.0
    assert order_service.status_progress("cancelled") == 0.0


def test_estimate_prep_minutes_scales_and_caps():
    one = [{"quantity": 1}]
    three = [{"quantity": 2}, {"quantity": 1}]
    huge = [{"quantity": 50}]
    assert order_service.estimate_prep_minutes(one) == order_service.BASE_PREP_MINUTES
    assert order_service.estimate_prep_minutes(three) > order_service.estimate_prep_minutes(one)
    assert order_service.estimate_prep_minutes(huge) == order_service.MAX_PREP_MINUTES


# ---------------------------------------------------------------------------
# Integration tests - require MongoDB
# ---------------------------------------------------------------------------
pytest_db = pytest.mark.skipif(not mongo.ping(), reason="MongoDB is not available")


def _any_user_id():
    users = user_service.list_users()
    return users[0]["id"] if users else None


@pytest_db
def test_place_list_and_cancel_order():
    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")

    items = [
        {"food_id": "x", "name": "Test Dish", "price": 100, "quantity": 2, "subtotal": 200}
    ]
    result = order_service.place_order(user_id, items)
    assert result["success"] is True
    order = result["order"]
    assert order["total"] == 200
    order_id = order.get("id") or order.get("_id")

    ids = [o.get("id") or o.get("_id") for o in order_service.list_user_orders(user_id)]
    assert order_id in ids

    cancel = order_service.cancel_order(user_id, order_id)
    assert cancel["success"] is True

    # Second cancel fails - it is no longer pending.
    again = order_service.cancel_order(user_id, order_id)
    assert again["success"] is False

    # Clean up the test order.
    from utils.id_utils import to_object_id

    mongo.orders().delete_one({"_id": to_object_id(order_id)})


@pytest_db
def test_cannot_cancel_another_users_order():
    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")
    result = order_service.place_order(
        user_id,
        [{"food_id": "x", "name": "D", "price": 50, "quantity": 1, "subtotal": 50}],
    )
    order_id = result["order"].get("id") or result["order"].get("_id")

    other = order_service.cancel_order("someone-else", order_id)
    assert other["success"] is False

    from utils.id_utils import to_object_id

    mongo.orders().delete_one({"_id": to_object_id(order_id)})


@pytest_db
def test_order_lifecycle_and_history():
    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")

    result = order_service.place_order(
        user_id,
        [{"food_id": "x", "name": "Lifecycle Dish", "price": 100, "quantity": 1, "subtotal": 100}],
    )
    order = result["order"]
    order_id = order.get("id") or order.get("_id")
    assert order["status"] == "placed"
    assert order["estimated_minutes"] >= order_service.BASE_PREP_MINUTES

    # Advance through the whole lifecycle.
    for expected in ["confirmed", "preparing", "ready", "delivered"]:
        res = order_service.update_status(order_id, expected)
        assert res["success"], res.get("error")

    from utils.id_utils import to_object_id

    stored = mongo.orders().find_one({"_id": to_object_id(order_id)})
    assert stored["status"] == "delivered"
    history_statuses = [h["status"] for h in stored.get("status_history", [])]
    assert history_statuses == ["placed", "confirmed", "preparing", "ready", "delivered"]

    # Terminal: no further changes, and the user cannot cancel it.
    assert order_service.update_status(order_id, "preparing")["success"] is False
    assert order_service.cancel_order(user_id, order_id)["success"] is False

    mongo.orders().delete_one({"_id": to_object_id(order_id)})


@pytest_db
def test_user_cannot_cancel_once_preparing():
    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")

    result = order_service.place_order(
        user_id,
        [{"food_id": "x", "name": "D", "price": 50, "quantity": 1, "subtotal": 50}],
    )
    order_id = result["order"].get("id") or result["order"].get("_id")

    # Cancellable while just confirmed...
    order_service.update_status(order_id, "confirmed")
    # ...but not once the kitchen starts.
    order_service.update_status(order_id, "preparing")
    assert order_service.cancel_order(user_id, order_id)["success"] is False

    from utils.id_utils import to_object_id

    mongo.orders().delete_one({"_id": to_object_id(order_id)})


@pytest_db
def test_place_order_rejects_empty_items():
    user_id = _any_user_id()
    if not user_id:
        pytest.skip("No users seeded")
    result = order_service.place_order(user_id, [])
    assert result["success"] is False


# ---------------------------------------------------------------------------
# Deletion tests.
#
# These use a throwaway user id (a fresh ObjectId that belongs to no real
# account) so the bulk-clear test can never touch a seeded demo user's orders.
# ---------------------------------------------------------------------------
def _throwaway_user_id():
    from bson import ObjectId

    return str(ObjectId())


def _place(user_id, name):
    res = order_service.place_order(
        user_id,
        [{"food_id": "x", "name": name, "price": 50, "quantity": 1, "subtotal": 50}],
    )
    return res["order"].get("id") or res["order"].get("_id")


@pytest_db
def test_delete_order_only_when_finished():
    from utils.id_utils import to_object_id

    user_id = _throwaway_user_id()
    order_id = _place(user_id, "Deletable")

    # An active order cannot be deleted - it must be cancelled first.
    blocked = order_service.delete_order(user_id, order_id)
    assert blocked["success"] is False
    assert mongo.orders().find_one({"_id": to_object_id(order_id)}) is not None

    # Once cancelled, deletion is allowed and the document is really gone.
    assert order_service.cancel_order(user_id, order_id)["success"] is True
    removed = order_service.delete_order(user_id, order_id)
    assert removed["success"] is True
    assert mongo.orders().find_one({"_id": to_object_id(order_id)}) is None

    # Deleting twice reports failure rather than raising.
    assert order_service.delete_order(user_id, order_id)["success"] is False


@pytest_db
def test_delivered_order_can_be_deleted():
    from utils.id_utils import to_object_id

    user_id = _throwaway_user_id()
    order_id = _place(user_id, "Delivered Dish")
    for step in ["confirmed", "preparing", "ready", "delivered"]:
        order_service.update_status(order_id, step)

    assert order_service.delete_order(user_id, order_id)["success"] is True
    assert mongo.orders().find_one({"_id": to_object_id(order_id)}) is None


@pytest_db
def test_cannot_delete_another_users_order():
    from utils.id_utils import to_object_id

    owner = _throwaway_user_id()
    order_id = _place(owner, "Theirs")
    order_service.cancel_order(owner, order_id)

    other = order_service.delete_order(_throwaway_user_id(), order_id)
    assert other["success"] is False
    assert mongo.orders().find_one({"_id": to_object_id(order_id)}) is not None

    mongo.orders().delete_one({"_id": to_object_id(order_id)})


@pytest_db
def test_delete_finished_orders_keeps_active_ones():
    from utils.id_utils import to_object_id

    user_id = _throwaway_user_id()
    finished_id = _place(user_id, "Finished Dish")
    order_service.cancel_order(user_id, finished_id)
    active_id = _place(user_id, "Active Dish")

    assert order_service.count_finished_orders(user_id) == 1

    removed = order_service.delete_finished_orders(user_id)
    assert removed == 1
    assert mongo.orders().find_one({"_id": to_object_id(finished_id)}) is None
    # The still-active order survives the bulk clear.
    assert mongo.orders().find_one({"_id": to_object_id(active_id)}) is not None
    assert order_service.count_finished_orders(user_id) == 0

    mongo.orders().delete_one({"_id": to_object_id(active_id)})


@pytest_db
def test_admin_delete_requires_a_finished_order():
    from utils.id_utils import to_object_id

    user_id = _throwaway_user_id()
    order_id = _place(user_id, "Admin Deletable")

    # A live kitchen ticket cannot be dropped, even by the admin.
    blocked = order_service.admin_delete_order(order_id)
    assert blocked["success"] is False
    assert mongo.orders().find_one({"_id": to_object_id(order_id)}) is not None

    order_service.update_status(order_id, "cancelled")
    assert order_service.admin_delete_order(order_id)["success"] is True
    assert mongo.orders().find_one({"_id": to_object_id(order_id)}) is None


@pytest_db
def test_admin_delete_works_regardless_of_owner():
    # The admin is not the customer, so there is no ownership check.
    from utils.id_utils import to_object_id

    order_id = _place(_throwaway_user_id(), "Someone Elses Order")
    order_service.update_status(order_id, "cancelled")

    assert order_service.admin_delete_order(order_id)["success"] is True
    assert mongo.orders().find_one({"_id": to_object_id(order_id)}) is None


@pytest_db
def test_admin_delete_rejects_bad_ids():
    assert order_service.admin_delete_order("not-an-id")["success"] is False
    from bson import ObjectId

    assert order_service.admin_delete_order(str(ObjectId()))["success"] is False
