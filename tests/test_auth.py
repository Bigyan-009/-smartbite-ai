"""Unit tests for password hashing and verification."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import auth_service


def test_password_hash_is_not_plaintext():
    hashed = auth_service.hash_password("secret123")
    assert hashed != "secret123"
    assert hashed.startswith("$2")  # bcrypt prefix


def test_correct_password_verifies():
    hashed = auth_service.hash_password("MyStrongPass1")
    assert auth_service.verify_password("MyStrongPass1", hashed) is True


def test_wrong_password_fails():
    hashed = auth_service.hash_password("MyStrongPass1")
    assert auth_service.verify_password("WrongPass", hashed) is False


def test_empty_password_fails_verification():
    hashed = auth_service.hash_password("something")
    assert auth_service.verify_password("", hashed) is False


def test_hashes_are_salted_and_unique():
    a = auth_service.hash_password("samepassword")
    b = auth_service.hash_password("samepassword")
    assert a != b  # different salts
    assert auth_service.verify_password("samepassword", a)
    assert auth_service.verify_password("samepassword", b)
