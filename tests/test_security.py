from datetime import timedelta

import pytest

from app.core.security import TokenError, create_access_token, decode_access_token, hash_password, verify_password


def test_hash_password_verifies_original_password():
    password_hash = hash_password("secret123")

    assert verify_password("secret123", password_hash)
    assert not verify_password("wrong-password", password_hash)


def test_access_token_round_trips_subject_and_email():
    token = create_access_token(subject="user-123", email="user@example.com")

    payload = decode_access_token(token)

    assert payload["sub"] == "user-123"
    assert payload["email"] == "user@example.com"


def test_expired_access_token_is_rejected():
    token = create_access_token(
        subject="user-123",
        email="user@example.com",
        expires_delta=timedelta(seconds=-1),
    )

    with pytest.raises(TokenError, match="expired"):
        decode_access_token(token)

