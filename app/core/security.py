from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from passlib.context import CryptContext

from app.core.config import configs


PASSWORD_CONTEXT = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


class TokenError(ValueError):
    """Raised when an access token is missing, invalid, or expired."""


def hash_password(password: str) -> str:
    """Hash a plain-text password for storage."""

    return PASSWORD_CONTEXT.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plain-text password against a stored hash."""

    return PASSWORD_CONTEXT.verify(password, password_hash)


def create_access_token(*, subject: str, email: str, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token for a user."""

    expire_at = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=configs.jwt_access_token_expire_minutes)
    )
    payload = {
        "sub": subject,
        "email": email,
        "exp": expire_at,
    }
    return jwt.encode(
        payload,
        configs.jwt_secret_key,
        algorithm=configs.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT access token."""

    try:
        payload = jwt.decode(
            token,
            configs.jwt_secret_key,
            algorithms=[configs.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Access token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Access token is invalid.") from exc

    subject = str(payload.get("sub", "")).strip()
    email = str(payload.get("email", "")).strip()
    if not subject:
        raise TokenError("Access token is missing subject.")
    if not email:
        raise TokenError("Access token is missing email.")
    return payload
