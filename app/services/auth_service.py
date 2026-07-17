from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.api.schemas.auth import AuthTokenResponse, LoginRequest, RegisterRequest, UserResponse
from app.core.database import Database
from app.core.security import TokenError, create_access_token, decode_access_token, hash_password, verify_password


class EmailAlreadyExistsError(ValueError):
    """Raised when attempting to register an email that already exists."""


class InvalidCredentialsError(ValueError):
    """Raised when a login attempt fails."""


class UserNotFoundError(ValueError):
    """Raised when a user id from the token cannot be resolved."""


class InactiveUserError(ValueError):
    """Raised when an inactive user attempts to authenticate."""


class AuthService:
    """Service responsible for local account registration and authentication."""

    def __init__(self) -> None:
        self._users = Database.get_users_collection()

    async def register(self, payload: RegisterRequest) -> UserResponse:
        """Create a new user record and return its public profile."""

        now = datetime.now(timezone.utc)
        document = {
            "email": self._normalize_email(payload.email),
            "password_hash": hash_password(payload.password),
            "full_name": self._normalize_full_name(payload.full_name),
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        }
        try:
            result = await self._users.insert_one(document)
        except DuplicateKeyError as exc:
            raise EmailAlreadyExistsError("Email already exists.") from exc

        user = {**document, "_id": result.inserted_id}
        return self._serialize_user(user)

    async def login(self, payload: LoginRequest) -> AuthTokenResponse:
        """Verify credentials and issue a Bearer access token."""

        user = await self.get_user_by_email(payload.email)
        if user is None or not verify_password(payload.password, str(user.get("password_hash", ""))):
            raise InvalidCredentialsError("Invalid email or password.")
        if not bool(user.get("is_active", True)):
            raise InactiveUserError("User account is inactive.")

        access_token = create_access_token(
            subject=str(user["_id"]),
            email=str(user["email"]),
        )
        return AuthTokenResponse(access_token=access_token, token_type="bearer")

    async def get_current_user_from_token(self, token: str) -> UserResponse:
        """Resolve the current user from a Bearer access token."""

        payload = decode_access_token(token)
        user = await self.get_user_by_id(str(payload["sub"]))
        if user is None:
            raise UserNotFoundError("User referenced by token was not found.")
        if not bool(user.get("is_active", True)):
            raise InactiveUserError("User account is inactive.")
        return self._serialize_user(user)

    async def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        """Find a user document by email address."""

        normalized_email = self._normalize_email(email)
        return await self._users.find_one({"email": normalized_email})

    async def get_user_by_id(self, user_id: str) -> Optional[dict[str, Any]]:
        """Find a user document by its MongoDB ObjectId string."""

        try:
            object_id = ObjectId(user_id)
        except Exception:
            return None
        return await self._users.find_one({"_id": object_id})

    @staticmethod
    def _normalize_email(email: str) -> str:
        return str(email).strip().lower()

    @staticmethod
    def _normalize_full_name(full_name: Optional[str]) -> Optional[str]:
        if full_name is None:
            return None
        normalized = str(full_name).strip()
        return normalized or None

    @classmethod
    def _serialize_user(cls, user: dict[str, Any]) -> UserResponse:
        return UserResponse(
            id=str(user["_id"]),
            email=cls._normalize_email(str(user["email"])),
            full_name=cls._normalize_full_name(user.get("full_name")),
            is_active=bool(user.get("is_active", True)),
        )

