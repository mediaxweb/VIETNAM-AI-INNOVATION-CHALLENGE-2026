from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.schemas.auth import UserResponse
from app.core.config import configs
from app.services.auth_service import (
    AuthService,
    InactiveUserError,
    UserNotFoundError,
)
from app.core.security import TokenError


bearer_scheme = HTTPBearer(auto_error=False)

OPENCLAW_API_NOT_CONFIGURED_DETAIL = "OpenClaw API access is not configured."
INVALID_OPENCLAW_API_KEY_DETAIL = "Invalid OpenClaw API key."
OPENCLAW_USER_ID_REQUIRED_DETAIL = "OpenClaw user id is required."


def get_auth_service() -> AuthService:
    """FastAPI dependency that instantiates the auth service."""

    return AuthService()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """Resolve the current user from the Authorization header."""

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return await auth_service.get_current_user_from_token(credentials.credentials)
    except (TokenError, UserNotFoundError, InactiveUserError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_openclaw_or_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
    openclaw_api_key: str | None = Header(default=None, alias="X-OpenClaw-Api-Key"),
    openclaw_user_id: str | None = Header(default=None, alias="X-OpenClaw-User-Id"),
) -> str:
    """Resolve user scope for OpenClaw tool routes.

    OpenClaw can use a short shared key plus an explicit user id, while existing
    clients can keep using the JWT Bearer token path.
    """

    provided_openclaw_key = (openclaw_api_key or "").strip()
    if provided_openclaw_key:
        expected_openclaw_key = configs.resolved_rag_brain_openclaw_api_key
        if not expected_openclaw_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=OPENCLAW_API_NOT_CONFIGURED_DETAIL,
            )

        if not secrets.compare_digest(provided_openclaw_key, expected_openclaw_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=INVALID_OPENCLAW_API_KEY_DETAIL,
            )

        resolved_openclaw_user_id = (openclaw_user_id or "").strip()
        if not resolved_openclaw_user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=OPENCLAW_USER_ID_REQUIRED_DETAIL,
            )
        return resolved_openclaw_user_id

    current_user = await get_current_user(
        credentials=credentials,
        auth_service=auth_service,
    )
    return current_user.id
