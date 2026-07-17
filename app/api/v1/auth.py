from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas.auth import AuthTokenResponse, LoginRequest, RegisterRequest, UserResponse
from app.core.dependencies import get_auth_service, get_current_user
from app.services.auth_service import (
    AuthService,
    EmailAlreadyExistsError,
    InactiveUserError,
    InvalidCredentialsError,
)


router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """Register a new user account."""

    try:
        return await auth_service.register(payload)
    except EmailAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/login", response_model=AuthTokenResponse)
async def login(
    payload: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthTokenResponse:
    """Authenticate a user and return a Bearer access token."""

    try:
        return await auth_service.login(payload)
    except (InvalidCredentialsError, InactiveUserError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


@router.get("/me", response_model=UserResponse)
async def me(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Return the currently authenticated user."""

    return current_user
