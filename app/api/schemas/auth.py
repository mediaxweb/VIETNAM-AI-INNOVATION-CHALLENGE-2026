from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """Request payload for creating a new local account."""

    email: EmailStr = Field(
        ...,
        description="User email address. Stored in lowercase and must be unique.",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Plain-text password used only during registration.",
    )
    full_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Optional display name for the user profile.",
    )


class LoginRequest(BaseModel):
    """Request payload for exchanging credentials for an access token."""

    email: EmailStr = Field(
        ...,
        description="User email address used to authenticate.",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Plain-text password to verify against the stored password hash.",
    )


class AuthTokenResponse(BaseModel):
    """Bearer token contract returned after successful authentication."""

    access_token: str = Field(
        ...,
        min_length=1,
        description="JWT access token to send in the Authorization header.",
    )
    token_type: str = Field(
        default="bearer",
        description="Token type returned to the client. Always 'bearer' for this API.",
    )


class UserResponse(BaseModel):
    """Public representation of the authenticated user."""

    id: str = Field(
        ...,
        min_length=1,
        description="Stable user identifier used as the JWT subject.",
    )
    email: EmailStr = Field(
        ...,
        description="Lowercased unique email address for the user.",
    )
    full_name: Optional[str] = Field(
        default=None,
        description="Optional display name for the user profile.",
    )
    is_active: bool = Field(
        ...,
        description="Whether the user account is active.",
    )

