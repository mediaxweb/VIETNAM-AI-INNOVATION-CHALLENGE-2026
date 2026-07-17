from bson import ObjectId

from app.services.auth_service import AuthService


def test_auth_service_normalizes_email_and_full_name():
    assert AuthService._normalize_email(" USER@Example.COM ") == "user@example.com"
    assert AuthService._normalize_full_name("  Alice Example  ") == "Alice Example"
    assert AuthService._normalize_full_name("   ") is None


def test_auth_service_serializes_public_user_fields():
    user_id = ObjectId()
    user = AuthService._serialize_user(
        {
            "_id": user_id,
            "email": "USER@Example.COM",
            "full_name": "  Alice Example  ",
            "is_active": True,
        }
    )

    assert user.id == str(user_id)
    assert user.email == "user@example.com"
    assert user.full_name == "Alice Example"
    assert user.is_active is True

