from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.schemas.auth import UserResponse
from app.api.schemas.loan_agent import CustomerResponse, UploadedDocumentResponse
from app.api.v1 import loan_agent


NOW = datetime(2026, 7, 17, tzinfo=timezone.utc)


class _FakeLoanAgentService:
    def __init__(self):
        self.created_customer_user_id = None
        self.uploaded_legal_doc = None

    async def create_customer(self, *, user_id, payload):
        self.created_customer_user_id = user_id
        return CustomerResponse(
            id="customer-1",
            full_name=payload.full_name,
            phone=payload.phone,
            email=str(payload.email) if payload.email else None,
            national_id=payload.national_id,
            date_of_birth=payload.date_of_birth,
            address=payload.address,
            metadata=payload.metadata,
            created_at=NOW,
            updated_at=NOW,
        )

    async def upload_legal_doc(self, *, user_id, loan_profile_id, file, doc_type, description):
        self.uploaded_legal_doc = {
            "user_id": user_id,
            "loan_profile_id": loan_profile_id,
            "file_name": file.filename,
            "doc_type": doc_type,
            "description": description,
        }
        return UploadedDocumentResponse(
            id="doc-1",
            loan_profile_id=loan_profile_id,
            category="legal_doc",
            file_name=file.filename,
            file_path="/tmp/doc-1.pdf",
            content_type=file.content_type,
            size_bytes=10,
            doc_type=doc_type,
            description=description,
            status="uploaded",
            created_at=NOW,
        )


def _build_client(fake_service):
    app = FastAPI()
    app.include_router(loan_agent.router, prefix="/api/v1/loan")

    async def fake_current_user():
        return UserResponse(
            id="user-123",
            email="user@example.com",
            full_name=None,
            is_active=True,
        )

    app.dependency_overrides[loan_agent.get_current_user] = fake_current_user
    app.dependency_overrides[loan_agent.get_loan_agent_service] = lambda: fake_service
    return TestClient(app)


def test_loan_agent_openapi_exposes_mvp_endpoints():
    app = FastAPI()
    app.include_router(loan_agent.router, prefix="/api/v1/loan")
    openapi = app.openapi()
    paths = openapi["paths"]

    expected_routes = {
        ("post", "/api/v1/loan/customers"),
        ("get", "/api/v1/loan/customers/{customer_id}"),
        ("patch", "/api/v1/loan/customers/{customer_id}"),
        ("post", "/api/v1/loan/loan-profiles"),
        ("get", "/api/v1/loan/loan-profiles/{loan_profile_id}"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/legal-docs"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/check-legal-docs"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/financial-reports"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/collaterals"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/check-financials"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/check-collateral"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/check-credit-rule"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/compliance-result"),
        ("patch", "/api/v1/loan/loan-profiles/{loan_profile_id}/status"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/checklist"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/calculate-limit"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/tasks"),
        ("post", "/api/v1/loan/loan-profiles/{loan_profile_id}/reports"),
        ("get", "/api/v1/loan/loan-profiles/{loan_profile_id}/reports"),
    }
    discovered_routes = {
        (method, path)
        for path, operations in paths.items()
        for method in operations
        if method in {"get", "post", "patch", "delete"}
    }

    assert expected_routes <= discovered_routes
    assert len(expected_routes) == 19
    assert "multipart/form-data" in paths[
        "/api/v1/loan/loan-profiles/{loan_profile_id}/legal-docs"
    ]["post"]["requestBody"]["content"]
    assert "multipart/form-data" in paths[
        "/api/v1/loan/loan-profiles/{loan_profile_id}/financial-reports"
    ]["post"]["requestBody"]["content"]
    assert "multipart/form-data" in paths[
        "/api/v1/loan/loan-profiles/{loan_profile_id}/collaterals"
    ]["post"]["requestBody"]["content"]


def test_create_customer_uses_authenticated_user_scope():
    fake_service = _FakeLoanAgentService()
    client = _build_client(fake_service)

    response = client.post(
        "/api/v1/loan/customers",
        json={
            "full_name": "Nguyen Van A",
            "phone": "0900000000",
            "email": "customer@example.com",
        },
    )

    assert response.status_code == 201
    assert response.json()["full_name"] == "Nguyen Van A"
    assert fake_service.created_customer_user_id == "user-123"


def test_upload_legal_doc_accepts_multipart_file():
    fake_service = _FakeLoanAgentService()
    client = _build_client(fake_service)

    response = client.post(
        "/api/v1/loan/loan-profiles/profile-1/legal-docs",
        data={
            "doc_type": "identity",
            "description": "Citizen identity card",
        },
        files={
            "file": (
                "identity.pdf",
                b"%PDF-1.4\nidentity",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["file_name"] == "identity.pdf"
    assert fake_service.uploaded_legal_doc == {
        "user_id": "user-123",
        "loan_profile_id": "profile-1",
        "file_name": "identity.pdf",
        "doc_type": "identity",
        "description": "Citizen identity card",
    }
