from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field


CheckStatus = Literal["passed", "warning", "failed"]
ComplianceDecision = Literal["approved", "conditional", "rejected", "needs_review"]
TaskPriority = Literal["low", "medium", "high", "urgent"]


class CustomerCreateRequest(BaseModel):
    full_name: str = Field(..., min_length=1)
    phone: str | None = None
    email: EmailStr | None = None
    national_id: str | None = None
    date_of_birth: str | None = None
    address: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CustomerUpdateRequest(BaseModel):
    full_name: str | None = Field(default=None, min_length=1)
    phone: str | None = None
    email: EmailStr | None = None
    national_id: str | None = None
    date_of_birth: str | None = None
    address: str | None = None
    metadata: dict[str, Any] | None = None


class CustomerResponse(BaseModel):
    id: str
    full_name: str
    phone: str | None = None
    email: str | None = None
    national_id: str | None = None
    date_of_birth: str | None = None
    address: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class LoanProfileCreateRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    loan_amount: float = Field(..., gt=0)
    loan_purpose: str = Field(..., min_length=1)
    term_months: int = Field(..., gt=0, le=360)
    product_type: str | None = None
    currency: str = Field(default="VND", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoanProfileResponse(BaseModel):
    id: str
    customer_id: str
    loan_amount: float
    loan_purpose: str
    term_months: int
    product_type: str | None = None
    currency: str = "VND"
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class UploadedDocumentResponse(BaseModel):
    id: str
    loan_profile_id: str
    category: str
    file_name: str
    file_path: str
    content_type: str | None = None
    size_bytes: int = Field(..., ge=0)
    doc_type: str | None = None
    description: str | None = None
    status: str
    created_at: datetime


class FinancialReportResponse(UploadedDocumentResponse):
    report_period: str | None = None
    declared_revenue: float | None = None
    declared_expense: float | None = None
    declared_monthly_income: float | None = None


class CollateralResponse(UploadedDocumentResponse):
    collateral_type: str | None = None
    estimated_value: float | None = None
    currency: str = "VND"


class CheckIssue(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "critical"] = "warning"


class CheckLegalDocsRequest(BaseModel):
    required_doc_types: list[str] = Field(
        default_factory=lambda: ["identity", "income", "residence"]
    )
    notes: str | None = None


class CheckRequest(BaseModel):
    notes: str | None = None


class CheckResultResponse(BaseModel):
    id: str
    loan_profile_id: str
    check_type: str
    status: CheckStatus
    score: float = Field(..., ge=0, le=100)
    issues: list[CheckIssue] = Field(default_factory=list)
    recommendation: str
    notes: str | None = None
    created_at: datetime


class ComplianceResultRequest(BaseModel):
    decision: ComplianceDecision
    score: float | None = Field(default=None, ge=0, le=100)
    notes: str | None = None
    conditions: list[str] = Field(default_factory=list)


class ComplianceResultResponse(BaseModel):
    id: str
    loan_profile_id: str
    decision: ComplianceDecision
    score: float | None = None
    notes: str | None = None
    conditions: list[str] = Field(default_factory=list)
    created_at: datetime


class CaseStatusUpdateRequest(BaseModel):
    status: str = Field(..., min_length=1)
    notes: str | None = None


class CaseStatusResponse(BaseModel):
    loan_profile_id: str
    status: str
    notes: str | None = None
    updated_at: datetime


class ChecklistItemInput(BaseModel):
    title: str = Field(..., min_length=1)
    status: str = "pending"
    owner_agent: str | None = None


class ChecklistCreateRequest(BaseModel):
    items: list[ChecklistItemInput] = Field(default_factory=list)
    notes: str | None = None


class ChecklistItemResponse(BaseModel):
    id: str
    title: str
    status: str
    owner_agent: str | None = None


class ChecklistResponse(BaseModel):
    id: str
    loan_profile_id: str
    items: list[ChecklistItemResponse]
    status: str
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


class LoanLimitCalculationRequest(BaseModel):
    monthly_income: float | None = Field(default=None, ge=0)
    monthly_debt: float | None = Field(default=None, ge=0)
    collateral_value: float | None = Field(default=None, ge=0)
    debt_to_income_ratio: float = Field(default=0.4, gt=0, le=1)
    collateral_ltv_ratio: float = Field(default=0.7, gt=0, le=1)


class LoanLimitCalculationResponse(BaseModel):
    id: str
    loan_profile_id: str
    requested_amount: float
    calculated_limit: float
    recommended_limit: float
    currency: str
    assumptions: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class TaskCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)
    description: str | None = None
    assignee_agent: str | None = None
    priority: TaskPriority = "medium"
    due_date: str | None = None


class TaskResponse(BaseModel):
    id: str
    loan_profile_id: str
    title: str
    description: str | None = None
    assignee_agent: str | None = None
    priority: TaskPriority
    due_date: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime


class ReportCreateRequest(BaseModel):
    report_type: str = Field(default="case_summary", min_length=1)
    title: str | None = None
    summary: str | None = None


class ReportResponse(BaseModel):
    id: str
    loan_profile_id: str
    report_type: str
    title: str
    summary: str
    report_body: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ReportsListResponse(BaseModel):
    total_count: int = Field(..., ge=0)
    reports: list[ReportResponse] = Field(default_factory=list)
