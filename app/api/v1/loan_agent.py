from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.schemas.auth import UserResponse
from app.api.schemas.loan_agent import (
    CaseStatusResponse,
    CaseStatusUpdateRequest,
    ChecklistCreateRequest,
    ChecklistResponse,
    CheckLegalDocsRequest,
    CheckRequest,
    CheckResultResponse,
    CollateralResponse,
    ComplianceResultRequest,
    ComplianceResultResponse,
    CustomerCreateRequest,
    CustomerResponse,
    CustomerUpdateRequest,
    FinancialReportResponse,
    LoanLimitCalculationRequest,
    LoanLimitCalculationResponse,
    LoanProfileCreateRequest,
    LoanProfileResponse,
    ReportCreateRequest,
    ReportResponse,
    ReportsListResponse,
    TaskCreateRequest,
    TaskResponse,
    UploadedDocumentResponse,
)
from app.core.dependencies import get_current_user
from app.services.loan_agent_service import (
    LoanAgentNotFoundError,
    LoanAgentService,
    LoanAgentValidationError,
)
from logs.logging_config import logger


router = APIRouter()


def get_loan_agent_service() -> LoanAgentService:
    """FastAPI dependency that instantiates the loan-agent service."""

    return LoanAgentService()


def _raise_loan_agent_http_error(exc: Exception):
    if isinstance(exc, LoanAgentNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, LoanAgentValidationError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.exception("Unexpected loan-agent API error: %s", exc)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Loan-agent operation failed.",
    ) from exc


@router.post(
    "/customers",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Credit Agent"],
)
async def create_customer(
    payload: CustomerCreateRequest,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> CustomerResponse:
    try:
        return await service.create_customer(user_id=current_user.id, payload=payload)
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.get(
    "/customers/{customer_id}",
    response_model=CustomerResponse,
    tags=["Credit Agent"],
)
async def get_customer(
    customer_id: str,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> CustomerResponse:
    try:
        return await service.get_customer(user_id=current_user.id, customer_id=customer_id)
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.patch(
    "/customers/{customer_id}",
    response_model=CustomerResponse,
    tags=["Credit Agent"],
)
async def update_customer(
    customer_id: str,
    payload: CustomerUpdateRequest,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> CustomerResponse:
    try:
        return await service.update_customer(
            user_id=current_user.id,
            customer_id=customer_id,
            payload=payload,
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles",
    response_model=LoanProfileResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Credit Agent"],
)
async def create_loan_profile(
    payload: LoanProfileCreateRequest,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> LoanProfileResponse:
    try:
        return await service.create_loan_profile(user_id=current_user.id, payload=payload)
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.get(
    "/loan-profiles/{loan_profile_id}",
    response_model=LoanProfileResponse,
    tags=["Compliance Agent"],
)
async def get_loan_profile(
    loan_profile_id: str,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> LoanProfileResponse:
    try:
        return await service.get_loan_profile(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/legal-docs",
    response_model=UploadedDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Credit Agent"],
)
async def upload_legal_doc(
    loan_profile_id: str,
    file: UploadFile = File(...),
    doc_type: str | None = Form(default=None),
    description: str | None = Form(default=None),
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> UploadedDocumentResponse:
    try:
        return await service.upload_legal_doc(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            file=file,
            doc_type=doc_type,
            description=description,
        )
    except Exception as exc:
        await file.close()
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/check-legal-docs",
    response_model=CheckResultResponse,
    tags=["Credit Agent"],
)
async def check_legal_docs(
    loan_profile_id: str,
    payload: CheckLegalDocsRequest | None = None,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> CheckResultResponse:
    try:
        request_payload = payload or CheckLegalDocsRequest()
        return await service.check_legal_docs(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            required_doc_types=request_payload.required_doc_types,
            notes=request_payload.notes,
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/financial-reports",
    response_model=FinancialReportResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Compliance Agent"],
)
async def upload_financial_report(
    loan_profile_id: str,
    file: UploadFile = File(...),
    report_period: str | None = Form(default=None),
    declared_revenue: float | None = Form(default=None),
    declared_expense: float | None = Form(default=None),
    declared_monthly_income: float | None = Form(default=None),
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> FinancialReportResponse:
    try:
        return await service.upload_financial_report(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            file=file,
            report_period=report_period,
            declared_revenue=declared_revenue,
            declared_expense=declared_expense,
            declared_monthly_income=declared_monthly_income,
        )
    except Exception as exc:
        await file.close()
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/collaterals",
    response_model=CollateralResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Compliance Agent"],
)
async def upload_collateral(
    loan_profile_id: str,
    file: UploadFile = File(...),
    collateral_type: str | None = Form(default=None),
    estimated_value: float | None = Form(default=None),
    currency: str = Form(default="VND"),
    description: str | None = Form(default=None),
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> CollateralResponse:
    try:
        return await service.upload_collateral(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            file=file,
            collateral_type=collateral_type,
            estimated_value=estimated_value,
            currency=currency,
            description=description,
        )
    except Exception as exc:
        await file.close()
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/check-financials",
    response_model=CheckResultResponse,
    tags=["Compliance Agent"],
)
async def check_financials(
    loan_profile_id: str,
    payload: CheckRequest | None = None,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> CheckResultResponse:
    try:
        return await service.check_financials(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            notes=payload.notes if payload else None,
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/check-collateral",
    response_model=CheckResultResponse,
    tags=["Compliance Agent"],
)
async def check_collateral(
    loan_profile_id: str,
    payload: CheckRequest | None = None,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> CheckResultResponse:
    try:
        return await service.check_collateral(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            notes=payload.notes if payload else None,
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/check-credit-rule",
    response_model=CheckResultResponse,
    tags=["Compliance Agent"],
)
async def check_credit_rule(
    loan_profile_id: str,
    payload: CheckRequest | None = None,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> CheckResultResponse:
    try:
        return await service.check_credit_rule(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            notes=payload.notes if payload else None,
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/compliance-result",
    response_model=ComplianceResultResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Compliance Agent"],
)
async def save_compliance_result(
    loan_profile_id: str,
    payload: ComplianceResultRequest,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> ComplianceResultResponse:
    try:
        return await service.save_compliance_result(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            payload=payload,
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.patch(
    "/loan-profiles/{loan_profile_id}/status",
    response_model=CaseStatusResponse,
    tags=["Operations Agent"],
)
async def update_case_status(
    loan_profile_id: str,
    payload: CaseStatusUpdateRequest,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> CaseStatusResponse:
    try:
        return await service.update_case_status(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            payload=payload,
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/checklist",
    response_model=ChecklistResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Operations Agent"],
)
async def create_checklist(
    loan_profile_id: str,
    payload: ChecklistCreateRequest | None = None,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> ChecklistResponse:
    try:
        return await service.create_checklist(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            payload=payload or ChecklistCreateRequest(),
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/calculate-limit",
    response_model=LoanLimitCalculationResponse,
    tags=["Operations Agent"],
)
async def calculate_loan_limit(
    loan_profile_id: str,
    payload: LoanLimitCalculationRequest | None = None,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> LoanLimitCalculationResponse:
    try:
        return await service.calculate_loan_limit(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            payload=payload or LoanLimitCalculationRequest(),
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Operations Agent"],
)
async def create_task(
    loan_profile_id: str,
    payload: TaskCreateRequest,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> TaskResponse:
    try:
        return await service.create_task(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            payload=payload,
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.post(
    "/loan-profiles/{loan_profile_id}/reports",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Operations Agent"],
)
async def create_report(
    loan_profile_id: str,
    payload: ReportCreateRequest | None = None,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> ReportResponse:
    try:
        return await service.create_report(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
            payload=payload or ReportCreateRequest(),
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)


@router.get(
    "/loan-profiles/{loan_profile_id}/reports",
    response_model=ReportsListResponse,
    tags=["Operations Agent"],
)
async def list_reports(
    loan_profile_id: str,
    service: LoanAgentService = Depends(get_loan_agent_service),
    current_user: UserResponse = Depends(get_current_user),
) -> ReportsListResponse:
    try:
        return await service.list_reports(
            user_id=current_user.id,
            loan_profile_id=loan_profile_id,
        )
    except Exception as exc:
        _raise_loan_agent_http_error(exc)
