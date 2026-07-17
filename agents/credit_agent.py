from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


RATIO_QUANTUM = Decimal("0.0001")
MetricName = Literal["dti", "ltv", "dscr", "debt_to_equity", "current_ratio"]


class PersonalLoanApplication(BaseModel):
    case_id: str = Field(min_length=1)
    loan_type: Literal["personal"]
    requested_amount: Decimal = Field(gt=0)
    term_months: int = Field(gt=0)
    purpose: str = Field(min_length=1)
    monthly_income: Decimal | None = Field(default=None, ge=0)
    monthly_debt_payment: Decimal | None = Field(default=None, ge=0)
    collateral_value: Decimal | None = Field(default=None, gt=0)


class SMELoanApplication(BaseModel):
    case_id: str = Field(min_length=1)
    loan_type: Literal["sme"]
    requested_amount: Decimal = Field(gt=0)
    term_months: int = Field(gt=0)
    purpose: str = Field(min_length=1)
    annual_net_operating_income: Decimal | None = None
    annual_debt_service: Decimal | None = Field(default=None, ge=0)
    total_debt: Decimal | None = Field(default=None, ge=0)
    equity: Decimal | None = None
    current_assets: Decimal | None = Field(default=None, ge=0)
    current_liabilities: Decimal | None = Field(default=None, ge=0)
    collateral_value: Decimal | None = Field(default=None, gt=0)


LoanApplication = Annotated[
    PersonalLoanApplication | SMELoanApplication,
    Field(discriminator="loan_type"),
]
LOAN_APPLICATION_ADAPTER = TypeAdapter(LoanApplication)


class MetricResult(BaseModel):
    name: MetricName
    value: str | None
    reason: str | None = None


def _ratio(
    name: MetricName,
    numerator: Decimal | None,
    numerator_field: str,
    denominator: Decimal | None,
    denominator_field: str,
) -> tuple[MetricResult, list[str]]:
    missing = [
        field_name
        for value, field_name in (
            (numerator, numerator_field),
            (denominator, denominator_field),
        )
        if value is None
    ]
    if missing:
        return MetricResult(name=name, value=None, reason=f"Missing: {', '.join(missing)}"), missing
    assert numerator is not None
    assert denominator is not None
    if denominator <= 0:
        return (
            MetricResult(
                name=name,
                value=None,
                reason=f"{denominator_field} must be greater than 0",
            ),
            [],
        )
    value = (numerator / denominator).quantize(RATIO_QUANTUM, rounding=ROUND_HALF_UP)
    return MetricResult(name=name, value=str(value)), []


def calculate_credit_metrics(
    application: LoanApplication,
) -> tuple[list[MetricResult], list[str]]:
    metrics: list[MetricResult] = []
    missing_data: list[str] = []

    if isinstance(application, PersonalLoanApplication):
        metric, missing = _ratio(
            "dti",
            application.monthly_debt_payment,
            "monthly_debt_payment",
            application.monthly_income,
            "monthly_income",
        )
        metrics.append(metric)
        missing_data.extend(missing)
    else:
        for arguments in (
            (
                "dscr",
                application.annual_net_operating_income,
                "annual_net_operating_income",
                application.annual_debt_service,
                "annual_debt_service",
            ),
            (
                "debt_to_equity",
                application.total_debt,
                "total_debt",
                application.equity,
                "equity",
            ),
            (
                "current_ratio",
                application.current_assets,
                "current_assets",
                application.current_liabilities,
                "current_liabilities",
            ),
        ):
            metric, missing = _ratio(*arguments)
            metrics.append(metric)
            missing_data.extend(missing)

    if application.collateral_value is not None:
        metric, missing = _ratio(
            "ltv",
            application.requested_amount,
            "requested_amount",
            application.collateral_value,
            "collateral_value",
        )
        metrics.append(metric)
        missing_data.extend(missing)

    return metrics, sorted(set(missing_data))


RiskLevel = Literal["low", "medium", "high", "undetermined"]
Recommendation = Literal[
    "proceed_to_manual_review",
    "request_more_information",
    "escalate_high_risk_review",
]


class KnowledgeEvidence(BaseModel):
    source_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    page: str | None = None
    excerpt: str = Field(min_length=1)


class CreditFinding(BaseModel):
    summary: str = Field(min_length=1)
    severity: Literal["info", "warning", "critical"]
    evidence_ids: list[str] = Field(min_length=1)


class CreditDecisionDraft(BaseModel):
    risk_level: RiskLevel
    recommendation: Recommendation
    findings: list[CreditFinding] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    evidence: list[KnowledgeEvidence] = Field(default_factory=list)


class CreditAssessment(BaseModel):
    case_id: str
    loan_type: Literal["personal", "sme"]
    risk_level: RiskLevel
    recommendation: Recommendation
    metrics: list[MetricResult]
    findings: list[CreditFinding]
    missing_data: list[str]
    evidence: list[KnowledgeEvidence]


def fail_closed_assessment(
    application: LoanApplication,
    metrics: list[MetricResult],
    missing_data: list[str],
) -> CreditAssessment:
    return CreditAssessment(
        case_id=application.case_id,
        loan_type=application.loan_type,
        risk_level="undetermined",
        recommendation="request_more_information",
        metrics=metrics,
        findings=[],
        missing_data=sorted(set(missing_data)),
        evidence=[],
    )


def assemble_credit_assessment(
    application: LoanApplication,
    metrics: list[MetricResult],
    missing_data: list[str],
    draft: CreditDecisionDraft,
) -> CreditAssessment:
    combined_missing_data = sorted(set([*missing_data, *draft.missing_data]))
    if combined_missing_data:
        return fail_closed_assessment(application, metrics, combined_missing_data)
    if not draft.evidence:
        return fail_closed_assessment(application, metrics, ["rag_evidence"])

    available_ids = {item.source_id for item in draft.evidence}
    referenced_ids = {
        evidence_id
        for finding in draft.findings
        for evidence_id in finding.evidence_ids
    }
    unknown_ids = sorted(referenced_ids - available_ids)
    if unknown_ids:
        raise ValueError(f"Unknown evidence ids: {', '.join(unknown_ids)}")

    return CreditAssessment(
        case_id=application.case_id,
        loan_type=application.loan_type,
        risk_level=draft.risk_level,
        recommendation=draft.recommendation,
        metrics=metrics,
        findings=draft.findings,
        missing_data=[],
        evidence=draft.evidence,
    )
