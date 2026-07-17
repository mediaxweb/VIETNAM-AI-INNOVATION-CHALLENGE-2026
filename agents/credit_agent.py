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
