import pytest
from pydantic import ValidationError

from credit_agent import LOAN_APPLICATION_ADAPTER, calculate_credit_metrics


def metric_map(metrics):
    return {metric.name: metric.value for metric in metrics}


def test_personal_application_metrics():
    application = LOAN_APPLICATION_ADAPTER.validate_python(
        {
            "case_id": "PERSONAL-001",
            "loan_type": "personal",
            "requested_amount": "800000000",
            "term_months": 60,
            "purpose": "Mua nhà",
            "monthly_income": "50000000",
            "monthly_debt_payment": "15000000",
            "collateral_value": "1000000000",
        }
    )

    metrics, missing_data = calculate_credit_metrics(application)

    assert metric_map(metrics) == {"dti": "0.3000", "ltv": "0.8000"}
    assert missing_data == []


def test_sme_application_metrics():
    application = LOAN_APPLICATION_ADAPTER.validate_python(
        {
            "case_id": "SME-001",
            "loan_type": "sme",
            "requested_amount": "4000000000",
            "term_months": 48,
            "purpose": "Bổ sung vốn lưu động",
            "annual_net_operating_income": "2400000000",
            "annual_debt_service": "1200000000",
            "total_debt": "3000000000",
            "equity": "2000000000",
            "current_assets": "1500000000",
            "current_liabilities": "1000000000",
            "collateral_value": "5000000000",
        }
    )

    metrics, missing_data = calculate_credit_metrics(application)

    assert metric_map(metrics) == {
        "dscr": "2.0000",
        "debt_to_equity": "1.5000",
        "current_ratio": "1.5000",
        "ltv": "0.8000",
    }
    assert missing_data == []


def test_missing_personal_income_is_reported_without_division():
    application = LOAN_APPLICATION_ADAPTER.validate_python(
        {
            "case_id": "PERSONAL-002",
            "loan_type": "personal",
            "requested_amount": "100000000",
            "term_months": 24,
            "purpose": "Tiêu dùng",
            "monthly_debt_payment": "5000000",
        }
    )

    metrics, missing_data = calculate_credit_metrics(application)

    assert metric_map(metrics) == {"dti": None}
    assert missing_data == ["monthly_income"]
    assert metrics[0].reason == "Missing: monthly_income"


def test_zero_sme_equity_returns_undefined_ratio_not_infinity():
    application = LOAN_APPLICATION_ADAPTER.validate_python(
        {
            "case_id": "SME-002",
            "loan_type": "sme",
            "requested_amount": "500000000",
            "term_months": 12,
            "purpose": "Vốn lưu động",
            "annual_net_operating_income": "100000000",
            "annual_debt_service": "50000000",
            "total_debt": "200000000",
            "equity": "0",
            "current_assets": "300000000",
            "current_liabilities": "150000000",
        }
    )

    metrics, missing_data = calculate_credit_metrics(application)
    by_name = {metric.name: metric for metric in metrics}

    assert by_name["debt_to_equity"].value is None
    assert by_name["debt_to_equity"].reason == "equity must be greater than 0"
    assert missing_data == []


def test_negative_requested_amount_is_rejected():
    with pytest.raises(ValidationError):
        LOAN_APPLICATION_ADAPTER.validate_python(
            {
                "case_id": "PERSONAL-003",
                "loan_type": "personal",
                "requested_amount": "-1",
                "term_months": 12,
                "purpose": "Tiêu dùng",
            }
        )
