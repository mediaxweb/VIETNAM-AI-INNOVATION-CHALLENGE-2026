import asyncio
import json

import pytest
from agents import Agent
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.tool import ToolOrigin, ToolOriginType
from agents.tool_context import ToolContext
from openai.types.responses import ResponseFunctionToolCall
from pydantic import ValidationError

from credit_agent import (
    CreditDecisionExecution,
    LOAN_APPLICATION_ADAPTER,
    CreditRAGRunHooks,
    CreditDecisionDraft,
    CreditFinding,
    KnowledgeEvidence,
    assemble_credit_assessment,
    build_mcp_server,
    calculate_credit_metrics,
    extract_trusted_evidence,
    fail_closed_assessment,
    run_credit_assessment,
    validate_search_knowledge_call,
)


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


def personal_application():
    return LOAN_APPLICATION_ADAPTER.validate_python(
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


def valid_draft():
    return CreditDecisionDraft(
        risk_level="low",
        recommendation="proceed_to_manual_review",
        findings=[
            CreditFinding(
                summary="DTI nằm trong phạm vi chính sách được truy xuất.",
                severity="info",
                evidence_ids=["credit-policy-1"],
            )
        ],
        missing_data=[],
        evidence=[
            KnowledgeEvidence(
                source_id="credit-policy-1",
                file_name="credit-policy.pdf",
                page="12",
                excerpt="Chính sách đánh giá tỷ lệ nghĩa vụ nợ trên thu nhập.",
            )
        ],
    )


def valid_execution(draft=None):
    trusted_evidence = valid_draft().evidence
    return CreditDecisionExecution(
        draft=draft or valid_draft(),
        trusted_evidence=trusted_evidence,
    )


def rag_run_items(
    *,
    arguments=None,
    output=None,
    origin=None,
):
    agent = Agent(name="Test Agent")
    tool_origin = origin or ToolOrigin(
        type=ToolOriginType.MCP,
        mcp_server_name="credit-rag",
    )
    call = ToolCallItem(
        agent=agent,
        raw_item=ResponseFunctionToolCall(
            arguments=json.dumps(
                arguments
                or {"domain": "credit", "query": "personal loan DTI", "top_k": 5}
            ),
            call_id="call-1",
            name="search_knowledge",
            type="function_call",
        ),
        tool_origin=tool_origin,
    )
    evidence_payload = [item.model_dump(mode="json") for item in valid_draft().evidence]
    tool_output = ToolCallOutputItem(
        agent=agent,
        raw_item={"type": "function_call_output", "call_id": "call-1"},
        output=(
            output
            if output is not None
            else {"type": "text", "text": json.dumps({"evidence": evidence_payload})}
        ),
        tool_origin=tool_origin,
    )
    return [call, tool_output]


def test_assessment_keeps_deterministic_metrics_and_valid_sources():
    application = personal_application()
    metrics, missing_data = calculate_credit_metrics(application)
    draft = valid_draft()

    result = assemble_credit_assessment(
        application,
        metrics,
        missing_data,
        draft,
        draft.evidence,
    )

    assert result.case_id == "PERSONAL-001"
    assert result.risk_level == "low"
    assert {metric.name: metric.value for metric in result.metrics} == {
        "dti": "0.3000",
        "ltv": "0.8000",
    }
    assert result.findings[0].evidence_ids == ["credit-policy-1"]


def test_unknown_evidence_reference_is_rejected():
    application = personal_application()
    metrics, missing_data = calculate_credit_metrics(application)
    draft = valid_draft()
    trusted_evidence = valid_draft().evidence
    draft.findings[0].evidence_ids = ["missing-source"]

    with pytest.raises(ValueError, match="Unknown evidence ids: missing-source"):
        assemble_credit_assessment(
            application,
            metrics,
            missing_data,
            draft,
            trusted_evidence,
        )


def test_empty_rag_evidence_fails_closed():
    application = personal_application()
    metrics, missing_data = calculate_credit_metrics(application)
    draft = valid_draft()
    draft.evidence = []
    draft.findings = []

    result = assemble_credit_assessment(
        application,
        metrics,
        missing_data,
        draft,
        [],
    )

    assert result.risk_level == "undetermined"
    assert result.recommendation == "request_more_information"
    assert result.missing_data == ["rag_evidence"]


def test_missing_financial_data_fails_closed():
    application = LOAN_APPLICATION_ADAPTER.validate_python(
        {
            "case_id": "PERSONAL-004",
            "loan_type": "personal",
            "requested_amount": "100000000",
            "term_months": 12,
            "purpose": "Tiêu dùng",
            "monthly_debt_payment": "5000000",
        }
    )
    metrics, missing_data = calculate_credit_metrics(application)

    result = fail_closed_assessment(application, metrics, missing_data)

    assert result.risk_level == "undetermined"
    assert result.recommendation == "request_more_information"
    assert result.missing_data == ["monthly_income"]


def test_runner_uses_executor_and_returns_assessment():
    application = personal_application()
    calls = []

    async def fake_executor(received_application, metrics, mcp_url, model):
        calls.append((received_application.case_id, mcp_url, model))
        return valid_execution()

    result = asyncio.run(
        run_credit_assessment(
            application,
            mcp_url="http://rag.test/mcp",
            model="test-model",
            decision_executor=fake_executor,
        )
    )

    assert calls == [("PERSONAL-001", "http://rag.test/mcp", "test-model")]
    assert result.risk_level == "low"


def test_runner_skips_executor_when_required_financial_data_is_missing():
    application = LOAN_APPLICATION_ADAPTER.validate_python(
        {
            "case_id": "PERSONAL-005",
            "loan_type": "personal",
            "requested_amount": "100000000",
            "term_months": 12,
            "purpose": "Tiêu dùng",
            "monthly_debt_payment": "5000000",
        }
    )

    async def executor_must_not_run(*args):
        raise AssertionError("executor should not run")

    result = asyncio.run(
        run_credit_assessment(application, decision_executor=executor_must_not_run)
    )

    assert result.risk_level == "undetermined"
    assert result.missing_data == ["monthly_income"]


def test_runner_fails_closed_when_runtime_raises(caplog):
    application = personal_application()

    async def failing_executor(*args):
        raise ConnectionError("RAG unavailable")

    result = asyncio.run(
        run_credit_assessment(application, decision_executor=failing_executor)
    )

    assert result.risk_level == "undetermined"
    assert result.recommendation == "request_more_information"
    assert result.missing_data == ["rag_or_agent_runtime"]
    assert "Credit assessment runtime/provenance failure" in caplog.text


def test_mcp_server_allowlist_exposes_only_search_knowledge():
    server = build_mcp_server("http://rag.test/mcp")

    assert server.tool_filter == {"allowed_tool_names": ["search_knowledge"]}


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("delete_loan", {"domain": "credit", "query": "DTI", "top_k": 5}),
        ("search_knowledge", {"domain": "hr", "query": "DTI", "top_k": 5}),
        ("search_knowledge", {"domain": "credit", "query": "DTI"}),
        ("search_knowledge", {"domain": "credit", "query": "DTI", "top_k": 4}),
        ("search_knowledge", {"domain": "credit", "query": "  ", "top_k": 5}),
        (
            "search_knowledge",
            {"domain": "credit", "query": "DTI", "top_k": 5, "extra": True},
        ),
    ],
)
def test_search_knowledge_contract_rejects_invalid_calls(tool_name, arguments):
    with pytest.raises(ValueError):
        validate_search_knowledge_call(tool_name, json.dumps(arguments))


def test_search_knowledge_contract_accepts_exact_arguments():
    validate_search_knowledge_call(
        "search_knowledge",
        json.dumps({"domain": "credit", "query": "DTI policy", "top_k": 5}),
    )


def test_run_hook_validates_arguments_before_tool_invocation():
    context = ToolContext(
        None,
        tool_name="search_knowledge",
        tool_call_id="call-1",
        tool_arguments=json.dumps({"domain": "credit", "query": "DTI", "top_k": 4}),
    )

    with pytest.raises(ValueError):
        asyncio.run(CreditRAGRunHooks().on_tool_start(context, Agent(name="test"), object()))


def test_extracts_trusted_evidence_from_credit_rag_call_and_output():
    evidence = extract_trusted_evidence(rag_run_items())

    assert evidence == valid_draft().evidence


def test_extracts_trusted_evidence_from_direct_list_output():
    payload = [item.model_dump(mode="json") for item in valid_draft().evidence]

    evidence = extract_trusted_evidence(rag_run_items(output=json.dumps(payload)))

    assert evidence == valid_draft().evidence


def test_evidence_extraction_rejects_missing_call():
    items = rag_run_items()

    with pytest.raises(ValueError, match="call"):
        extract_trusted_evidence([items[1]])


@pytest.mark.parametrize(
    "origin",
    [
        ToolOrigin(type=ToolOriginType.FUNCTION),
        ToolOrigin(type=ToolOriginType.MCP, mcp_server_name="other-rag"),
    ],
)
def test_evidence_extraction_rejects_non_credit_rag_origin(origin):
    with pytest.raises(ValueError, match="credit-rag"):
        extract_trusted_evidence(rag_run_items(origin=origin))


def test_evidence_extraction_rejects_invalid_arguments():
    items = rag_run_items(
        arguments={"domain": "credit", "query": "DTI", "top_k": 4},
    )

    with pytest.raises(ValueError):
        extract_trusted_evidence(items)


def test_evidence_extraction_rejects_malformed_output():
    with pytest.raises(ValueError):
        extract_trusted_evidence(rag_run_items(output="not json"))


def test_evidence_extraction_rejects_conflicting_source_ids():
    evidence = valid_draft().evidence[0].model_dump(mode="json")
    conflicting = {**evidence, "excerpt": "Altered excerpt"}

    with pytest.raises(ValueError, match="Duplicate evidence source_id"):
        extract_trusted_evidence(rag_run_items(output=json.dumps([evidence, conflicting])))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source_id", "fabricated-source"),
        ("excerpt", "Altered excerpt"),
    ],
)
def test_runner_fails_closed_for_fabricated_or_altered_model_evidence(field_name, value):
    draft = valid_draft()
    setattr(draft.evidence[0], field_name, value)

    async def fake_executor(*args):
        return valid_execution(draft)

    result = asyncio.run(
        run_credit_assessment(personal_application(), decision_executor=fake_executor)
    )

    assert result.risk_level == "undetermined"
    assert result.missing_data == ["rag_or_agent_runtime"]


def test_runner_fails_closed_for_finding_without_trusted_evidence():
    draft = valid_draft()
    draft.findings[0].evidence_ids = ["fabricated-source"]

    async def fake_executor(*args):
        return valid_execution(draft)

    result = asyncio.run(
        run_credit_assessment(personal_application(), decision_executor=fake_executor)
    )

    assert result.risk_level == "undetermined"
    assert result.missing_data == ["rag_or_agent_runtime"]


@pytest.mark.parametrize(
    "draft",
    [
        CreditDecisionDraft(
            risk_level="undetermined",
            recommendation="proceed_to_manual_review",
        ),
        CreditDecisionDraft(
            risk_level="undetermined",
            recommendation="request_more_information",
            findings=valid_draft().findings,
            evidence=valid_draft().evidence,
        ),
        CreditDecisionDraft(
            risk_level="low",
            recommendation="request_more_information",
            findings=valid_draft().findings,
            evidence=valid_draft().evidence,
        ),
        CreditDecisionDraft(
            risk_level="low",
            recommendation="proceed_to_manual_review",
            evidence=valid_draft().evidence,
        ),
    ],
)
def test_runner_fails_closed_for_contradictory_draft(draft):
    async def fake_executor(*args):
        return valid_execution(draft)

    result = asyncio.run(
        run_credit_assessment(personal_application(), decision_executor=fake_executor)
    )

    assert result.risk_level == "undetermined"
    assert result.missing_data == ["rag_or_agent_runtime"]


def test_consistent_undetermined_draft_stays_fail_closed():
    application = personal_application()
    metrics, missing_data = calculate_credit_metrics(application)
    draft = CreditDecisionDraft(
        risk_level="undetermined",
        recommendation="request_more_information",
    )

    result = assemble_credit_assessment(
        application,
        metrics,
        missing_data,
        draft,
        valid_draft().evidence,
    )

    assert result.risk_level == "undetermined"
    assert result.missing_data == ["agent_undetermined"]


def extreme_decimal_application():
    return LOAN_APPLICATION_ADAPTER.validate_python(
        {
            "case_id": "PERSONAL-EXTREME",
            "loan_type": "personal",
            "requested_amount": "1",
            "term_months": 12,
            "purpose": "Decimal boundary",
            "monthly_income": "1",
            "monthly_debt_payment": "1e999999",
        }
    )


def test_extreme_decimal_metric_is_undefined_instead_of_raising():
    metrics, missing_data = calculate_credit_metrics(extreme_decimal_application())

    assert metrics[0].value is None
    assert metrics[0].reason == "Calculation failed"
    assert missing_data == []


def test_runner_skips_executor_for_undefined_metric():
    async def executor_must_not_run(*args):
        raise AssertionError("executor should not run")

    result = asyncio.run(
        run_credit_assessment(
            extreme_decimal_application(),
            decision_executor=executor_must_not_run,
        )
    )

    assert result.risk_level == "undetermined"
    assert result.missing_data == ["invalid_financial_metrics"]
