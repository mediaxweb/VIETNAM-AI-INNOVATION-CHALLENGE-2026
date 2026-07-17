from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from decimal import Decimal, DecimalException, ROUND_HALF_UP
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from agents import Agent, Runner
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.lifecycle import RunHooksBase
from agents.mcp import MCPServerStreamableHttp
from agents.tool import FunctionTool, ToolOriginType, get_function_tool_origin
from agents.tool_context import ToolContext
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


RATIO_QUANTUM = Decimal("0.0001")
DEFAULT_RAG_MCP_URL = "http://127.0.0.1:8766/mcp"
DEFAULT_MODEL = "gpt-5.4-mini"
CREDIT_MCP_TOOL_NAMES = [
    "search_knowledge",
    "get_document_page",
    "get_loan_profile",
    "get_customer",
    "list_reports",
]
MetricName = Literal["dti", "ltv", "dscr", "debt_to_equity", "current_ratio"]
logger = logging.getLogger(__name__)


class PersonalLoanApplication(BaseModel):
    case_id: str = Field(min_length=1)
    loan_profile_id: str | None = Field(default=None, min_length=1)
    loan_type: Literal["personal"]
    requested_amount: Decimal = Field(gt=0)
    term_months: int = Field(gt=0)
    purpose: str = Field(min_length=1)
    monthly_income: Decimal | None = Field(default=None, ge=0)
    monthly_debt_payment: Decimal | None = Field(default=None, ge=0)
    collateral_value: Decimal | None = Field(default=None, gt=0)


class SMELoanApplication(BaseModel):
    case_id: str = Field(min_length=1)
    loan_profile_id: str | None = Field(default=None, min_length=1)
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
    try:
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
    except DecimalException:
        return MetricResult(name=name, value=None, reason="Calculation failed"), []
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
    model_config = ConfigDict(extra="forbid")

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


class CreditDecisionExecution(BaseModel):
    draft: CreditDecisionDraft
    trusted_evidence: list[KnowledgeEvidence]


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
    trusted_evidence: list[KnowledgeEvidence],
) -> CreditAssessment:
    if not trusted_evidence:
        return fail_closed_assessment(application, metrics, ["rag_evidence"])

    trusted_by_id = _evidence_by_id(trusted_evidence)
    draft_by_id = _evidence_by_id(draft.evidence)
    for source_id, item in draft_by_id.items():
        if trusted_by_id.get(source_id) != item:
            raise ValueError(f"Untrusted model evidence: {source_id}")

    referenced_ids = {
        evidence_id
        for finding in draft.findings
        for evidence_id in finding.evidence_ids
    }
    unknown_ids = sorted(referenced_ids - trusted_by_id.keys())
    if unknown_ids:
        raise ValueError(f"Unknown evidence ids: {', '.join(unknown_ids)}")

    if draft.risk_level == "undetermined":
        if draft.recommendation != "request_more_information" or draft.findings:
            raise ValueError("Contradictory undetermined credit decision")
    elif draft.recommendation == "request_more_information" or not draft.findings:
        raise ValueError("Contradictory determinate credit decision")

    combined_missing_data = sorted(set([*missing_data, *draft.missing_data]))
    if combined_missing_data:
        return fail_closed_assessment(application, metrics, combined_missing_data)
    if draft.risk_level == "undetermined":
        return fail_closed_assessment(application, metrics, ["agent_undetermined"])

    return CreditAssessment(
        case_id=application.case_id,
        loan_type=application.loan_type,
        risk_level=draft.risk_level,
        recommendation=draft.recommendation,
        metrics=metrics,
        findings=draft.findings,
        missing_data=[],
        evidence=trusted_evidence,
    )


DecisionExecutor: TypeAlias = Callable[
    [LoanApplication, list[MetricResult], str, str],
    Awaitable[CreditDecisionExecution],
]


def _evidence_by_id(
    evidence: list[KnowledgeEvidence],
) -> dict[str, KnowledgeEvidence]:
    by_id: dict[str, KnowledgeEvidence] = {}
    for item in evidence:
        if item.source_id in by_id:
            raise ValueError(f"Duplicate evidence source_id: {item.source_id}")
        by_id[item.source_id] = item
    return by_id


def validate_search_knowledge_call(tool_name: str, raw_arguments: str) -> dict[str, Any]:
    if tool_name != "search_knowledge":
        raise ValueError("Only search_knowledge may be invoked")
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("search_knowledge arguments must be valid JSON") from error
    if not isinstance(arguments, dict) or set(arguments) != {"domain", "query", "top_k"}:
        raise ValueError("search_knowledge requires exactly domain, query, and top_k")
    if arguments["domain"] != "credit":
        raise ValueError("search_knowledge domain must be credit")
    if not isinstance(arguments["query"], str) or not arguments["query"].strip():
        raise ValueError("search_knowledge query must be a non-empty string")
    if type(arguments["top_k"]) is not int or arguments["top_k"] != 5:
        raise ValueError("search_knowledge top_k must be 5")
    return arguments


def validate_document_page_call(tool_name: str, raw_arguments: str) -> dict[str, Any]:
    if tool_name != "get_document_page":
        raise ValueError("Only get_document_page may be invoked")
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("get_document_page arguments must be valid JSON") from error
    if not isinstance(arguments, dict) or set(arguments) != {"domain", "source_id"}:
        raise ValueError("get_document_page requires exactly domain and source_id")
    if arguments["domain"] != "credit":
        raise ValueError("get_document_page domain must be credit")
    if not isinstance(arguments["source_id"], str) or not arguments["source_id"].strip():
        raise ValueError("get_document_page source_id must be a non-empty string")
    return arguments


def validate_loan_data_call(tool_name: str, raw_arguments: str) -> dict[str, Any]:
    id_fields = {
        "get_loan_profile": "loan_profile_id",
        "get_customer": "customer_id",
        "list_reports": "loan_profile_id",
    }
    field_name = id_fields.get(tool_name)
    if field_name is None:
        raise ValueError("Unsupported loan data tool")
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{tool_name} arguments must be valid JSON") from error
    if not isinstance(arguments, dict) or set(arguments) != {field_name}:
        raise ValueError(f"{tool_name} requires exactly {field_name}")
    if not isinstance(arguments[field_name], str) or not arguments[field_name].strip():
        raise ValueError(f"{tool_name} {field_name} must be a non-empty string")
    return arguments


def _validate_credit_tool_call(tool_name: str, raw_arguments: str) -> dict[str, Any]:
    if tool_name == "search_knowledge":
        return validate_search_knowledge_call(tool_name, raw_arguments)
    if tool_name == "get_document_page":
        return validate_document_page_call(tool_name, raw_arguments)
    return validate_loan_data_call(tool_name, raw_arguments)


class CreditRAGRunHooks(RunHooksBase):
    async def on_tool_start(self, context, agent, tool) -> None:
        if not isinstance(context, ToolContext):
            raise ValueError("Credit Agent tool calls require ToolContext")
        _validate_credit_tool_call(context.tool_name, context.tool_arguments)
        if not isinstance(tool, FunctionTool):
            raise ValueError("Credit Agent may only invoke the credit-rag MCP tool")
        origin = get_function_tool_origin(tool)
        if (
            origin is None
            or origin.type != ToolOriginType.MCP
            or origin.mcp_server_name != "credit-rag"
        ):
            raise ValueError("Credit Agent may only invoke the credit-rag MCP tool")


def _parse_evidence_output(output: Any) -> list[KnowledgeEvidence]:
    payload = output
    if isinstance(payload, dict) and payload.get("type") == "text":
        payload = payload.get("text")
    elif (
        isinstance(payload, list)
        and len(payload) == 1
        and isinstance(payload[0], dict)
        and payload[0].get("type") == "text"
    ):
        payload = payload[0].get("text")

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("search_knowledge returned invalid JSON") from error
    if isinstance(payload, dict):
        if set(payload) != {"evidence"}:
            raise ValueError("search_knowledge output envelope must contain only evidence")
        payload = payload["evidence"]
    if not isinstance(payload, list):
        raise ValueError("search_knowledge output must be an evidence list")

    evidence = TypeAdapter(list[KnowledgeEvidence]).validate_python(payload)
    _evidence_by_id(evidence)
    return evidence


def _require_credit_rag_origin(item: ToolCallItem | ToolCallOutputItem) -> None:
    origin = item.tool_origin
    if (
        origin is None
        or origin.type != ToolOriginType.MCP
        or origin.mcp_server_name != "credit-rag"
    ):
        raise ValueError("Tool item did not originate from credit-rag MCP")


def extract_trusted_evidence(new_items: list[Any]) -> list[KnowledgeEvidence]:
    calls: dict[str, tuple[ToolCallItem, dict[str, Any]]] = {}
    outputs: dict[str, ToolCallOutputItem] = {}
    for item in new_items:
        if isinstance(item, ToolCallItem):
            _require_credit_rag_origin(item)
            call_id = item.call_id
            raw_arguments = (
                item.raw_item.get("arguments")
                if isinstance(item.raw_item, dict)
                else getattr(item.raw_item, "arguments", None)
            )
            if not call_id or not isinstance(raw_arguments, str):
                raise ValueError("credit-rag tool call is missing its call ID or arguments")
            if call_id in calls:
                raise ValueError(f"Duplicate credit-rag call ID: {call_id}")
            arguments = _validate_credit_tool_call(item.tool_name or "", raw_arguments)
            calls[call_id] = (item, arguments)
        elif isinstance(item, ToolCallOutputItem):
            _require_credit_rag_origin(item)
            call_id = item.call_id
            if not call_id:
                raise ValueError("credit-rag tool output is missing its call ID")
            if call_id in outputs:
                raise ValueError(f"Duplicate credit-rag output ID: {call_id}")
            outputs[call_id] = item

    if not calls:
        raise ValueError("No credit-rag search_knowledge call found")

    trusted_evidence: list[KnowledgeEvidence] = []
    search_evidence: list[KnowledgeEvidence] = []
    for call_id, (call, arguments) in calls.items():
        output = outputs.get(call_id)
        if output is None:
            raise ValueError(f"Missing credit-rag output for call: {call_id}")
        if call.tool_name in {"get_loan_profile", "get_customer", "list_reports"}:
            continue
        tool_evidence = _parse_evidence_output(output.output)
        if call.tool_name == "search_knowledge":
            search_evidence.extend(tool_evidence)
        else:
            search_item = next(
                (
                    item
                    for item in search_evidence
                    if item.source_id == arguments["source_id"]
                ),
                None,
            )
            if search_item is None:
                raise ValueError("get_document_page requires prior search evidence")
            if len(tool_evidence) != 1:
                raise ValueError("get_document_page must return exactly one evidence item")
            page_evidence = tool_evidence[0]
            if (
                page_evidence.file_name,
                page_evidence.page,
                page_evidence.source_id,
            ) != (
                search_item.file_name,
                search_item.page,
                f"page:{arguments['source_id']}",
            ):
                raise ValueError("get_document_page returned a different requested document page")
        trusted_evidence.extend(tool_evidence)
    if set(outputs) != set(calls):
        raise ValueError("credit-rag output does not match a tool call")
    if not search_evidence:
        raise ValueError("No credit-rag search_knowledge call found")
    if not trusted_evidence:
        raise ValueError("credit-rag returned no evidence")
    _evidence_by_id(trusted_evidence)
    return trusted_evidence


def build_credit_agent(server: MCPServerStreamableHttp, model: str) -> Agent:
    return Agent(
        name="Credit Agent",
        instructions=(
            "Assess one personal or SME loan application. "
            "When loan_profile_id is supplied, use get_loan_profile to read the persisted "
            "case; you may then use its customer_id with get_customer and use list_reports "
            "for existing case history. Treat loan data tool results as supplemental context "
            "and never create or update records. "
            "Before making any policy finding, call search_knowledge with "
            "domain='credit' and top_k=5. If a returned excerpt lacks enough context, "
            "call get_document_page with the exact source_id from that search evidence. "
            "Never read a page for a source_id not returned by search_knowledge. "
            "Use only evidence returned by these tools. "
            "Copy source_id, file_name, page, and excerpt into evidence, and make every "
            "finding reference existing evidence_ids. Treat supplied metrics as immutable. "
            "Never approve, reject, or update a loan. Return undetermined and request more "
            "information when data or evidence is insufficient."
        ),
        model=model,
        mcp_servers=[server],
        output_type=CreditDecisionDraft,
    )


def build_agent_input(
    application: LoanApplication,
    metrics: list[MetricResult],
) -> str:
    return json.dumps(
        {
            "application": application.model_dump(mode="json"),
            "metrics": [metric.model_dump(mode="json") for metric in metrics],
        },
        ensure_ascii=False,
    )


def build_mcp_server(mcp_url: str) -> MCPServerStreamableHttp:
    return MCPServerStreamableHttp(
        params={
            "url": mcp_url,
            "timeout": 30,
            "sse_read_timeout": 30,
        },
        name="credit-rag",
        cache_tools_list=True,
        client_session_timeout_seconds=30,
        tool_filter={
            "allowed_tool_names": CREDIT_MCP_TOOL_NAMES
        },
        use_structured_content=True,
    )


async def execute_credit_decision(
    application: LoanApplication,
    metrics: list[MetricResult],
    mcp_url: str,
    model: str,
) -> CreditDecisionExecution:
    async with build_mcp_server(mcp_url) as server:
        tools = await server.list_tools()
        if {tool.name for tool in tools} != set(CREDIT_MCP_TOOL_NAMES):
            raise RuntimeError(
                "Credit MCP server exposes an unexpected tool set"
            )
        agent = build_credit_agent(server, model)
        result = await Runner.run(
            agent,
            build_agent_input(application, metrics),
            hooks=CreditRAGRunHooks(),
        )

    if not isinstance(result.final_output, CreditDecisionDraft):
        raise TypeError("Credit Agent returned an invalid structured output")
    return CreditDecisionExecution(
        draft=result.final_output,
        trusted_evidence=extract_trusted_evidence(result.new_items),
    )


async def run_credit_assessment(
    application: LoanApplication,
    *,
    mcp_url: str = DEFAULT_RAG_MCP_URL,
    model: str = DEFAULT_MODEL,
    decision_executor: DecisionExecutor | None = None,
) -> CreditAssessment:
    metrics, missing_data = calculate_credit_metrics(application)
    if missing_data:
        return fail_closed_assessment(application, metrics, missing_data)
    if any(metric.value is None for metric in metrics):
        logger.warning("Credit assessment failed closed: invalid_financial_metrics")
        return fail_closed_assessment(
            application,
            metrics,
            ["invalid_financial_metrics"],
        )

    executor = decision_executor or execute_credit_decision
    try:
        execution = await executor(application, metrics, mcp_url, model)
        return assemble_credit_assessment(
            application,
            metrics,
            [],
            execution.draft,
            execution.trusted_evidence,
        )
    except Exception as error:
        logger.error(
            "Credit assessment runtime/provenance failure [%s]",
            type(error).__name__,
        )
        return fail_closed_assessment(
            application,
            metrics,
            ["rag_or_agent_runtime"],
        )


def load_application(path: str) -> LoanApplication:
    raw_json = Path(path).read_text(encoding="utf-8")
    return LOAN_APPLICATION_ADAPTER.validate_json(raw_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MediaX Credit Agent.")
    parser.add_argument("--input", required=True, help="Path to a normalized loan JSON file.")
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("RAG_MCP_URL", DEFAULT_RAG_MCP_URL),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_AGENT_MODEL", DEFAULT_MODEL),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")
    application = load_application(args.input)
    assessment = asyncio.run(
        run_credit_assessment(
            application,
            mcp_url=args.mcp_url,
            model=args.model,
        )
    )
    print(assessment.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
