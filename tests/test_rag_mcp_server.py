import asyncio
from types import SimpleNamespace

import pytest

from app import rag_mcp_server
from app.rag_mcp_server import retrieve_credit_evidence, retrieve_document_page


class FakeKnowledgeBaseService:
    def __init__(
        self,
        chunks=None,
        error=None,
        chunk_result=None,
        chunk_error=None,
        page_result=None,
        page_error=None,
    ):
        self.chunks = chunks or []
        self.error = error
        self.chunk_result = chunk_result
        self.chunk_error = chunk_error
        self.page_result = page_result
        self.page_error = page_error
        self.calls = []
        self.chunk_calls = []
        self.page_calls = []

    def retrieve_chunks(self, query, conversation_history=None, *, user_id):
        self.calls.append((query, conversation_history, user_id))
        if self.error:
            raise self.error
        return {"question": query, "chunks": self.chunks}

    def get_chunk_detail(self, chunk_id, *, user_id):
        self.chunk_calls.append((chunk_id, user_id))
        if self.chunk_error:
            raise self.chunk_error
        return self.chunk_result

    def get_document_text(self, document_path, *, page_label, user_id):
        self.page_calls.append((document_path, page_label, user_id))
        if self.page_error:
            raise self.page_error
        return self.page_result


class FakeLoanAgentService:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def get_loan_profile(self, *, user_id, loan_profile_id):
        self.calls.append(("get_loan_profile", user_id, loan_profile_id))
        if self.error:
            raise self.error
        return SimpleNamespace(id=loan_profile_id, customer_id="customer-1")

    async def get_customer(self, *, user_id, customer_id):
        self.calls.append(("get_customer", user_id, customer_id))
        if self.error:
            raise self.error
        return SimpleNamespace(id=customer_id, full_name="Customer")

    async def list_reports(self, *, user_id, loan_profile_id):
        self.calls.append(("list_reports", user_id, loan_profile_id))
        if self.error:
            raise self.error
        return SimpleNamespace(total_count=0, reports=[])


def chunk(index, *, window=None, text=None):
    return {
        "chunk_id": f"source-{index}",
        "file_name": f"policy-{index}.pdf",
        "page_label": str(index),
        "window": window,
        "text": text or f"text {index}",
    }


def test_retrieval_maps_five_evidence_items_and_uses_server_user_id():
    service = FakeKnowledgeBaseService(
        [chunk(1, window="window 1")] + [chunk(index) for index in range(2, 7)]
    )

    result = asyncio.run(
        retrieve_credit_evidence(
            "credit",
            "  DTI policy  ",
            5,
            user_id=" credit-policy-user ",
            service=service,
        )
    )

    assert service.calls == [("DTI policy", None, "credit-policy-user")]
    assert len(result.evidence) == 5
    assert result.evidence[0].model_dump() == {
        "source_id": "source-1",
        "file_name": "policy-1.pdf",
        "page": "1",
        "excerpt": "window 1",
    }
    assert result.evidence[1].excerpt == "text 2"


@pytest.mark.parametrize(
    ("domain", "query", "top_k", "user_id"),
    [
        ("hr", "DTI", 5, "credit-user"),
        ("credit", "  ", 5, "credit-user"),
        ("credit", "DTI", 4, "credit-user"),
        ("credit", "DTI", True, "credit-user"),
        ("credit", "DTI", 5, "  "),
    ],
)
def test_invalid_contract_is_rejected_before_retrieval(domain, query, top_k, user_id):
    service = FakeKnowledgeBaseService([chunk(1)])

    with pytest.raises(ValueError):
        asyncio.run(
            retrieve_credit_evidence(
                domain,
                query,
                top_k,
                user_id=user_id,
                service=service,
            )
        )

    assert service.calls == []


@pytest.mark.parametrize(
    "invalid_chunk",
    [
        {"file_name": "policy.pdf", "text": "evidence"},
        {"chunk_id": "source-1", "text": "evidence"},
        {"chunk_id": "source-1", "file_name": "policy.pdf"},
    ],
)
def test_missing_provenance_is_rejected(invalid_chunk):
    service = FakeKnowledgeBaseService([invalid_chunk])

    with pytest.raises(ValueError):
        asyncio.run(
            retrieve_credit_evidence(
                "credit", "DTI", 5, user_id="credit-user", service=service
            )
        )


def test_duplicate_source_ids_are_rejected():
    service = FakeKnowledgeBaseService([chunk(1), chunk(1)])

    with pytest.raises(ValueError, match="Duplicate evidence source_id"):
        asyncio.run(
            retrieve_credit_evidence(
                "credit", "DTI", 5, user_id="credit-user", service=service
            )
        )


def test_empty_evidence_is_rejected():
    service = FakeKnowledgeBaseService([])

    with pytest.raises(ValueError, match="no evidence"):
        asyncio.run(
            retrieve_credit_evidence(
                "credit", "DTI", 5, user_id="credit-user", service=service
            )
        )


def test_retrieval_error_is_redacted():
    service = FakeKnowledgeBaseService(error=RuntimeError("secret collection detail"))

    with pytest.raises(RuntimeError) as error:
        asyncio.run(
            retrieve_credit_evidence(
                "credit", "DTI", 5, user_id="credit-user", service=service
            )
        )

    assert str(error.value) == "Credit knowledge retrieval failed"


def test_document_page_returns_full_page_with_server_user_scope():
    service = FakeKnowledgeBaseService(
        chunk_result=SimpleNamespace(
            metadata={"file_name": "credit-policy.pdf", "page_label": "12"}
        ),
        page_result=SimpleNamespace(
            document_path="credit-policy.pdf",
            page_label="12",
            text="Full policy page text",
        )
    )

    result = asyncio.run(
        retrieve_document_page(
            "credit",
            " source-1 ",
            user_id=" credit-policy-user ",
            service=service,
        )
    )

    assert service.chunk_calls == [("source-1", "credit-policy-user")]
    assert service.page_calls == [
        ("credit-policy.pdf", "12", "credit-policy-user")
    ]
    assert result.evidence[0].model_dump() == {
        "source_id": "page:source-1",
        "file_name": "credit-policy.pdf",
        "page": "12",
        "excerpt": "Full policy page text",
    }


@pytest.mark.parametrize(
    ("domain", "source_id", "user_id"),
    [
        ("hr", "source-1", "credit-user"),
        ("credit", "  ", "credit-user"),
        ("credit", "source-1", "  "),
    ],
)
def test_document_page_rejects_invalid_contract_before_lookup(
    domain, source_id, user_id
):
    service = FakeKnowledgeBaseService()

    with pytest.raises(ValueError):
        asyncio.run(
            retrieve_document_page(
                domain,
                source_id,
                user_id=user_id,
                service=service,
            )
        )

    assert service.chunk_calls == []
    assert service.page_calls == []


def test_document_page_error_is_redacted():
    service = FakeKnowledgeBaseService(
        chunk_result=SimpleNamespace(
            metadata={"file_name": "policy.pdf", "page_label": "1"}
        ),
        page_error=RuntimeError("secret page and collection detail")
    )

    with pytest.raises(RuntimeError) as error:
        asyncio.run(
            retrieve_document_page(
                "credit",
                "source-1",
                user_id="credit-user",
                service=service,
            )
        )

    assert str(error.value) == "Document page retrieval failed"


def test_document_page_rejects_a_different_page_from_rag():
    service = FakeKnowledgeBaseService(
        chunk_result=SimpleNamespace(
            metadata={"file_name": "policy.pdf", "page_label": "1"}
        ),
        page_result=SimpleNamespace(
            document_path="policy.pdf",
            page_label="2",
            text="Wrong page",
        )
    )

    with pytest.raises(ValueError, match="different document page"):
        asyncio.run(
            retrieve_document_page(
                "credit",
                "source-1",
                user_id="credit-user",
                service=service,
            )
        )


def test_fastmcp_exposes_five_credit_read_tools():
    tools = asyncio.run(rag_mcp_server.mcp.list_tools())

    assert [tool.name for tool in tools] == [
        "search_knowledge",
        "get_document_page",
        "get_loan_profile",
        "get_customer",
        "list_reports",
    ]


def test_fastmcp_tool_reads_user_scope_from_environment(monkeypatch):
    service = FakeKnowledgeBaseService([chunk(1)])
    monkeypatch.setattr(rag_mcp_server, "knowledge_base_service", service)
    monkeypatch.setenv("RAG_MCP_USER_ID", "credit-policy-user")

    result = asyncio.run(rag_mcp_server.search_knowledge("credit", "DTI", 5))

    assert result.evidence[0].source_id == "source-1"
    assert service.calls == [("DTI", None, "credit-policy-user")]


def test_fastmcp_tool_fails_without_user_scope(monkeypatch):
    monkeypatch.delenv("RAG_MCP_USER_ID", raising=False)

    with pytest.raises(ValueError, match="RAG_MCP_USER_ID"):
        asyncio.run(rag_mcp_server.search_knowledge("credit", "DTI", 5))


def test_fastmcp_page_tool_reads_user_scope_from_environment(monkeypatch):
    service = FakeKnowledgeBaseService(
        chunk_result=SimpleNamespace(
            metadata={"file_name": "policy.pdf", "page_label": "2"}
        ),
        page_result=SimpleNamespace(
            document_path="policy.pdf",
            page_label="2",
            text="Full page",
        )
    )
    monkeypatch.setattr(rag_mcp_server, "knowledge_base_service", service)
    monkeypatch.setenv("RAG_MCP_USER_ID", "credit-policy-user")

    result = asyncio.run(
        rag_mcp_server.get_document_page("credit", "source-1")
    )

    assert result.evidence[0].excerpt == "Full page"
    assert service.chunk_calls == [("source-1", "credit-policy-user")]
    assert service.page_calls == [
        ("policy.pdf", "2", "credit-policy-user")
    ]


def test_loan_data_tools_read_server_user_scope(monkeypatch):
    service = FakeLoanAgentService()
    monkeypatch.setattr(rag_mcp_server, "loan_agent_service", service)
    monkeypatch.setenv("LOAN_DATA_MCP_USER_ID", "loan-user")

    profile = asyncio.run(rag_mcp_server.get_loan_profile(" profile-1 "))
    customer = asyncio.run(rag_mcp_server.get_customer(" customer-1 "))
    reports = asyncio.run(rag_mcp_server.list_reports(" profile-1 "))

    assert profile.id == "profile-1"
    assert customer.id == "customer-1"
    assert reports.total_count == 0
    assert service.calls == [
        ("get_loan_profile", "loan-user", "profile-1"),
        ("get_customer", "loan-user", "customer-1"),
        ("list_reports", "loan-user", "profile-1"),
    ]


def test_loan_data_tool_fails_without_user_scope(monkeypatch):
    monkeypatch.delenv("LOAN_DATA_MCP_USER_ID", raising=False)

    with pytest.raises(ValueError, match="LOAN_DATA_MCP_USER_ID"):
        asyncio.run(rag_mcp_server.get_loan_profile("profile-1"))


def test_loan_data_tool_redacts_service_error(monkeypatch):
    service = FakeLoanAgentService(error=RuntimeError("secret database detail"))
    monkeypatch.setattr(rag_mcp_server, "loan_agent_service", service)
    monkeypatch.setenv("LOAN_DATA_MCP_USER_ID", "loan-user")

    with pytest.raises(RuntimeError) as error:
        asyncio.run(rag_mcp_server.get_loan_profile("profile-1"))

    assert str(error.value) == "Loan profile retrieval failed"
