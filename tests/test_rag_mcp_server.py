import asyncio

import pytest

from app.rag_mcp_server import retrieve_credit_evidence


class FakeKnowledgeBaseService:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.calls = []

    def retrieve_chunks(self, query, conversation_history=None, *, user_id):
        self.calls.append((query, conversation_history, user_id))
        if self.error:
            raise self.error
        return {"question": query, "chunks": self.chunks}


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
