from pathlib import Path

import pytest

from app.api.schemas.knowledge_base import KnowledgeBaseProcessDocumentRequest
from app.services import knowledge_base_service
from app.services.knowledge_base_service import KnowledgeBaseProcessResult, KnowledgeBaseService


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, pages):
        self.pages = pages


def test_pdf_has_extractable_text_when_sample_contains_text(monkeypatch):
    monkeypatch.setattr(
        knowledge_base_service,
        "PdfReader",
        lambda _path: _FakeReader([_FakePage("A" * 60)]),
    )

    assert KnowledgeBaseService._pdf_has_extractable_text(Path("dummy.pdf"))


def test_pdf_without_extractable_text_is_not_accepted(monkeypatch):
    monkeypatch.setattr(
        knowledge_base_service,
        "PdfReader",
        lambda _path: _FakeReader([_FakePage(""), _FakePage("   ")]),
    )

    assert not KnowledgeBaseService._pdf_has_extractable_text(Path("dummy.pdf"))


def test_empty_pdf_raises_validation_error(monkeypatch):
    monkeypatch.setattr(
        knowledge_base_service,
        "PdfReader",
        lambda _path: _FakeReader([]),
    )

    with pytest.raises(ValueError, match="is empty"):
        KnowledgeBaseService._pdf_has_extractable_text(Path("dummy.pdf"))


def test_process_document_rejects_pdf_without_extractable_text(monkeypatch, tmp_path):
    pdf_path = tmp_path / "empty_text.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%")

    monkeypatch.setattr(knowledge_base_service, "_configure_embed_model", lambda: None)
    monkeypatch.setattr(
        KnowledgeBaseService,
        "_pdf_has_extractable_text",
        staticmethod(lambda _path: False),
    )

    service = KnowledgeBaseService()
    request = KnowledgeBaseProcessDocumentRequest(document_path=str(pdf_path))

    with pytest.raises(ValueError, match="only supports text-based PDF files"):
        service.process_document(request, user_id="user-123")


def test_process_document_ingests_text_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "text.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%")

    monkeypatch.setattr(knowledge_base_service, "_configure_embed_model", lambda: None)
    monkeypatch.setattr(
        KnowledgeBaseService,
        "_pdf_has_extractable_text",
        staticmethod(lambda _path: True),
    )

    class _FakeReader:
        def __init__(self, input_files):
            self.input_files = input_files

        def load_data(self):
            return ["document-page"]

    captured = {}

    def fake_ingest(self, documents, request, collection_name, *, source_description):
        captured["documents"] = documents
        captured["request"] = request
        captured["collection_name"] = collection_name
        captured["source_description"] = source_description
        return KnowledgeBaseProcessResult(
            document_count=len(documents),
            node_count=1,
            collection_name=collection_name,
            message="ok",
        )

    monkeypatch.setattr(knowledge_base_service, "SimpleDirectoryReader", _FakeReader)
    monkeypatch.setattr(KnowledgeBaseService, "_ingest_documents", fake_ingest)

    service = KnowledgeBaseService()
    request = KnowledgeBaseProcessDocumentRequest(document_path=str(pdf_path))

    result = service.process_document(request, user_id="user-123")

    assert result.document_count == 1
    assert result.node_count == 1
    assert result.collection_name == "qa_collection__user__user-123"
    assert captured["documents"] == ["document-page"]
    assert captured["source_description"] == str(pdf_path)

