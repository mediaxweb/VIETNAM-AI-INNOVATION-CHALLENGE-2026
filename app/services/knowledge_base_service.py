from __future__ import annotations

import asyncio
import json
import os
import shutil
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, List, Optional, Sequence
from urllib.parse import urlparse

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from pypdf import PdfReader


from app.api.schemas.knowledge_base import (
    KnowledgeBaseProcessDocumentRequest,
)
from app.core.config import configs
from app.core.database import (
    KB_DOCUMENT_INDEX_STATUS_DELETED,
    KB_DOCUMENT_INDEX_STATUS_FAILED,
    KB_DOCUMENT_INDEX_STATUS_INDEXED,
    Database,
)
from logs.logging_config import logger

load_dotenv()

HF_CACHE_DIR = os.getenv("HF_HOME", "./hf_cache")


def _embedding_details() -> tuple[str, str]:
    provider = (configs.llama_embed_provider or "huggingface").strip().lower()
    if provider in {"huggingface", "local"}:
        return provider, configs.llama_embed_model or "unknown"
    if provider == "openai":
        return provider, configs.openai_embed_model
    return provider, "unknown"


def _build_embed_model():
    provider = (configs.llama_embed_provider or "huggingface").strip().lower()

    if provider in {"huggingface", "local"}:
        try:
            huggingface_module = import_module("llama_index.embeddings.huggingface")
        except ImportError as exc:
            raise ValueError(
                "HuggingFace embeddings are not installed. Install requirements-local-embeddings.txt "
                "to use LLAMA_EMBED_PROVIDER='huggingface' or 'local'."
            ) from exc
        return huggingface_module.HuggingFaceEmbedding(
            model_name=configs.llama_embed_model,
            cache_folder=HF_CACHE_DIR,
        )

    if provider == "openai":
        if not configs.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLAMA_EMBED_PROVIDER is set to 'openai'."
            )
        return OpenAIEmbedding(
            model=configs.openai_embed_model,
            api_key=configs.openai_api_key,
        )

    raise ValueError(
        "Unsupported embedding provider "
        f"'{configs.llama_embed_provider}'. Use 'huggingface' or 'openai'."
    )


def _configure_embed_model() -> None:
    Settings.embed_model = _build_embed_model()


_configure_embed_model()

@dataclass
class KnowledgeBaseProcessResult:
    """Domain model describing a knowledge base ingestion run."""

    document_count: int
    node_count: int
    # document_nodes: list
    collection_name: str
    message: str
    summary: Optional[str] = None
    nodes: Optional[list] = None
    group_summaries: Optional[List[dict]] = None


@dataclass
class KnowledgeBaseDocumentTextResult:
    """Domain model for one indexed page read back from ChromaDB."""

    document_path: str
    page_label: str
    text: str
    chunk_count: int
    collection_name: str
    metadata: List[dict]


@dataclass
class KnowledgeBaseChunkDetailResult:
    """Domain model for one indexed chunk read back from ChromaDB."""

    chunk_id: str
    text: str
    collection_name: str
    metadata: dict


@dataclass
class KnowledgeBaseIndexedFileResult:
    """Domain model describing one indexed file in a ChromaDB collection."""

    file_name: str | None
    file_path: str | None
    document_id: str | None
    index_status: str
    chunk_count: int
    page_count: int
    last_error: str | None = None


@dataclass
class KnowledgeBaseDeleteFileResult:
    """Domain model describing a user-scoped indexed file delete."""

    status: str
    message: str
    document_path: str
    deleted_node_count: int
    collection_name: str


class KnowledgeBaseService:
    """Service responsible for orchestrating document ingestion with LlamaIndex."""

    ANSWER_CONTEXT_TOP_K = 4
    DOMINANT_SOURCE_SCORE_COUNT = 2
    DOMINANT_SOURCE_RATIO = 2.0
    RETRIEVED_CHUNKS_TOP_K = 10

    def __init__(self) -> None:
        self._last_retriever = None

    @property
    def last_retriever(self):
        """Return the most recently built retriever instance, if available."""

        return self._last_retriever

    def question_and_answer(
        self,
        query: str,
        conversation_history: Optional[Sequence[object]] = None,
        *,
        user_id: str,
    ):
        collection_name = self._collection_name_for_user(user_id)
        return self.question_and_answer_for_collection(
            query,
            conversation_history,
            collection_name=collection_name,
        )

    def question_and_answer_for_collection(
        self,
        query: str,
        conversation_history: Optional[Sequence[object]] = None,
        *,
        collection_name: str,
    ):
        _configure_embed_model()
        embed_provider, embed_model = _embedding_details()
        logger.info(
            "Question answering with embedding provider='%s' model='%s' collection='%s'",
            embed_provider,
            embed_model,
            collection_name,
        )
        chroma_client = self._create_chroma_client()
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        if chroma_collection.count() == 0:
            raise ValueError(
                f"Knowledge base collection '{collection_name}' is empty. "
                "Ingest documents before asking questions."
            )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
        )
        vector_retrieve = index.as_retriever(similarity_top_k=10)
        bm25_persist_dir = self._bm25_persist_dir(collection_name)
        if not bm25_persist_dir.exists():
            raise ValueError(
                f"BM25 index for collection '{collection_name}' is not initialized. "
                "Re-ingest the collection before asking questions."
            )
        bm25_retriever = BM25Retriever.from_persist_dir(str(bm25_persist_dir))
        fusion_retriever = QueryFusionRetriever(
            [vector_retrieve, bm25_retriever],
            num_queries=1,
            similarity_top_k=10,
            use_async=False,
        )
        retrieval_query = self._build_retrieval_query(query, conversation_history)
        nodes = fusion_retriever.retrieve(retrieval_query)
        selected_nodes = self._select_answer_nodes(nodes)
        context_texts = [
            f"Page {node.metadata.get('page_label', '-')}, File {node.metadata.get('file_name', '-')}\n{node.metadata.get('window', '-')}"
            for node in selected_nodes
        ]
        source_metadata = []
        for i, node in enumerate(selected_nodes):
            source_metadata.append({
                "page_label": node.metadata.get('page_label', '-'),
                "file_name": node.metadata.get('file_name', '-'),
                "window": node.metadata.get('window', '-'),
            })
            print(f"\n--- 🔹 Chunk {i+1} ---")
            print(node.text)  # Limit to 500 chars for readability
            print("📎 Metadata:", node.metadata)
            if hasattr(node, "score"):
                print("📊 Similarity Score:", node.score)

        return {
            "context": "\n\n".join(context_texts),
            "metadata_list": source_metadata
        }

    def retrieve_chunks(
        self,
        query: str,
        conversation_history: Optional[Sequence[object]] = None,
        *,
        user_id: str,
    ):
        collection_name = self._collection_name_for_user(user_id)
        return self.retrieve_chunks_for_collection(
            query,
            conversation_history,
            collection_name=collection_name,
        )

    def retrieve_chunks_for_collection(
        self,
        query: str,
        conversation_history: Optional[Sequence[object]] = None,
        *,
        collection_name: str,
    ):
        _configure_embed_model()
        embed_provider, embed_model = _embedding_details()
        logger.info(
            "Retrieving chunks with embedding provider='%s' model='%s'",
            embed_provider,
            embed_model,
        )
        chroma_client = self._create_chroma_client()
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        if chroma_collection.count() == 0:
            raise ValueError(
                f"Knowledge base collection '{collection_name}' is empty. "
                "Ingest documents before retrieving chunks."
            )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        vector_retrieve = index.as_retriever(similarity_top_k=self.RETRIEVED_CHUNKS_TOP_K)
        bm25_persist_dir = self._bm25_persist_dir(collection_name)
        if not bm25_persist_dir.exists():
            raise ValueError(
                f"BM25 index for collection '{collection_name}' is not initialized. "
                "Re-ingest the collection before retrieving chunks."
            )
        bm25_retriever = BM25Retriever.from_persist_dir(str(bm25_persist_dir))
        fusion_retriever = QueryFusionRetriever(
            [vector_retrieve, bm25_retriever],
            num_queries=1,
            similarity_top_k=self.RETRIEVED_CHUNKS_TOP_K,
            use_async=False,
        )
        retrieval_query = self._build_retrieval_query(query, conversation_history)
        nodes = fusion_retriever.retrieve(retrieval_query)
        return {
            "question": query,
            "chunks": [self._serialize_retrieved_chunk(node) for node in nodes],
        }

    @classmethod
    def _select_answer_nodes(cls, nodes: Sequence[object]) -> List[object]:
        ranked_nodes = list(nodes)
        if not ranked_nodes:
            return []

        source_to_nodes: dict[str, List[object]] = defaultdict(list)
        for node in ranked_nodes:
            source_to_nodes[cls._node_source_key(node)].append(node)

        if len(source_to_nodes) > 1:
            ranked_sources = sorted(
                (
                    (
                        cls._source_strength(source_nodes),
                        cls._node_score(source_nodes[0]),
                        source_key,
                    )
                    for source_key, source_nodes in source_to_nodes.items()
                ),
                reverse=True,
            )
            top_strength, _, top_source = ranked_sources[0]
            second_strength = ranked_sources[1][0]
            if top_strength > 0 and top_strength >= second_strength * cls.DOMINANT_SOURCE_RATIO:
                logger.info(
                    "Using dominant source '%s' for answer context (%.3f vs %.3f)",
                    top_source,
                    top_strength,
                    second_strength,
                )
                return source_to_nodes[top_source][: cls.ANSWER_CONTEXT_TOP_K]

        return ranked_nodes[: cls.ANSWER_CONTEXT_TOP_K]

    @classmethod
    def _source_strength(cls, nodes: Sequence[object]) -> float:
        top_scores = sorted(
            (cls._node_score(node) for node in nodes),
            reverse=True,
        )[: cls.DOMINANT_SOURCE_SCORE_COUNT]
        return sum(top_scores)

    @staticmethod
    def _node_source_key(node: object) -> str:
        metadata = KnowledgeBaseService._node_metadata(node)
        return (
            metadata.get("file_name")
            or metadata.get("file_path")
            or "unknown"
        )

    @staticmethod
    def _node_score(node: object) -> float:
        score = getattr(node, "score", 0.0)
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _node_text(node: object) -> str:
        text = getattr(node, "text", None)
        if text is None and hasattr(node, "node"):
            text = getattr(node.node, "text", None)
        return text or ""

    @staticmethod
    def _node_metadata(node: object) -> dict:
        metadata = getattr(node, "metadata", None)
        if metadata is None and hasattr(node, "node"):
            metadata = getattr(node.node, "metadata", None)
        return metadata or {}

    @classmethod
    def _serialize_retrieved_chunk(cls, node: object) -> dict:
        metadata = cls._node_metadata(node)
        score = getattr(node, "score", None)
        try:
            normalized_score = None if score is None else float(score)
        except (TypeError, ValueError):
            normalized_score = None
        return {
            "chunk_id": cls._retrieved_node_key(node),
            "text": cls._node_text(node),
            "score": normalized_score,
            "page_label": metadata.get("page_label"),
            "file_name": metadata.get("file_name"),
            "window": metadata.get("window"),
            "metadata": metadata,
        }

    def get_document_text(
        self,
        document_path: str,
        *,
        page_label: str,
        user_id: str,
    ) -> KnowledgeBaseDocumentTextResult:
        """Read indexed ChromaDB chunks for one PDF page in a user's collection."""

        collection_name = self._collection_name_for_user(user_id)
        return self.get_document_text_for_collection(
            document_path,
            page_label=page_label,
            collection_name=collection_name,
        )

    async def list_indexed_files(self, *, user_id: str) -> List[KnowledgeBaseIndexedFileResult]:
        """Return indexed files from the user-scoped registry without scanning Chroma."""

        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            raise ValueError("user_id is required.")

        registry = Database.get_user_indexed_files_collection()
        cursor = registry.find(
            {"user_id": normalized_user_id},
            {"_id": 0},
        ).sort("updated_at", -1)
        documents = await cursor.to_list(length=None)
        return [
            KnowledgeBaseIndexedFileResult(
                file_name=self._normalize_optional_metadata_value(document.get("file_name")),
                file_path=self._normalize_optional_metadata_value(document.get("file_path")),
                document_id=self._normalize_optional_metadata_value(document.get("document_id")),
                index_status=str(document.get("index_status") or "indexed"),
                chunk_count=int(document.get("chunk_count") or 0),
                page_count=int(document.get("page_count") or 0),
                last_error=self._normalize_optional_metadata_value(document.get("last_error")),
            )
            for document in documents
        ]

    async def record_user_file_indexing(
        self,
        *,
        user_id: str,
        document_path: str,
    ) -> None:
        """Mark a user-scoped file as currently indexing."""

        await self._write_user_indexed_file_registry(
            user_id=user_id,
            document_path=document_path,
            index_status="indexing",
            collection_name=self._collection_name_for_user(user_id),
            chunk_count=0,
            page_count=0,
            last_error=None,
        )

    async def record_user_indexed_file(
        self,
        *,
        user_id: str,
        document_path: str,
        result: KnowledgeBaseProcessResult,
    ) -> None:
        """Persist one user-scoped indexed file registry row after ingestion succeeds."""

        await self._write_user_indexed_file_registry(
            user_id=user_id,
            document_path=document_path,
            index_status="indexed",
            collection_name=result.collection_name,
            chunk_count=result.node_count,
            page_count=result.document_count,
            last_error=None,
            mark_indexed=True,
        )

    async def record_user_indexed_file_failed(
        self,
        *,
        user_id: str,
        document_path: str,
        error: str,
    ) -> None:
        """Mark a user-scoped file as failed after ingestion errors."""

        await self._write_user_indexed_file_registry(
            user_id=user_id,
            document_path=document_path,
            index_status="failed",
            collection_name=self._collection_name_for_user(user_id),
            chunk_count=0,
            page_count=0,
            last_error=error,
        )

    async def delete_user_indexed_file(
        self,
        *,
        user_id: str,
        document_path: str,
    ) -> KnowledgeBaseDeleteFileResult:
        """Delete a user-scoped file from Chroma/docstore/BM25 and mark registry deleted."""

        normalized_user_id = str(user_id or "").strip()
        normalized_document_path = str(document_path or "").strip()
        if not normalized_user_id:
            raise ValueError("user_id is required.")
        if not normalized_document_path:
            raise ValueError("document_path is required.")

        collection_name = self._collection_name_for_user(normalized_user_id)
        registry = Database.get_user_indexed_files_collection()
        existing = await registry.find_one(
            {
                "user_id": normalized_user_id,
                "file_key": normalized_document_path,
            }
        )
        if existing and existing.get("index_status") == KB_DOCUMENT_INDEX_STATUS_DELETED:
            return KnowledgeBaseDeleteFileResult(
                status=KB_DOCUMENT_INDEX_STATUS_DELETED,
                message="File index is already deleted.",
                document_path=normalized_document_path,
                deleted_node_count=0,
                collection_name=collection_name,
            )

        try:
            deleted_node_count = await self._delete_user_file_index_layers(
                collection_name=collection_name,
                document_path=normalized_document_path,
            )
        except Exception as exc:
            await self._write_user_indexed_file_registry(
                user_id=normalized_user_id,
                document_path=normalized_document_path,
                index_status=KB_DOCUMENT_INDEX_STATUS_FAILED,
                collection_name=collection_name,
                chunk_count=int((existing or {}).get("chunk_count") or 0),
                page_count=int((existing or {}).get("page_count") or 0),
                last_error=f"Delete failed: {exc}",
            )
            raise

        if not existing and deleted_node_count == 0:
            return KnowledgeBaseDeleteFileResult(
                status="not_found",
                message="File is not registered and no indexed nodes were found.",
                document_path=normalized_document_path,
                deleted_node_count=0,
                collection_name=collection_name,
            )

        await self._write_user_indexed_file_registry(
            user_id=normalized_user_id,
            document_path=normalized_document_path,
            index_status=KB_DOCUMENT_INDEX_STATUS_DELETED,
            collection_name=collection_name,
            chunk_count=0,
            page_count=0,
            last_error=None,
            mark_deleted=True,
        )
        return KnowledgeBaseDeleteFileResult(
            status=KB_DOCUMENT_INDEX_STATUS_DELETED,
            message="File index was deleted.",
            document_path=normalized_document_path,
            deleted_node_count=deleted_node_count,
            collection_name=collection_name,
        )

    async def _write_user_indexed_file_registry(
        self,
        *,
        user_id: str,
        document_path: str,
        index_status: str,
        collection_name: str,
        chunk_count: int,
        page_count: int,
        last_error: str | None,
        mark_indexed: bool = False,
        mark_deleted: bool = False,
    ) -> None:
        normalized_user_id = str(user_id or "").strip()
        normalized_document_path = str(document_path or "").strip()
        if not normalized_user_id:
            raise ValueError("user_id is required.")
        if not normalized_document_path:
            raise ValueError("document_path is required.")

        file_name = Path(urlparse(normalized_document_path).path).name or Path(normalized_document_path).name
        now = datetime.now(timezone.utc)
        set_fields = {
            "user_id": normalized_user_id,
            "file_key": normalized_document_path,
            "file_name": file_name or None,
            "file_path": normalized_document_path,
            "document_id": None,
            "collection_name": collection_name,
            "index_status": index_status,
            "chunk_count": chunk_count,
            "page_count": page_count,
            "last_error": last_error,
            "updated_at": now,
        }
        if mark_indexed:
            set_fields["last_indexed_at"] = now
        if mark_deleted:
            set_fields["last_deleted_at"] = now

        registry = Database.get_user_indexed_files_collection()
        await registry.update_one(
            {
                "user_id": normalized_user_id,
                "file_key": normalized_document_path,
            },
            {
                "$set": set_fields,
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )

    def get_document_text_for_collection(
        self,
        document_path: str,
        *,
        page_label: str,
        collection_name: str,
    ) -> KnowledgeBaseDocumentTextResult:
        normalized_document_path = str(document_path or "").strip()
        if not normalized_document_path:
            raise ValueError("document_path is required.")
        normalized_page_label = str(page_label or "").strip()
        if not normalized_page_label:
            raise ValueError("page_label is required.")

        chroma_client = self._create_chroma_client()
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        entries = self._get_chroma_document_entries(
            chroma_collection,
            normalized_document_path,
        )
        if not entries:
            raise ValueError(
                f"Document '{normalized_document_path}' was not found in collection '{collection_name}'."
            )

        entries = [
            entry
            for entry in entries
            if str(entry["metadata"].get("page_label", "")).strip() == normalized_page_label
        ]
        if not entries:
            raise ValueError(
                f"Page '{normalized_page_label}' for document '{normalized_document_path}' "
                f"was not found in collection '{collection_name}'."
            )

        entries.sort(key=self._document_text_entry_sort_key)
        text_parts = [entry["text"] for entry in entries if entry["text"]]
        full_text = "\n\n".join(text_parts).strip()

        return KnowledgeBaseDocumentTextResult(
            document_path=normalized_document_path,
            page_label=normalized_page_label,
            text=full_text,
            chunk_count=len(entries),
            collection_name=collection_name,
            metadata=[entry["metadata"] for entry in entries],
        )

    def get_chunk_detail(
        self,
        chunk_id: str,
        *,
        user_id: str,
    ) -> KnowledgeBaseChunkDetailResult:
        """Read one indexed ChromaDB chunk from a user's collection."""

        collection_name = self._collection_name_for_user(user_id)
        return self.get_chunk_detail_for_collection(
            chunk_id,
            collection_name=collection_name,
        )

    def get_chunk_detail_for_collection(
        self,
        chunk_id: str,
        *,
        collection_name: str,
    ) -> KnowledgeBaseChunkDetailResult:
        normalized_chunk_id = str(chunk_id or "").strip()
        if not normalized_chunk_id:
            raise ValueError("chunk_id is required.")

        chroma_client = self._create_chroma_client()
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        result = chroma_collection.get(
            ids=[normalized_chunk_id],
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        if not ids:
            raise ValueError(
                f"Chunk '{normalized_chunk_id}' was not found in collection '{collection_name}'."
            )

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        text = documents[0] if documents else ""
        metadata = metadatas[0] if metadatas else {}
        return KnowledgeBaseChunkDetailResult(
            chunk_id=str(ids[0]),
            text=str(text or "").strip(),
            collection_name=collection_name,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def _get_chroma_document_entries(
        cls,
        chroma_collection,
        document_path: str,
    ) -> List[dict]:
        candidates = cls._document_lookup_candidates(document_path)
        entries_by_id: dict[str, dict] = {}

        for metadata_key, metadata_value in candidates:
            result = chroma_collection.get(
                where={metadata_key: {"$eq": metadata_value}},
                include=["documents", "metadatas"],
            )
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            for index, chunk_id in enumerate(ids):
                if chunk_id in entries_by_id:
                    continue
                text = documents[index] if index < len(documents) else ""
                metadata = metadatas[index] if index < len(metadatas) else {}
                entries_by_id[str(chunk_id)] = {
                    "id": str(chunk_id),
                    "text": str(text or "").strip(),
                    "metadata": dict(metadata or {}),
                }

        return list(entries_by_id.values())

    @staticmethod
    def _document_lookup_candidates(document_path: str) -> List[tuple[str, str]]:
        normalized_document_path = str(document_path).strip()
        candidates = [
            ("file_path", normalized_document_path),
            ("file_name", normalized_document_path),
            ("document_id", normalized_document_path),
        ]

        file_name = Path(normalized_document_path).name
        if file_name and file_name != normalized_document_path:
            candidates.append(("file_name", file_name))

        unique_candidates: List[tuple[str, str]] = []
        seen = set()
        for metadata_key, metadata_value in candidates:
            candidate_key = (metadata_key, metadata_value)
            if metadata_value and candidate_key not in seen:
                unique_candidates.append(candidate_key)
                seen.add(candidate_key)
        return unique_candidates

    @staticmethod
    def _document_text_entry_sort_key(entry: dict) -> tuple:
        metadata = entry.get("metadata") or {}
        page_label = str(metadata.get("page_label", "")).strip()
        try:
            page_value: tuple[int, int | str] = (0, int(page_label))
        except ValueError:
            page_value = (1, page_label)
        start_char_idx, end_char_idx = KnowledgeBaseService._node_text_offsets(metadata)
        return (
            page_value,
            start_char_idx is None,
            start_char_idx if start_char_idx is not None else 0,
            end_char_idx is None,
            end_char_idx if end_char_idx is not None else 0,
            str(entry.get("id", "")),
        )

    @staticmethod
    def _node_text_offsets(metadata: dict) -> tuple[int | None, int | None]:
        start_char_idx = KnowledgeBaseService._coerce_optional_int(
            metadata.get("start_char_idx")
        )
        end_char_idx = KnowledgeBaseService._coerce_optional_int(
            metadata.get("end_char_idx")
        )
        if start_char_idx is not None or end_char_idx is not None:
            return start_char_idx, end_char_idx

        node_content = metadata.get("_node_content")
        if not node_content:
            return None, None

        try:
            parsed_node = json.loads(node_content)
        except (TypeError, ValueError):
            return None, None

        if not isinstance(parsed_node, dict):
            return None, None

        return (
            KnowledgeBaseService._coerce_optional_int(parsed_node.get("start_char_idx")),
            KnowledgeBaseService._coerce_optional_int(parsed_node.get("end_char_idx")),
        )

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_optional_metadata_value(value: Any) -> str | None:
        normalized_value = str(value or "").strip()
        if not normalized_value or normalized_value == "-":
            return None
        return normalized_value

    @staticmethod
    def _retrieved_node_key(node: object) -> str:
        base_node = getattr(node, "node", node)
        node_id = getattr(base_node, "node_id", None) or getattr(base_node, "id_", None)
        if node_id:
            return str(node_id)
        metadata = getattr(base_node, "metadata", {}) or {}
        return repr((metadata.get("document_id"), metadata.get("page_label"), getattr(base_node, "text", "")))

    @staticmethod
    def _build_retrieval_query(
        query: str,
        conversation_history: Optional[Sequence[object]] = None,
    ) -> str:
        history_lines: List[str] = []
        for message in conversation_history or []:
            role = getattr(message, "role", None)
            if role is None and isinstance(message, dict):
                role = message.get("role")
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if not content:
                continue
            role_label = str(role or "user").strip().capitalize()
            history_lines.append(f"{role_label}: {str(content).strip()}")

        if not history_lines:
            return query

        return (
            "Conversation history:\n"
            + "\n".join(history_lines)
            + f"\n\nCurrent question: {query}"
        )

    def process_document(
        self,
        request: KnowledgeBaseProcessDocumentRequest,
        *,
        user_id: str,
    ) -> KnowledgeBaseProcessResult:
        collection_name = self._collection_name_for_user(user_id)
        return self.process_document_for_collection(request, collection_name=collection_name)

    def process_document_for_collection(
        self,
        request: KnowledgeBaseProcessDocumentRequest,
        *,
        collection_name: str,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> KnowledgeBaseProcessResult:
        """Execute the ingestion pipeline for a single document path."""

        _configure_embed_model()
        embed_provider, embed_model = _embedding_details()
        document_path = Path(request.document_path)
        if not document_path.exists() or not document_path.is_file():
            raise ValueError(
                f"Document '{document_path}' does not exist or is not accessible."
            )

        file_ext = document_path.suffix.lower()
        logger.info(f"Starting ingestion for {document_path} (type={file_ext})")
        logger.info(
            "Document ingestion using embedding provider='%s' model='%s' collection='%s'",
            embed_provider,
            embed_model,
            collection_name,
        )
        if file_ext == ".pdf":
            if not self._pdf_has_extractable_text(document_path):
                raise ValueError(
                    f"Document '{document_path}' only supports text-based PDF files. "
                    "PDF files without extractable text are not supported."
                )

            documents = SimpleDirectoryReader(
                input_files=[request.document_path]
            ).load_data()
            logger.info(f"Assigned real PDF page numbers: {len(documents)} pages")
        else:
            raise ValueError(
                f"Unsupported document type '{file_ext or 'unknown'}'. Only PDF files are supported."
            )

        if not documents:
            raise ValueError(
                f"Document '{document_path}' could not be loaded or is empty."
            )

        ingest_kwargs: dict[str, Any] = {"source_description": str(document_path)}
        if metadata_overrides:
            ingest_kwargs["metadata_overrides"] = metadata_overrides

        ingest_result = self._ingest_documents(
            documents,
            request,
            collection_name,
            **ingest_kwargs,
        )

        return KnowledgeBaseProcessResult(
            document_count=ingest_result.document_count,
            node_count=ingest_result.node_count,
            collection_name=ingest_result.collection_name,
            message=ingest_result.message,
        )

    @staticmethod
    def _normalize_mime_type(mime_type: str | None) -> str | None:
        if not mime_type:
            return None
        return mime_type.split(";", 1)[0].strip().lower() or None

    async def _delete_user_file_index_layers(
        self,
        *,
        collection_name: str,
        document_path: str,
    ) -> int:
        return await asyncio.to_thread(
            self._delete_user_file_index_layers_sync,
            collection_name=collection_name,
            document_path=document_path,
        )

    def _delete_user_file_index_layers_sync(
        self,
        *,
        collection_name: str,
        document_path: str,
    ) -> int:
        """Delete user file nodes from Chroma/docstore and rebuild BM25."""

        logger.info(
            "Deleting user file index document_path='%s' collection='%s'",
            document_path,
            collection_name,
        )
        candidates = self._document_lookup_candidates(document_path)
        where_filter = self._metadata_candidates_where_filter(candidates)

        docstore_persist_dir = self._docstore_persist_dir(collection_name)
        bm25_persist_dir = self._bm25_persist_dir(collection_name)
        docstore_path = docstore_persist_dir / "docstore.json"

        chroma_client = self._create_chroma_client()
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        chroma_matches = chroma_collection.get(where=where_filter, include=[])
        chroma_ids = list(chroma_matches.get("ids") or [])
        if chroma_ids:
            chroma_collection.delete(ids=chroma_ids)

        if not docstore_path.exists():
            if bm25_persist_dir.exists():
                self._remove_collection_persist_dir(bm25_persist_dir, collection_name=collection_name)
            return len(chroma_ids)

        docstore = SimpleDocumentStore.from_persist_dir(str(docstore_persist_dir))
        matching_docstore_ids = [
            node_id
            for node_id, node in list(docstore.docs.items())
            if self._node_matches_metadata_candidates(node, candidates)
        ]
        for node_id in matching_docstore_ids:
            docstore.delete_document(node_id, raise_error=False)

        docstore_persist_dir.mkdir(parents=True, exist_ok=True)
        docstore.persist(str(docstore_path))
        remaining_nodes = list(docstore.docs.values())
        if remaining_nodes:
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=docstore,
                similarity_top_k=10,
            )
            bm25_persist_dir.mkdir(parents=True, exist_ok=True)
            bm25_retriever.persist(path=str(bm25_persist_dir))
        elif bm25_persist_dir.exists():
            self._remove_collection_persist_dir(bm25_persist_dir, collection_name=collection_name)

        return max(len(chroma_ids), len(matching_docstore_ids))

    @staticmethod
    def _metadata_candidates_where_filter(candidates: Sequence[tuple[str, str]]) -> dict:
        filters = [{key: {"$eq": value}} for key, value in candidates]
        if len(filters) == 1:
            return filters[0]
        return {"$or": filters}

    @classmethod
    def _node_matches_metadata_candidates(
        cls,
        node: object,
        candidates: Sequence[tuple[str, str]],
    ) -> bool:
        metadata = cls._node_metadata(node)
        return any(
            str(metadata.get(metadata_key, "")).strip() == metadata_value
            for metadata_key, metadata_value in candidates
        )

    @staticmethod
    def _remove_collection_persist_dir(directory: Path, *, collection_name: str) -> None:
        resolved_directory = directory.resolve()
        if resolved_directory.name != collection_name:
            raise ValueError(
                f"Refusing to remove unexpected persist directory '{resolved_directory}'."
            )
        shutil.rmtree(resolved_directory)

    @staticmethod
    def _collection_base_name() -> str:
        base_collection_name = (configs.llama_collection_name or "qa_collection").strip() or "qa_collection"
        return base_collection_name

    @staticmethod
    def _normalize_collection_component(value: str, *, label: str) -> str:
        normalized_value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip()).strip("_")
        if not normalized_value:
            raise ValueError(f"{label} is required to resolve a knowledge base namespace.")
        return normalized_value

    @classmethod
    def _collection_name_for_user(cls, user_id: str) -> str:
        normalized_user_id = cls._normalize_collection_component(user_id, label="User id")
        base_collection_name = cls._collection_base_name()
        return f"{base_collection_name}__user__{normalized_user_id}"

    @staticmethod
    def _pdf_has_extractable_text(document_path: Path, sample_pages: int = 5, min_text_chars: int = 40) -> bool:
        """Return whether a PDF has enough extractable text to ingest."""
        reader = PdfReader(str(document_path))
        pages_to_sample = min(len(reader.pages), sample_pages)
        if pages_to_sample == 0:
            raise ValueError(f"Document '{document_path}' is empty.")

        extracted_text_chars = 0
        for page_index in range(pages_to_sample):
            page_text = reader.pages[page_index].extract_text() or ""
            extracted_text_chars += len(page_text.strip())

        return extracted_text_chars >= min_text_chars

    @staticmethod
    def _create_chroma_client():
        """Create a Chroma client using persistent storage when configured."""
        persist_dir: Optional[str] = configs.resolved_llama_chroma_persist_dir
        if persist_dir:
            return chromadb.PersistentClient(path=persist_dir)
        return chromadb.Client()

    @staticmethod
    def _docstore_persist_dir(collection_name: str) -> Path:
        return Path(configs.resolved_docstore_persist_dir) / collection_name

    @staticmethod
    def _bm25_persist_dir(collection_name: str) -> Path:
        return Path(configs.resolved_bm25_persist_dir) / collection_name

    def _ingest_documents(
        self,
        documents,
        request: KnowledgeBaseProcessDocumentRequest,
        chroma_collection_name: str,
        *,
        source_description: str,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> KnowledgeBaseProcessResult:

        chroma_client = self._create_chroma_client()
        chroma_collection = chroma_client.get_or_create_collection(chroma_collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        parser = SentenceWindowNodeParser.from_defaults(window_size=3)
        nodes = parser.get_nodes_from_documents(documents)
        self._enrich_node_metadata(nodes, metadata_overrides)

        VectorStoreIndex(nodes, storage_context=storage_context)

        docstore_persist_dir = self._docstore_persist_dir(chroma_collection_name)
        bm25_persist_dir = self._bm25_persist_dir(chroma_collection_name)
        docstore_path = docstore_persist_dir / "docstore.json"

        if docstore_path.exists():
            docstore = SimpleDocumentStore.from_persist_dir(str(docstore_persist_dir))
        else:
            docstore = SimpleDocumentStore()

        docstore.add_documents(nodes)
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=docstore,
            similarity_top_k=10,
        )
        docstore_persist_dir.mkdir(parents=True, exist_ok=True)
        bm25_persist_dir.mkdir(parents=True, exist_ok=True)
        docstore.persist(str(docstore_path))
        bm25_retriever.persist(path=str(bm25_persist_dir))

        # vector_retriever = vector_index.as_retriever(
        #     similarity_top_k=30
        # )
        # if doc_count > 0:
        #     keyword_retriever = BM25Retriever.from_defaults(
        #         docstore=vector_index.docstore,
        #         similarity_top_k=30
        #     )
        #     logger.info(f"Initialized BM25 retriever with {doc_count} documents.")
        # else:
        #     keyword_retriever = None
        #     logger.warning("Skipping BM25 retriever: docstore is empty.")

        # self._last_retriever = QueryFusionRetriever(
        #     [vector_retriever, keyword_retriever],
        #     num_queries=settings.llama_num_queries,
        #     similarity_top_k=settings.llama_fusion_similarity_top_k,
        #     use_async=True,
        # )

        message = (
            f"Processed {len(documents)} documents from {source_description} into "
            f"collection '{chroma_collection_name}' → generated {len(nodes)} nodes."
        )
        logger.info(message)

        return KnowledgeBaseProcessResult(
            document_count=len(documents),
            node_count=len(nodes),
            collection_name=chroma_collection_name,
            message=message,
            nodes=nodes,
        )

    @classmethod
    def _enrich_node_metadata(
        cls,
        nodes,
        metadata_overrides: dict[str, Any] | None,
    ) -> None:
        if not metadata_overrides:
            return

        normalized_metadata = {
            key: cls._metadata_value_for_storage(value)
            for key, value in metadata_overrides.items()
            if value is not None
        }

        for node in nodes:
            current_metadata = dict(getattr(node, "metadata", {}) or {})

            if not current_metadata.get("file_name") and normalized_metadata.get("file_name"):
                current_metadata["file_name"] = normalized_metadata["file_name"]
            current_metadata.setdefault("file_name", "-")

            if not current_metadata.get("page_label"):
                current_metadata["page_label"] = normalized_metadata.get("page_label", "-")

            if "file_path" in normalized_metadata:
                current_metadata["file_path"] = normalized_metadata["file_path"]

            node.metadata = current_metadata

    @staticmethod
    def _metadata_value_for_storage(value: Any) -> str | int | float | bool:
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return json.dumps([str(item) for item in value], ensure_ascii=False)
        return str(value)
