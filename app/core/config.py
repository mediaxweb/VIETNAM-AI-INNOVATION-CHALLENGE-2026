from pathlib import PurePosixPath
from typing import Optional

from pydantic_settings import BaseSettings


class Configs(BaseSettings):
    jwt_secret_key: str = "change-me-to-a-long-random-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 10080
    llama_collection_name: str = "qa_collection"
    llama_embed_provider: str = "huggingface"
    llama_embed_model: Optional[str] = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    openai_api_key: Optional[str] = None
    openai_embed_model: str = "text-embedding-3-small"
    storage_root: Optional[str] = None
    railway_volume_mount_path: Optional[str] = None
    llama_chroma_persist_dir: Optional[str] = None
    bm25_persist_dir: Optional[str] = None
    docstore_persist_dir: Optional[str] = None
    rag_brain_openclaw_api_key: Optional[str] = None
    openai_qna_model: str = "gpt-5.4-mini"
    temp_kb_dir: str = "tmp/knowledge_base"
    loan_upload_dir: Optional[str] = None

    @staticmethod
    def _normalize_optional_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _normalize_optional_secret(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _resolve_persist_dir(self, explicit_path: Optional[str], default_dir_name: str) -> str:
        explicit = self._normalize_optional_path(explicit_path)
        if explicit:
            return explicit

        storage_root = self.resolved_storage_root
        if storage_root:
            return str(PurePosixPath(storage_root) / default_dir_name)

        return f"./{default_dir_name}"

    @property
    def resolved_storage_root(self) -> Optional[str]:
        return self._normalize_optional_path(self.storage_root) or self._normalize_optional_path(
            self.railway_volume_mount_path
        )

    @property
    def resolved_llama_chroma_persist_dir(self) -> str:
        return self._resolve_persist_dir(self.llama_chroma_persist_dir, "chroma_db")

    @property
    def resolved_bm25_persist_dir(self) -> str:
        return self._resolve_persist_dir(self.bm25_persist_dir, "bm25_storage")

    @property
    def resolved_docstore_persist_dir(self) -> str:
        return self._resolve_persist_dir(self.docstore_persist_dir, "docstore")

    @property
    def resolved_rag_brain_openclaw_api_key(self) -> Optional[str]:
        return self._normalize_optional_secret(self.rag_brain_openclaw_api_key)

    @property
    def resolved_loan_upload_dir(self) -> str:
        explicit = self._normalize_optional_path(self.loan_upload_dir)
        if explicit:
            return explicit

        storage_root = self.resolved_storage_root
        if storage_root:
            return str(Path(storage_root) / "loan_uploads")

        return "./loan_uploads"

    class Config:
        env_file = (".env",)
        extra = "allow"


configs = Configs()
