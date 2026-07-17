# rag brain 

`rag brain` is a FastAPI service for document ingestion, retrieval, and
retrieval-augmented question answering. It ingests documents into a searchable
knowledge base with LlamaIndex + Chroma and exposes a small API for downstream
apps and workflows.

## Core features

- FastAPI endpoints under `/api/v1` for document ingestion and question answering.
- JWT-based auth endpoints under `/api/v1/auth` for local register/login/me flows.
- Text-based PDF ingestion that can chunk documents into sentence windows, build
  embeddings, and index content into Chroma.
- Hybrid retrieval for Q&A, combining vector search with BM25 before generating
  final answers with OpenAI.
- Persistent storage for Chroma, BM25, and the LlamaIndex document store so
  retrievers can be reused across runs.

## Service architecture

```text
FastAPI routes
  /api/v1/auth/register
  /api/v1/auth/login
  /api/v1/auth/me
  /api/v1/knowledge-base/process-document
  /api/v1/qna/question_and_answer
            |
            v
KnowledgeBaseService
  - loads documents
  - parses and indexes content
  - retrieves relevant context
  - tracks user indexed files
            |
            v
LlamaIndex + Chroma + BM25
MongoDB user_indexed_files registry
```

## API surface

### `POST /api/v1/auth/register`

Create a local account with an email and password.

| Field | Type | Description |
|-------|------|-------------|
| `email` | string | User email address. Stored in lowercase and must be unique. |
| `password` | string | Plain-text password. Minimum length is 8 characters. |
| `full_name` | string or null | Optional display name. |

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
        "email": "user@example.com",
        "password": "secret123",
        "full_name": "Test User"
      }'
```

Returns `201 Created` on success and `409 Conflict` when the email already exists.

### `POST /api/v1/auth/login`

Authenticate with email and password to get a Bearer access token.

| Field | Type | Description |
|-------|------|-------------|
| `email` | string | User email address. |
| `password` | string | Plain-text password. |

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
        "email": "user@example.com",
        "password": "secret123"
      }'
```

Returns:

```json
{
  "access_token": "<jwt>",
  "token_type": "bearer"
}
```

Returns `401 Unauthorized` when credentials are invalid.

### `GET /api/v1/auth/me`

Resolve the currently authenticated user from the Bearer token.

Example:

```bash
curl "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer <jwt>"
```

Returns `200 OK` with the user profile when the token is valid.

### `POST /api/v1/knowledge-base/process-document`

Upload and ingest a single text-based PDF into the knowledge base. This endpoint
requires a Bearer access token and accepts `multipart/form-data`.

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Text-based PDF file to ingest. PDFs without extractable text are rejected. |

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge-base/process-document" \
  -H "Authorization: Bearer <jwt>" \
  -F "file=@./documents/report.pdf;type=application/pdf"
```

### `POST /api/v1/qna/question_and_answer`

Ask a question against the latest ingested knowledge base. This endpoint requires
a Bearer access token.

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | Natural-language question to answer from retrieved context. |
| `conversation_history` | array | Optional prior chat messages used to interpret follow-up questions. |

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/qna/question_and_answer" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <jwt>" \
  -d '{
        "question": "What about the deadline?",
        "conversation_history": [
          {"role": "user", "content": "Tell me about the project timeline."},
          {"role": "assistant", "content": "The project has multiple milestones."}
        ]
      }'
```

### OpenClaw tool access

`POST /api/v1/qna/retrieve-chunks` and
`POST /api/v1/knowledge-base/document-text` still support the normal
`Authorization: Bearer <jwt>` user auth. For OpenClaw tool calls, they also
accept a short shared key while keeping the same user-scoped collection lookup:

```bash
curl -X POST "http://localhost:8000/api/v1/qna/retrieve-chunks" \
  -H "Content-Type: application/json" \
  -H "X-OpenClaw-Api-Key: <openclaw-key>" \
  -H "X-OpenClaw-User-Id: <user-id>" \
  -d '{
        "question": "What about the deadline?",
        "conversation_history": []
      }'
```

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge-base/document-text" \
  -H "Content-Type: application/json" \
  -H "X-OpenClaw-Api-Key: <openclaw-key>" \
  -H "X-OpenClaw-User-Id: <user-id>" \
  -d '{
        "document_path": "report.pdf",
        "page_label": "2"
      }'
```

Configure the shared key with `RAG_BRAIN_OPENCLAW_API_KEY`.

## Getting started

### Prerequisites

- Python 3.11
- An OpenAI API key for answer generation

### Installation

Install the core application dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to use local HuggingFace embeddings, install the extra local
embedding dependencies as well:

```bash
pip install -r requirements.txt -r requirements-local-embeddings.txt
```

### Environment configuration

Create a `.env` file in the project root and configure the values you need.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Disables interactive docs in production. |
| `MONGO_URI` | — | Required MongoDB connection string used for startup client initialization. |
| `MONGO_DB_NAME` | `rag_brain` | Fallback database name used when `MONGO_URI` omits the database path. |
| `JWT_SECRET_KEY` | `change-me-to-a-long-random-secret-key` | Secret used to sign and verify JWT access tokens. |
| `JWT_ALGORITHM` | `HS256` | Signing algorithm for access tokens. |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `10080` | Access-token lifetime in minutes. |
| `OPENAI_API_KEY` | — | Required for question answering. |
| `LLAMA_COLLECTION_NAME` | `qa_collection` | Default Chroma collection name. |
| `LLAMA_EMBED_PROVIDER` | `huggingface` | Embedding backend. Supported values: `huggingface` and `openai`. |
| `LLAMA_EMBED_MODEL` | `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` | Local embedding model used when `LLAMA_EMBED_PROVIDER=huggingface`. |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model used when `LLAMA_EMBED_PROVIDER=openai`. |
| `OPENAI_QNA_MODEL` | `gpt-5.4-mini` | OpenAI chat model used for QnA answer generation. |
| `STORAGE_ROOT` | unset | Shared persistent storage root. When set, Chroma/BM25/docstore default under this path. |
| `RAILWAY_VOLUME_MOUNT_PATH` | unset | Optional fallback mount path if you prefer Railway-specific naming over `STORAGE_ROOT`. |
| `LLAMA_CHROMA_PERSIST_DIR` | derived | Explicit Chroma persistence path. Overrides `STORAGE_ROOT` when set. |
| `HF_HOME` / `TRANSFORMERS_CACHE` / `HF_HUB_CACHE` | `./hf_cache` | Local cache directories for HuggingFace assets. |
| `BM25_PERSIST_DIR` | derived | Explicit BM25 persistence path. Overrides `STORAGE_ROOT` when set. |
| `DOCSTORE_PERSIST_DIR` | derived | Explicit path used to persist the LlamaIndex document store. Overrides `STORAGE_ROOT` when set. |
| `RAG_BRAIN_OPENCLAW_API_KEY` | unset | Short shared secret accepted with `X-OpenClaw-Api-Key` for OpenClaw access to `retrieve-chunks` and `document-text`; callers must also send `X-OpenClaw-User-Id`. |
| `TEMP_KB_DIR` | `tmp/knowledge_base` | Ephemeral working directory for uploaded PDF files during ingestion. |

Persistence resolution rules:

- If `LLAMA_CHROMA_PERSIST_DIR`, `BM25_PERSIST_DIR`, or `DOCSTORE_PERSIST_DIR` are set, the app uses them as-is.
- Otherwise, if `STORAGE_ROOT` is set, the app derives `chroma_db`, `bm25_storage`, and `docstore` under that root.
- Otherwise, if `RAILWAY_VOLUME_MOUNT_PATH` is set, the app derives the same subdirectories under that mount path.
- Otherwise, local development falls back to `./chroma_db`, `./bm25_storage`, and `./docstore`.

### Embedding providers

- `LLAMA_EMBED_PROVIDER=huggingface` uses the local model configured in
  `LLAMA_EMBED_MODEL`.
- `LLAMA_EMBED_PROVIDER=openai` uses OpenAI embeddings and requires
  `OPENAI_API_KEY`.

Installation guidance:

- `LLAMA_EMBED_PROVIDER=openai` only needs `requirements.txt`.
- `LLAMA_EMBED_PROVIDER=huggingface` should install both
  `requirements.txt` and `requirements-local-embeddings.txt`.

When switching embedding providers or models, do not reuse the same persisted
Chroma collection without re-ingesting documents, because vector dimensions may
change.

### Running the service

```bash
uvicorn app.main:app --reload
```

Interactive API docs are available at `http://localhost:8000/docs` in
non-production environments.

### Running tests

```bash
pytest -q
```

## Railway persistence

For Railway, attach a Volume and mount it at `/app/data`, then set either:

```env
STORAGE_ROOT=/app/data
```

or the three explicit paths:

```env
LLAMA_CHROMA_PERSIST_DIR=/app/data/chroma_db
BM25_PERSIST_DIR=/app/data/bm25_storage
DOCSTORE_PERSIST_DIR=/app/data/docstore
```

`TEMP_KB_DIR` can stay ephemeral because uploaded PDF files are only kept while
the ingestion request is running.

## Development notes

- The root endpoint `/` returns a simple readiness message.
- Static files are mounted at `/static`.
- Phase 2 now protects knowledge-base ingestion and question answering with Bearer auth.
- Knowledge-base storage is now scoped per authenticated user namespace.
- Shared legacy knowledge-base data is not migrated automatically; users should re-ingest documents into their own namespace.

## License

This project is distributed under the [Business Source License 1.1](LICENSE).
