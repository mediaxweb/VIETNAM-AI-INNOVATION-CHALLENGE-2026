from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("LLAMA_EMBED_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017/rag_brain_test")

AGENTS_DIR = Path(__file__).resolve().parents[1] / "agents"
sys.path.insert(0, str(AGENTS_DIR))
