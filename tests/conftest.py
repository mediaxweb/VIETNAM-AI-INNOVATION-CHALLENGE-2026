from __future__ import annotations

import sys
from pathlib import Path


AGENTS_DIR = Path(__file__).resolve().parents[1] / "agents"
sys.path.insert(0, str(AGENTS_DIR))
