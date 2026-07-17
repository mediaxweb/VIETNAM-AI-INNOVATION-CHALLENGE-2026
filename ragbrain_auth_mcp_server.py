"""Simple streamable HTTP MCP server for RagBrain auth endpoints.

Tools:
- ragbrain_register
- ragbrain_login
- ragbrain_me

This server uses MCP Streamable HTTP transport at /mcp by default.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from mcp.server.fastmcp import FastMCP


DEFAULT_API_BASE = "https://ragbrain-production.up.railway.app"
API_BASE = os.getenv("RAGBRAIN_API_BASE", DEFAULT_API_BASE).rstrip("/")
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8765"))
MCP_PATH = os.getenv("MCP_PATH", "/mcp")

mcp = FastMCP(
    "ragbrain-auth",
    host=MCP_HOST,
    port=MCP_PORT,
    streamable_http_path=MCP_PATH,
)
_last_access_token: str | None = None


def _request_json(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    url = f"{API_BASE}{path}"
    data = None
    headers = {
        "Accept": "application/json",
        "User-Agent": "ragbrain-auth-mcp/0.1",
    }

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            if isinstance(parsed, dict):
                return parsed
            return {"data": parsed}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail: Any = json.loads(raw)
        except json.JSONDecodeError:
            detail = raw
        return {
            "error": True,
            "status_code": exc.code,
            "reason": exc.reason,
            "detail": detail,
        }
    except Exception as exc:
        print(f"RagBrain MCP request failed: {exc}", file=sys.stderr)
        return {
            "error": True,
            "status_code": None,
            "reason": type(exc).__name__,
            "detail": str(exc),
        }


@mcp.tool()
def ragbrain_register(email: str, password: str, full_name: str | None = None) -> str:
    """Register a new RagBrain user account.

    Args:
        email: User email address. Must be unique.
        password: Plain-text password. API requires at least 8 characters.
        full_name: Optional display name for the user profile.
    """
    payload: dict[str, Any] = {"email": email, "password": password}
    if full_name:
        payload["full_name"] = full_name

    result = _request_json("POST", "/api/v1/auth/register", body=payload)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def ragbrain_login(email: str, password: str) -> str:
    """Login to RagBrain and return a Bearer access token.

    The token is also cached in this MCP process so ragbrain_me can be called
    without passing access_token immediately after a successful login.

    Args:
        email: User email address.
        password: Plain-text password.
    """
    global _last_access_token

    result = _request_json(
        "POST",
        "/api/v1/auth/login",
        body={"email": email, "password": password},
    )
    token = result.get("access_token")
    if isinstance(token, str) and token:
        _last_access_token = token

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def ragbrain_me(access_token: str | None = None) -> str:
    """Return the currently authenticated RagBrain user.

    Args:
        access_token: Optional Bearer token. If omitted, this uses the token
            cached by the latest successful ragbrain_login call in this server
            process.
    """
    token = access_token or _last_access_token
    if not token:
        return json.dumps(
            {
                "error": True,
                "message": "No access token. Call ragbrain_login first or pass access_token.",
            },
            ensure_ascii=False,
            indent=2,
        )

    result = _request_json("GET", "/api/v1/auth/me", access_token=token)
    return json.dumps(result, ensure_ascii=False, indent=2)


def main() -> None:
    print(
        f"Starting RagBrain auth MCP server at http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}",
        file=sys.stderr,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
