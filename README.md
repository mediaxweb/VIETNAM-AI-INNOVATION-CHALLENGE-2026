# RagBrain auth MCP server

This project initializes a Streamable HTTP Model Context Protocol (MCP) server for the
RagBrain authentication API. It exposes the register, login, and current-user endpoints
as MCP tools so an MCP-compatible client or OpenAI Agents SDK agent can authenticate
against RagBrain without calling the REST API directly.

The MCP server starts locally, reads its runtime configuration from environment
variables, registers each auth operation as a tool, and forwards tool calls to the
RagBrain production API:

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`

API base:

```text
https://ragbrain-production.up.railway.app
```

Override with:

```bash
export RAGBRAIN_API_BASE="https://ragbrain-production.up.railway.app"
```

## Setup

```bash
cd /home/mediax/research/mcp_research
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run streamable HTTP server

```bash
python ragbrain_auth_mcp_server.py
```

Default MCP endpoint:

```text
http://127.0.0.1:8765/mcp
```

Override host, port, or path:

```bash
MCP_HOST=0.0.0.0 MCP_PORT=8765 MCP_PATH=/mcp python ragbrain_auth_mcp_server.py
```

## MCP initialization flow

1. Load `RAGBRAIN_API_BASE`, `MCP_HOST`, `MCP_PORT`, and `MCP_PATH` from the
   environment, using safe local defaults when they are not provided.
2. Create a `FastMCP` server named `ragbrain-auth`.
3. Register `ragbrain_register`, `ragbrain_login`, and `ragbrain_me` as callable MCP
   tools using Python type hints and docstrings.
4. Start the server with the `streamable-http` transport.
5. Accept MCP tool calls at `http://127.0.0.1:8765/mcp` by default and proxy the
   corresponding requests to the RagBrain auth API.

## Tools

### `ragbrain_register`

Inputs:

- `email`
- `password`
- `full_name` optional

### `ragbrain_login`

Inputs:

- `email`
- `password`

Returns the API `access_token` and caches it in the server process.

### `ragbrain_me`

Inputs:

- `access_token` optional

If `access_token` is omitted, it uses the token cached by the latest successful `ragbrain_login`.

## Streamable HTTP client config example

```json
{
  "mcpServers": {
    "ragbrain-auth": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

Start the server first:

```bash
cd /home/mediax/research/mcp_research
.venv/bin/python ragbrain_auth_mcp_server.py
```

Note: some clients only support local STDIO MCP servers. Use a client that supports MCP Streamable HTTP.

## OpenAI Agents SDK client

This repo also includes a simple OpenAI Agents SDK agent that uses the MCP server above.

Start the MCP server:

```bash
cd /home/mediax/research/mcp_research
.venv/bin/python ragbrain_auth_mcp_server.py
```

In another terminal, verify the agent can see the MCP tools without calling OpenAI:

```bash
cd /home/mediax/research/mcp_research
.venv/bin/python ragbrain_auth_agent.py --list-tools
```

Run the agent with OpenAI:

```bash
cd /home/mediax/research/mcp_research
export OPENAI_API_KEY="..."
.venv/bin/python ragbrain_auth_agent.py "Login to RagBrain with email user@example.com and password ..."
```

Optional config:

```bash
export RAGBRAIN_MCP_URL="http://127.0.0.1:8765/mcp"
export OPENAI_AGENT_MODEL="gpt-5.6"
```

## Credit Agent

`agents/credit_agent.py` evaluates normalized personal and SME loan applications.
It calculates financial ratios in Python, retrieves credit policy evidence from the
RAG MCP server, and returns a structured recommendation for manual review. It does
not approve, reject, or update a loan.

The RAG MCP server must expose:

```text
search_knowledge(domain="credit", query: str, top_k: int = 5)
```

The RAG corpus contains text documents or PDFs with a text layer. OCR is outside
this project scope.

Run the personal example:

```powershell
$env:OPENAI_API_KEY="..."
$env:RAG_MCP_URL="http://127.0.0.1:8766/mcp"
& '.\.venv\Scripts\python.exe' agents/credit_agent.py --input examples/credit_personal.json
```

Run the SME example:

```powershell
& '.\.venv\Scripts\python.exe' agents/credit_agent.py --input examples/credit_sme.json
```
