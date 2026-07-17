# MCP build server notes

Source: https://modelcontextprotocol.io/docs/develop/build-server

## Summary

The official tutorial builds a simple MCP weather server exposing two tools:

- `get_alerts`
- `get_forecast`

The sample connects the server to an MCP host such as Claude Desktop, but the server can be used by any MCP-compatible client.

## Core MCP capabilities

MCP servers can expose three main capability types:

- Resources: file-like data clients can read, such as API responses or file contents.
- Tools: functions the LLM can call, usually with user approval.
- Prompts: reusable prompt templates for common workflows.

The build-server quickstart mainly focuses on tools.

## Important STDIO rule

For STDIO-based MCP servers, do not write logs to stdout.

Reason: stdout carries JSON-RPC messages. Any normal stdout log can corrupt the protocol stream and break the server.

Use stderr or file logging instead:

- Python: `print("message", file=sys.stderr)` or logging configured away from stdout.
- TypeScript: `console.error(...)`, not `console.log(...)`.

For HTTP-based MCP servers, normal stdout logging is acceptable because it does not interfere with HTTP responses.

## Python server shape

Requirements:

- Python 3.10+
- Python MCP SDK 1.2.0+

Typical setup:

```bash
uv init weather
cd weather
uv venv
source .venv/bin/activate
uv add "mcp[cli]" httpx
touch weather.py
```

Basic server pattern:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state."""
    ...

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

`FastMCP` uses Python type hints and docstrings to generate tool definitions.

## TypeScript server shape

Requirements:

- Node.js 16+
- `@modelcontextprotocol/sdk`
- `zod@3`
- TypeScript

Typical setup:

```bash
mkdir weather
cd weather
npm init -y
npm install @modelcontextprotocol/sdk zod@3
npm install -D @types/node typescript
mkdir src
touch src/index.ts
```

Basic server pattern:

```ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "weather",
  version: "1.0.0",
});

server.registerTool(
  "get_alerts",
  {
    description: "Get weather alerts for a state",
    inputSchema: {
      state: z.string().length(2).describe("Two-letter state code"),
    },
  },
  async ({ state }) => ({
    content: [{ type: "text", text: `Alerts for ${state}` }],
  }),
);
```

## Claude Desktop config pattern

Claude Desktop launches local MCP servers via config:

```json
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather",
        "run",
        "weather.py"
      ]
    }
  }
}
```

Use absolute paths. If `uv` is not found by Claude Desktop, use the full path from `which uv`.

## Practical direction for our research

For our own MCP experiments, start with a STDIO server because it is the simplest host integration path. Use Python FastMCP first unless we specifically need TypeScript package/distribution ergonomics.

Potential first tools:

- `search_project_files(query: str)`
- `read_project_file(path: str)`
- `ocr_image_url(url: str)` calling our OCR API
- `railway_cost_summary()` using Railway CLI/API output
