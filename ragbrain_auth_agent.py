"""OpenAI Agents SDK client for the RagBrain auth MCP server.

Run the MCP server first:
    python ragbrain_auth_mcp_server.py

List MCP tools without calling OpenAI:
    python ragbrain_auth_agent.py --list-tools

Run the agent:
    OPENAI_API_KEY=... python ragbrain_auth_agent.py "Login with email ..."
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp


DEFAULT_MCP_URL = "http://127.0.0.1:8765/mcp"
DEFAULT_MODEL = "gpt-5.4-mini"


def load_dotenv() -> None:
    """Load simple KEY=VALUE lines from .env beside this script."""
    env_path = Path(__file__).resolve().with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a RagBrain auth agent backed by a Streamable HTTP MCP server."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Task for the agent. Example: Login to RagBrain with email ...",
    )
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("RAGBRAIN_MCP_URL", DEFAULT_MCP_URL),
        help=f"MCP Streamable HTTP URL. Default: {DEFAULT_MCP_URL}",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_AGENT_MODEL", DEFAULT_MODEL),
        help=f"OpenAI model for the agent. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Only connect to MCP and list available tools. Does not call OpenAI.",
    )
    return parser


def build_agent(server: MCPServerStreamableHttp, model: str) -> Agent:
    return Agent(
        name="RagBrain auth agent",
        instructions=(
            "You help operate the RagBrain auth API through MCP tools. "
            "Use ragbrain_register for account creation, ragbrain_login for login, "
            "and ragbrain_me to fetch the current user. "
            "Do not invent API results. If credentials or a token are missing, ask for them. "
            "Never print passwords back to the user."
        ),
        model=model,
        mcp_servers=[server],
    )


async def list_tools(server: MCPServerStreamableHttp) -> None:
    tools = await server.list_tools()
    for tool in tools:
        print(f"- {tool.name}: {tool.description or ''}")


async def run_agent(args: argparse.Namespace) -> None:
    if not args.list_tools and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required unless you use --list-tools.")

    params = {
        "url": args.mcp_url,
        "timeout": 30,
        "sse_read_timeout": 30,
    }

    async with MCPServerStreamableHttp(
        name="ragbrain-auth",
        params=params,
        cache_tools_list=True,
        client_session_timeout_seconds=30,
    ) as server:
        if args.list_tools:
            await list_tools(server)
            return

        prompt = " ".join(args.prompt).strip()
        if not prompt:
            raise SystemExit("Prompt is required unless you use --list-tools.")

        agent = build_agent(server, args.model)
        result = await Runner.run(agent, prompt)
        print(result.final_output)


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
