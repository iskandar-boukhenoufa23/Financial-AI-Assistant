import os
import shutil
import sys
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

# =========================================================
# Environment setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ALPHAVANTAGE_API_KEY = (
    os.getenv("ALPHAVANTAGE_API_KEY")
    or os.getenv("ALPHA_VANTAGE_API_KEY")
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

UVX_COMMAND = shutil.which("uvx") or "uvx"


def _require_env(name: str, value: str | None) -> str:
    """Return an environment value or raise a readable setup error."""
    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )
    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """Preserve the environment and add financial MCP API keys."""
    env = os.environ.copy()
    for key, value in updates.items():
        if value:
            env[key] = value
    return env


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=_require_env("GROQ_API_KEY", GROQ_API_KEY),
)


# =========================================================
# MCP client - Configured for Financial Data & News
# =========================================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY or ''}"
            ),
        },
        "alphavantage": {
            "transport": "stdio",
            "command": UVX_COMMAND,
            "args": [
                "alpha-vantage-mcp",
            ],
            "env": _subprocess_env(
                ALPHA_VANTAGE_API_KEY=ALPHAVANTAGE_API_KEY,
            ),
        },
    }
)


async def _get_server_tool(
    server_name: str,
    tool_name: str,
):
    """Load one tool from the selected financial/search MCP server."""
    if server_name == "tavily":
        _require_env("TAVILY_API_KEY", TAVILY_API_KEY)
    elif server_name == "alphavantage":
        _require_env("ALPHAVANTAGE_API_KEY", ALPHAVANTAGE_API_KEY)
        if shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found. Install uv and ensure it's in your PATH."
            )

    tools = await client.get_tools(
        server_name=server_name,
    )

    tool = next(
        (
            item
            for item in tools
            if item.name == tool_name
        ),
        None,
    )

    if tool is None:
        available_tools = (
            ", ".join(
                sorted(item.name for item in tools)
            )
            or "none"
        )

        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found "
            f"on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    return tool


# =========================================================
# Tavily MCP (For Financial News and Market Sentiment)
# =========================================================

async def tavily_mcp_search(query: str):
    search_tool = await _get_server_tool(
        "tavily",
        "tavily_search",
    )

    return await search_tool.ainvoke(
        {
            "query": query,
        }
    )


# =========================================================
# Alpha Vantage MCP (For Quantitative Financial Data)
# =========================================================

async def financial_mcp_call(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
):
    """
    Calls Alpha Vantage financial tools (e.g., stock quotes, 
    income statements, balance sheets).
    """
    fin_tool = await _get_server_tool(
        "alphavantage",
        tool_name,
    )

    return await fin_tool.ainvoke(
        tool_args or {}
    )


# =========================================================
# Stock Ticker Extractor Helper
# =========================================================

def extract_stock_ticker(query: str) -> str:
    prompt = f"""
Extract only the stock ticker symbol (e.g., AAPL, TSLA, MSFT, GOOGL) from the financial query.

Financial query:
{query}

Return only the ticker symbol uppercase. Do not add any explanation.
"""

    response = llm.invoke(prompt)
    ticker = str(response.content).strip().upper()

    if not ticker or len(ticker) > 5:
        raise ValueError(
            "The stock ticker could not be accurately extracted from the query."
        )

    return ticker