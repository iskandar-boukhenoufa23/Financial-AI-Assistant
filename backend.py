import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    financial_mcp_call,
    extract_stock_ticker,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing in .env")
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)


class FinanceState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    financial_constraints: dict[str, Any]
    supervisor_reasoning: str

    market_data: str
    fundamentals_data: str
    news_sentiment: str
    valuation_report: str

    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str
    llm_calls: int


KNOWN_AGENTS = {
    "market_data_agent",
    "fundamentals_agent",
    "valuation_agent",
}

AGENT_ORDER = [
    "market_data_agent",
    "fundamentals_agent",
    "valuation_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model did not return valid JSON.")
    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "ticker": "",
        "analysis_type": "",
        "timeframe": "",
        "focus_metrics": [],
    }


def supervisor_agent(state: FinanceState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the request pertains to stock market analysis, financial statements, 
earnings reports, company valuation, or financial data inquiries. 
Block unrelated or harmful queries.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""
    try:
        guardrail_raw = _llm_text(
            "Input guardrail for financial assistant. Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception:
        allowed = True
        guardrail_reason = "Fallback allowed financial request."

    if not allowed:
        reason = guardrail_reason or "Request blocked by financial guardrail."
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "financial_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=reason)],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
Choose the appropriate financial specialist agents for the query.

Available agents:
- market_data_agent: stock quotes, historical prices, volume, and market metrics
- fundamentals_agent: income statements, balance sheets, cash flow, and earnings
- valuation_agent: synthesizes findings into a professional financial report and must always be included

Return strict JSON only:
{{
  "selected_agents": ["market_data_agent", "fundamentals_agent", "valuation_agent"],
  "financial_constraints": {{
    "ticker": "",
    "analysis_type": "",
    "timeframe": "",
    "focus_metrics": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""
    try:
        sup_raw = _llm_text("Financial supervisor routing. Return strict JSON only.", supervisor_prompt)
        parsed = _json_from_llm(sup_raw)
        requested = parsed.get("selected_agents", [])
        selected_agents = [name for name in AGENT_ORDER if name in requested and name in KNOWN_AGENTS]
        if "valuation_agent" not in selected_agents:
            selected_agents.append("valuation_agent")

        constraints = _empty_constraints()
        if isinstance(parsed.get("financial_constraints"), dict):
            constraints.update(parsed["financial_constraints"])

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception:
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = "Fallback selected full financial workflow."

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "financial_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Financial routing plan created.")],
        "llm_calls": llm_calls,
    }


def guardrail_blocked_agent(state: FinanceState):
    reason = state.get("final_response") or "Request blocked."
    return {"final_response": reason, "messages": [AIMessage(content=reason)]}


def market_data_agent(state: FinanceState):
    ticker = extract_stock_ticker(state["user_query"])
    try:
        quote = asyncio.run(financial_mcp_call("get_stock_quote", {"symbol": ticker}))
        news = asyncio.run(tavily_mcp_search(f"Latest financial news and sentiment for {ticker}"))
        market_data = f"Stock Quote for {ticker}:\n{quote}\n\nRecent News:\n{news}"
    except Exception as exc:
        market_data = f"Market data retrieval error for {ticker}: {exc}"

    return {
        "market_data": market_data,
        "messages": [AIMessage(content="Market data fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def fundamentals_agent(state: FinanceState):
    ticker = extract_stock_ticker(state["user_query"])
    try:
        income = asyncio.run(financial_mcp_call("get_income_statement", {"symbol": ticker}))
        balance = asyncio.run(financial_mcp_call("get_balance_sheet", {"symbol": ticker}))
        fundamentals_data = f"Income Statement:\n{income}\n\nBalance Sheet:\n{balance}"
    except Exception as exc:
        fundamentals_data = f"Fundamentals retrieval error: {exc}"

    return {
        "fundamentals_data": fundamentals_data,
        "messages": [AIMessage(content="Fundamentals processed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def valuation_agent(state: FinanceState):
    prompt = f"""
Create a comprehensive financial analysis report.

User Query:
{state['user_query']}

Constraints:
{state.get('financial_constraints', {})}

Market Data:
{state.get('market_data', '')}

Fundamentals:
{state.get('fundamentals_data', '')}

Prepare a detailed draft report ready for review.
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are an expert senior equity research analyst."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = "Please review the financial draft report. Approve to finalize or provide revision feedback."

    return {
        "valuation_report": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Valuation draft created for review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def human_approval_agent(state: FinanceState):
    review = interrupt(
        {
            "question": "Do you approve this financial report?",
            "draft_report": state.get("valuation_report", ""),
            "approval_request": state.get("approval_request", ""),
            "expected_response": {"approved": True, "feedback": ""},
        }
    )
    return {
        "approved": bool(review.get("approved", False)),
        "human_feedback": str(review.get("feedback", "")).strip(),
        "messages": [AIMessage(content="Approval step completed.")],
    }


def final_agent(state: FinanceState):
    if state.get("approved", False):
        review_instruction = "User approved draft. Polish the final report."
    else:
        review_instruction = f"Apply revision feedback: {state.get('human_feedback', '')}"

    prompt = f"""
Generate final financial report.
Review Instructions: {review_instruction}
Query: {state['user_query']}
Market Data: {state.get('market_data', '')}
Fundamentals: {state.get('fundamentals_data', '')}
Draft: {state.get('valuation_report', '')}

Format using sections:
1. Executive Summary
2. Market Performance & Sentiment
3. Financial Health & Statements (Income Statement, Balance Sheet)
4. Valuation & Key Metrics
5. Investment Recommendation / Conclusion
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are a certified financial analyst."),
            HumanMessage(content=prompt),
        ]
    )
    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "market_data_agent": "market_data_agent",
    "fundamentals_agent": "fundamentals_agent",
    "valuation_agent": "valuation_agent",
}


def _selected_agents(state: FinanceState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: FinanceState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"
    selected = _selected_agents(state)
    return selected[0] if selected else "valuation_agent"


def route_after_agent(current_agent: str):
    def route(state: FinanceState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)
        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent
        return "valuation_agent"
    return route


graph = StateGraph(FinanceState)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("market_data_agent", market_data_agent)
graph.add_node("fundamentals_agent", fundamentals_agent)
graph.add_node("valuation_agent", valuation_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)
graph.add_conditional_edges("market_data_agent", route_after_agent("market_data_agent"), ROUTE_MAP)
graph.add_conditional_edges("fundamentals_agent", route_after_agent("fundamentals_agent"), ROUTE_MAP)
graph.add_edge("valuation_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

DATABASE_URL = get_database_url()
_conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

finance_graph = graph.compile(checkpointer=checkpointer)


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    first = interrupts[0]
    payload = getattr(first, "value", first)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_msg = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_msg
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_report") or result.get("valuation_report", "")

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": interrupt_payload.get("approval_request", "") if interrupt_payload else result.get("approval_request", ""),
        "market_data": result.get("market_data", ""),
        "fundamentals_data": result.get("fundamentals_data", ""),
        "valuation_report": result.get("valuation_report", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_finance_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    result = finance_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "selected_agents": [],
            "financial_constraints": _empty_constraints(),
            "llm_calls": 0,
        },
        config=config,
    )
    return _serialize_result(result, thread_id)


def resume_finance_agent(thread_id: str, approved: bool, feedback: str = ""):
    if not thread_id:
        raise ValueError("thread_id is required.")
    config = {"configurable": {"thread_id": thread_id}}
    result = finance_graph.invoke(
        Command(resume={"approved": approved, "feedback": feedback.strip()}),
        config=config,
    )
    return _serialize_result(result, thread_id)