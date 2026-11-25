from pathlib import Path
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from backend import run_finance_agent, resume_finance_agent

import nest_asyncio

nest_asyncio.apply()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="FinSight AI",
    description="Multi-Agent Financial Analyst with Market Data MCP, Fundamentals, and HITL",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class FinanceRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/api/finance")
async def financial_analysis(request_data: FinanceRequest):
    try:
        user_message = request_data.message.strip()
        if not user_message:
            return JSONResponse(status_code=400, content={"success": False, "error": "Query cannot be empty."})

        result = run_finance_agent(user_input=user_message, thread_id=request_data.thread_id)
        return JSONResponse(content={"success": True, **result})
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.post("/api/finance/approve")
async def approve_financial_report(request_data: ApprovalRequest):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(status_code=400, content={"success": False, "error": "Please provide revision feedback."})

        result = resume_finance_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )
        return JSONResponse(content={"success": True, **result})
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "FinSight AI API is running"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)