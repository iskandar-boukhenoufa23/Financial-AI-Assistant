# FinSight AI — Multi-Agent Financial Analyst

A demo multi-agent financial system built with **LangGraph** and **MCP (Model Context Protocol)** to implement an AI-powered financial analyst assistant with Supervisor orchestration, input Guardrails, and Human-in-the-Loop (HITL) approval workflows.

The project demonstrates how specialized financial agents, external financial data tools, and human approval mechanisms can work together to create a structured, safe, and reviewable financial analysis pipeline.

## Key Concepts

- **Multi-Agent Architecture:** Coordinate specialized AI agents using LangGraph.
- **Supervisor Agent:** Manage and orchestrate complex financial analysis workflows.
- **Input Guardrails:** Validate incoming requests and restrict unsupported topics.
- **Human-in-the-Loop (HITL):** Require human approval before finalizing generated financial reports.
- **Model Context Protocol (MCP):** Connect AI agents to external financial data and research tools.

## Features

- Interactive web interface for stock and company research.
- AI-powered financial analysis and report drafting.
- Supervisor-based multi-agent orchestration.
- Input validation and financial-topic guardrails.
- Human approval workflow for generated reports.
- Alpha Vantage MCP integration for stock quotes and financial fundamentals.
- Tavily MCP integration for financial news and market research.
- FastAPI endpoints for analysis and report approval.
- Persistent conversation threads for managing analysis workflows.

## Architecture

The system follows a supervisor-based multi-agent architecture in which the Supervisor coordinates financial research and analysis tasks.

```text
                    User
                     |
                     v
              FastAPI Application
                     |
                     v
               Input Guardrails
                     |
                     v
              Supervisor Agent
                     |
          +----------+----------+
          |          |          |
          v          v          v
      Financial   Market     Research
      Analysis    Data       Agent
       Agent      Agent        |
          |          |          |
          v          v          v
       Alpha      Alpha       Tavily
      Vantage    Vantage        MCP
        MCP        MCP
          |          |          |
          +----------+----------+
                     |
                     v
              Analysis Results
                     |
                     v
               Report Draft
                     |
                     v
             Human Approval
                     |
              +------+------+
              |             |
              v             v
           Approved      Revisions
              |             |
              v             v
          Final Report   Supervisor
```

The diagram illustrates the intended workflow; the actual agent implementation and routing depend on the configured LangGraph graph.

## Project Structure

```text
FinSight-AI/
│
├── app.py                  # FastAPI application and API endpoints
├── backend.py              # Core agent orchestration and financial logic
├── mcp_client.py           # MCP client helpers
│
├── templates/              # HTML templates
│
├── static/                 # CSS, JavaScript, and frontend assets
│
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
├── .env.example            # Example environment configuration
├── .gitignore              # Git ignore rules
└── README.md
```

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| FastAPI | Backend application and REST API |
| LangGraph | Agent orchestration and workflow management |
| LangChain | LLM and tool integration |
| MCP | External tool communication |
| Alpha Vantage | Stock market data and financial fundamentals |
| Tavily | Web search and market research |
| HTML, CSS, JavaScript | Web interface |
| Uvicorn | ASGI application server |

## Prerequisites

Before running the project, make sure you have:

- Python 3.10 or newer.
- Git.
- Alpha Vantage API key.
- Tavily API key.
- A virtual environment tool such as `venv`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/FinSight-AI.git

cd FinSight-AI
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

```env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Add any additional LLM provider credentials or configuration required by your implementation.

**Important:** Never commit API keys, credentials, or other sensitive information to GitHub.

### 5. Run the application

There are two ways to start the FastAPI application.

**Option A — Run the application directly:**

```bash
python app.py
```

**Option B — Run with Uvicorn:**

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 6. Open the web application

Once the server is running, open:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

You can now interact with the FinSight AI financial analyst through the web interface.

## API Documentation

The application exposes the following REST API endpoints.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/finance` | Submit a financial analysis request |
| `POST` | `/api/finance/approve` | Approve or request revisions to a report |
| `GET` | `/health` | Check application health and available features |

### 1. Financial Analysis

**Endpoint:**

```http
POST /api/finance
```

**Request body:**

```json
{
  "message": "Analyze NVIDIA's financial performance and market outlook.",
  "thread_id": "optional-thread-id"
}
```

The endpoint processes the financial research request and manages the corresponding analysis thread.

### 2. Human Approval

**Endpoint:**

```http
POST /api/finance/approve
```

**Request body:**

```json
{
  "thread_id": "your-thread-id",
  "approved": true,
  "feedback": "The financial analysis looks good."
}
```

The approval endpoint allows a human reviewer to approve a generated report or reject it and provide feedback for revisions.

### 3. Health Check

**Endpoint:**

```http
GET /health
```

Returns the application's health status and supported features.

## Human-in-the-Loop Workflow

FinSight AI incorporates human approval into the financial reporting process.

The workflow is designed to:

1. Receive a financial analysis request.
2. Validate the request using input guardrails.
3. Delegate tasks to specialized agents through the Supervisor.
4. Retrieve financial data and market research through MCP.
5. Generate a preliminary financial report.
6. Pause execution and request human approval.
7. Accept approval or feedback for revisions.
8. Continue the workflow based on the human decision.

This approach makes the analysis process more transparent and allows human reviewers to inspect generated reports before finalization.

## MCP Integrations

### Alpha Vantage MCP

Used to retrieve financial market information, including:

- Stock quotes.
- Company fundamentals.
- Financial statements.
- Market-related indicators.

### Tavily MCP

Used to retrieve external research and market information, including:

- Financial news.
- Company-related articles.
- Market developments.
- Relevant web research.

These integrations allow the financial agents to access external information rather than relying exclusively on the language model's internal knowledge.

## Development Notes

- FastAPI provides the web application and API endpoints.
- LangGraph manages agent coordination and workflow execution.
- MCP provides a standardized interface for external tools.
- `backend.py` contains the core financial analysis and orchestration logic.
- `mcp_client.py` provides helpers for communicating with external MCP servers.
- `nest_asyncio` is applied in `app.py` to support synchronous convenience wrappers that interact with asynchronous MCP helpers.
- Tests are not currently included.

## Future Improvements

Potential improvements include:

- Adding automated financial report evaluation.
- Introducing specialized risk assessment agents.
- Supporting portfolio analysis and asset allocation.
- Adding historical market data analysis.
- Implementing structured financial report generation.
- Improving agent observability and workflow tracing.
- Adding automated testing for guardrails and approval workflows.

## Disclaimer

FinSight AI is an educational and experimental project demonstrating multi-agent AI architectures for financial research and analysis.

Generated reports may contain inaccurate or incomplete information. They are not professional financial advice and should not be used as the sole basis for investment decisions.

Always verify financial information using reliable sources before making investment decisions.

## Contributing

Contributions are welcome!

If you have ideas for improvements, bug fixes, or additional integrations:

1. Fork the repository.
2. Create a new feature branch.
3. Implement your changes.
4. Submit a pull request.

## License

This project follows the license specified in the `LICENSE` file.

## Acknowledgements

Built as a demonstration of:

- LangGraph multi-agent orchestration.
- Model Context Protocol (MCP).
- Supervisor-based agent architectures.
- Input guardrails.
- Human-in-the-Loop workflows.

---

**FinSight AI — Building safer and more transparent AI-powered financial analysis systems.**
