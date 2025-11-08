# Hackathon FastAPI + Mounted Frontend

## Prerequisites
- Python 3.10+ recommended
- `pip` available in your shell

## Quickstart (with virtual environment)

### 1) Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> To deactivate later, run `deactivate`.

### 2) Install dependencies and setup API keys
```bash
pip install -r requirements.txt
cp .env.example .env             # FILL VALUES in the .env with YOUR API keys
```

### 3) Run the server (with auto-reload for dev)
```bash
uvicorn app:app --reload
```

### 4) Open the frontend
- Visit: http://127.0.0.1:8000  
  This serves `frontend/index.html` as the root page and the static files (`/styles.css`, `/app.js`).

## Project Structure
```
JIRA-Descriptions/
├─ app.py                # FastAPI app with endpoints
├─ requirements.txt
├─ README.md
└─ frontend/
   ├─ index.html         # Basic page mounted at /
   ├─ styles.css         # Styling
   └─ app.js             # Calls the two endpoints and renders JSON
```

## Examples

### Example Request (JSON)
```json
{
  "repo_url": "https://github.com/tiangolo/fastapi",
  "jira_title": "Add endpoint to bulk import users with role mapping and validation",
  "max_context_chunks": 12,
  "chunk_size": 1200,
  "overlap": 150
}
```