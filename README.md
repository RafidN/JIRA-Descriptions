# Hackathon FastAPI + Mounted Frontend

A minimal scaffold with a FastAPI backend and a static frontend served from `frontend/`.
The frontend is mounted at `/`, and it calls two JSON endpoints: `/endpoint-1` and `/endpoint-2`.

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

### 2) Install dependencies
```bash
pip install -r requirements.txt
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
hackathon_fastapi_frontend/
├─ app.py                # FastAPI app with endpoints
├─ requirements.txt
├─ README.md
└─ frontend/
   ├─ index.html         # Basic page mounted at /
   ├─ styles.css         # Styling
   └─ script.js             # Calls the two endpoints and renders JSON
```

## Notes
- The frontend uses **relative fetches** (`/endpoint-1`, `/endpoint-2`), so no extra CORS setup is needed during local dev.
- If you change the port (e.g., `uvicorn app:app --reload --port 9000`), just open `http://127.0.0.1:9000/`.
- For production, consider running via a process manager (e.g., `gunicorn` with `uvicorn.workers.UvicornWorker`) and a reverse proxy (e.g., Nginx).
