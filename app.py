from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Hackathon FastAPI + Frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/generateDescription")
async def endpoint_1(ticket_title: str = Query(..., alias="ticket-title")):
    
    return {
        "message": f"Hello from endpoint-1 — got '{ticket_title}'",
        "ok": True,
        "ticket-title": ticket_title
    }