import os
import time
import uuid
import logging

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load .env early
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(usecwd=True) or ".env")
except Exception:
    pass

from python_scripts.clonegit import check_repos
from python_scripts.embed import build_context_for_ticket
from python_scripts.geminicall import call_gemini

logger = logging.getLogger("app")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

app = FastAPI(
    title="PM Action Items Generator + Frontend",
    description="Given a JIRA ticket title + GitHub repo, generate dev action items.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ActionItemsRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")
    jira_title: str = Field(..., description="JIRA ticket title or short description")
    max_context_chunks: int = Field(12, ge=3, le=40)
    chunk_size: int = Field(1200, ge=300, le=4000)
    overlap: int = Field(150, ge=0, le=1000)

class ActionItemsResponse(BaseModel):
    action_items_markdown: str
    used_files: list[str]
    used_chunks: int

@app.post("/action-items", response_model=ActionItemsResponse)
def action_items(body: ActionItemsRequest):
    req_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    logger.info(f"[{req_id}] START repo_url={body.repo_url} jira_title='{body.jira_title}' "
                f"max_k={body.max_context_chunks} chunk_size={body.chunk_size} overlap={body.overlap}")
    repo_dir = check_repos(body.repo_url)
    logger.info(f"[{req_id}] cloned -> {repo_dir}")
    try:
        t_ctx = time.time()
        logger.info(f"[{req_id}] building retrieval context")
        context_md, used_files, used_chunks = build_context_for_ticket(
            repo_dir=repo_dir,
            query=body.jira_title,
            max_k=body.max_context_chunks,
            chunk_size=body.chunk_size,
            overlap=body.overlap,
        )
        logger.info(f"[{req_id}] context built chunks={used_chunks} files={len(used_files)} "
                    f"elapsed={time.time()-t_ctx:.2f}s")
        if used_chunks == 0:
            logger.warning(f"[{req_id}] no chunks found")
            raise HTTPException(status_code=422, detail="No text/code found to index in repo.")

        t_llm = time.time()
        logger.info(f"[{req_id}] calling Gemini LLM")
        plan_md = call_gemini(jira_title=body.jira_title, context_markdown=context_md)
        logger.info(f"[{req_id}] LLM done elapsed={time.time()-t_llm:.2f}s response_len={len(plan_md)}")
        return ActionItemsResponse(
            action_items_markdown=plan_md,
            used_files=used_files,
            used_chunks=used_chunks,
        )
    except HTTPException:
        logger.exception(f"[{req_id}] HTTPException")
        raise
    except Exception:
        logger.exception(f"[{req_id}] Unhandled exception")
        raise
    finally:
        logger.info(f"[{req_id}] TOTAL elapsed={time.time()-t0:.2f}s")

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")