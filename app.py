"""Standalone FastAPI service for the issue triage pipeline."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Load .env from the project root before reading any os.getenv() calls.
# This is a no-op when the variables are already set (e.g. in Docker / CI).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv is optional; env vars must be set another way

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import Column, DateTime, Integer, JSON, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from starlette.concurrency import run_in_threadpool

LOG = logging.getLogger("triage")

APP_ENV = os.getenv("APP_ENV", "development").lower()
IS_PRODUCTION = APP_ENV == "production"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./triage.db")
API_KEY = os.getenv("TRIAGE_API_KEY")
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")
WEBHOOK_ENABLED = os.getenv("GITHUB_WEBHOOK_ENABLED", "false").lower() == "true"
WRITEBACK_ENABLED = os.getenv("GITHUB_WRITEBACK", "false").lower() == "true"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
MAX_WEBHOOK_BYTES = 1_000_000

# ---------------------------------------------------------------------------
# Rate limiter — simple in-process sliding-window counter.
# Default: 60 requests per 60 seconds per API key (or per remote IP when no
# key is configured).  Override via env vars RATE_LIMIT_REQUESTS /
# RATE_LIMIT_WINDOW_SECONDS.
# ---------------------------------------------------------------------------
_RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
_RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

_rate_buckets: dict[str, deque] = {}
_rate_lock = threading.Lock()


def _check_rate_limit(client_id: str) -> None:
    """Raise HTTP 429 if *client_id* has exceeded the sliding-window quota."""
    now = time.monotonic()
    cutoff = now - _RATE_LIMIT_WINDOW
    with _rate_lock:
        bucket = _rate_buckets.setdefault(client_id, deque())
        # Evict timestamps outside the current window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded — max {_RATE_LIMIT_REQUESTS} requests "
                    f"per {_RATE_LIMIT_WINDOW}s."
                ),
                headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
            )
        bucket.append(now)


Base = declarative_base()


class TriageResult(Base):
    __tablename__ = "triage_results"

    id = Column(Integer, primary_key=True)
    repo = Column(String(300), nullable=False, index=True)
    issue_number = Column(Integer, nullable=False, index=True)
    title = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False)
    duplicate_of = Column(Integer, nullable=True)
    files = Column(JSON, nullable=True)
    suspect_sha = Column(String(64), nullable=True)
    assignees = Column(JSON, nullable=True)
    repro_test = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class IssueIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=500)
    body: str = Field(default="", max_length=50_000)


def persist(session_factory, state: dict[str, Any], repo: str) -> None:
    issue = state["issue"]
    with session_factory() as session:
        session.add(TriageResult(
            repo=repo,
            issue_number=issue["id"],
            title=issue["title"],
            severity=state["severity"],
            duplicate_of=(state.get("duplicate") or {}).get("issue_id"),
            files=state.get("files"),
            suspect_sha=(state.get("suspects") or [{}])[0].get("sha"),
            assignees=state.get("assignees"),
            repro_test=state.get("repro_test"),
        ))
        session.commit()


def post_comment(repo: str, number: int, report: str) -> None:
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=503, detail="GitHub write-back requires GITHUB_TOKEN")
    response = requests.post(
        f"https://api.github.com/repos/{repo}/issues/{number}/comments",
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"body": report},
        timeout=(5, 20),
    )
    if not response.ok:
        LOG.error("GitHub comment write failed: %s %s", response.status_code, response.text[:500])
        raise HTTPException(status_code=502, detail="GitHub comment write failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if IS_PRODUCTION and not API_KEY:
        raise RuntimeError("TRIAGE_API_KEY must be set in production")
    if WEBHOOK_ENABLED and not WEBHOOK_SECRET:
        raise RuntimeError("GITHUB_WEBHOOK_SECRET is required when webhooks are enabled")
    if WRITEBACK_ENABLED and not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is required when GitHub write-back is enabled")
    if os.getenv("TRIAGE_MODE", "synthetic").lower() == "github" and not os.getenv("GITHUB_REPOSITORY"):
        raise RuntimeError("GITHUB_REPOSITORY is required when TRIAGE_MODE=github")

    db = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(db)
    app.state.db = db
    app.state.sessions = sessionmaker(bind=db, expire_on_commit=False)

    # Heavy corpus/model/index initialization happens once per service process.
    import triage_engine as pipeline
    app.state.pipeline = pipeline
    app.state.pipeline_lock = threading.Lock()
    LOG.info(
        "Triage pipeline ready (mode=%s, embeddings=%s)",
        pipeline.CFG["MODE"],
        pipeline.EMBEDDING_BACKEND,
    )
    try:
        yield
    finally:
        db.dispose()
        try:
            pipeline.qdr.close()
        except Exception:
            pass


app = FastAPI(
    title="Agentic Issue Triage",
    version="1.0.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
    lifespan=lifespan,
)


def require_api_key(supplied: str | None) -> None:
    """Validate the supplied API key using a constant-time comparison.

    Rules:
    - If TRIAGE_API_KEY is set, it is **always** enforced (dev and production).
    - In production mode, the key must also be configured or the service is
      considered mis-configured and returns 503.
    """
    if API_KEY:
        if not supplied or not hmac.compare_digest(supplied, API_KEY):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    elif IS_PRODUCTION:
        # Production with no key configured → fail closed
        raise HTTPException(status_code=503, detail="API key is not configured")


def run_triage(app: FastAPI, issue: dict[str, Any], repo: str) -> dict[str, Any]:
    pipeline = app.state.pipeline
    with app.state.pipeline_lock:
        state = pipeline.triage_issue(issue)
    persist(app.state.sessions, state, repo)
    return state


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready(request: Request):
    if not hasattr(request.app.state, "pipeline"):
        raise HTTPException(status_code=503, detail="Triage pipeline is not initialized")
    try:
        with request.app.state.db.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        LOG.exception("Readiness database check failed")
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ready", "mode": request.app.state.pipeline.CFG["MODE"]}


@app.post("/triage")
def triage(
    issue: IssueIn,
    request: Request,
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)
    # Rate-limit by API key when present, otherwise by remote IP
    client_id = x_api_key or (request.client.host if request.client else "unknown")
    _check_rate_limit(client_id)

    state = run_triage(
        request.app,
        {
            "id": issue.number,
            "title": issue.title,
            "body": issue.body,
            "created": time.time(),
            "severity": None,
        },
        repo="api",
    )
    return {
        "severity": state["severity"],
        "duplicate": state.get("duplicate"),
        "files": state.get("files"),
        "suspects": state.get("suspects"),
        "assignees": state.get("assignees"),
        "repro_test": state.get("repro_test"),
        "report": state["report"],
    }


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
):
    if not WEBHOOK_ENABLED:
        raise HTTPException(status_code=404, detail="GitHub webhook is disabled")

    # Guard: WEBHOOK_SECRET must be set (lifespan enforces this, but be
    # explicit here so an AttributeError can never surface as a 500).
    if not WEBHOOK_SECRET:
        LOG.error("Webhook received but GITHUB_WEBHOOK_SECRET is not configured")
        raise HTTPException(status_code=503, detail="Webhook secret is not configured")

    declared_size = request.headers.get("content-length")
    if declared_size and declared_size.isdigit() and int(declared_size) > MAX_WEBHOOK_BYTES:
        raise HTTPException(status_code=413, detail="Webhook payload is too large")

    body = await request.body()
    if len(body) > MAX_WEBHOOK_BYTES:
        raise HTTPException(status_code=413, detail="Webhook payload is too large")

    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    if not x_hub_signature_256 or not hmac.compare_digest(x_hub_signature_256, expected):
        raise HTTPException(status_code=401, detail="Invalid GitHub webhook signature")

    if x_github_event != "issues":
        return {"ignored": True, "reason": "unsupported event"}

    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    if payload.get("action") != "opened" or "issue" not in payload:
        return {"ignored": True, "reason": "not a newly opened issue"}

    issue = payload["issue"]
    repo = payload.get("repository", {}).get("full_name")

    if isinstance(issue, dict) and "pull_request" in issue:
        return {"ignored": True, "reason": "pull requests are not issue tickets"}

    if not isinstance(issue, dict) or not repo:
        raise HTTPException(status_code=400, detail="Missing repository or issue fields")

    try:
        normalized_issue = IssueIn.model_validate({
            "number": issue.get("number"),
            "title": issue.get("title"),
            "body": issue.get("body") or "",
        })
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="Invalid GitHub issue fields") from exc

    state = await run_in_threadpool(
        run_triage,
        request.app,
        {
            "id": normalized_issue.number,
            "title": normalized_issue.title,
            "body": normalized_issue.body,
            "created": time.time(),
            "severity": None,
        },
        repo,
    )

    if WRITEBACK_ENABLED:
        await run_in_threadpool(post_comment, repo, normalized_issue.number, state["report"])

    return {"ok": True, "severity": state["severity"], "report": state["report"]}
