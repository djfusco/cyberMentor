"""FastAPI app entrypoint for the beta export ONLY (see
scripts/beta_manifest.txt and scripts/export_beta.py).

This is a new, additive module -- app/main.py is NOT modified. It mounts
only the routers needed for the Practice flow (view/start an exercise, set
difficulty on an Open exercise, in-session mentor chat, finish + see
results, export your own signed result, import an exercise) and omits
everything instructor-only, authoring, analytics, research, settings, and
review-related. The normal app (python run.py) is completely unaffected and
continues to use app/main.py as before.
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.database import init_db
from app.routes import exercises, mentor, pages_beta, sessions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AI Exercise Mentor (beta)")

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

app.include_router(pages_beta.router)
app.include_router(exercises.router)
app.include_router(sessions.router)
app.include_router(mentor.router)


@app.on_event("startup")
async def on_startup() -> None:
    init_db()


@app.get("/api/health")
async def health():
    """Same shape as app/main.py's health endpoint (evidence provider status
    + mentor chat backend status), sharing app/services/evidence_status.py
    and app/services/chat_status.py rather than duplicating that logic here."""
    from app.dependencies import get_evidence_provider, get_mentor_chat_backend
    from app.services.chat_status import get_chat_status
    from app.services.evidence_status import get_evidence_status

    provider = get_evidence_provider()
    chat_backend = get_mentor_chat_backend()

    evidence_status = await get_evidence_status(provider)
    chat_status = await get_chat_status(chat_backend)

    return {
        "evidence": evidence_status,
        "chat": chat_status,
    }
