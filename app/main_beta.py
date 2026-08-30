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
    """Minimal health check for the beta build -- Ollama connectivity only.
    See app/main.py's fuller _evidence_status for the normal app's version."""
    from app.dependencies import get_ollama_service
    from app.config import get_settings

    settings = get_settings()
    ollama = get_ollama_service()
    ollama_ok = await ollama.health()
    return {
        "ollama": {
            "connected": ollama_ok,
            "url": settings.ollama_base_url,
            "model": settings.ollama_model,
            "hint": None if ollama_ok else "Start it with: ollama serve",
        },
    }
