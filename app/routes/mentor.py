import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from app.config import get_settings
from app.database import get_session
from app.dependencies import get_evidence_provider, get_exercise_service, get_mentor_service
from app.models.session import ExerciseSession, SessionStatus
from app.services.capture_manifest import ManifestWriter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["mentor"])


class MentorQuestionRequest(BaseModel):
    question: str
    help_level: Optional[int] = None


@router.post("/{session_id}/mentor")
async def ask_mentor(session_id: int, payload: MentorQuestionRequest, db: Session = Depends(get_session)):
    session = db.get(ExerciseSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    if session.status != SessionStatus.active:
        raise HTTPException(status_code=400, detail="Session is not active")

    exercise = get_exercise_service().get_exercise(session.exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{session.exercise_id}' not found")
    if not exercise.mentor.allow_questions:
        raise HTTPException(status_code=400, detail="This exercise does not allow mentor questions")

    settings = get_settings()
    provider = get_evidence_provider()
    run_id = session.capture_run_id or str(session.id)
    manifest: Optional[ManifestWriter] = None
    if hasattr(provider, "output_dir"):
        capture_dir = Path(provider.output_dir) / run_id
        try:
            manifest = ManifestWriter(
                capture_dir=capture_dir,
                db_session_id=session.id,
                capture_run_id=run_id,
                exercise_id=session.exercise_id,
                provider_name=type(provider).__name__,
                diagnostics_enabled=getattr(settings, "mentor_diagnostics", False),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create ManifestWriter for session %s: %s", session.id, exc)

    answer = await get_mentor_service().ask(
        db, session, exercise, payload.question, payload.help_level, manifest=manifest
    )
    return {"answer": answer}


@router.get("/{session_id}/mentor/history")
async def mentor_history(session_id: int, db: Session = Depends(get_session)):
    session = db.get(ExerciseSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    history = get_mentor_service().get_history(db, session_id)
    return [{"role": m.role.value, "message": m.message, "created_at": m.created_at} for m in history]
