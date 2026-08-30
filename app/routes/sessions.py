import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.database import get_session
from app.dependencies import (
    get_evaluator_service,
    get_evidence_provider,
    get_evidence_service,
    get_exercise_service,
)
from app.models.evaluation import Evaluation, EvaluationResult
from app.models.exercise import DifficultyLevel, Exercise
from app.models.session import ExerciseSession, SessionStatus
from app.services import settings_service
from app.services.evidence_provider import EvidenceProviderError
from app.services.submission import build_submission_export

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    exercise_id: str
    # Only meaningful (and required) when the exercise's own difficulty is
    # "open" -- the student's per-session choice. Ignored otherwise.
    student_difficulty: Optional[DifficultyLevel] = None


def _session_or_404(db: Session, session_id: int) -> ExerciseSession:
    session = db.get(ExerciseSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


def _resolve_student_difficulty(
    exercise: Exercise, requested: Optional[DifficultyLevel]
) -> Optional[DifficultyLevel]:
    """For an Open exercise, the student must choose beginner/intermediate/
    advanced before starting -- that choice applies to this session only
    (see ExerciseSession.student_difficulty). Non-open exercises ignore any
    requested value; the instructor's own difficulty always applies.
    """
    if exercise.difficulty != DifficultyLevel.open:
        return None
    if requested is None or requested == DifficultyLevel.open:
        raise ValueError(
            "This exercise is Open difficulty -- choose beginner, intermediate, or advanced to start."
        )
    return requested


@router.post("")
async def create_session(payload: CreateSessionRequest, db: Session = Depends(get_session)):
    exercise = get_exercise_service().get_exercise(payload.exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{payload.exercise_id}' not found")

    try:
        student_difficulty = _resolve_student_difficulty(exercise, payload.student_difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    session = ExerciseSession(
        exercise_id=exercise.id,
        student_name=settings_service.get_student_name(db),
        started_at=datetime.now(timezone.utc),
        status=SessionStatus.active,
        student_difficulty=student_difficulty,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    try:
        await get_evidence_provider().start_session(str(session.id))
    except Exception as exc:  # noqa: BLE001 -- capture startup must never block session creation
        logger.warning("Evidence provider could not start capture for session %s: %s", session.id, exc)

    return session


@router.get("/{session_id}")
async def get_session_detail(session_id: int, db: Session = Depends(get_session)):
    return _session_or_404(db, session_id)


@router.get("/{session_id}/evidence")
async def get_session_evidence(session_id: int, db: Session = Depends(get_session)):
    session = _session_or_404(db, session_id)
    try:
        events = await get_evidence_service().get_session_activity(session.started_at, session.ended_at)
    except EvidenceProviderError as exc:
        return {"error": str(exc), "events": [], "count": 0}
    return {"error": None, "events": [e.to_dict() for e in events], "count": len(events)}


@router.get("/{session_id}/evaluation")
async def get_session_evaluation(session_id: int, db: Session = Depends(get_session)):
    session = _session_or_404(db, session_id)
    statement = select(Evaluation).where(Evaluation.session_id == session.id)
    evaluation = db.exec(statement).first()
    if evaluation is None:
        raise HTTPException(status_code=404, detail="No evaluation yet for this session")
    return {
        "score": evaluation.score,
        "summary": evaluation.summary,
        "details": json.loads(evaluation.evaluation_json),
        "created_at": evaluation.created_at,
    }


@router.get("/{session_id}/export")
async def export_session(session_id: int, db: Session = Depends(get_session)):
    """Download a signed result bundle to share with an instructor.

    Signing is best-effort tamper-evidence (see app/services/submission.py) --
    it catches accidental edits to the exported file, not a student willing
    to modify their own app before running the exercise.
    """
    session = _session_or_404(db, session_id)
    if session.status != SessionStatus.completed:
        raise HTTPException(status_code=400, detail="Finish the exercise before exporting a result")

    exercise = get_exercise_service().get_exercise(session.exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{session.exercise_id}' not found")

    statement = select(Evaluation).where(Evaluation.session_id == session.id)
    evaluation_row = db.exec(statement).first()
    if evaluation_row is None:
        raise HTTPException(status_code=404, detail="No evaluation yet for this session")

    evaluation = EvaluationResult.model_validate(json.loads(evaluation_row.evaluation_json))
    bundle = build_submission_export(session, evaluation, exercise)
    filename = f"submission-{exercise.id}-{session.id}.json"
    return JSONResponse(bundle, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.post("/{session_id}/finish")
async def finish_session(session_id: int, db: Session = Depends(get_session)):
    session = _session_or_404(db, session_id)
    if session.status != SessionStatus.active:
        raise HTTPException(status_code=400, detail=f"Session is already {session.status.value}")

    exercise = get_exercise_service().get_exercise(session.exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{session.exercise_id}' not found")

    session.ended_at = datetime.now(timezone.utc)

    try:
        await get_evidence_provider().stop_session(str(session.id))
    except Exception as exc:  # noqa: BLE001 -- must not block finishing/evaluating the session
        logger.warning("Evidence provider could not stop capture for session %s: %s", session.id, exc)

    evidence_error: str | None = None
    try:
        events = await get_evidence_service().get_session_activity(session.started_at, session.ended_at)
    except EvidenceProviderError as exc:
        logger.warning("Evidence retrieval failed at finish: %s", exc)
        events = []
        evidence_error = str(exc)

    result = await get_evaluator_service().evaluate(
        exercise, events, evidence_error=evidence_error, session_id=str(session.id)
    )

    evaluation = Evaluation(
        session_id=session.id,
        score=result.score,
        summary=result.summary,
        evaluation_json=result.model_dump_json(),
    )
    session.status = SessionStatus.completed

    db.add(session)
    db.add(evaluation)
    db.commit()
    db.refresh(session)
    db.refresh(evaluation)

    return {
        "session": session,
        "evaluation": {
            "score": evaluation.score,
            "summary": evaluation.summary,
            "details": result.model_dump(),
        },
    }
