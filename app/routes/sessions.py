import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import get_settings
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
from app.services.capture_manifest import ManifestWriter
from app.services.evidence_provider import EvidenceProviderError
from app.services.submission import build_submission_export
from app.services.verifier import deserialize_verification, run_verification, serialize_verification

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

    # Snapshot deterministic verification BEFORE the student does anything
    # this session, so Finish can tell "produced this session" apart from
    # "already there from a prior attempt" -- see
    # EvaluatorService._score_filesystem_outcome. Must never block starting
    # a session (e.g. a transient filesystem error reading state).
    try:
        baseline_verification_json = serialize_verification(run_verification(exercise))
    except Exception as exc:  # noqa: BLE001 -- baseline capture must never block starting a session
        logger.warning("Baseline verification failed for exercise %s: %s", exercise.id, exc)
        baseline_verification_json = None

    capture_run_id = str(uuid.uuid4())
    session = ExerciseSession(
        exercise_id=exercise.id,
        student_name=settings_service.get_student_name(db),
        started_at=datetime.now(timezone.utc),
        status=SessionStatus.active,
        student_difficulty=student_difficulty,
        capture_run_id=capture_run_id,
        baseline_verification_json=baseline_verification_json,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    try:
        await get_evidence_provider().start_session(capture_run_id)
    except Exception as exc:  # noqa: BLE001 -- capture startup must never block session creation
        logger.warning("Evidence provider could not start capture for session %s (run %s): %s", session.id, capture_run_id, exc)

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

    run_id = session.capture_run_id or str(session.id)
    try:
        await get_evidence_provider().stop_session(run_id)
    except Exception as exc:  # noqa: BLE001 -- must not block finishing/evaluating the session
        logger.warning("Evidence provider could not stop capture for session %s (run %s): %s", session.id, run_id, exc)

    evidence_error: str | None = None
    try:
        events = await get_evidence_service().get_session_activity(session.started_at, session.ended_at)
    except EvidenceProviderError as exc:
        logger.warning("Evidence retrieval failed at finish: %s", exc)
        events = []
        evidence_error = str(exc)

    settings = get_settings()
    provider = get_evidence_provider()
    capture_dir = Path(provider.output_dir) / run_id if hasattr(provider, "output_dir") else None
    manifest: Optional[ManifestWriter] = None
    if capture_dir is not None:
        manifest = ManifestWriter(
            capture_dir=capture_dir,
            db_session_id=session.id,
            capture_run_id=run_id,
            exercise_id=session.exercise_id,
            provider_name=type(provider).__name__,
            diagnostics_enabled=getattr(settings, "mentor_diagnostics", False),
        )
        manifest.set_session_metadata(
            started_at=session.started_at,
            ended_at=session.ended_at,
            raw_event_count=len(events),
            normalized_event_count=len(events),
            available_frames=[
                {"path": e.frame_path, "timestamp": e.timestamp.isoformat()}
                for e in events if e.frame_path
            ],
        )

    baseline_verification = deserialize_verification(session.baseline_verification_json)
    result = await get_evaluator_service().evaluate(
        exercise, events, evidence_error=evidence_error, session_id=run_id, manifest=manifest,
        baseline_verification=baseline_verification,
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


@router.post("/{session_id}/retry-evaluation")
async def retry_evaluation(session_id: int, db: Session = Depends(get_session)):
    """Re-run evaluation for an already-completed session from its PERSISTED
    evidence (capture_sessions/<run_id>/events.jsonl) -- no new capture
    session is started, and stop_session() is never called again (the
    capture already ended when this session was finished).

    Exists so a student whose evaluation came back "unavailable" (e.g. the
    AI evaluator timed out, even after the built-in retry with a smaller
    payload) is never stuck redoing the entire exercise just to get a real
    score for work that was already captured. Replaces any existing
    Evaluation row for this session, mirroring finish_session's persistence.
    """
    session = _session_or_404(db, session_id)
    if session.status != SessionStatus.completed:
        raise HTTPException(
            status_code=400, detail="Only a completed session's evaluation can be retried"
        )

    exercise = get_exercise_service().get_exercise(session.exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{session.exercise_id}' not found")

    run_id = session.capture_run_id or str(session.id)
    events = await get_evidence_service().get_persisted_session_activity(run_id)

    baseline_verification = deserialize_verification(session.baseline_verification_json)
    result = await get_evaluator_service().evaluate(
        exercise, events, session_id=run_id, baseline_verification=baseline_verification,
    )

    existing = db.exec(select(Evaluation).where(Evaluation.session_id == session.id)).all()
    for row in existing:
        db.delete(row)
    evaluation = Evaluation(
        session_id=session.id,
        score=result.score,
        summary=result.summary,
        evaluation_json=result.model_dump_json(),
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    return {
        "session": session,
        "evaluation": {
            "score": evaluation.score,
            "summary": evaluation.summary,
            "details": result.model_dump(),
        },
    }
