from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from app.database import get_session
from app.dependencies import get_exercise_service, get_mentor_service
from app.models.session import ExerciseSession, SessionStatus

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

    answer = await get_mentor_service().ask(db, session, exercise, payload.question, payload.help_level)
    return {"answer": answer}


@router.get("/{session_id}/mentor/history")
async def mentor_history(session_id: int, db: Session = Depends(get_session)):
    session = db.get(ExerciseSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    history = get_mentor_service().get_history(db, session_id)
    return [{"role": m.role.value, "message": m.message, "created_at": m.created_at} for m in history]
