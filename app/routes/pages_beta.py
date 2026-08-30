"""Practice-only page routes for the beta export (see scripts/beta_manifest.txt
and scripts/export_beta.py).

This is a new, additive module -- app/routes/pages.py is NOT modified and is
NOT exported to the beta build. It bundles Practice pages together with
Settings/Lab Chat/Mentor Review/History/Instructor pages in one file, so
including it (or splitting it) was out of scope for this change; instead,
this module duplicates just the four Practice-relevant route functions
(index, exercise detail, active session, results) so the beta app can run
standalone without ever shipping pages.py's instructor-page source.

Used only by app/main_beta.py; the normal app (app/main.py, run.py) is
completely unaffected and continues to use app/routes/pages.py as before.
"""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from app.database import get_session
from app.dependencies import get_exercise_service
from app.models.session import ExerciseSession

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    exercises = get_exercise_service().list_exercises()
    return templates.TemplateResponse(request, "index.html", {"exercises": exercises})


@router.get("/exercises/{exercise_id}", response_class=HTMLResponse)
async def exercise_detail(request: Request, exercise_id: str):
    exercise = get_exercise_service().get_exercise(exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail="Exercise not found")
    return templates.TemplateResponse(request, "exercise.html", {"exercise": exercise})


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_page(request: Request, session_id: int, db: Session = Depends(get_session)):
    session = db.get(ExerciseSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    exercise = get_exercise_service().get_exercise(session.exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail="Exercise not found")
    return templates.TemplateResponse(
        request, "session.html", {"session": session, "exercise": exercise}
    )


@router.get("/sessions/{session_id}/results", response_class=HTMLResponse)
async def results_page(request: Request, session_id: int, db: Session = Depends(get_session)):
    session = db.get(ExerciseSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    exercise = get_exercise_service().get_exercise(session.exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail="Exercise not found")
    return templates.TemplateResponse(
        request, "results.html", {"session": session, "exercise": exercise}
    )
