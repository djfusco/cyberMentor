from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from sqlmodel import Session, select as sqlmodel_select

from app.database import get_session
from app.dependencies import get_exercise_service
from app.models.session import ExerciseSession
from app.services.exercises import ExerciseLoadError, ExerciseService

router = APIRouter(prefix="/api/exercises", tags=["exercises"])


@router.get("")
async def list_exercises():
    exercises = get_exercise_service().list_exercises()
    return [
        {
            "id": e.id,
            "title": e.title,
            "description": e.description,
            "environment": e.environment.type.value,
            "difficulty": e.difficulty.value,
        }
        for e in exercises
    ]


@router.get("/{exercise_id}")
async def get_exercise(exercise_id: str):
    exercise = get_exercise_service().get_exercise(exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    return exercise.model_dump()


@router.get("/{exercise_id}/export")
async def export_exercise(exercise_id: str):
    """Download the raw exercise YAML -- the distribution mechanism for
    giving an exercise to anyone else running this app locally."""
    service = get_exercise_service()
    path = service.get_exercise_file_path(exercise_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    return PlainTextResponse(
        path.read_text(),
        media_type="application/x-yaml",
        headers={"Content-Disposition": f'attachment; filename="{exercise_id}.yaml"'},
    )


@router.post("/import")
async def import_exercise(file: UploadFile = File(...)):
    """Receive an exercise YAML file (e.g. sent by an instructor) and install it locally."""
    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="File must be UTF-8 encoded YAML") from exc

    try:
        exercise = get_exercise_service().save_exercise_yaml(raw_text)
    except ExerciseLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"exercise_id": exercise.id, "title": exercise.title}


@router.delete("/{exercise_id}", status_code=204)
async def delete_exercise(
    exercise_id: str,
    db: Session = Depends(get_session),
    service: ExerciseService = Depends(get_exercise_service),
):
    """Delete a published exercise. Blocked if any student sessions exist."""
    if service.get_exercise(exercise_id) is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    session_count = len(
        list(db.exec(sqlmodel_select(ExerciseSession).where(ExerciseSession.exercise_id == exercise_id)).all())
    )
    if session_count > 0:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete: {session_count} student session(s) reference this exercise. Archive instead.",
        )
    try:
        service.delete_exercise(exercise_id)
    except ExerciseLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{exercise_id}/duplicate")
async def duplicate_exercise(
    exercise_id: str,
    service: ExerciseService = Depends(get_exercise_service),
):
    """Copy a published exercise with a new ID and '(Copy)' title suffix."""
    try:
        exercise = service.duplicate_exercise(exercise_id)
    except ExerciseLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"exercise_id": exercise.id, "title": exercise.title}


@router.post("/{exercise_id}/edit")
async def edit_published_exercise(
    exercise_id: str,
    db: Session = Depends(get_session),
    service: ExerciseService = Depends(get_exercise_service),
):
    """Create a new authoring session seeded from the published exercise YAML.

    The published YAML in exercises/ is NOT modified — it stays live until
    the instructor explicitly publishes the revised draft.
    """
    path = service.get_exercise_file_path(exercise_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    published_yaml = path.read_text()
    exercise = service.get_exercise(exercise_id)

    from app.dependencies import get_exercise_author_service
    from app.models.authoring import AuthoringStatus
    author_service = get_exercise_author_service()
    authoring_session = author_service.start(db)
    authoring_session.draft_yaml = published_yaml
    authoring_session.status = AuthoringStatus.editing
    authoring_session.is_edit_of = exercise_id
    authoring_session.exercise_title = exercise.title if exercise else exercise_id
    authoring_session.updated_at = datetime.now(timezone.utc)
    db.add(authoring_session)
    db.commit()
    db.refresh(authoring_session)
    return {"authoring_session_id": authoring_session.id}
