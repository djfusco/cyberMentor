from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.dependencies import get_exercise_service
from app.services.exercises import ExerciseLoadError

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
