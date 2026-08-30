"""Loads and validates exercise definitions from YAML files.

Dropping a new *.yaml file into the exercises directory makes it available
throughout the app automatically -- no Python code changes are required.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import ValidationError

from app.config import get_settings
from app.models.exercise import Exercise

logger = logging.getLogger(__name__)


class ExerciseLoadError(Exception):
    """Raised when a single exercise file fails to parse or validate."""


class ExerciseService:
    def __init__(self, exercises_dir: Optional[str] = None):
        settings = get_settings()
        self.exercises_dir = Path(exercises_dir or settings.exercises_dir)

    def _iter_yaml_files(self):
        if not self.exercises_dir.exists():
            return
        seen = set()
        for pattern in ("*.yaml", "*.yml"):
            for path in sorted(self.exercises_dir.glob(pattern)):
                if path not in seen:
                    seen.add(path)
                    yield path

    def list_exercises(self) -> List[Exercise]:
        exercises: Dict[str, Exercise] = {}
        for path in self._iter_yaml_files():
            try:
                exercise = self._load_file(path)
            except ExerciseLoadError as exc:
                logger.warning("Skipping invalid exercise file %s: %s", path, exc)
                continue
            exercises[exercise.id] = exercise
        return sorted(exercises.values(), key=lambda e: e.title)

    def get_exercise(self, exercise_id: str) -> Optional[Exercise]:
        for exercise in self.list_exercises():
            if exercise.id == exercise_id:
                return exercise
        return None

    def get_exercise_file_path(self, exercise_id: str) -> Optional[Path]:
        """Locate the YAML file backing a given exercise ID, for export."""
        for path in self._iter_yaml_files():
            try:
                exercise = self._load_file(path)
            except ExerciseLoadError:
                continue
            if exercise.id == exercise_id:
                return path
        return None

    def save_exercise_yaml(self, raw_yaml: str) -> Exercise:
        """Validate raw YAML text and write it into the exercises directory.

        Used both by exercise import (a student receiving a file from an
        instructor) and by the chat-authoring "save" step. The exercise's
        own id is used as the filename, so saving again with the same id
        intentionally overwrites the previous version.

        Raises ExerciseLoadError if the YAML is invalid or fails schema
        validation -- a bad draft can never reach exercises/.
        """
        try:
            raw = yaml.safe_load(raw_yaml)
        except yaml.YAMLError as exc:
            raise ExerciseLoadError(f"invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ExerciseLoadError("exercise file must contain a YAML mapping")
        try:
            exercise = Exercise.model_validate(raw)
        except ValidationError as exc:
            raise ExerciseLoadError(str(exc)) from exc

        self.exercises_dir.mkdir(parents=True, exist_ok=True)
        destination = self.exercises_dir / f"{exercise.id}.yaml"
        destination.write_text(raw_yaml)
        return exercise

    def _load_file(self, path: Path) -> Exercise:
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ExerciseLoadError(f"invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ExerciseLoadError("exercise file must contain a YAML mapping")
        try:
            return Exercise.model_validate(raw)
        except ValidationError as exc:
            raise ExerciseLoadError(str(exc)) from exc
