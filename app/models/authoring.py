"""Runtime state for the conversational exercise-authoring flow.

Mirrors the ExerciseSession/MentorMessage chat pattern in session.py, but
for an instructor designing a new exercise rather than a student working
one.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel

from app.models.session import utcnow


class AuthoringStatus(str, Enum):
    drafting = "drafting"
    finalized = "finalized"
    saved = "saved"


class ExerciseAuthoringSession(SQLModel, table=True):
    __tablename__ = "exercise_authoring_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    status: AuthoringStatus = Field(default=AuthoringStatus.drafting)
    # Populated once the instructor finalizes -- the LLM-generated exercise
    # YAML, shown as a preview before being written to exercises/.
    draft_yaml: Optional[str] = None
    saved_exercise_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class AuthoringMessageRole(str, Enum):
    instructor = "instructor"
    assistant = "assistant"


class ExerciseAuthoringMessage(SQLModel, table=True):
    __tablename__ = "exercise_authoring_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    authoring_session_id: int = Field(index=True, foreign_key="exercise_authoring_sessions.id")
    role: AuthoringMessageRole
    message: str
    created_at: datetime = Field(default_factory=utcnow)
