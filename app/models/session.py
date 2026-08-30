"""Runtime state: exercise sessions and mentor chat messages."""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel

from app.models.exercise import DifficultyLevel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(str, Enum):
    active = "active"
    completed = "completed"
    cancelled = "cancelled"


class ExerciseSession(SQLModel, table=True):
    __tablename__ = "exercise_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    exercise_id: str = Field(index=True)
    # Snapshotted from AppSettings.student_name at creation time, so renaming
    # the local student identity later doesn't rewrite past sessions.
    student_name: Optional[str] = None
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: SessionStatus = Field(default=SessionStatus.active)
    # Only set (and only meaningful) when the exercise's own difficulty is
    # "open" -- the student's choice of beginner/intermediate/advanced for
    # this session only. Never a standing preference; not read for any
    # other session.
    student_difficulty: Optional[DifficultyLevel] = None
    created_at: datetime = Field(default_factory=utcnow)


class AppSettings(SQLModel, table=True):
    """Single-row local settings table (no accounts/auth -- just a remembered name)."""

    __tablename__ = "app_settings"

    id: Optional[int] = Field(default=1, primary_key=True)
    student_name: Optional[str] = None
    # Optional local label used only to scope the anonymous mentor-export
    # identifier (see AnonymousIdentity) -- not an accounts/course system.
    # Defaults to "default" wherever it's read, so export works with zero setup.
    course_id: Optional[str] = None
    updated_at: datetime = Field(default_factory=utcnow)


class AnonymousIdentity(SQLModel, table=True):
    """A random, course-scoped identifier used only for the anonymous mentor
    data export (see app/services/mentor_evidence.py). Never derived from the
    student's name, OS user, or machine -- see settings_service.get_or_create_anonymous_id.
    One stable id per course_id, so exports from the same course correlate to
    the same anonymous learner but exports across unrelated courses do not.
    """

    __tablename__ = "anonymous_identities"

    course_id: str = Field(primary_key=True)
    anonymous_id: str
    created_at: datetime = Field(default_factory=utcnow)


class MentorMessageRole(str, Enum):
    student = "student"
    mentor = "mentor"


class MentorMessage(SQLModel, table=True):
    __tablename__ = "mentor_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(index=True, foreign_key="exercise_sessions.id")
    role: MentorMessageRole
    message: str
    created_at: datetime = Field(default_factory=utcnow)
