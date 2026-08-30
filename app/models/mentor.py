"""Instructor-side storage for imported anonymous mentor data exports.

Mirrors app.models.evaluation.Submission's shape and intent: a local,
append-only record of a file the instructor manually imported, kept
separate from any of this install's own data (which is the student's
local ExerciseSession/Evaluation/MentorMessage rows -- never imported here).
"""
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel

from app.models.session import utcnow


class AnonymousLearnerImport(SQLModel, table=True):
    """One imported anonymous mentor-export file (see app/services/mentor_evidence.py
    for the export format and app/services/mentor_insights.py for import handling).

    Multiple rows may share the same anonymous_student_id (e.g. a student
    re-exports later in the course) -- app.services.mentor_insights groups by
    that id and uses the most recent row per learner for analysis, while
    still keeping earlier imports for the instructor's own record.
    """

    __tablename__ = "anonymous_learner_imports"

    id: Optional[int] = Field(default=None, primary_key=True)
    anonymous_student_id: str = Field(index=True)
    course_id: Optional[str] = None
    # The raw, already-anonymized JSON export, stored verbatim (same pattern
    # as Submission.evaluation_json) -- re-parsed on demand, not normalized
    # into a bespoke analytics schema.
    mentor_json: str
    source_filename: Optional[str] = None
    imported_at: datetime = Field(default_factory=utcnow)
