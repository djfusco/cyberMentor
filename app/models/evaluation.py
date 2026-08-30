"""Evaluation persistence and structured evaluation result schemas."""
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from pydantic import Field as PydanticField
from sqlmodel import Field, SQLModel

from app.models.session import utcnow


class Confidence(str, Enum):
    verified = "verified"
    strongly_observed = "strongly_observed"
    inferred = "inferred"
    unknown = "unknown"


class OutcomeResult(BaseModel):
    id: str
    passed: bool
    score: float
    max_score: float
    confidence: Confidence = Confidence.unknown
    evidence: str = ""
    # Populated for multi-step exercises so results can be grouped by step;
    # None for single-task exercises (see Exercise.get_steps()).
    step_id: Optional[str] = None
    step_title: Optional[str] = None


class EvaluationResult(BaseModel):
    """The full, final evaluation returned to the client and persisted."""

    score: float
    outcomes: List[OutcomeResult]
    summary: str = ""
    strengths: List[str] = PydanticField(default_factory=list)
    improvements: List[str] = PydanticField(default_factory=list)
    observed_approach: List[str] = PydanticField(default_factory=list)
    alternative_approaches: List[str] = PydanticField(default_factory=list)
    # Previously computed by the LLM but discarded before reaching the
    # student or the persisted record -- now kept end-to-end since it also
    # feeds the cross-student common-errors dashboard.
    risky_or_unnecessary_steps: List[str] = PydanticField(default_factory=list)
    # Set when the evidence provider failed during this evaluation, so
    # "no activity observed" is never confused with "evidence retrieval
    # failed".
    evidence_error: Optional[str] = None


class OutcomeJudgment(BaseModel):
    """The LLM's per-outcome judgment for a non-filesystem (observed_behavior
    or process) outcome -- e.g. "did the learner click Submit and see the
    confirmation message". Generalizes what used to be a single hardcoded
    terminal-verification-keyword check into a judgment about any described
    behavior, in any environment.

    The LLM should only ever choose `inferred` or `unknown` for confidence;
    `strongly_observed` is reserved for when deterministic evidence (e.g. a
    literal verification command) supplements the judgment -- see
    EvaluatorService._score_from_judgment.
    """

    id: str
    observed: bool
    confidence: Confidence = Confidence.inferred
    evidence: str = ""


class LLMEvaluationInsights(BaseModel):
    """The subset of the evaluation that Ollama is responsible for producing.

    Deliberately excludes pass/fail on objective outcomes -- that is decided
    by deterministic verification and must never be overridden by the LLM.
    """

    summary: str = ""
    strengths: List[str] = PydanticField(default_factory=list)
    improvements: List[str] = PydanticField(default_factory=list)
    observed_approach: List[str] = PydanticField(default_factory=list)
    alternative_approaches: List[str] = PydanticField(default_factory=list)
    risky_or_unnecessary_steps: List[str] = PydanticField(default_factory=list)
    # One judgment per non-filesystem outcome id -- see OutcomeJudgment.
    outcome_judgments: List[OutcomeJudgment] = PydanticField(default_factory=list)
    # Deprecated: superseded by outcome_judgments, kept only so old cached
    # evaluation_json rows still deserialize.
    approach_valid: Optional[bool] = None
    verification_observed: Optional[bool] = None
    verification_evidence: str = ""


class Evaluation(SQLModel, table=True):
    __tablename__ = "evaluations"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(index=True, foreign_key="exercise_sessions.id")
    score: float
    summary: str
    evaluation_json: str
    created_at: datetime = Field(default_factory=utcnow)


class Submission(SQLModel, table=True):
    """An instructor-side, imported record of a student's exported result.

    Deliberately separate from ExerciseSession/Evaluation (which represent
    this install's own local runs) so imported roster data can never be
    confused with the instructor's own practice sessions.
    """

    __tablename__ = "submissions"

    id: Optional[int] = Field(default=None, primary_key=True)
    exercise_id: str = Field(index=True)
    exercise_title: str
    student_name: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    score: float
    summary: str
    evaluation_json: str
    # None when the matching exercise (with its signing_key) isn't available
    # locally to verify against -- distinct from a confirmed bad signature.
    signature_valid: Optional[bool] = None
    signature_note: str = ""
    imported_at: datetime = Field(default_factory=utcnow)
    source_filename: Optional[str] = None
