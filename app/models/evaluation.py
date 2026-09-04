"""Evaluation persistence and structured evaluation result schemas."""
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, field_validator
from pydantic import Field as PydanticField
from sqlmodel import Field, SQLModel

from app.models.exercise import VerificationState
from app.models.session import utcnow


class Confidence(str, Enum):
    verified = "verified"
    strongly_observed = "strongly_observed"
    inferred = "inferred"
    unknown = "unknown"


class EvidenceBasis(str, Enum):
    """What KIND of evidence backs one LLM outcome judgment -- distinguishing
    proof the STUDENT PERFORMED an action this session from evidence that
    only shows a resulting STATE (which may predate the session). For a
    filesystem outcome already satisfied at baseline, state evidence can
    never earn credit on its own; only action evidence can -- see
    EvaluatorService._score_filesystem_outcome. Set by the LLM per judgment
    (app.services.prompts.EVALUATION_SYSTEM_PROMPT), then enforced
    mechanically rather than trusted from `observed` alone.
    """

    action = "action"
    state = "state"
    unclear = "unclear"


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
    # new additive fields — safe defaults for backward compat
    verification_state: Optional[VerificationState] = None
    feedback: Optional[str] = None
    # Only meaningful for outcomes with a deterministic (filesystem) check.
    # final_state_verified: does the check pass RIGHT NOW (the state fact).
    # demonstrated_this_session: is there current-session attribution for
    # that state -- a baseline->final transition, or observed evidence of
    # the action (the demonstration fact). These are independent: a
    # compliant state can be verified while not demonstrated this session
    # (stale state left over from an earlier attempt), which must score
    # zero -- see EvaluatorService._score_filesystem_outcome. None when the
    # distinction doesn't apply (no deterministic check for this outcome).
    final_state_verified: Optional[bool] = None
    demonstrated_this_session: Optional[bool] = None


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
    or process) outcome. For enriched labs, also carries structured criterion
    assessment. criteria_implicit is set by result-parsing code from
    outcome.has_explicit_criteria — never trusted from the LLM.
    """

    @field_validator("verification_state", "evidence_basis", mode="before")
    @classmethod
    def normalise_verification_state(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    id: str
    observed: bool
    confidence: Confidence = Confidence.inferred
    evidence: str = ""
    # additive fields — safe defaults for backward compat
    verification_state: Optional[VerificationState] = None  # required in enriched labs; nullable for legacy
    criteria_met: List[str] = PydanticField(default_factory=list)
    criteria_not_met: List[str] = PydanticField(default_factory=list)
    feedback: Optional[str] = None
    criteria_implicit: bool = False  # set by result parser from outcome.has_explicit_criteria
    # Nullable: legacy/malformed responses omit it, in which case the
    # baseline-already-satisfied gate in EvaluatorService._score_filesystem_outcome
    # treats it as not proven action evidence (fails closed, not open).
    evidence_basis: Optional[EvidenceBasis] = None


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
