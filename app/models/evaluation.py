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
    # Set when the AI evaluation call itself could not be completed (e.g. a
    # model timeout), even after one retry with a smaller evidence packet --
    # distinct from evidence_error (evidence RETRIEVAL failing upstream).
    # Every LLM-judged outcome scores 0 in this case, but that 0 must never
    # be read as "the learner didn't do it" -- see
    # EvaluatorService._score_from_judgment, which threads this into each
    # affected outcome's own evidence text and verification_state.
    ai_unavailable: Optional[str] = None


class CriterionJudgment(BaseModel):
    """One outcome's success_criteria entry, judged individually with its own
    evidence -- not just an unstructured pass/fail bucket. When the criterion
    asserts an exact literal (a field name, command, value, count, or
    identifier -- typically backtick-quoted in the authored criterion text),
    `quote` should carry the EXACT text the model claims to have read, so the
    scoring layer can mechanically cross-check it against captured OCR/AX
    text rather than trusting the claim outright (see
    EvaluatorService._evaluate_criteria). Optional and additive: an LLM
    response that omits this per-criterion structure entirely still scores
    via the legacy criteria_met/criteria_not_met matching on OutcomeJudgment.

    Reuses EvidenceBasis (action/state/unclear) -- the same enum and values
    as the outer OutcomeJudgment.evidence_basis, deliberately -- rather than
    a separate text/visual vocabulary: a prior version used a different enum
    here and the model, seeing two fields both named "evidence_basis" in the
    same schema, filled this one with the OUTER field's values ("action"/
    "state"), which strict enum validation then rejected and silently
    dropped the ENTIRE structured response (observed live: 3/3 evaluation
    runs failed this way). The lenient before-validator below is an
    additional safety net: an unrecognized value here falls back to
    "unclear" instead of invalidating the whole judgment.
    """

    @field_validator("evidence_basis", mode="before")
    @classmethod
    def _coerce_evidence_basis(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v not in ("action", "state", "unclear"):
                return "unclear"
        return v

    criterion: str
    supported: bool
    # Set true when the judge explicitly could not determine support either
    # way (evidence was shown but was inconclusive) -- distinct from
    # `supported=False`, which means the evidence affirmatively shows the
    # criterion was NOT met. Takes priority over `supported` when true (see
    # EvaluatorService._evaluate_criteria): an inconclusive read must never
    # be scored as a confident fail. Default False so every existing caller
    # (the legacy combined-call path, which never sets this) is unaffected.
    unverifiable: bool = False
    evidence_basis: EvidenceBasis = EvidenceBasis.unclear
    # The exact quoted text/value the model claims appears in the evidence --
    # required (by the prompt) whenever the criterion names a specific exact
    # value; left empty for criteria with no exact-value claim (e.g. "at
    # least one result is returned") or genuinely visual ones.
    quote: Optional[str] = None
    # Which attached screenshot or timeline moment this came from -- a
    # 1-based image index (as captioned in the prompt) or an ISO timestamp.
    # Free text; used only for diagnostics, never parsed for scoring.
    frame_reference: Optional[str] = None
    note: Optional[str] = None


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
    # Structured, per-criterion judgments -- preferred over criteria_met/
    # criteria_not_met when present (see EvaluatorService._evaluate_criteria).
    # Empty by default: an LLM/response that never populates this falls back
    # to the legacy string-list matching unchanged.
    criterion_judgments: List[CriterionJudgment] = PydanticField(default_factory=list)
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


class VisualCriterionState(str, Enum):
    """One success criterion's state as self-reported by a SINGLE-OUTCOME
    visual scoring call (see EvaluatorService._score_visual_outcome) -- a
    deliberately small, 3-way vocabulary for the model itself to choose
    from. The scoring layer never trusts this alone for an exact-value
    criterion (see CriterionJudgment.unverifiable / _evaluate_criteria);
    it is converted into the richer, mechanically-derived
    CriterionEvidenceState before it can affect a score.
    """

    supported = "supported"
    contradicted = "contradicted"
    unverifiable = "unverifiable"


class VisualCriterionResult(BaseModel):
    """One criterion's judgment from a per-outcome visual scoring call.
    Deliberately minimal -- see the per-outcome schema requirements in
    EvaluatorService._score_visual_outcome: only what scoring needs, no
    outcome-level narrative, no unrelated outcomes."""

    criterion: str
    state: VisualCriterionState
    quote: Optional[str] = None
    frame_ref: Optional[str] = None


class VisualOutcomeJudgment(BaseModel):
    """The full response schema for ONE outcome's dedicated visual scoring
    call -- supplied to Ollama's `format` request property as a JSON Schema
    (via .model_json_schema()) so the response is grammar-constrained, not
    merely requested in prompt text. See EvaluatorService._score_visual_outcome.
    """

    outcome_id: str
    criteria: List[VisualCriterionResult] = PydanticField(default_factory=list)
    feedback: str = ""


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
    # Mirrors the exported bundle's own top-level "evaluation_status" (see
    # app.services.submission.build_submission_export) -- "unavailable" when
    # the AI evaluator could not obtain valid judgments for this attempt, so
    # the instructor-facing submissions table can show that instead of a
    # numeric score that would otherwise read as a legitimate zero. Optional
    # so existing rows (imported before this field existed) fall back to
    # None, rendered the same as "complete" -- see loadSubmissionsTable().
    evaluation_status: Optional[str] = None
    # None when the matching exercise (with its signing_key) isn't available
    # locally to verify against -- distinct from a confirmed bad signature.
    signature_valid: Optional[bool] = None
    signature_note: str = ""
    imported_at: datetime = Field(default_factory=utcnow)
    source_filename: Optional[str] = None
