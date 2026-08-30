"""Combines deterministic verification, observed evidence, and LLM analysis
into a final, weighted evaluation.

Rule that must never be violated: for outcomes covered by deterministic
verification, the LLM's opinion is never used to decide pass/fail.
"""
import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.config import BASE_DIR, get_settings
from app.models.evaluation import Confidence, EvaluationResult, LLMEvaluationInsights, OutcomeResult
from app.models.exercise import EnvironmentType, Exercise, ExpectedOutcome, OutcomeType
from app.services.evidence import EvidenceEvent, EvidenceService
from app.services.ollama import OllamaError, OllamaService
from app.services.prompts import EVALUATION_SYSTEM_PROMPT, build_evaluation_user_prompt
from app.services.token_budget import fit_to_budget
from app.services.verifier import VerificationDetail, run_verification

logger = logging.getLogger(__name__)

MAX_VISION_FRAMES = 4
# Maximum number of recent evidence events considered for the evaluation
# prompt before context budgeting (app/services/token_budget.py) may shrink
# this further to stay within the model's context window.
EVALUATION_MAX_EVENTS = 80
# Floor below which the evidence window is never shrunk further, even if
# the prompt is still estimated to exceed budget -- an evaluation with
# almost no evidence at all is not useful.
EVALUATION_MIN_EVENTS = 10
# llava:latest's default context window (32768 tokens, per its Ollama model
# metadata) is too small for a full evaluation prompt (system + instructions
# + up to 80 events of captured text) plus attached screenshots -- observed
# failing at ~60,342 tokens with a 400 "exceed_context_size_error". Used
# both as the num_ctx override for vision calls and as the context budget
# those prompts are trimmed to fit -- an oversized prompt is fixed by
# trimming evidence (see token_budget.py), not by raising this further.
VISION_NUM_CTX = 65536

# A GUI/SIEM/web exercise is only routed to the vision model when captured
# text/AX/OCR evidence is too sparse to judge outcomes from alone -- these
# thresholds define "sparse". environment.type is a routing hint, not the
# sole rule (see _select_model).
MIN_SUBSTANTIVE_TEXT_EVENTS = 3
MIN_SUBSTANTIVE_TEXT_CHARS = 40


def _select_frame_paths(events: List[EvidenceEvent], max_frames: int = MAX_VISION_FRAMES) -> List[str]:
    """Pick a small, deduplicated set of screenshot paths, preferring the most
    recent ones (closest to the final state being evaluated)."""
    paths: List[str] = []
    for event in reversed(events):
        if event.frame_path and event.frame_path not in paths:
            paths.append(event.frame_path)
        if len(paths) >= max_frames:
            break
    return list(reversed(paths))


def _resolve_frame_path(path: str, session_id: Optional[str]) -> Path:
    """Native capture (app/services/native_mac.py) emits frame_path relative
    to its own session output directory (e.g. "frames/000003.jpg", relative
    to capture_sessions/<session_id>/), not to the project root. Any
    already-absolute path is returned unchanged."""
    candidate = Path(path)
    if candidate.is_absolute() or session_id is None:
        return candidate
    capture_output = Path(get_settings().native_capture_output)
    if not capture_output.is_absolute():
        capture_output = (BASE_DIR / capture_output).resolve()
    return capture_output / str(session_id) / candidate


def _encode_images(paths: List[str], session_id: Optional[str] = None) -> List[str]:
    encoded: List[str] = []
    for path in paths:
        resolved = _resolve_frame_path(path, session_id)
        try:
            with open(resolved, "rb") as f:
                encoded.append(base64.b64encode(f.read()).decode("ascii"))
        except OSError as exc:
            logger.warning("Could not read screenshot frame %s for vision evaluation: %s", resolved, exc)
    return encoded


@dataclass
class _ModelRouting:
    model: str
    use_vision: bool
    reason: str


def _has_sufficient_text_evidence(events: List[EvidenceEvent]) -> bool:
    """True if enough substantive (non-trivial) text was captured to judge a
    visual exercise from text/AX/OCR alone, without needing screenshots."""
    substantive = sum(1 for e in events if len(e.text.strip()) >= MIN_SUBSTANTIVE_TEXT_CHARS)
    return substantive >= MIN_SUBSTANTIVE_TEXT_EVENTS


def _select_model(exercise: Exercise, events: List[EvidenceEvent], settings) -> _ModelRouting:
    """Chooses which Ollama model evaluates this session, and why.

    Terminal exercises have no separate visual state beyond what is already
    captured as text, so they never need the vision model. GUI/SIEM/web
    exercises only fall back to the vision model when captured text
    evidence is too sparse to judge outcomes from alone AND screenshots are
    actually available -- environment.type is a hint, not the sole rule.
    """
    if exercise.environment.type == EnvironmentType.terminal:
        return _ModelRouting(
            settings.ollama_model, False, "terminal exercise: text/AX evidence is authoritative"
        )
    if not settings.ollama_vision_model:
        return _ModelRouting(settings.ollama_model, False, "vision model not configured")
    if not _select_frame_paths(events):
        return _ModelRouting(settings.ollama_model, False, "no screenshots captured for this session")
    if _has_sufficient_text_evidence(events):
        return _ModelRouting(settings.ollama_model, False, "text evidence sufficient")
    return _ModelRouting(
        settings.ollama_vision_model,
        True,
        "insufficient text evidence for a visual exercise; using screenshots",
    )


# Supplementary confidence booster only -- not the primary scoring mechanism.
# Literal verification commands seen in evidence bump an already-positive LLM
# judgment from `inferred` to `strongly_observed`, but a missing keyword never
# fails an outcome by itself (that would only work for terminal exercises).
VERIFICATION_KEYWORDS = ["ls -l", "ls -la", "ls -al", "stat ", "getfacl", "namei -l"]


def _evidence_mentions_verification(events: List[EvidenceEvent]) -> bool:
    return any(
        keyword in event.text.lower() for event in events for keyword in VERIFICATION_KEYWORDS
    )


class EvaluatorService:
    def __init__(self, ollama: OllamaService):
        self.ollama = ollama

    async def evaluate(
        self,
        exercise: Exercise,
        events: List[EvidenceEvent],
        verification: Optional[Dict[str, VerificationDetail]] = None,
        evidence_error: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> EvaluationResult:
        if verification is None:
            verification = run_verification(exercise)

        if exercise.environment.type == EnvironmentType.terminal:
            events = EvidenceService.filter_terminal_events(events)

        llm_insights = await self._get_llm_insights(exercise, events, verification, evidence_error, session_id)

        outcomes: List[OutcomeResult] = []
        for step in exercise.get_steps():
            is_real_step = step.id != "_default"
            for outcome in step.expected_outcomes:
                result = self._score_outcome(outcome, verification, events, llm_insights, evidence_error)
                if is_real_step:
                    result.step_id = step.id
                    result.step_title = step.title
                outcomes.append(result)
        total_score = sum(o.score for o in outcomes)

        return EvaluationResult(
            score=round(total_score, 1),
            outcomes=outcomes,
            summary=llm_insights.summary,
            strengths=llm_insights.strengths,
            improvements=llm_insights.improvements,
            observed_approach=llm_insights.observed_approach,
            alternative_approaches=llm_insights.alternative_approaches,
            risky_or_unnecessary_steps=llm_insights.risky_or_unnecessary_steps,
            evidence_error=evidence_error,
        )

    async def _get_llm_insights(
        self,
        exercise: Exercise,
        events: List[EvidenceEvent],
        verification: Dict[str, VerificationDetail],
        evidence_error: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> LLMEvaluationInsights:
        settings = get_settings()
        routing = _select_model(exercise, events, settings)
        use_vision, model, reason = routing.use_vision, routing.model, routing.reason

        images: List[str] = []
        if use_vision:
            images = _encode_images(_select_frame_paths(events), session_id)
            if not images:
                use_vision, model = False, settings.ollama_model
                reason = "screenshots could not be read; falling back to text-only"

        context_size = VISION_NUM_CTX if use_vision else settings.ollama_model_num_ctx

        def render(count: int):
            prompt, per_event_truncated = build_evaluation_user_prompt(
                exercise, events, verification, evidence_error, has_images=use_vision, max_events=count
            )
            return EVALUATION_SYSTEM_PROMPT, prompt, per_event_truncated

        system_prompt, prompt, selected_count, estimated_tokens, truncated = fit_to_budget(
            render, EVALUATION_MAX_EVENTS, context_size,
            image_count=len(images) if use_vision else 0, min_count=EVALUATION_MIN_EVENTS,
        )

        logger.info(
            "Evaluation routing:\n"
            "exercise_type=%s\n"
            "model=%s\n"
            "reason=%s\n"
            "raw_events=%d\n"
            "selected_events=%d\n"
            "screenshots=%d\n"
            "estimated_tokens=%d\n"
            "context=%d\n"
            "truncated=%s",
            exercise.environment.type.value,
            model,
            reason,
            len(events),
            min(selected_count, len(events)),
            len(images),
            estimated_tokens,
            context_size,
            str(truncated).lower(),
        )

        try:
            if use_vision:
                return await self.ollama.evaluate(
                    system_prompt, prompt, images=images, model=model, num_ctx=context_size,
                )
            return await self.ollama.evaluate(system_prompt, prompt, num_ctx=context_size)
        except OllamaError as exc:
            logger.warning("LLM evaluation unavailable: %s", exc)
            if events:
                # Capture worked -- only the AI evaluation call failed. Must not
                # be worded as if activity capture itself was unavailable.
                summary = (
                    f"Activity was captured successfully, but AI evaluation failed ({exc}). "
                    "Scoring below relies on deterministic verification and observed evidence only."
                )
            else:
                summary = (
                    f"AI-based process analysis was unavailable ({exc}), and no activity evidence "
                    "was captured either. Scoring below relies on deterministic verification only."
                )
            return LLMEvaluationInsights(summary=summary)

    def _score_outcome(
        self,
        outcome: ExpectedOutcome,
        verification: Dict[str, VerificationDetail],
        events: List[EvidenceEvent],
        llm_insights: LLMEvaluationInsights,
        evidence_error: Optional[str] = None,
    ) -> OutcomeResult:
        if outcome.type == OutcomeType.filesystem and outcome.id in verification:
            detail = verification[outcome.id]
            return OutcomeResult(
                id=outcome.id,
                passed=detail.passed,
                score=outcome.weight if detail.passed else 0.0,
                max_score=outcome.weight,
                confidence=Confidence.verified,
                evidence=detail.note or "Verified via filesystem check.",
            )

        # observed_behavior AND process outcomes (and any filesystem outcome
        # without a matching deterministic check) all use the same generalized
        # per-outcome LLM judgment -- there is no app-specific ground truth to
        # fall back on, so this is the only mechanism available for anything
        # that isn't a declared filesystem check.
        return self._score_from_judgment(outcome, events, llm_insights, evidence_error)

    def _score_from_judgment(
        self,
        outcome: ExpectedOutcome,
        events: List[EvidenceEvent],
        llm_insights: LLMEvaluationInsights,
        evidence_error: Optional[str] = None,
    ) -> OutcomeResult:
        if evidence_error:
            return OutcomeResult(
                id=outcome.id,
                passed=False,
                score=0.0,
                max_score=outcome.weight,
                confidence=Confidence.unknown,
                evidence=f"Evidence retrieval failed, so this could not be evaluated: {evidence_error}",
            )

        judgment = next((j for j in llm_insights.outcome_judgments if j.id == outcome.id), None)

        # A literal verification-style command in evidence is a strong, standalone
        # signal -- checked first regardless of whether the LLM also judged this
        # outcome, mirroring how deterministic evidence should outrank inference.
        if _evidence_mentions_verification(events):
            return OutcomeResult(
                id=outcome.id,
                passed=True,
                score=outcome.weight,
                max_score=outcome.weight,
                confidence=Confidence.strongly_observed,
                evidence=(judgment.evidence if judgment and judgment.observed else None)
                or "Observed a verification command in captured activity.",
            )

        if judgment is not None and judgment.observed:
            return OutcomeResult(
                id=outcome.id,
                passed=True,
                score=round(outcome.weight * 0.5, 1),
                max_score=outcome.weight,
                confidence=Confidence.inferred,
                evidence=judgment.evidence or "AI inferred this occurred from the activity timeline.",
            )

        if judgment is None:
            evidence = (
                "There was insufficient captured evidence to confidently evaluate this behavior."
                if not events
                else "The AI evaluator did not return a judgment for this outcome."
            )
        else:
            evidence = judgment.evidence or "Not observed in captured activity."

        return OutcomeResult(
            id=outcome.id,
            passed=False,
            score=0.0,
            max_score=outcome.weight,
            confidence=Confidence.unknown,
            evidence=evidence,
        )
