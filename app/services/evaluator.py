"""Combines deterministic verification, observed evidence, and LLM analysis
into a final, weighted evaluation.

Rule that must never be violated: for outcomes covered by deterministic
verification, the LLM's opinion is never used to decide pass/fail.
"""
import base64
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.config import BASE_DIR, get_settings
from app.models.evaluation import Confidence, EvaluationResult, LLMEvaluationInsights, OutcomeResult
from app.models.exercise import EnvironmentType, Exercise, ExpectedOutcome, OutcomeType, VerificationState
from app.services.capture_manifest import ManifestWriter
from app.services.evidence import EvidenceEvent, EvidenceService
from app.services.ollama import OllamaError, OllamaService
from app.services.prompts import (
    EVALUATION_SYSTEM_PROMPT,
    build_evaluation_user_prompt,
    build_missing_judgments_prompt,
)
from app.services.keyframes import SelectedFrame, format_frame_captions, select_keyframes
from app.services.token_budget import fit_to_budget
from app.services.verifier import VerificationDetail, needs_llm_attribution, run_verification

logger = logging.getLogger(__name__)

MAX_VISION_FRAMES = 4
# Number of screenshots attached to a FINISH-time evaluation. Higher than
# MAX_VISION_FRAMES (used for latency-sensitive live mentor chat / session
# Q&A) because evaluation runs once and needs to see every phase of the
# session, not just the final state. Frames are downscaled before encoding
# (see _downscale_image_bytes), so 16 downsampled frames cost roughly what
# 8 full-resolution frames did before -- buying ~2x coverage of the session
# timeline without raising wall-clock time. See _select_spread_frame_paths.
EVALUATION_MAX_VISION_FRAMES = 16
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

# Used by session-query routing (app/services/session_query.py), NOT by
# evaluation routing. Session Q&A is interactive and cost-sensitive, so it
# only attaches screenshots when the question is visual or genuine captured
# text is sparse; evaluation instead treats screenshots as the primary
# source of truth whenever they exist (see _select_model). Kept here so
# session_query can reuse the same "is there enough real text to answer
# without images?" heuristic without a second copy.
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


def _select_spread_frame_paths(
    events: List[EvidenceEvent], max_frames: int = EVALUATION_MAX_VISION_FRAMES
) -> List[str]:
    """Pick a small, deduplicated set of screenshot paths spread evenly across
    the WHOLE session, so the vision model sees every phase of the work -- not
    just the final state.

    A multi-step lab satisfies different outcomes at different points in time
    (generate the pcap early, upload it next, inspect packets later, write
    answers last). Sampling only the most recent frames -- as
    _select_frame_paths does for latency-sensitive live chat -- leaves early
    and mid-session steps with no screenshot evidence at all, which is how an
    upload visible at frame 12 of 37 was never seen by the vision model.
    Spreading the selection across the full timeline ensures each phase is
    represented, always including the first and last frames.
    """
    distinct: List[str] = []
    seen = set()
    for event in events:
        if event.frame_path and event.frame_path not in seen:
            seen.add(event.frame_path)
            distinct.append(event.frame_path)
    if len(distinct) <= max_frames:
        return distinct
    if max_frames <= 1:
        return [distinct[-1]]
    # Evenly sample across the full span, always including the first and last.
    n = len(distinct)
    indices = sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)})
    return [distinct[i] for i in indices]


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


# Screenshots are captured at native display resolution (e.g. 4096x2304 on a
# retina mac) but vision models downsample internally anyway -- sending full
# resolution costs ~10x the tokens/time for detail the model can't use. Resize
# to this width (preserving aspect ratio) before base64-encoding. Wide enough
# to keep on-screen text/buttons/confirmations legible to the model.
VISION_FRAME_MAX_WIDTH = 1280


def _downscale_image_bytes(raw: bytes, max_width: int = VISION_FRAME_MAX_WIDTH) -> bytes:
    """Downscale an image to max_width (aspect ratio preserved), returning
    JPEG bytes. Falls back to the original bytes unchanged when Pillow is not
    installed or the input isn't a decodable image -- so callers (and tests
    using fake bytes) never hard-fail on resizing, only lose the optimization.
    """
    try:
        import io

        from PIL import Image
    except ImportError:
        return raw
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:  # noqa: BLE001 -- non-image bytes / corrupt frames: fall back, don't crash
        return raw
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, max(1, round(img.height * ratio)))
        img = img.resize(new_size, Image.LANCZOS)
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=85)
    return out.getvalue()


def _encode_images(paths: List[str], session_id: Optional[str] = None) -> List[str]:
    encoded: List[str] = []
    for path in paths:
        resolved = _resolve_frame_path(path, session_id)
        try:
            with open(resolved, "rb") as f:
                raw = f.read()
        except OSError as exc:
            logger.warning("Could not read screenshot frame %s for vision evaluation: %s", resolved, exc)
            continue
        encoded.append(base64.b64encode(_downscale_image_bytes(raw)).decode("ascii"))
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
    captured as text, so they never need the vision model.

    For every other exercise type (gui/siem/web/...), captured screenshots
    are the PRIMARY source of truth for what the learner actually did. The
    text timeline is only supplementary: window titles, click coordinates,
    and key-count summaries carry no element names or typed content, and OCR
    is frequently garbled (e.g. a real capture turned a Word answer document
    into "TIC\nLUIL\nVICVV..."). Routing on a raw text-LENGTH heuristic
    suppressed vision whenever window titles were long enough -- i.e.
    exactly when screenshots were most needed. So whenever a vision model is
    configured AND screenshots were captured, route to the vision model and
    attach them. Fall back to the text model only when there are no
    screenshots to look at (or no vision model is configured).
    """
    if exercise.environment.type == EnvironmentType.terminal:
        return _ModelRouting(
            settings.ollama_model, False, "terminal exercise: text/AX evidence is authoritative"
        )
    if not settings.ollama_vision_model:
        return _ModelRouting(settings.ollama_model, False, "vision model not configured")
    if not any(e.frame_path for e in events):
        return _ModelRouting(settings.ollama_model, False, "no screenshots captured for this session")
    return _ModelRouting(
        settings.ollama_vision_model,
        True,
        "visual exercise with screenshots: using screenshots as primary evidence",
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
        manifest: Optional[ManifestWriter] = None,
        baseline_verification: Optional[Dict[str, VerificationDetail]] = None,
    ) -> EvaluationResult:
        if verification is None:
            verification = run_verification(exercise)

        if exercise.environment.type == EnvironmentType.terminal:
            events = EvidenceService.filter_terminal_events(events)

        llm_insights = await self._get_llm_insights(
            exercise, events, verification, evidence_error, session_id, manifest=manifest,
            baseline_verification=baseline_verification,
        )

        outcomes: List[OutcomeResult] = []
        for step in exercise.get_steps():
            is_real_step = step.id != "_default"
            for outcome in step.expected_outcomes:
                result = self._score_outcome(
                    outcome, verification, events, llm_insights, evidence_error,
                    baseline_verification=baseline_verification,
                )
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
        manifest: Optional[ManifestWriter] = None,
        baseline_verification: Optional[Dict[str, VerificationDetail]] = None,
    ) -> LLMEvaluationInsights:
        settings = get_settings()
        routing = _select_model(exercise, events, settings)
        use_vision, model, reason = routing.use_vision, routing.model, routing.reason

        images: List[str] = []
        frame_captions: Optional[str] = None
        selected_frames: List[SelectedFrame] = []
        if use_vision:
            selected_frames = select_keyframes(events, EVALUATION_MAX_VISION_FRAMES)
            images = _encode_images([sf.path for sf in selected_frames], session_id)
            if not images:
                use_vision, model = False, settings.ollama_model
                reason = "screenshots could not be read; falling back to text-only"
                selected_frames = []
            else:
                frame_captions = format_frame_captions(selected_frames)

        context_size = VISION_NUM_CTX if use_vision else settings.ollama_model_num_ctx
        # Vision evaluation attaches up to 8 screenshots and asks for a large
        # structured JSON object -- that takes well over the default 60s chat
        # timeout on a local vision model, so use the dedicated (longer)
        # evaluation timeout. Text-only evaluation is lighter but still gets
        # the same room; the setting is evaluation-scoped, not vision-scoped.
        eval_timeout = settings.evaluation_timeout_seconds

        def render(count: int):
            prompt, per_event_truncated = build_evaluation_user_prompt(
                exercise, events, verification, evidence_error,
                has_images=use_vision, frame_captions=frame_captions, max_events=count,
                baseline_verification=baseline_verification,
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

        # Record evaluation evidence snapshot in manifest before calling the model.
        model_request_timestamp = datetime.now(timezone.utc).isoformat()
        if manifest is not None:
            available_frames = [
                {
                    "path": e.frame_path,
                    "timestamp": e.timestamp.isoformat(),
                    "application": e.application,
                }
                for e in events
                if e.frame_path
            ]
            try:
                manifest.record_final_evaluation(
                    model_request_timestamp=model_request_timestamp,
                    model=model,
                    routing_reason=reason,
                    event_count=len(events),
                    normalized_event_count=len(events),
                    available_frames=available_frames,
                    selected_frames=selected_frames,
                    images_attached=bool(images),
                    use_vision=use_vision,
                )
                if use_vision:
                    manifest.save_diagnostic_payload(
                        consumer="final_evaluation",
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        selected_frames=selected_frames,
                    )
            except Exception as exc:  # noqa: BLE001 -- manifest errors must never fail evaluation
                logger.warning("Evaluation manifest write failed: %s", exc)

        try:
            if use_vision:
                insights = await self.ollama.evaluate(
                    system_prompt, prompt, images=images, model=model,
                    num_ctx=context_size, timeout=eval_timeout,
                )
            else:
                insights = await self.ollama.evaluate(
                    system_prompt, prompt, num_ctx=context_size, timeout=eval_timeout,
                )
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

        # Vision models (e.g. llava) frequently return a coherent summary but
        # omit the structured outcome_judgments array, or fill it only partially.
        # Because outcome_judgments defaults to an empty list and the JSON is
        # otherwise valid, the parse/repair path never fires -- and every
        # non-filesystem outcome silently scores "did not return a judgment".
        # Detect missing required IDs and make ONE focused follow-up call asking
        # only for those, then merge. Reuses the same screenshots/model/context.
        if not evidence_error:
            required_ids = [
                o.id for o in exercise.get_all_outcomes()
                if o.id not in verification
                or needs_llm_attribution(o.id, verification, baseline_verification)
            ]
            returned_ids = {j.id for j in insights.outcome_judgments}
            missing_ids = [rid for rid in required_ids if rid not in returned_ids]
            if missing_ids:
                insights = await self._fill_missing_judgments(
                    insights, exercise, events, missing_ids,
                    images=images if use_vision else None,
                    frame_captions=frame_captions,
                    model=model, num_ctx=context_size, timeout=eval_timeout,
                )

        return insights

    async def _fill_missing_judgments(
        self,
        insights: LLMEvaluationInsights,
        exercise: Exercise,
        events: List[EvidenceEvent],
        missing_ids: List[str],
        images: Optional[List[str]] = None,
        frame_captions: Optional[str] = None,
        model: Optional[str] = None,
        num_ctx: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> LLMEvaluationInsights:
        """One focused follow-up call asking only for the outcome IDs the first
        evaluation skipped. Smaller, simpler ask -> higher compliance from
        weaker vision models. Merges the returned judgments into `insights`
        (filling gaps; never overwriting a judgment the first call did return).
        """
        focused_prompt, _truncated = build_missing_judgments_prompt(
            exercise, events, missing_ids,
            has_images=images is not None, frame_captions=frame_captions,
        )
        logger.info(
            "Evaluation follow-up for %d missing outcome judgment(s): %s",
            len(missing_ids), ", ".join(missing_ids),
        )
        try:
            followup = await self.ollama.evaluate(
                EVALUATION_SYSTEM_PROMPT, focused_prompt,
                images=images, model=model, num_ctx=num_ctx, timeout=timeout,
            )
        except OllamaError as exc:
            logger.warning("Judgment follow-up unavailable: %s", exc)
            return insights

        by_id = {j.id: j for j in insights.outcome_judgments}
        for judgment in followup.outcome_judgments:
            # Only fill gaps -- never overwrite a judgment the first pass produced.
            if judgment.id not in by_id:
                by_id[judgment.id] = judgment
        insights.outcome_judgments = list(by_id.values())
        return insights

    def _score_outcome(
        self,
        outcome: ExpectedOutcome,
        verification: Dict[str, VerificationDetail],
        events: List[EvidenceEvent],
        llm_insights: LLMEvaluationInsights,
        evidence_error: Optional[str] = None,
        baseline_verification: Optional[Dict[str, VerificationDetail]] = None,
    ) -> OutcomeResult:
        if outcome.type == OutcomeType.filesystem and outcome.id in verification:
            return self._score_filesystem_outcome(
                outcome, verification, baseline_verification, events, llm_insights, evidence_error
            )

        # observed_behavior AND process outcomes (and any filesystem outcome
        # without a matching deterministic check) all use the same generalized
        # per-outcome LLM judgment -- there is no app-specific ground truth to
        # fall back on, so this is the only mechanism available for anything
        # that isn't a declared filesystem check.
        return self._score_from_judgment(outcome, events, llm_insights, evidence_error)

    def _score_filesystem_outcome(
        self,
        outcome: ExpectedOutcome,
        verification: Dict[str, VerificationDetail],
        baseline_verification: Optional[Dict[str, VerificationDetail]],
        events: List[EvidenceEvent],
        llm_insights: LLMEvaluationInsights,
        evidence_error: Optional[str] = None,
    ) -> OutcomeResult:
        """Deterministic verification proves a final state exists; it says
        nothing about whether THIS session produced it. Distinguish the two
        facts explicitly:

        - final_state_verified: does the check pass right now.
        - demonstrated_this_session: is there current-session attribution
          for that state, via a baseline->final transition (nonexistent ->
          exists, incorrect -> correct) or observed evidence of the action.

        A failing check is always final and never overridable by the LLM --
        that rule is unchanged. A passing check that also represents a real
        transition (or has no baseline to compare against, e.g. a session
        created before baselines were captured) is deterministic credit, as
        before. A passing check whose state already existed at session
        start falls back to the same evidence-based judgment used for
        outcomes with no deterministic check at all -- state alone cannot
        prove attribution, only activity can.
        """
        detail = verification[outcome.id]
        if not detail.passed:
            return OutcomeResult(
                id=outcome.id,
                passed=False,
                score=0.0,
                max_score=outcome.weight,
                confidence=Confidence.verified,
                evidence=detail.note or "Verified via filesystem check.",
                verification_state=VerificationState.incorrect,
                final_state_verified=False,
                demonstrated_this_session=False,
            )

        if not needs_llm_attribution(outcome.id, verification, baseline_verification):
            # Either no baseline was recorded for this outcome (permissive
            # fallback, matching behavior before baselines existed), or the
            # baseline shows this outcome was NOT yet compliant at session
            # start -- so the current pass represents a real transition
            # produced during this session.
            return OutcomeResult(
                id=outcome.id,
                passed=True,
                score=outcome.weight,
                max_score=outcome.weight,
                confidence=Confidence.verified,
                evidence=detail.note or "Verified via filesystem check.",
                verification_state=VerificationState.verified,
                final_state_verified=True,
                demonstrated_this_session=True,
            )

        # The compliant state already existed before this session began --
        # only current-session evidence of the action can still earn credit.
        result = self._score_from_judgment(outcome, events, llm_insights, evidence_error)
        result.final_state_verified = True
        result.demonstrated_this_session = result.passed
        if not result.passed and not evidence_error:
            result.evidence = (
                "This state already existed when the session started, and no "
                "current-session activity demonstrated it again -- not "
                "demonstrated this attempt."
            )
        return result

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
                verification_state=VerificationState.unverifiable,
            )

        judgment = next((j for j in llm_insights.outcome_judgments if j.id == outcome.id), None)

        # Set criteria_implicit deterministically — do not trust what the LLM claimed.
        if judgment is not None:
            judgment.criteria_implicit = not outcome.has_explicit_criteria

        if judgment is not None and judgment.observed:
            # Screenshots are the primary source of truth for visual exercises
            # (see _select_model): when the vision model directly observes an
            # action on screen, that is direct evidence -- not inference -- so it
            # earns full credit, the same as a deterministic filesystem check.
            # The strict evidence rules in EVALUATION_SYSTEM_PROMPT guard against
            # false positives; if those prove too loose in practice, tighten the
            # prompt rather than re-capping real observations at half credit.
            return OutcomeResult(
                id=outcome.id,
                passed=True,
                score=outcome.weight,
                max_score=outcome.weight,
                confidence=Confidence.strongly_observed,
                evidence=judgment.evidence or "Observed in captured screenshots/activity.",
                verification_state=judgment.verification_state,
                feedback=judgment.feedback,
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
            verification_state=judgment.verification_state if judgment else None,
            feedback=judgment.feedback if judgment else None,
        )
