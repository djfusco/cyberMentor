"""Combines deterministic verification, observed evidence, and LLM analysis
into a final, weighted evaluation.

Rule that must never be violated: for outcomes covered by deterministic
verification, the LLM's opinion is never used to decide pass/fail.
"""
import base64
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

from app.config import BASE_DIR, get_settings
from app.models.evaluation import (
    Confidence,
    EvaluationResult,
    EvidenceBasis,
    LLMEvaluationInsights,
    OutcomeJudgment,
    OutcomeResult,
)
from app.models.exercise import EnvironmentType, Exercise, ExpectedOutcome, OutcomeType, VerificationState
from app.services.capture_manifest import ManifestWriter
from app.services.evidence import EvidenceEvent, EvidenceService
from app.services.ollama import OllamaError, OllamaService
from app.services.prompts import (
    EVALUATION_SYSTEM_PROMPT,
    TARGETED_FOLLOWUP_SYSTEM_PROMPT,
    build_evaluation_user_prompt,
    build_missing_judgments_prompt,
)
from app.services.keyframes import (
    SelectedFrame,
    format_frame_captions,
    select_keyframes,
    select_keyframes_with_coverage,
)
from app.services.text_relevance import (
    backtick_literals,
    build_text_corpus,
    distinctive_word_sets,
    normalize_literal,
    ocr_text_available,
    overlap_score,
    significant_words,
)
from app.services.token_budget import fit_to_budget
from app.services.verifier import (
    EvidenceFact,
    VerificationDetail,
    extract_evidence_facts,
    needs_llm_attribution,
    run_verification,
)

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

# How many screenshots the evaluation payload budgets per expected outcome
# when selecting RELEVANT evidence (see select_keyframes(..., outcomes=...)),
# not per session. The actual frame budget for a given exercise is
# min(EVALUATION_MAX_VISION_FRAMES, num_outcomes * EVALUATION_FRAMES_PER_OUTCOME)
# -- e.g. a 3-outcome exercise budgets 6 frames, not the full 16-frame
# ceiling, while a much larger exercise is still capped at the ceiling. This
# is what replaced blindly spreading a large frame count evenly across the
# whole session (which reliably diluted the request with irrelevant frames
# and pushed prompt size, and therefore wall-clock time, past the model's
# timeout -- see EVALUATION_TIMEOUT_RETRY_* below for what happens if it
# still doesn't fit).
EVALUATION_FRAMES_PER_OUTCOME = 2
# Hard ceiling on total base64-encoded image bytes attached to one
# evaluation request, regardless of how many outcomes an exercise has. When
# the relevance-selected frame set exceeds this, the LEAST relevant frames
# (by SelectedFrame.relevance_score) are dropped first -- never the most
# relevant ones -- down to EVALUATION_MIN_IMAGES_FLOOR. ~333KB/image is a
# generous allowance for a 1280px-wide JPEG at quality 85 (see
# _downscale_image_bytes); six such images comfortably fits under 2MB.
EVALUATION_MAX_IMAGE_BYTES = 3_000_000
# Never trim relevance-selected frames below this floor when bounding bytes
# -- an evaluation with literally zero screenshots defeats the point of a
# visual exercise, so byte-bounding degrades resolution/count gracefully
# rather than all the way to nothing.
EVALUATION_MIN_IMAGES_FLOOR = 2
# When the model call fails (timeout or otherwise), retry EXACTLY ONCE with
# a strictly smaller payload -- half the frames (kept: the most RELEVANT
# half, not an arbitrary temporal subset) and half the event window -- never
# the same oversized request twice. See _get_llm_insights.
EVALUATION_RETRY_FRAME_DIVISOR = 2
EVALUATION_RETRY_EVENT_DIVISOR = 2

# Maximum screenshots attached to the ONE targeted follow-up call (see
# _run_targeted_followup) that re-checks exact-value success criteria the
# main pass could not verify -- either no relevant evidence was selected for
# them, or captured OCR/AX text was too sparse this session to trust an
# "absent" reading. Small and separate from EVALUATION_MAX_VISION_FRAMES: a
# handful of screenshots spread across the WHOLE session (not
# criterion-targeted -- there was no reliable way to pre-identify the right
# one lexically) is enough for the model to spot an answer if it's visible
# anywhere, without paying for a second full evaluation-sized request.
EVALUATION_FOLLOWUP_MAX_FRAMES = 4

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


def _encode_frames(
    frames: List[SelectedFrame], session_id: Optional[str] = None
) -> "tuple[List[str], List[SelectedFrame]]":
    """Like _encode_images, but keeps the returned image list and the
    SelectedFrame list it came from aligned 1:1 (unreadable frames are
    dropped from BOTH, together) -- needed so byte-bounding and retry-time
    frame trimming can reason about relevance_score per actual image sent,
    not just per originally-selected path.
    """
    encoded: List[str] = []
    kept: List[SelectedFrame] = []
    for sf in frames:
        resolved = _resolve_frame_path(sf.path, session_id)
        try:
            with open(resolved, "rb") as f:
                raw = f.read()
        except OSError as exc:
            logger.warning("Could not read screenshot frame %s for vision evaluation: %s", resolved, exc)
            continue
        encoded.append(base64.b64encode(_downscale_image_bytes(raw)).decode("ascii"))
        kept.append(sf)
    return encoded, kept


def _bound_images_by_bytes(
    images: List[str],
    frames: List[SelectedFrame],
    max_bytes: int,
    min_images: int = EVALUATION_MIN_IMAGES_FLOOR,
) -> "tuple[List[str], List[SelectedFrame]]":
    """Drop the LEAST relevant images (by SelectedFrame.relevance_score)
    first when the total base64 payload exceeds max_bytes, never below
    min_images (or below however many images/frames exist, if fewer).
    Preserves the original chronological order of whatever remains.
    """
    total = sum(len(img) for img in images)
    if total <= max_bytes or len(images) <= min_images:
        return images, frames

    paired = list(zip(images, frames))
    # Sort ascending by relevance so the least relevant are dropped first;
    # ties keep their original (chronological) relative order (stable sort).
    by_relevance_asc = sorted(range(len(paired)), key=lambda i: paired[i][1].relevance_score)

    keep_indices = set(range(len(paired)))
    for idx in by_relevance_asc:
        if len(keep_indices) <= min_images:
            break
        candidate_total = sum(len(paired[i][0]) for i in keep_indices if i != idx)
        if candidate_total <= max_bytes:
            keep_indices.discard(idx)
            total = candidate_total
            if total <= max_bytes:
                break
        else:
            # Removing this one alone doesn't get under budget yet, but
            # removing it is still progress toward the floor -- keep going.
            keep_indices.discard(idx)
            total = candidate_total

    kept_pairs = [paired[i] for i in sorted(keep_indices)]
    return [p[0] for p in kept_pairs], [p[1] for p in kept_pairs]


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


class CriterionEvidenceState(str, Enum):
    """The scoring layer's own, mechanically-derived conclusion for ONE
    success criterion -- never trusted blindly from the model's self-report.
    Only the first three may contribute to a COMPLETED assessment decision
    (a genuine pass or fail); the last two mean the assessment could not
    reach a conclusion at all and must never be scored as if the learner
    failed (see EvaluatorService._evaluate_criteria / _score_from_judgment).
    """

    # Positively confirmed: the judgment's claim (or the targeted follow-up's
    # independent reading) matches the criterion, and for an exact-value
    # criterion, matches with genuine corroborating evidence.
    supported = "supported"
    # Evidence was evaluated (corpus was sufficient, or a targeted follow-up
    # ran) and it shows something DIFFERENT than the criterion requires.
    contradicted = "contradicted"
    # Evidence was evaluated and the claimed text/value genuinely does not
    # appear anywhere in the complete captured session.
    not_found_in_session = "not_found_in_session"
    # No relevant evidence was ever selected/considered for this criterion
    # (selection failure) -- the assessment never actually looked.
    not_evaluated = "not_evaluated"
    # Relevant evidence exists but is of insufficient quality to confirm OR
    # deny an exact-value claim (e.g. this session captured no substantive
    # OCR/AX text at all, so a corpus miss proves nothing either way), and a
    # targeted follow-up (if attempted) did not resolve it either.
    unavailable = "unavailable"


# States that may contribute to a COMPLETED (genuinely decided) assessment.
# Never `not_evaluated`/`unavailable` -- see CriterionEvidenceState.
_COMPLETING_CRITERION_STATES = frozenset({
    CriterionEvidenceState.supported,
    CriterionEvidenceState.contradicted,
    CriterionEvidenceState.not_found_in_session,
})


@dataclass
class _CriteriaEvaluation:
    """Result of mechanically evaluating one outcome's success_criteria
    against a judgment -- see EvaluatorService._evaluate_criteria. This is
    the scoring layer's OWN conclusion, derived from (and where necessary,
    overriding) the model's self-reported criteria_met/criteria_not_met."""

    fully_satisfied: bool
    met: List[str]
    not_met: List[str]
    # Criteria whose evidence state is not_evaluated/unavailable -- selection
    # failure or insufficient evidence quality, NEVER treated as a failure.
    pending: List[str]
    rejected_claims: List[str]
    states: Dict[str, CriterionEvidenceState]


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


# -- Report-consistency reconciliation ----------------------------------------
#
# The LLM's free-text summary/strengths/observed_approach are generated in
# the SAME response as its structured outcome_judgments, but the SCORE comes
# from outcomes that may have since been overridden by deterministic rules
# the LLM never sees applied (baseline attribution, the evidence_basis gate,
# a failed final-state check). Nothing previously reconciled the narrative
# against that final, authoritative outcome list, so a locally-correct-
# sounding sentence ("the learner set permissions and verified them") could
# survive alongside a 0-scored outcome for the exact same thing. This
# section makes the scored OutcomeResult list the single source of truth for
# what the narrative is allowed to claim -- not a second, independent
# interpretation.
#
# Deliberately NOT sentence-level NLP: a small, explainable keyword-overlap
# check scoped to each outcome's OWN authored wording (description /
# student_demonstration), used only to detect and replace/drop narrative
# that credits a specific NOT-PASSED outcome -- never to rewrite prose.

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "this", "that", "of", "to", "in", "on",
    "for", "and", "or", "with", "student", "learner", "their", "its", "it", "as", "by",
    "during", "session", "already", "e", "g",
}
# Words that, alongside topic overlap with a NOT-PASSED outcome and no
# nearby negation, mark a summary sentence as affirmatively (wrongly)
# claiming that outcome was completed.
_COMPLETION_CUES = {
    "created", "creates", "creating", "set", "sets", "configured", "configures",
    "verified", "verifies", "verifying", "completed", "completes", "demonstrated",
    "demonstrates", "successfully", "did", "performed", "performs", "ran", "runs",
    "established", "establishes", "confirmed", "confirms", "applied", "applies",
}
_NEGATION_CUES = {
    "not", "never", "no", "without", "unable", "fail", "fails", "failed", "missing",
    "absent", "cannot", "couldn't", "didn't", "wasn't", "weren't", "isn't", "aren't",
    "n't",
}


def _significant_words(text: Optional[str]) -> set:
    if not text:
        return set()
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _outcome_reference_words(outcome: ExpectedOutcome) -> set:
    # `description` only, deliberately: it's short and topic-focused (e.g.
    # "~/secure-data exists"), which is what makes a ratio-based overlap
    # check meaningful. `student_demonstration` is a full paragraph written
    # for prompt guidance (see format_outcomes_for_evaluation) -- unioning
    # it in here would dilute the ratio so much that almost nothing could
    # ever cross the threshold.
    return _significant_words(outcome.description)


def _mentions_outcome(text_words: set, outcome_words: set, threshold: float = 0.5) -> bool:
    """True when `text_words` closely echoes an outcome's OWN authored
    wording -- at least two overlapping significant words, and at least
    `threshold` of the outcome's reference words present. Requiring both a
    minimum count and a minimum ratio keeps a single common word (or a
    short description) from triggering a match on its own.
    """
    if not outcome_words:
        return False
    overlap = text_words & outcome_words
    return len(overlap) >= 2 and len(overlap) / len(outcome_words) >= threshold


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


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

        # Structural, deterministic facts about captured evidence (e.g. "a
        # chmod-style command ran", "a long-format permission listing was
        # produced") -- [] for exercises with no registered extractor, so
        # this has zero effect anywhere else. See app.services.verifier and
        # _score_from_judgment, which treats a present, action-basis fact as
        # sufficient on its own rather than relying solely on the LLM to
        # re-derive the same fact from noisy OCR text.
        evidence_facts = extract_evidence_facts(exercise, events, verification)

        llm_insights, ai_unavailable_reason, uncovered_criteria, followup_answers = await self._get_llm_insights(
            exercise, events, verification, evidence_error, session_id, manifest=manifest,
            baseline_verification=baseline_verification, evidence_facts=evidence_facts,
        )
        ocr_sufficient = ocr_text_available(events)

        outcomes: List[OutcomeResult] = []
        for step in exercise.get_steps():
            is_real_step = step.id != "_default"
            for outcome in step.expected_outcomes:
                result = self._score_outcome(
                    outcome, verification, events, llm_insights, evidence_error,
                    baseline_verification=baseline_verification, evidence_facts=evidence_facts,
                    ai_unavailable_reason=ai_unavailable_reason,
                    uncovered_criteria=set(uncovered_criteria.get(outcome.id, [])),
                    followup_answers=followup_answers, ocr_sufficient=ocr_sufficient,
                )
                if is_real_step:
                    result.step_id = step.id
                    result.step_title = step.title
                outcomes.append(result)
        total_score = sum(o.score for o in outcomes)

        summary, strengths, observed_approach = self._reconcile_narrative(
            exercise, outcomes, llm_insights.summary, llm_insights.strengths, llm_insights.observed_approach,
        )

        return EvaluationResult(
            score=round(total_score, 1),
            outcomes=outcomes,
            summary=summary,
            strengths=strengths,
            improvements=llm_insights.improvements,
            observed_approach=observed_approach,
            alternative_approaches=llm_insights.alternative_approaches,
            risky_or_unnecessary_steps=llm_insights.risky_or_unnecessary_steps,
            evidence_error=evidence_error,
            ai_unavailable=ai_unavailable_reason,
        )

    def _select_evaluation_frames(
        self, exercise: Exercise, events: List[EvidenceEvent], session_id: Optional[str], frame_budget: int,
    ) -> "tuple[List[str], List[SelectedFrame], Dict[str, List[str]]]":
        """Criterion-COVERAGE-ranked frame selection, bounded by both count
        and total byte size -- see select_keyframes_with_coverage(...) and
        _bound_images_by_bytes. Also returns which (outcome_id -> [criterion
        text]) pairs received NO relevant evidence at all, so the scoring
        layer can mark them not_evaluated (never a silent learner failure)
        instead of pretending the packet covered everything.
        """
        outcomes = exercise.get_all_outcomes()
        selected_frames, uncovered = select_keyframes_with_coverage(events, frame_budget, outcomes)
        images, kept_frames = _encode_frames(selected_frames, session_id)
        images, kept_frames = _bound_images_by_bytes(images, kept_frames, EVALUATION_MAX_IMAGE_BYTES)
        return images, kept_frames, uncovered

    async def _call_ollama(
        self, system_prompt: str, prompt: str, images: List[str], model: str,
        context_size: int, timeout: float, use_vision: bool,
    ) -> LLMEvaluationInsights:
        if use_vision:
            return await self.ollama.evaluate(
                system_prompt, prompt, images=images, model=model,
                num_ctx=context_size, timeout=timeout,
            )
        return await self.ollama.evaluate(
            system_prompt, prompt, num_ctx=context_size, timeout=timeout,
        )

    @staticmethod
    def _evenly_sample_paths(paths: List[str], budget: int) -> List[str]:
        n = len(paths)
        if n <= budget:
            return paths
        if budget <= 1:
            return [paths[-1]]
        indices = sorted({round(i * (n - 1) / (budget - 1)) for i in range(budget)})
        return [paths[i] for i in indices]

    async def _run_targeted_followup(
        self,
        pending: List["tuple[str, str]"],
        events: List[EvidenceEvent],
        session_id: Optional[str],
        model: str,
        context_size: int,
        timeout: float,
    ) -> Dict[str, str]:
        """ONE additional vision call asking, in NEUTRAL terms (no expected
        value is ever named), what is actually visible relevant to each
        pending exact-value criterion. Attaches a handful of screenshots
        spread across the WHOLE session -- there is no reliable way to
        lexically pre-identify which specific frame shows the answer for a
        text-sparse capture (that is exactly why these criteria are
        pending), so the model is given breadth instead of a single guess.

        Returns {criterion_text: reported_answer}; entries are ABSENT for
        any item the model didn't answer or that came back malformed --
        callers must not treat a missing entry as either confirmation or
        denial. `pending` is a list of (outcome_id, criterion_text) --
        outcome_id is accepted for future diagnostics but not otherwise used
        here (the answer is keyed purely by criterion_text since a criterion
        already uniquely identifies itself within one exercise's outcomes).
        """
        if not pending:
            return {}
        all_paths: List[str] = []
        seen: set = set()
        for e in events:
            if e.frame_path and e.frame_path not in seen:
                seen.add(e.frame_path)
                all_paths.append(e.frame_path)
        if not all_paths:
            return {}
        sample_paths = self._evenly_sample_paths(all_paths, EVALUATION_FOLLOWUP_MAX_FRAMES)
        images = _encode_images(sample_paths, session_id)
        if not images:
            return {}

        questions = []
        for i, (_oid, criterion) in enumerate(pending, 1):
            neutral = re.sub(r"`[^`]+`", "[a specific value]", criterion)
            questions.append(f"{i}. {neutral}")

        user_prompt = (
            f"Attached are {len(images)} screenshots from one learner's session, "
            "spread across the session and not necessarily all relevant.\n\n"
            "For EACH numbered item below, report exactly what is visible in "
            "the screenshots relevant to it.\n\n" + "\n".join(questions) +
            "\n\nRespond with ONLY the JSON object described in the system prompt, "
            f"with exactly {len(pending)} entries in \"answers\", in order."
        )

        try:
            data = await self.ollama.chat_json(
                TARGETED_FOLLOWUP_SYSTEM_PROMPT, user_prompt,
                temperature=get_settings().evaluation_temperature,
                images=images, model=model, num_ctx=context_size, timeout=timeout,
            )
        except OllamaError as exc:
            logger.warning(
                "Targeted follow-up evidence check failed (%s); %d criteria remain unavailable",
                exc, len(pending),
            )
            return {}

        if not data or not isinstance(data.get("answers"), list):
            return {}
        answers = data["answers"]
        result: Dict[str, str] = {}
        for (_oid, criterion), answer in zip(pending, answers):
            if isinstance(answer, str) and answer.strip():
                result[criterion] = answer
        return result

    async def _get_llm_insights(
        self,
        exercise: Exercise,
        events: List[EvidenceEvent],
        verification: Dict[str, VerificationDetail],
        evidence_error: Optional[str] = None,
        session_id: Optional[str] = None,
        manifest: Optional[ManifestWriter] = None,
        baseline_verification: Optional[Dict[str, VerificationDetail]] = None,
        evidence_facts: Optional[List[EvidenceFact]] = None,
    ) -> "tuple[LLMEvaluationInsights, Optional[str], Dict[str, List[str]], Dict[str, str]]":
        """Returns (insights, ai_unavailable_reason, uncovered_criteria,
        followup_answers).

        ai_unavailable_reason is None on success (including a successful
        retry); otherwise it names why the LLM call could not be completed
        even after one retry with a smaller evidence packet -- callers must
        score affected outcomes as unavailable/unscored, never as a silent 0
        that reads like the learner didn't do the work (see
        _score_from_judgment).

        uncovered_criteria maps outcome_id -> [criterion text] for every
        success criterion that received NO relevant evidence in the selected
        packet at all (see select_keyframes_with_coverage) -- the scoring
        layer must treat these as not_evaluated, never as a failure.

        followup_answers maps criterion text -> the targeted follow-up's
        independently-observed answer, for exact-value criteria the main
        pass could not verify (uncovered, or insufficient OCR/AX text this
        session) -- see _run_targeted_followup. Empty when no follow-up was
        needed or it could not be completed.
        """
        settings = get_settings()
        routing = _select_model(exercise, events, settings)
        use_vision, model, reason = routing.use_vision, routing.model, routing.reason

        outcomes = exercise.get_all_outcomes()
        # Budget ~1 frame per authored success criterion (not per outcome --
        # coverage of individual criteria is what evidence selection now
        # optimizes for, see select_keyframes_with_coverage), capped at
        # EVALUATION_MAX_VISION_FRAMES so a much larger exercise is still
        # bounded. A criterion may still receive a 2nd frame when it has
        # multiple strong candidates (see MAX_FRAMES_PER_CRITERION in
        # keyframes.py); this is only the TARGET, not a hard per-criterion
        # cap. Nothing here is specific to any one exercise or vocabulary.
        total_criteria = sum(len(o.get_success_criteria()) for o in outcomes) if outcomes else 0
        frame_budget = min(EVALUATION_MAX_VISION_FRAMES, max(1, total_criteria)) if outcomes else EVALUATION_MAX_VISION_FRAMES

        images: List[str] = []
        frame_captions: Optional[str] = None
        selected_frames: List[SelectedFrame] = []
        uncovered_criteria: Dict[str, List[str]] = {}
        if use_vision:
            images, selected_frames, uncovered_criteria = self._select_evaluation_frames(
                exercise, events, session_id, frame_budget,
            )
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

        def render(count: int, img_count: int, captions: Optional[str]):
            prompt, per_event_truncated = build_evaluation_user_prompt(
                exercise, events, verification, evidence_error,
                has_images=use_vision, frame_captions=captions, max_events=count,
                baseline_verification=baseline_verification, evidence_facts=evidence_facts,
            )
            return EVALUATION_SYSTEM_PROMPT, prompt, per_event_truncated

        system_prompt, prompt, selected_count, estimated_tokens, truncated = fit_to_budget(
            lambda c: render(c, len(images), frame_captions), EVALUATION_MAX_EVENTS, context_size,
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

        available_frames = [
            {
                "path": e.frame_path,
                "timestamp": e.timestamp.isoformat(),
                "application": e.application,
            }
            for e in events
            if e.frame_path
        ]

        def _record_manifest(frames: List[SelectedFrame], sys_p: str, user_p: str,
                              retry_used: bool, ai_unavailable: Optional[str]) -> None:
            if manifest is None:
                return
            try:
                manifest.record_final_evaluation(
                    model_request_timestamp=datetime.now(timezone.utc).isoformat(),
                    model=model,
                    routing_reason=reason,
                    event_count=len(events),
                    normalized_event_count=len(events),
                    available_frames=available_frames,
                    selected_frames=frames,
                    images_attached=bool(images),
                    use_vision=use_vision,
                    retry_used=retry_used,
                    ai_unavailable=ai_unavailable,
                )
                if use_vision:
                    manifest.save_diagnostic_payload(
                        consumer="final_evaluation_retry" if retry_used else "final_evaluation",
                        system_prompt=sys_p,
                        user_prompt=user_p,
                        selected_frames=frames,
                    )
            except Exception as exc:  # noqa: BLE001 -- manifest errors must never fail evaluation
                logger.warning("Evaluation manifest write failed: %s", exc)

        try:
            insights = await self._call_ollama(system_prompt, prompt, images, model, context_size, eval_timeout, use_vision)
            _record_manifest(selected_frames, system_prompt, prompt, retry_used=False, ai_unavailable=None)
        except OllamaError as exc:
            logger.warning(
                "Initial evaluation call failed (%s); retrying once with a smaller, "
                "still-relevant evidence packet instead of repeating the same request", exc,
            )
            retry_images, retry_frames, retry_captions = images, selected_frames, frame_captions
            if use_vision and len(selected_frames) > 1:
                retry_frame_count = max(1, len(selected_frames) // EVALUATION_RETRY_FRAME_DIVISOR)
                # Keep the MOST relevant frames (highest relevance_score),
                # never an arbitrary temporal subset -- a timeout must not
                # be "fixed" by discarding the evidence that mattered.
                ranked = sorted(selected_frames, key=lambda sf: sf.relevance_score, reverse=True)
                keep_paths = {sf.path for sf in ranked[:retry_frame_count]}
                retry_frames = [sf for sf in selected_frames if sf.path in keep_paths]
                retry_images, retry_frames = _encode_frames(retry_frames, session_id)
                retry_captions = format_frame_captions(retry_frames) if retry_frames else None
            retry_event_ceiling = max(EVALUATION_MIN_EVENTS, selected_count // EVALUATION_RETRY_EVENT_DIVISOR)

            retry_system_prompt, retry_prompt, retry_count, retry_tokens, retry_truncated = fit_to_budget(
                lambda c: render(c, len(retry_images), retry_captions), retry_event_ceiling, context_size,
                image_count=len(retry_images) if use_vision else 0, min_count=EVALUATION_MIN_EVENTS,
            )
            logger.info(
                "Evaluation retry payload: screenshots=%d (was %d), events<=%d (was %d), estimated_tokens=%d",
                len(retry_images), len(images), retry_count, selected_count, retry_tokens,
            )

            try:
                insights = await self._call_ollama(
                    retry_system_prompt, retry_prompt, retry_images, model, context_size, eval_timeout, use_vision,
                )
                images, selected_frames, frame_captions = retry_images, retry_frames, retry_captions
                _record_manifest(retry_frames, retry_system_prompt, retry_prompt, retry_used=True, ai_unavailable=None)
            except OllamaError as exc2:
                logger.warning("Retry evaluation call also failed (%s); marking evaluation unavailable", exc2)
                _record_manifest(retry_frames, retry_system_prompt, retry_prompt, retry_used=True, ai_unavailable=str(exc2))
                if events:
                    # Capture worked -- only the AI evaluation call failed. Must not
                    # be worded as if activity capture itself was unavailable.
                    summary = (
                        f"Activity was captured successfully, but AI evaluation failed ({exc2}). "
                        "Scoring below relies on deterministic verification and observed evidence only."
                    )
                else:
                    summary = (
                        f"AI-based process analysis was unavailable ({exc2}), and no activity evidence "
                        "was captured either. Scoring below relies on deterministic verification only."
                    )
                return LLMEvaluationInsights(summary=summary), str(exc2), uncovered_criteria, {}

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

        # Targeted follow-up: for any EXACT-VALUE criterion (backtick-quoted
        # literal) that either received no evidence in the main packet at
        # all, or whose absence from the text corpus cannot be trusted
        # (this session captured no substantive OCR/AX text to confirm
        # absence one way or the other -- see ocr_text_available), ask ONE
        # additional neutral question rather than silently leaving it
        # unresolved. Never fires for non-exact-value criteria (a visual
        # claim with nothing to quote) or when OCR text this session WAS
        # substantive enough that a corpus miss is trustworthy on its own.
        followup_answers: Dict[str, str] = {}
        if use_vision and not evidence_error:
            ocr_sufficient = ocr_text_available(events)
            pending: List["tuple[str, str]"] = []
            for outcome in outcomes:
                outcome_uncovered = set(uncovered_criteria.get(outcome.id, []))
                for criterion in outcome.get_success_criteria():
                    if not backtick_literals(criterion):
                        continue
                    if criterion in outcome_uncovered or not ocr_sufficient:
                        pending.append((outcome.id, criterion))
            if pending:
                followup_answers = await self._run_targeted_followup(
                    pending, events, session_id, model, context_size, eval_timeout,
                )

        return insights, None, uncovered_criteria, followup_answers

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

    @staticmethod
    def _matching_action_fact(
        outcome_id: str, evidence_facts: Optional[List[EvidenceFact]]
    ) -> Optional[EvidenceFact]:
        for fact in evidence_facts or []:
            if fact.outcome_id == outcome_id and fact.present and fact.basis == EvidenceBasis.action:
                return fact
        return None

    # Feedback text shown when a filesystem outcome's compliant state already
    # existed at baseline and no current-session action demonstrated it --
    # deliberately overrides whatever the LLM's own per-outcome feedback said,
    # since that's exactly the situation where an LLM is prone to suggesting
    # `mkdir`/`touch` to "prove" creation of something that already exists.
    # No files are deleted or reset -- see app.routes.sessions -- this only
    # ever recommends a clean workspace, never performs one.
    _ALREADY_EXISTED_FEEDBACK = (
        "This already existed before the session started, so nothing this "
        "session could prove creating it. If demonstrating creation matters "
        "here, start from a clean/reset lab workspace rather than deleting "
        "or recreating existing files by hand."
    )

    def _score_outcome(
        self,
        outcome: ExpectedOutcome,
        verification: Dict[str, VerificationDetail],
        events: List[EvidenceEvent],
        llm_insights: LLMEvaluationInsights,
        evidence_error: Optional[str] = None,
        baseline_verification: Optional[Dict[str, VerificationDetail]] = None,
        evidence_facts: Optional[List[EvidenceFact]] = None,
        ai_unavailable_reason: Optional[str] = None,
        uncovered_criteria: Optional[Set[str]] = None,
        followup_answers: Optional[Dict[str, str]] = None,
        ocr_sufficient: bool = True,
    ) -> OutcomeResult:
        if outcome.type == OutcomeType.filesystem and outcome.id in verification:
            return self._score_filesystem_outcome(
                outcome, verification, baseline_verification, events, llm_insights, evidence_error,
                evidence_facts, ai_unavailable_reason=ai_unavailable_reason,
                uncovered_criteria=uncovered_criteria, followup_answers=followup_answers,
                ocr_sufficient=ocr_sufficient,
            )

        # observed_behavior AND process outcomes (and any filesystem outcome
        # without a matching deterministic check) all use the same generalized
        # per-outcome LLM judgment -- there is no app-specific ground truth to
        # fall back on, so this is the only mechanism available for anything
        # that isn't a declared filesystem check.
        return self._score_from_judgment(
            outcome, events, llm_insights, evidence_error, evidence_facts,
            ai_unavailable_reason=ai_unavailable_reason,
            uncovered_criteria=uncovered_criteria, followup_answers=followup_answers,
            ocr_sufficient=ocr_sufficient,
        )

    def _score_filesystem_outcome(
        self,
        outcome: ExpectedOutcome,
        verification: Dict[str, VerificationDetail],
        baseline_verification: Optional[Dict[str, VerificationDetail]],
        events: List[EvidenceEvent],
        llm_insights: LLMEvaluationInsights,
        evidence_error: Optional[str] = None,
        evidence_facts: Optional[List[EvidenceFact]] = None,
        ai_unavailable_reason: Optional[str] = None,
        uncovered_criteria: Optional[Set[str]] = None,
        followup_answers: Optional[Dict[str, str]] = None,
        ocr_sufficient: bool = True,
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
        result = self._score_from_judgment(
            outcome, events, llm_insights, evidence_error, evidence_facts,
            ai_unavailable_reason=ai_unavailable_reason,
            uncovered_criteria=uncovered_criteria, followup_answers=followup_answers,
            ocr_sufficient=ocr_sufficient,
        )
        result.final_state_verified = True

        fact = self._matching_action_fact(outcome.id, evidence_facts)
        if fact is not None and result.passed:
            # _score_from_judgment already credited this from a structural
            # fact, not an LLM judgment -- that fact IS the action evidence,
            # so the gate below (which exists to distrust a bare LLM
            # observed=true) does not apply here.
            result.demonstrated_this_session = True
            return result

        # Mechanical backstop: never trust a bare observed=true here. Look up
        # the judgment's own evidence_basis and require it to be "action" --
        # state evidence (a listing, a window title, a status display) can
        # prove the state exists, which nobody disputes, but not that THIS
        # session produced it. This holds regardless of what the LLM set
        # `observed` to, so a judgment that got the classification wrong
        # cannot smuggle state evidence through as a pass.
        judgment = next((j for j in llm_insights.outcome_judgments if j.id == outcome.id), None)
        if result.passed and (judgment is None or judgment.evidence_basis != EvidenceBasis.action):
            result.passed = False
            result.score = 0.0
            result.confidence = Confidence.unknown
            result.verification_state = VerificationState.not_observed
            result.evidence = (
                "The evidence found only shows the resulting state (e.g. a "
                "listing or window title), which already existed before this "
                "session began -- it does not show the action being "
                "performed during this session, so this is not demonstrated "
                "this attempt."
            )
            result.feedback = self._ALREADY_EXISTED_FEEDBACK
        elif not result.passed and not evidence_error:
            result.evidence = (
                "This state already existed when the session started, and no "
                "current-session activity demonstrated it again -- not "
                "demonstrated this attempt."
            )
            result.feedback = self._ALREADY_EXISTED_FEEDBACK

        result.demonstrated_this_session = result.passed
        return result

    @staticmethod
    def _criterion_claimed_support(
        criterion: str, judgment: OutcomeJudgment, distinctive_words: "Optional[Set[str]]" = None,
    ) -> "tuple[Optional[bool], Optional[str]]":
        """What did the judgment claim for ONE specific authored criterion?
        Returns (claimed_supported, quote). claimed_supported is None when
        the criterion was never addressed at all (neither confirmed nor
        denied) -- that counts as unmet (missing criterion judgments fail
        closed), handled by the caller.

        Prefers a structured CriterionJudgment (matched by normalized
        criterion text) over the legacy criteria_met/criteria_not_met string
        lists, so a response using either shape scores identically.

        `distinctive_words`: this criterion's significant words that do NOT
        also appear in any OTHER criterion of the same outcome (see
        _evaluate_criteria) -- used only by the tier-3 fallback below.
        """
        norm_c = normalize_literal(criterion)
        # Tier 1: structured criterion_judgments, near-exact text match.
        for cj in judgment.criterion_judgments:
            norm_cj = normalize_literal(cj.criterion)
            if norm_cj == norm_c or norm_c in norm_cj or norm_cj in norm_c:
                return cj.supported, cj.quote

        # Tier 2: legacy criteria_met/criteria_not_met, near-exact text match.
        def _matches(entry: str) -> bool:
            norm_e = normalize_literal(entry)
            return norm_e == norm_c or norm_c in norm_e or norm_e in norm_c

        if any(_matches(e) for e in judgment.criteria_met):
            return True, None
        if any(_matches(e) for e in judgment.criteria_not_met):
            return False, None

        # Tier 3: fallback for a judgment that PARAPHRASES or MERGES several
        # criteria into one summarized statement instead of copying the
        # authored text verbatim -- a real, observed llava:latest failure
        # mode despite the prompt explicitly asking for exact criterion text
        # ("prompt instructions alone are insufficient" applies here too).
        # Associate this criterion with whichever candidate statement (from
        # EITHER list) shares the most significant words with it -- see
        # _best_overlap_claim for the exact acceptance rule.
        candidates: List["tuple[str, bool, Optional[str]]"] = [
            (cj.criterion, cj.supported, cj.quote) for cj in judgment.criterion_judgments
        ]
        candidates += [(e, True, None) for e in judgment.criteria_met]
        candidates += [(e, False, None) for e in judgment.criteria_not_met]
        return EvaluatorService._best_overlap_claim(criterion, distinctive_words or set(), candidates)

    # Minimum shared significant words to associate a paraphrased statement
    # with a criterion when NONE of the shared words are distinctive to that
    # specific criterion (see below) -- requiring 2+ generic shared words
    # keeps a single incidental common term (e.g. two sibling criteria of
    # the SAME outcome both happening to mention "total") from cross-
    # crediting the wrong one; genuine paraphrases of the SAME criterion
    # reliably share several content words even when generic ones are
    # excluded.
    _FALLBACK_OVERLAP_MIN_WORDS = 2

    @staticmethod
    def _best_overlap_claim(
        criterion: str, distinctive_words: "Set[str]", candidates: List["tuple[str, bool, Optional[str]]"],
    ) -> "tuple[Optional[bool], Optional[str]]":
        """A candidate statement is accepted as addressing `criterion` if it
        shares >=2 significant words with it, OR shares just ONE word that
        is DISTINCTIVE to this criterion (i.e. not also present in any
        sibling criterion of the same outcome). The single-distinctive-word
        allowance is what lets a short, on-topic paraphrase (e.g. "the
        upload summary shows success" for a criterion about an "upload
        completion message") register as a real match even though it only
        shares one content word with the criterion's own wording -- while
        still refusing to cross-credit on a single GENERIC word two
        criteria happen to share (that word is, by construction, excluded
        from being "distinctive").
        """
        criterion_words = significant_words(criterion)
        best: Optional["tuple[int, bool, Optional[str]]"] = None
        for text, supported, quote in candidates:
            overlap_words = criterion_words & significant_words(text)
            if not overlap_words:
                continue
            if len(overlap_words) < EvaluatorService._FALLBACK_OVERLAP_MIN_WORDS and not (overlap_words & distinctive_words):
                continue
            score = len(overlap_words)
            if best is None or score > best[0]:
                best = (score, supported, quote)
        if best is None:
            return None, None
        return best[1], best[2]

    def _evaluate_criteria(
        self,
        outcome: ExpectedOutcome,
        judgment: OutcomeJudgment,
        events: List[EvidenceEvent],
        uncovered_criteria: Optional[Set[str]] = None,
        followup_answers: Optional[Dict[str, str]] = None,
        ocr_sufficient: bool = True,
    ) -> "_CriteriaEvaluation":
        """Mechanically derive per-criterion evidence STATE -- the SCORING
        layer's own conclusion, never trusted blindly from the model's
        self-report. See CriterionEvidenceState for the five possible states
        and which three may complete an assessment.

        For a criterion naming an exact literal value (a field name,
        command, value, count, or identifier -- conventionally
        backtick-quoted, e.g. "the field `src_ip` appears"):
        - If NO relevant evidence was ever selected for it (`criterion in
          uncovered_criteria`) or this session captured no substantive
          OCR/AX text at all (`not ocr_sufficient`) -- a text-corpus miss
          proves nothing either way -- the claim is neither trusted NOR
          rejected outright. If the targeted follow-up (see
          _run_targeted_followup) independently observed a value, that
          value (never the model's own possibly-parroted quote) decides
          supported/contradicted; otherwise the criterion is `unavailable`.
        - Otherwise (evidence WAS selected and OCR this session was
          substantive), a claim of support is trusted ONLY if that exact
          literal (case/whitespace-normalized, no fuzzy/semantic matching --
          `dest_ip` is not `dst_ip`) is found verbatim in the corpus; a miss
          here IS meaningful (`not_found_in_session`), since the corpus was
          actually capable of confirming it.

        Criteria with no backtick-quoted literal (e.g. "at least one result
        is returned", or a genuinely visual claim like a graph shape) are
        never subject to any corpus/follow-up check -- there is no exact
        value to verify, so the model's own claim (structured or legacy)
        is trusted as-is; `not_evaluated` still applies if evidence was
        never selected for it at all.
        """
        uncovered_criteria = uncovered_criteria or set()
        followup_answers = followup_answers or {}
        corpus = build_text_corpus(events)
        met: List[str] = []
        not_met: List[str] = []
        pending: List[str] = []
        rejected: List[str] = []
        states: Dict[str, CriterionEvidenceState] = {}
        per_criterion_words = [significant_words(c) for c in outcome.success_criteria]
        per_criterion_distinctive = distinctive_word_sets(per_criterion_words)

        for i, criterion in enumerate(outcome.success_criteria):
            distinctive_words = per_criterion_distinctive[i]
            claimed_supported, quote = self._criterion_claimed_support(criterion, judgment, distinctive_words)
            literals = backtick_literals(criterion)
            is_uncovered = criterion in uncovered_criteria

            if not literals:
                if claimed_supported is True:
                    state = CriterionEvidenceState.supported
                elif is_uncovered:
                    # No relevant evidence was ever selected for this
                    # criterion -- a selection failure, not a checked-and-
                    # failed claim.
                    state = CriterionEvidenceState.not_evaluated
                else:
                    # Evidence WAS selected/available; the judgment either
                    # explicitly denied it or silently never addressed it --
                    # both are a genuine (checked) non-confirmation, not a
                    # selection failure, so this stays a real not-met rather
                    # than being read as merely incomplete.
                    state = CriterionEvidenceState.not_found_in_session
            else:
                followup_answer = followup_answers.get(criterion)
                if followup_answer is not None:
                    norm_answer = normalize_literal(followup_answer)
                    if not norm_answer or norm_answer in ("not visible", "none", "n/a"):
                        state = CriterionEvidenceState.unavailable
                    elif all(normalize_literal(lit) in norm_answer for lit in literals):
                        state = CriterionEvidenceState.supported
                    else:
                        state = CriterionEvidenceState.contradicted
                        rejected.append(
                            f'"{criterion}" -- a targeted follow-up check independently observed '
                            f"\"{followup_answer}\", not the expected {', '.join(literals)} -- treated as contradicted."
                        )
                elif is_uncovered or not ocr_sufficient:
                    # No selected evidence, or this session's OCR/AX text is
                    # too sparse to trust an absence reading either way --
                    # the follow-up either wasn't attempted or didn't
                    # resolve this specific criterion.
                    state = CriterionEvidenceState.unavailable
                elif claimed_supported:
                    missing = [lit for lit in literals if normalize_literal(lit) not in corpus]
                    if missing:
                        state = CriterionEvidenceState.not_found_in_session
                        seen_note = f' (model quoted: "{quote}")' if quote else ""
                        rejected.append(
                            f'"{criterion}" was claimed supported, but '
                            f"{', '.join(missing)} not found verbatim in captured evidence text{seen_note} -- rejected."
                        )
                    else:
                        state = CriterionEvidenceState.supported
                else:
                    # claimed_supported is False, or None (never addressed
                    # despite evidence being available and OCR being
                    # sufficient) -- both are a genuine checked
                    # non-confirmation, not a selection failure.
                    state = CriterionEvidenceState.not_found_in_session

            states[criterion] = state
            if state == CriterionEvidenceState.supported:
                met.append(criterion)
            elif state in (CriterionEvidenceState.not_evaluated, CriterionEvidenceState.unavailable):
                pending.append(criterion)
            else:
                not_met.append(criterion)

        fully = not not_met and not pending and len(met) == len(outcome.success_criteria)
        return _CriteriaEvaluation(
            fully_satisfied=fully, met=met, not_met=not_met, pending=pending,
            rejected_claims=rejected, states=states,
        )

    @staticmethod
    def _criteria_evidence_text(judgment: OutcomeJudgment, evaluation: "_CriteriaEvaluation") -> str:
        """Student/instructor-facing explanation for why an outcome the model
        marked observed=true still did not earn full credit -- must name the
        actual gap (unmet criteria, pending/unresolved criteria, and any
        specific claim mechanically rejected for an unsupported exact value)
        so feedback never contradicts the score by looking like a generic
        failure when real partial progress was found, and never reads as a
        confident failure when the true issue is missing/insufficient
        evidence rather than a checked-and-failed criterion."""
        parts: List[str] = [judgment.evidence] if judgment.evidence else []
        if evaluation.not_met:
            parts.append("Unmet success criteria: " + "; ".join(evaluation.not_met))
        if evaluation.pending:
            parts.append(
                "Could not be assessed (no relevant evidence selected, or evidence quality was "
                "insufficient to confirm or deny): " + "; ".join(evaluation.pending)
            )
        if evaluation.rejected_claims:
            parts.append("Rejected unsupported claims: " + " ".join(evaluation.rejected_claims))
        total = len(evaluation.met) + len(evaluation.not_met) + len(evaluation.pending)
        if evaluation.not_met or evaluation.pending:
            parts.append(
                f"{len(evaluation.met)} of {total} required success criteria were confirmed supported."
            )
        if not parts:
            parts.append("Not all required success criteria were supported by the available evidence.")
        return " ".join(parts)

    def _score_from_judgment(
        self,
        outcome: ExpectedOutcome,
        events: List[EvidenceEvent],
        llm_insights: LLMEvaluationInsights,
        evidence_error: Optional[str] = None,
        evidence_facts: Optional[List[EvidenceFact]] = None,
        ai_unavailable_reason: Optional[str] = None,
        uncovered_criteria: Optional[Set[str]] = None,
        followup_answers: Optional[Dict[str, str]] = None,
        ocr_sufficient: bool = True,
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

        fact = self._matching_action_fact(outcome.id, evidence_facts)
        if fact is not None:
            # A structurally-derived, deterministic fact (see
            # app.services.verifier) establishes current-session action
            # evidence on its own -- not solely the LLM's structured output,
            # which is what this corroborates against: a local model can
            # correctly reason about an action in its free-text summary yet
            # still return an inconsistent/missing structured judgment for
            # the same outcome. Trust the fact even when that happens.
            return OutcomeResult(
                id=outcome.id,
                passed=True,
                score=outcome.weight,
                max_score=outcome.weight,
                confidence=Confidence.verified,
                evidence=fact.detail,
                verification_state=VerificationState.verified,
            )

        judgment = next((j for j in llm_insights.outcome_judgments if j.id == outcome.id), None)

        # Set criteria_implicit deterministically — do not trust what the LLM claimed.
        if judgment is not None:
            judgment.criteria_implicit = not outcome.has_explicit_criteria

        if judgment is not None and judgment.observed:
            # Screenshots are the primary source of truth for visual exercises
            # (see _select_model): when the vision model directly observes an
            # action on screen, that is direct evidence -- not inference. But
            # for an ENRICHED outcome (explicit success_criteria authored),
            # observed=true by itself is not sufficient for FULL credit --
            # see _criteria_fully_satisfied. Legacy outcomes (no explicit
            # criteria -- nothing more granular was ever authored to check)
            # keep the original behavior unchanged: observed=true earns full
            # credit outright.
            if not outcome.has_explicit_criteria:
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
            # Enriched outcome: the SCORING layer, not the model, derives the
            # final state from mechanically-validated per-criterion results
            # (see _evaluate_criteria) -- a self-reported observed=true or
            # verification_state="verified" is never sufficient on its own.
            criteria_evaluation = self._evaluate_criteria(
                outcome, judgment, events, uncovered_criteria=uncovered_criteria,
                followup_answers=followup_answers, ocr_sufficient=ocr_sufficient,
            )
            if criteria_evaluation.pending and not criteria_evaluation.not_met:
                # Every unresolved criterion is pending (not_evaluated /
                # unavailable) and NONE were genuinely checked-and-failed --
                # this outcome's assessment is INCOMPLETE, not a confident
                # failure. Never present this as a learner failure: the
                # evidence to decide it one way or the other simply was not
                # selected, or was of insufficient quality to confirm or
                # deny an exact-value claim.
                return OutcomeResult(
                    id=outcome.id,
                    passed=False,
                    score=0.0,
                    max_score=outcome.weight,
                    confidence=Confidence.unknown,
                    evidence=self._criteria_evidence_text(judgment, criteria_evaluation),
                    verification_state=VerificationState.unverifiable,
                    feedback=judgment.feedback,
                )
            if criteria_evaluation.fully_satisfied:
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
            # Some required criteria are unmet, unaddressed, or were claimed
            # met on an exact-value quote that the captured evidence does not
            # actually contain. The YAML defines no numeric partial-credit
            # scale for this outcome, so this fails closed at 0 rather than
            # inventing a fraction -- partial criteria coverage (or an
            # unsupported exact-value claim) must never silently become full
            # credit.
            return OutcomeResult(
                id=outcome.id,
                passed=False,
                score=0.0,
                max_score=outcome.weight,
                confidence=Confidence.unknown,
                evidence=self._criteria_evidence_text(judgment, criteria_evaluation),
                verification_state=judgment.verification_state,
                feedback=judgment.feedback,
            )

        if judgment is None:
            outcome_uncovered = (uncovered_criteria or set()) & set(outcome.success_criteria)
            all_uncovered = bool(outcome.success_criteria) and outcome_uncovered == set(outcome.success_criteria)
            if ai_unavailable_reason:
                # The LLM call itself failed (even after one retry with a
                # smaller evidence packet) -- this is an EVALUATOR failure,
                # not evidence the learner didn't do the work, and must read
                # and score differently from "the model looked and found
                # nothing" (the `else` branch below). See _get_llm_insights.
                evidence = (
                    f"AI evaluation could not be completed for this outcome ({ai_unavailable_reason}), "
                    "even after retrying with a smaller evidence packet. This reflects an evaluator "
                    "failure, not evidence that the learner did not complete this outcome."
                )
                verification_state = VerificationState.unverifiable
            elif all_uncovered:
                # No relevant evidence was selected for ANY of this
                # outcome's success criteria -- a selection gap, not
                # evidence the learner didn't complete it, and consistent
                # with why the model returned no judgment at all: it never
                # saw anything to judge.
                evidence = (
                    "No relevant evidence was selected for any of this outcome's success criteria, "
                    "so it could not be evaluated this attempt. This reflects a selection gap, not "
                    "evidence that the learner did not complete this outcome."
                )
                verification_state = VerificationState.unverifiable
            elif not events:
                evidence = "There was insufficient captured evidence to confidently evaluate this behavior."
                verification_state = None
            else:
                # Evidence was selected and the LLM call itself did not error
                # or return unparseable JSON overall, yet no judgment for
                # THIS outcome id came back even after the missing-judgment
                # follow-up (see _fill_missing_judgments). This is still an
                # evaluator gap, not observed evidence of failure -- must not
                # read as a confident fail (verification_state=None rendered
                # as "failed" in the UI/report). See _get_llm_insights.
                evidence = (
                    "The AI evaluator did not return a judgment for this outcome, even after a "
                    "follow-up request specifically asking for it. This reflects an evaluator gap, "
                    "not evidence that the learner did not complete this outcome."
                )
                verification_state = VerificationState.unverifiable
            return OutcomeResult(
                id=outcome.id,
                passed=False,
                score=0.0,
                max_score=outcome.weight,
                confidence=Confidence.unknown,
                evidence=evidence,
                verification_state=verification_state,
            )

        evidence = judgment.evidence or "Not observed in captured activity."
        return OutcomeResult(
            id=outcome.id,
            passed=False,
            score=0.0,
            max_score=outcome.weight,
            confidence=Confidence.unknown,
            evidence=evidence,
            verification_state=judgment.verification_state,
            feedback=judgment.feedback,
        )

    def _reconcile_narrative(
        self,
        exercise: Exercise,
        outcomes: List[OutcomeResult],
        summary: str,
        strengths: List[str],
        observed_approach: List[str],
    ) -> tuple:
        """Make the final, scored `outcomes` list authoritative over the
        LLM's free-text narrative -- see the module docstring above
        _significant_words for why this is needed. `strengths` and
        `observed_approach` are dropped entry-by-entry when an entry closely
        echoes a NOT-PASSED outcome's own wording (every entry in these two
        fields is, by the prompt's own contract, an affirmative "the learner
        did this" claim, so topic overlap with a failed outcome is enough --
        no separate negation check is needed the way it is for `summary`,
        which legitimately mixes positive and hedged/negative sentences).
        `summary` is replaced WHOLESALE with a deterministic, outcome-derived
        fallback only if a specific sentence both echoes a NOT-PASSED
        outcome AND contains a completion verb AND contains no negation --
        never edited in place, to avoid producing mangled prose.
        """
        not_passed = [o for o in outcomes if not o.passed]
        if not not_passed:
            return summary, strengths, observed_approach

        id_to_outcome = {o.id: o for o in exercise.get_all_outcomes()}
        not_passed_words = [
            _outcome_reference_words(id_to_outcome[o.id])
            for o in not_passed if o.id in id_to_outcome
        ]
        not_passed_words = [w for w in not_passed_words if w]
        if not not_passed_words:
            return summary, strengths, observed_approach

        def _echoes_a_failed_outcome(text: str) -> bool:
            words = _significant_words(text)
            return any(_mentions_outcome(words, ref) for ref in not_passed_words)

        clean_strengths = [s for s in strengths if not _echoes_a_failed_outcome(s)]
        clean_observed = [s for s in observed_approach if not _echoes_a_failed_outcome(s)]

        contradicts = False
        for sentence in _split_sentences(summary):
            words = _significant_words(sentence)
            if words & _NEGATION_CUES:
                continue
            if not (words & _COMPLETION_CUES):
                continue
            if any(_mentions_outcome(words, ref) for ref in not_passed_words):
                contradicts = True
                break

        final_summary = self._generate_fallback_summary(outcomes, id_to_outcome) if contradicts else summary
        return final_summary, clean_strengths, clean_observed

    @staticmethod
    def _generate_fallback_summary(
        outcomes: List[OutcomeResult], id_to_outcome: Dict[str, ExpectedOutcome]
    ) -> str:
        """Deterministic, code-generated summary built only from the scored
        outcomes -- used exclusively as a safety net when the LLM's own
        summary contradicts them, so it can never itself be inconsistent
        with the score.
        """
        def _label(o: OutcomeResult) -> str:
            outcome = id_to_outcome.get(o.id)
            return outcome.description if outcome is not None else o.id

        passed = [o for o in outcomes if o.passed]
        not_passed = [o for o in outcomes if not o.passed]
        total = round(sum(o.score for o in outcomes), 1)
        max_total = round(sum(o.max_score for o in outcomes), 1)

        parts = [f"This attempt scored {total}/{max_total}."]
        if passed:
            parts.append("Demonstrated this session: " + "; ".join(_label(o) for o in passed) + ".")
        if not_passed:
            parts.append("Not demonstrated this session: " + "; ".join(_label(o) for o in not_passed) + ".")
        return " ".join(parts)
