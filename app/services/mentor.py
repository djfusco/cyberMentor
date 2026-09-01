"""Mentor chat: answers student questions grounded in observed evidence.

The mentor is a teacher, not an autonomous agent -- it never executes
commands on the student's behalf.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from sqlmodel import Session, select

from app.config import get_settings
from app.models.exercise import DifficultyLevel, EnvironmentType, Exercise
from app.models.session import ExerciseSession, MentorMessage, MentorMessageRole
from app.services.capture_manifest import ManifestWriter
from app.services.evidence import EvidenceEvent, EvidenceService
from app.services.evidence_provider import EvidenceProviderError
from app.services.evaluator import VISION_NUM_CTX, MAX_VISION_FRAMES, _encode_images
from app.services.keyframes import format_frame_captions, select_keyframes
from app.services.ollama import OllamaError, OllamaService
from app.services.prompts import build_mentor_system_prompt, build_mentor_user_prompt
from app.services.token_budget import fit_to_budget
from app.services.verifier import run_verification

logger = logging.getLogger(__name__)

# Maximum number of recent evidence events considered for the mentor
# prompt before context budgeting (app/services/token_budget.py) may
# shrink this further; floor below which shrinking stops.
MENTOR_MAX_EVENTS = 40
MENTOR_MIN_EVENTS = 5


def _effective_difficulty(exercise: Exercise, session: ExerciseSession) -> str:
    """instructor difficulty, unless the exercise is Open -- in which case
    the student's own per-session choice applies instead (see
    ExerciseSession.student_difficulty). Falls back to intermediate if an
    Open exercise somehow has no recorded student choice."""
    if exercise.difficulty != DifficultyLevel.open:
        return exercise.difficulty.value
    chosen = session.student_difficulty or DifficultyLevel.intermediate
    return chosen.value if isinstance(chosen, DifficultyLevel) else str(chosen)


@dataclass
class _MentorModelRouting:
    # model is None on the text-only path so whichever chat backend is active
    # (OllamaService OR the opt-in FrontierChatService) uses its own default
    # model. It is only set on the vision path, which is exclusively Ollama.
    model: Optional[str]
    use_vision: bool
    reason: str


def _route_mentor_model(
    exercise: Exercise, events: List[EvidenceEvent], settings
) -> _MentorModelRouting:
    """Decide whether the in-exercise mentor chat attaches screenshots.

    Mirrors the evaluator's _select_model: terminal exercises never need
    vision (text/AX evidence is already authoritative for them); every other
    exercise type routes to the vision model whenever one is configured AND
    screenshots were captured. Screenshots are the primary source of truth
    for on-screen actions in a GUI/web/SIEM app -- the text timeline alone
    (window titles, click coordinates, key-count summaries) cannot confirm
    what a learner actually did, which is why the mentor previously told
    students it "wasn't sure" they did the steps.

    One difference from the evaluator: mentor chat can run on the opt-in
    frontier (e.g. OpenAI) backend, which does NOT accept image attachments
    (FrontierChatService.chat raises if given images). When a non-Ollama
    backend is active, mentor chat stays text-only even for visual exercises
    -- screenshots are still used at Finish-time evaluation, which always
    runs on local Ollama regardless of MENTOR_CHAT_PROVIDER.
    """
    if exercise.environment.type == EnvironmentType.terminal:
        return _MentorModelRouting(
            None, False, "terminal exercise: text/AX evidence is authoritative"
        )
    if settings.mentor_chat_provider != "ollama":
        return _MentorModelRouting(
            None, False, "non-Ollama mentor chat backend; image attachments unsupported"
        )
    if not settings.ollama_vision_model:
        return _MentorModelRouting(None, False, "vision model not configured")
    if not any(e.frame_path for e in events):
        return _MentorModelRouting(
            None, False, "no screenshots captured for this session"
        )
    return _MentorModelRouting(
        settings.ollama_vision_model,
        True,
        "visual exercise with screenshots: using screenshots as primary evidence",
    )


class MentorService:
    def __init__(self, evidence_service: EvidenceService, ollama: OllamaService):
        self.evidence_service = evidence_service
        self.ollama = ollama

    async def ask(
        self,
        db: Session,
        session: ExerciseSession,
        exercise: Exercise,
        question: str,
        help_level: Optional[int] = None,
        manifest: Optional["ManifestWriter"] = None,
    ) -> str:
        question_timestamp = datetime.now(timezone.utc).isoformat()

        level = exercise.mentor.default_help_level if help_level is None else help_level
        level = max(0, min(5, level))
        if level == 5 and not exercise.mentor.reveal_answer:
            level = 4

        evidence_error: Optional[str] = None
        try:
            events = await self.evidence_service.get_session_activity(session.started_at)
        except EvidenceProviderError as exc:
            logger.warning("Mentor could not retrieve evidence: %s", exc)
            events = []
            evidence_error = str(exc)

        # Capture latest event timestamp right after first read, before any filtering.
        latest_at_read = events[-1].timestamp.isoformat() if events else None

        if exercise.environment.type == EnvironmentType.terminal:
            events = EvidenceService.filter_terminal_events(events)

        latest_after_processing = events[-1].timestamp.isoformat() if events else None

        verification = run_verification(exercise)

        effective_difficulty = _effective_difficulty(exercise, session)
        system_prompt = build_mentor_system_prompt(exercise, level, effective_difficulty)

        settings = get_settings()
        routing = _route_mentor_model(exercise, events, settings)
        use_vision, model, reason = routing.use_vision, routing.model, routing.reason

        images: List[str] = []
        frame_captions: Optional[str] = None
        selected_frames: List = []
        if use_vision:
            selected_frames = select_keyframes(events, MAX_VISION_FRAMES)
            images = _encode_images([sf.path for sf in selected_frames], str(session.id))
            if not images:
                use_vision, model = False, None
                reason = "screenshots could not be read; falling back to text-only"
                selected_frames = []
            else:
                frame_captions = format_frame_captions(selected_frames)

        context_size = VISION_NUM_CTX if use_vision else settings.ollama_model_num_ctx

        def render(count: int):
            prompt, truncated = build_mentor_user_prompt(
                exercise, events, verification, question, evidence_error,
                has_images=use_vision, frame_captions=frame_captions, max_events=count,
            )
            return system_prompt, prompt, truncated

        system_prompt, user_prompt, _selected, _tokens, _truncated = fit_to_budget(
            render, MENTOR_MAX_EVENTS, context_size,
            image_count=len(images) if use_vision else 0, min_count=MENTOR_MIN_EVENTS,
        )

        logger.info(
            "Mentor routing: exercise_type=%s model=%s reason=%s "
            "raw_events=%d screenshots=%d",
            exercise.environment.type.value, model, reason, len(events), len(images),
        )

        # Record evidence snapshot in manifest before sending to the model.
        model_request_timestamp = datetime.now(timezone.utc).isoformat()
        if manifest is not None:
            _INTERACTION_TYPES = frozenset({"mouse_click", "scroll", "app_change", "window_change"})
            last_interaction = next(
                (e for e in reversed(events) if e.type in _INTERACTION_TYPES), None
            )
            most_recent_interaction: Optional[dict] = None
            later_frame_exists = False
            if last_interaction is not None:
                most_recent_interaction = {
                    "type": last_interaction.type,
                    "application": last_interaction.application,
                    "timestamp": last_interaction.timestamp.isoformat(),
                }
                later_frame_exists = any(
                    e.frame_path
                    and e.timestamp > last_interaction.timestamp
                    and e.application == last_interaction.application
                    for e in events
                )
            try:
                manifest.record_mentor_question(
                    question=question,
                    question_timestamp=question_timestamp,
                    model_request_timestamp=model_request_timestamp,
                    latest_event_at_read=latest_at_read,
                    latest_event_after_processing=latest_after_processing,
                    most_recent_interaction=most_recent_interaction,
                    later_frame_from_same_app_exists=later_frame_exists,
                    selected_frames=selected_frames,
                    images_attached=bool(images),
                    model=model,
                    routing_reason=reason,
                    use_vision=use_vision,
                )
                if use_vision:
                    manifest.save_diagnostic_payload(
                        consumer="mentor_chat",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        selected_frames=selected_frames,
                    )
            except Exception as exc:  # noqa: BLE001 -- manifest errors must never fail a request
                logger.warning("Mentor manifest write failed: %s", exc)

        db.add(MentorMessage(session_id=session.id, role=MentorMessageRole.student, message=question))
        db.commit()

        try:
            answer = await self.ollama.chat(
                system_prompt, user_prompt,
                images=images if use_vision else None,
                model=model,
                num_ctx=context_size,
            )
        except OllamaError as exc:
            answer = (
                f"I can't reach the AI model right now ({exc}). "
                "Please check that Ollama is running and try again."
            )

        # Deterministically prepend a warning rather than relying on the LLM to
        # always mention the retrieval failure -- this must never be silent.
        if evidence_error:
            answer = f"[Screen activity could not be retrieved: {evidence_error}]\n\n{answer}"

        db.add(MentorMessage(session_id=session.id, role=MentorMessageRole.mentor, message=answer))
        db.commit()

        return answer

    @staticmethod
    def get_history(db: Session, session_id: int, limit: int = 50) -> List[MentorMessage]:
        statement = (
            select(MentorMessage)
            .where(MentorMessage.session_id == session_id)
            .order_by(MentorMessage.created_at)
        )
        results = list(db.exec(statement).all())
        return results[-limit:]
