"""Constructs and caches service instances used by routes.

Kept centralized so routes never instantiate capture/Ollama clients
directly.
"""
from functools import lru_cache

from app.config import get_settings
from app.services.evaluator import EvaluatorService
from app.services.evidence import EvidenceService
from app.services.evidence_provider import EvidenceProvider
from app.services.exercises import ExerciseService
from app.services.mentor import MentorService
from app.services.native_mac import NativeMacEvidenceProvider
from app.services.native_rust import RustCaptureEvidenceProvider
from app.services.native_windows import NativeWindowsEvidenceProvider
from app.services.ollama import OllamaService

# ExerciseAuthorService, ReferenceService, and SessionQueryService are
# imported lazily inside their factory functions below (not at module
# level) so that importing app.dependencies -- required by every route,
# including the student-facing exercises/sessions/mentor routes -- never
# depends on the instructor-only authoring/reference/session-query service
# modules being present. This matters for the trimmed beta export (see
# scripts/beta_manifest.txt), which does not ship those modules. No change
# in behavior when all modules are present: each service is still
# constructed once and cached via @lru_cache exactly as before.


@lru_cache
def get_exercise_service() -> ExerciseService:
    return ExerciseService()


def create_evidence_provider() -> EvidenceProvider:
    """Picks the EvidenceProvider implementation from EVIDENCE_PROVIDER.

    Everything downstream (EvidenceService, mentor, evaluator) depends only
    on the EvidenceProvider interface, so switching providers is purely a
    configuration change -- see README "Evidence Providers".
    """
    settings = get_settings()
    provider = settings.evidence_provider
    if provider == "native_mac":
        return NativeMacEvidenceProvider()
    if provider == "native_windows":
        return NativeWindowsEvidenceProvider()
    if provider == "rust":
        return RustCaptureEvidenceProvider()
    raise RuntimeError(
        f"Unknown EVIDENCE_PROVIDER '{provider}' -- expected 'native_mac', 'native_windows', or 'rust'"
    )


@lru_cache
def get_evidence_provider() -> EvidenceProvider:
    return create_evidence_provider()


@lru_cache
def get_evidence_service() -> EvidenceService:
    return EvidenceService(get_evidence_provider())


@lru_cache
def get_ollama_service() -> OllamaService:
    return OllamaService()


@lru_cache
def get_mentor_chat_backend():
    """The LLM backend for mentor chat, chosen by MENTOR_CHAT_PROVIDER --
    local Ollama (default) or an opt-in frontier API key (see
    app/services/frontier_chat.py). This is the ONLY AI-backed feature
    affected by that setting; every other service below is unconditionally
    Ollama-only. Exposed as its own function (not just an attribute on
    MentorService) so status reporting (app/services/chat_status.py) can
    reuse it without constructing a second instance.
    """
    settings = get_settings()
    if settings.mentor_chat_provider == "frontier":
        from app.services.frontier_chat import FrontierChatService

        return FrontierChatService()
    return get_ollama_service()


@lru_cache
def get_mentor_service() -> MentorService:
    return MentorService(get_evidence_service(), get_mentor_chat_backend())


@lru_cache
def get_evaluator_service() -> EvaluatorService:
    return EvaluatorService(get_ollama_service())


@lru_cache
def get_reference_service():
    from app.services.references import ReferenceService

    return ReferenceService(get_ollama_service())


@lru_cache
def get_session_query_service():
    from app.services.session_query import SessionQueryService

    return SessionQueryService(get_evidence_service(), get_ollama_service())


@lru_cache
def get_exercise_author_service():
    from app.services.exercise_author import ExerciseAuthorService

    settings = get_settings()
    authoring_ollama = OllamaService(timeout=settings.authoring_timeout_seconds)
    return ExerciseAuthorService(authoring_ollama, get_exercise_service(), get_reference_service())
