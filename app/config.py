"""Central application configuration, loaded from environment variables.

All Ollama endpoints and the Ollama model are configurable here so they
are never hardcoded elsewhere in the application.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8080

    database_url: str = f"sqlite:///{BASE_DIR / 'mentor.db'}"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-coder-next:latest"
    # Multimodal model used for vision-assisted evaluation at Finish time AND
    # for in-exercise mentor chat on visual (non-terminal) exercises when
    # screenshots were captured (see app/services/evaluator.py and
    # app/services/mentor.py). Must be pulled separately (e.g.
    # `ollama pull llava`). Set to empty string to disable vision-assisted
    # evaluation and mentor-chat screenshots entirely.
    ollama_vision_model: str = "llava:latest"
    # Assumed context window (tokens) of the default text model, used to
    # budget/estimate prompt size before a request (see
    # app/services/token_budget.py) and passed as num_ctx on text-path
    # calls so the estimate matches what is actually requested. Adjust if
    # your local qwen3-coder-next build uses a different context size --
    # this is not a substitute for trimming oversized evidence.
    ollama_model_num_ctx: int = 32768

    ollama_timeout_seconds: float = 60.0
    # Per-call timeout for SessionQueryService LLM calls only (the
    # "Ask About This Session" feature). Separate from
    # ollama_timeout_seconds so a slower, evidence-grounded session query can
    # run longer without also raising the mentor/evaluator timeouts. Override
    # with SESSION_QUERY_TIMEOUT_SECONDS.
    session_query_timeout_seconds: float = 180.0

    # Authoring-only frontier research assistant (see app/services/research.py).
    # This is a deliberate, narrow exception to the local-only design -- it
    # is never used for student session data, only instructor authoring
    # research.
    frontier_provider: str = "openai"
    frontier_api_key: Optional[str] = None
    frontier_model: str = "gpt-4o"

    # Which backend answers in-exercise mentor chat questions: "ollama"
    # (default, fully local) or "frontier" (opt-in -- uses FRONTIER_PROVIDER/
    # FRONTIER_API_KEY/FRONTIER_MODEL above, the same settings the authoring
    # research feature uses). Intended for people who can't run Ollama
    # locally and are comfortable sending exercise/evidence context to a
    # third-party model of their own choosing. Only mentor chat is affected;
    # evaluation, session Q&A, authoring, and mentor review stay Ollama-only
    # regardless of this setting. See app/services/frontier_chat.py.
    mentor_chat_provider: str = "ollama"

    exercises_dir: str = str(BASE_DIR / "exercises")
    reference_library_dir: str = str(BASE_DIR / "reference_library")

    # Which EvidenceProvider backs the app: "native_mac" (default on macOS,
    # no external service required), "native_windows" (Windows, same model
    # backed by mentor-capture.exe), or "rust" (cross-platform Rust capture
    # helper; intended to eventually replace the platform-specific native
    # helpers). See app/dependencies.py::create_evidence_provider.
    evidence_provider: str = "native_mac"
    # Relative paths are resolved against the project root (BASE_DIR) by the
    # native providers, not here, so a plain relative value in .env (e.g.
    # "./native_capture/.build/debug/mentor-capture") works regardless of the
    # working directory the app is started from.
    native_capture_executable: str = str(BASE_DIR / "native_capture" / ".build" / "debug" / "mentor-capture")
    # Rust capture helper (native_capture/rust), built with `cargo build`.
    # Cross-platform (scap: ScreenCaptureKit on macOS, Windows.Graphics.Capture
    # on Windows). Only used when evidence_provider == "rust". The Rust helper
    # emits screenshot events (no OCR); the provider OCRs the saved frames
    # app-side (see app/services/native_rust.py + app/services/ocr.py).
    native_rust_capture_executable: str = str(BASE_DIR / "native_capture" / "rust" / "target" / "debug" / "cyberalfred-capture")
    # mentor-capture.exe, built/published from native_capture/windows. Default
    # assumes `dotnet publish -o bin` from that directory; override with an
    # absolute path if you publish elsewhere. Only used when
    # evidence_provider == "native_windows".
    native_windows_capture_executable: str = str(BASE_DIR / "native_capture" / "windows" / "bin" / "mentor-capture.exe")
    # Shared by both native providers: capture_sessions/<session_id>/{events.jsonl,frames/}
    native_capture_output: str = str(BASE_DIR / "capture_sessions")


@lru_cache
def get_settings() -> Settings:
    return Settings()
