"""Diagnostic traceability for capture sessions.

Writes a manifest.json into each session's capture directory recording:
  - database session id and capture_run_id (the immutable link between the
    DB row and its evidence directory)
  - exercise, timing, evidence provider
  - all available frame paths and timestamps
  - per-model-call snapshots: selected keyframes with full provenance,
    routing decision, whether vision was used, and timestamps
  - per-mentor-question evidence snapshot: what the model saw, whether the
    most recent interaction had a post-event frame, etc.

When MENTOR_DIAGNOSTICS=true also writes per-call prompt payloads to
capture_sessions/<capture_run_id>/diags/ for debugging evidence selection.
The diagnostic files contain the text prompt and image metadata only --
never base64 screenshot content or API secrets.

Privacy guarantee: this module never stores raw keystrokes, terminal
input/output, browser DOM data, shell history, or API credentials. It
records only the application-generated evidence payload already sent to
the local Ollama model.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel used when a timestamp is not available.
_MISSING = "unknown"


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _frame_record(sf, consumer: str) -> Dict[str, Any]:
    """Convert a SelectedFrame into a JSON-serialisable dict for the manifest.

    Includes all provenance fields required by the traceability spec.
    Never includes base64 image data.
    """
    anchor_ts = getattr(sf, "anchor_timestamp", None)
    return {
        "path": sf.path,
        "timestamp": _iso(sf.timestamp) if hasattr(sf.timestamp, "isoformat") else str(sf.timestamp),
        "application": sf.application,
        "window_title": sf.window_title,
        "anchor_event_type": sf.trigger_type if sf.role != "spread" else None,
        "anchor_timestamp": _iso(anchor_ts) if isinstance(anchor_ts, datetime) else None,
        "role": sf.role,
        "consumer": consumer,
        # Populated only when selection was outcome-relevance-aware (see
        # app.services.keyframes.select_keyframes(..., outcomes=...)) --
        # 0.0 / "" for the generic mentor-chat/session-query path, which
        # never sets these. Answers "why was this frame kept" for diagnostics
        # (requirement: record which frames were selected and why).
        "relevance_score": getattr(sf, "relevance_score", 0.0),
        "matched_evidence": getattr(sf, "matched_evidence", ""),
    }


class ManifestWriter:
    """Reads, updates, and writes the per-capture manifest.json.

    Each public method reads the existing manifest (or bootstraps an empty
    one), applies its update, and writes atomically via a .tmp rename so a
    crash between calls leaves the last complete state intact.

    Multiple instances pointing at the same capture_dir are safe for
    sequential (not concurrent) writes -- the app is single-user/local
    with one active session at a time.
    """

    def __init__(
        self,
        capture_dir: Path,
        db_session_id: int,
        capture_run_id: str,
        exercise_id: str,
        provider_name: str,
        diagnostics_enabled: bool = False,
    ) -> None:
        self._capture_dir = capture_dir
        self._path = capture_dir / "manifest.json"
        self._diag_dir = capture_dir / "diags"
        self._diag = diagnostics_enabled
        self._seed: Dict[str, Any] = {
            "db_session_id": db_session_id,
            "capture_run_id": capture_run_id,
            "exercise_id": exercise_id,
            "evidence_provider": provider_name,
            "started_at": None,
            "ended_at": None,
            "raw_event_count": None,
            "normalized_event_count": None,
            "available_frames": [],
            "mentor_questions": [],
            "final_evaluation": None,
        }

    # ------------------------------------------------------------------
    # Session-level metadata (called from finish_session route)
    # ------------------------------------------------------------------

    def set_session_metadata(
        self,
        started_at: Optional[datetime],
        ended_at: Optional[datetime],
        raw_event_count: int,
        normalized_event_count: int,
        available_frames: List[Dict[str, Any]],
    ) -> None:
        data = self._load()
        data["started_at"] = _iso(started_at)
        data["ended_at"] = _iso(ended_at)
        data["raw_event_count"] = raw_event_count
        data["normalized_event_count"] = normalized_event_count
        data["available_frames"] = available_frames
        self._save(data)

    # ------------------------------------------------------------------
    # Per-model-call snapshots
    # ------------------------------------------------------------------

    def record_mentor_question(
        self,
        question: str,
        question_timestamp: str,
        model_request_timestamp: str,
        latest_event_at_read: Optional[str],
        latest_event_after_processing: Optional[str],
        most_recent_interaction: Optional[Dict[str, Any]],
        later_frame_from_same_app_exists: bool,
        selected_frames: list,  # List[SelectedFrame]
        images_attached: bool,
        model: Optional[str],
        routing_reason: str,
        use_vision: bool,
    ) -> None:
        snapshot = {
            "question": question,
            "question_timestamp": question_timestamp,
            "model_request_timestamp": model_request_timestamp,
            "prompt_type": "mentor_chat",
            "latest_event_at_read": latest_event_at_read,
            "latest_event_after_processing": latest_event_after_processing,
            "most_recent_interaction": most_recent_interaction,
            "later_frame_from_same_app_exists": later_frame_from_same_app_exists,
            "selected_frames": [_frame_record(sf, "mentor_chat") for sf in selected_frames],
            "images_attached": images_attached,
            "model": model,
            "routing_reason": routing_reason,
            "use_vision": use_vision,
        }
        data = self._load()
        data.setdefault("mentor_questions", []).append(snapshot)
        self._save(data)

    def record_final_evaluation(
        self,
        model_request_timestamp: str,
        model: str,
        routing_reason: str,
        event_count: int,
        normalized_event_count: int,
        available_frames: List[Dict[str, Any]],
        selected_frames: list,  # List[SelectedFrame]
        images_attached: bool,
        use_vision: bool,
        retry_used: bool = False,
        ai_unavailable: Optional[str] = None,
    ) -> None:
        snapshot = {
            "model_request_timestamp": model_request_timestamp,
            "model": model,
            "routing_reason": routing_reason,
            "prompt_type": "final_evaluation",
            "event_count": event_count,
            "normalized_event_count": normalized_event_count,
            "available_frames": available_frames,
            "selected_frames": [_frame_record(sf, "final_evaluation") for sf in selected_frames],
            "images_attached": images_attached,
            "use_vision": use_vision,
            # Set when the first (larger) request failed and this snapshot
            # reflects the smaller retry payload instead -- see
            # EvaluatorService._get_llm_insights. ai_unavailable is the
            # reason the retry ALSO failed (None on success), so a human
            # reading the manifest can immediately tell "unscored due to an
            # evaluator failure" apart from a normal successful evaluation.
            "retry_used": retry_used,
            "ai_unavailable": ai_unavailable,
        }
        data = self._load()
        data["final_evaluation"] = snapshot
        self._save(data)

    # ------------------------------------------------------------------
    # Diagnostic prompt payload (opt-in via MENTOR_DIAGNOSTICS=true)
    # ------------------------------------------------------------------

    def save_diagnostic_payload(
        self,
        consumer: str,
        system_prompt: str,
        user_prompt: str,
        selected_frames: list,  # List[SelectedFrame] — metadata only, no base64
    ) -> None:
        if not self._diag:
            return
        try:
            self._diag_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            image_metadata = [
                {
                    "path": sf.path,
                    "timestamp": str(sf.timestamp),
                    "application": sf.application,
                    "window_title": sf.window_title,
                    "role": sf.role,
                }
                for sf in selected_frames
            ]
            payload = {
                "consumer": consumer,
                "recorded_at": ts,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_metadata": image_metadata,
                "image_count": len(image_metadata),
            }
            path = self._diag_dir / f"{consumer}_{ts}.json"
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            logger.warning("capture_manifest: could not write diagnostic payload: %s", exc)

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("capture_manifest: could not read %s: %s; starting fresh", self._path, exc)
        return dict(self._seed)

    def _save(self, data: Dict[str, Any]) -> None:
        try:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as exc:
            logger.warning("capture_manifest: could not write %s: %s", self._path, exc)
