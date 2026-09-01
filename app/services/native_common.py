"""Shared layer for native (per-session subprocess) evidence providers.

Both the macOS helper (``mentor-capture``, Swift -- see
native_capture/Sources/MentorCapture) and the Windows helper
(``mentor-capture.exe``, C#/.NET -- see native_capture/windows) emit the
*same* flat newline-delimited JSON event schema (one ``CaptureEvent`` per
line, snake_case keys; see ``Models.swift`` / ``CaptureEvent.cs``). This
module owns the parts of the provider that are identical for both platforms
-- translating a native event into the raw-dict shape
``app.services.evidence.EvidenceNormalizer`` already understands, the
per-session bookkeeping, stdout/stderr ingestion, and the time-windowed
``get_activity``/``search`` readers -- so each platform provider only
implements its own process lifecycle (how the helper is started and, in
particular, stopped) plus its health/permission checks.

Everything downstream (EvidenceNormalizer, EvidenceService, mentor,
evaluator, Ollama) is unchanged: native providers are just another
EvidenceProvider feeding the same normalizer.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import BASE_DIR
from app.services.evidence_provider import EvidenceProvider, EvidenceProviderError

logger = logging.getLogger(__name__)

# Lifecycle/diagnostic native event types that aren't learner activity and
# therefore never become evidence records.
_NON_EVIDENCE_TYPES = {"session_start", "session_stop", "error"}

_SIGINT_TIMEOUT_SECONDS = 5.0
_TERMINATE_TIMEOUT_SECONDS = 3.0
# Windows stops the helper with an explicit `stop` subcommand (it can't be
# delivered SIGINT); these govern that cross-process stop handshake.
_STOP_COMMAND_TIMEOUT_SECONDS = 6.0
_PROCESS_EXIT_TIMEOUT_SECONDS = 5.0
_READER_DRAIN_TIMEOUT_SECONDS = 3.0


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def _synthesize_text(event_type: str, raw: Dict[str, Any]) -> str:
    """Every native event type must map to a non-empty text description --
    EvidenceNormalizer drops any raw item with no text (mirroring how
    non-text "input" records already get a synthesized description via
    _describe_input_event in evidence.py).
    """
    if event_type == "text_observed":
        return (raw.get("text") or "").strip()
    if event_type == "app_change":
        return f"Switched to {raw.get('app_name') or 'an application'}"
    if event_type == "window_change":
        return f'Focused window: "{raw.get("window_title") or "unknown"}"'
    if event_type == "screen_change":
        diff = raw.get("screen_difference")
        if isinstance(diff, (int, float)):
            return f"Screen content changed (difference={diff:.2f})"
        return "Screen content changed"
    if event_type == "mouse_click":
        x, y = raw.get("mouse_x"), raw.get("mouse_y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return f"Clicked at ({x:.0f}, {y:.0f})"
        return "Clicked"
    if event_type == "scroll":
        count = raw.get("key_count")
        return f"Scrolled ({count} events)" if count else "Scrolled"
    if event_type == "key_activity":
        # Category + count only -- never raw characters (see key_category/
        # key_count in native_capture; no typed text ever reaches this app).
        category = raw.get("key_category") or "keys"
        count = raw.get("key_count") or 0
        return f"Keyboard activity: {category} x{count}"
    return ""


def _translate_event(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate one native_capture JSON event into a raw dict shaped like
    what EvidenceNormalizer already expects from a provider. Returns None
    for lifecycle events or anything that fails to yield useful text.
    """
    event_type = raw.get("type")
    timestamp = raw.get("timestamp")
    if not event_type or not timestamp or event_type in _NON_EVIDENCE_TYPES:
        return None

    text = _synthesize_text(event_type, raw)
    if not text:
        return None

    return {
        "type": event_type,
        "timestamp": timestamp,
        "app_name": raw.get("app_name"),
        "window_title": raw.get("window_title"),
        "text": text,
        # EvidenceNormalizer looks for "file_path"/"frame_name"; native calls
        # it "frame_path" -- remap the key, not the normalizer.
        "file_path": raw.get("frame_path"),
        # No browser URL extraction in this version (see native_mac README
        # notes) -- left absent so EvidenceEvent.browser_url stays None.
    }


class NativeCaptureSession:
    """Per-session capture bookkeeping shared by all native providers."""
    def __init__(self, session_id: str, output_dir: Path):
        self.session_id = session_id
        self.output_dir = output_dir
        self.process: Optional[asyncio.subprocess.Process] = None
        self.events: List[Dict[str, Any]] = []
        self.stdout_task: Optional[asyncio.Task] = None
        self.stderr_task: Optional[asyncio.Task] = None
        self.error: Optional[str] = None


class NativeEvidenceProvider(EvidenceProvider):
    """Common base for native (session-bound subprocess) evidence providers.

    Subclasses implement ``health``, ``permission_status``, and
    ``stop_session`` -- the parts that differ by platform (macOS stops the
    helper with SIGINT; Windows stops it with an explicit ``stop`` command
    because Win32 asyncio subprocesses can't deliver SIGINT). Everything
    else -- launching ``<exe> start --session <id> --output <dir>``, reading
    the helper's newline-delimited JSON stdout, and serving the time-windowed
    activity/search queries -- is identical and lives here.
    """

    def __init__(self, executable: str, output_dir: str, log_prefix: str = "native"):
        self.executable = _resolve_path(executable)
        self.output_dir = _resolve_path(output_dir)
        self._sessions: Dict[str, NativeCaptureSession] = {}
        # This app is single-user/local and, in practice, has at most one
        # active exercise session at a time. get_activity()/search() take no
        # session_id (that's the existing EvidenceProvider contract, shared
        # with any continuous-timeline provider), so we track whichever
        # session was started most recently as "the" active one rather than
        # building a multi-session query layer.
        self._active_session_id: Optional[str] = None
        self._log_prefix = log_prefix

    # -- Session lifecycle (shared) ------------------------------------------

    async def start_session(self, capture_run_id: str) -> None:
        if capture_run_id in self._sessions:
            return  # already capturing this run

        # The capture directory is named after the immutable capture_run_id so
        # two sessions that share a numeric DB id (e.g. after a DB reset) can
        # never write to the same directory.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cs_dir = self.output_dir / capture_run_id
        if cs_dir.exists():
            raise EvidenceProviderError(
                f"Capture directory already exists: {cs_dir}. "
                f"Each session must have a unique capture_run_id."
            )
        cs_dir.mkdir(parents=True)

        cs = NativeCaptureSession(capture_run_id, cs_dir)
        self._sessions[capture_run_id] = cs
        self._active_session_id = capture_run_id

        if not self.executable.exists():
            cs.error = f"Native capture executable not found at {self.executable}"
            logger.error("%s: %s", self._log_prefix, cs.error)
            return

        try:
            process = await asyncio.create_subprocess_exec(
                str(self.executable), "start",
                "--session", capture_run_id,
                "--output", str(self.output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            cs.error = f"Could not launch native capture: {exc}"
            logger.error("%s: %s", self._log_prefix, cs.error)
            return

        cs.process = process
        cs.stdout_task = asyncio.create_task(self._read_stdout(cs))
        cs.stderr_task = asyncio.create_task(self._read_stderr(cs))
        logger.info("%s: started capture for run %s (pid=%s)", self._log_prefix, capture_run_id, process.pid)

    async def _drain_readers(self, cs: NativeCaptureSession) -> None:
        # readline() returns b"" once the process exits and closes its pipes,
        # so the reader tasks finish on their own shortly after -- just make
        # sure any remaining buffered stdout has actually been consumed
        # before returning, so evaluation sees the final events.
        for task in (cs.stdout_task, cs.stderr_task):
            if task is None:
                continue
            try:
                await asyncio.wait_for(task, timeout=_READER_DRAIN_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                task.cancel()
            except asyncio.CancelledError:
                pass

    async def stop_session(self, capture_run_id: str) -> None:
        # Platform-specific (macOS: SIGINT, Windows: stop command). Subclasses
        # override this and call _drain_readers(cs) once the process has exited.
        cs = self._sessions.get(capture_run_id)
        if cs is None or cs.process is None:
            return
        await self._drain_readers(cs)

    # -- Stdout/stderr ingestion (shared) ------------------------------------

    async def _read_stdout(self, cs: NativeCaptureSession) -> None:
        assert cs.process is not None and cs.process.stdout is not None
        try:
            while True:
                line = await cs.process.stdout.readline()
                if not line:
                    break
                self._ingest_line(cs, line)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a reader crash must not kill the session
            logger.warning("%s: stdout reader error for session %s: %s", self._log_prefix, cs.session_id, exc)

    def _ingest_line(self, cs: NativeCaptureSession, line: bytes) -> None:
        try:
            text = line.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            logger.warning("%s: skipping non-UTF8 stdout line: %s", self._log_prefix, exc)
            return
        if not text:
            return
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("%s: skipping malformed JSON line (%s): %.200s", self._log_prefix, exc, text)
            return
        if not isinstance(raw, dict):
            return
        translated = _translate_event(raw)
        if translated is not None:
            cs.events.append(translated)

    async def _read_stderr(self, cs: NativeCaptureSession) -> None:
        assert cs.process is not None and cs.process.stderr is not None
        try:
            while True:
                line = await cs.process.stderr.readline()
                if not line:
                    break
                message = line.decode("utf-8", errors="replace").rstrip()
                if message:
                    # Human-readable diagnostics only -- never learner evidence.
                    logger.info("mentor-capture[%s]: %s", cs.session_id, message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: stderr reader error for session %s: %s", self._log_prefix, cs.session_id, exc)

    # -- EvidenceProvider interface (shared) ---------------------------------

    async def get_activity(
        self, start_time: datetime, end_time: datetime, limit: int = 200
    ) -> List[Dict[str, Any]]:
        cs = self._sessions.get(self._active_session_id) if self._active_session_id else None
        if cs is None:
            return []
        items = [e for e in cs.events if self._within_window(e, start_time, end_time)]
        return items[-limit:] if limit else items

    async def search(
        self,
        query: str = "",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        content_type: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        cs = self._sessions.get(self._active_session_id) if self._active_session_id else None
        if cs is None:
            return []
        items = cs.events
        if start_time or end_time:
            window_start = start_time or datetime.min.replace(tzinfo=timezone.utc)
            window_end = end_time or datetime.now(timezone.utc)
            items = [e for e in items if self._within_window(e, window_start, window_end)]
        if content_type:
            items = [e for e in items if e.get("type") == content_type]
        if query:
            items = [e for e in items if query.lower() in (e.get("text") or "").lower()]
        return items[offset : offset + limit] if limit else items[offset:]

    @staticmethod
    def _within_window(item: Dict[str, Any], start_time: datetime, end_time: datetime) -> bool:
        raw_ts = item.get("timestamp")
        if not raw_ts:
            return True
        try:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        except ValueError:
            return True
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        return start_time <= ts <= end_time

    # -- Persisted evidence (historical queries after restart) ---------------

    async def load_persisted_events(self, capture_run_id: str) -> List[Dict[str, Any]]:
        """Read a completed session's events.jsonl back from disk and translate
        each line into the same raw-dict shape EvidenceNormalizer expects --
        reusing the exact same _translate_event the live stdout path uses, so
        persisted evidence normalizes identically to what the evaluator saw at
        finish time.

        capture_run_id is the immutable UUID stored on ExerciseSession. For
        sessions created before this field was introduced, the caller passes
        str(session.id) and the file is read from the legacy numeric directory.

        This is a read-only historical path; it does not touch capture
        (writing) behavior or the in-memory get_activity/search path used
        during an active session. Returns [] if the file is absent or unreadable
        so callers can fall back to the live provider.
        """
        path = self.output_dir / capture_run_id / "events.jsonl"
        if not path.exists():
            return []
        items: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        raw = json.loads(text)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "%s: skipping malformed JSON line in %s: %s",
                            self._log_prefix, path, exc,
                        )
                        continue
                    if not isinstance(raw, dict):
                        continue
                    translated = _translate_event(raw)
                    if translated is not None:
                        items.append(translated)
        except OSError as exc:
            logger.warning(
                "%s: could not read persisted events %s: %s", self._log_prefix, path, exc
            )
            return []
        return items

    # -- Platform-specific (override in subclasses) --------------------------

    async def permission_status(self) -> Dict[str, bool]:
        raise NotImplementedError
