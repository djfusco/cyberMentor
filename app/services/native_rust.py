"""EvidenceProvider backed by the Rust capture helper
(native_capture/rust/target/debug/cyberalfred-capture).

The Rust helper is a true drop-in for the Swift ``mentor-capture`` (native_mac)
helper: same session-bound subprocess model and the SAME newline-delimited
JSON event schema (see native_capture/rust/src/events.rs, which mirrors
native_capture/Sources/MentorCapture/Models.swift exactly). It emits
``text_observed`` (Accessibility text, Vision OCR fallback), ``screen_change``,
``app_change``, ``window_change``, ``mouse_click``, ``scroll``, ``key_activity``
-- the full native_mac event set -- so the shared ``_translate_event`` /
``_synthesize_text`` in app.services.native_common translate Rust events
unchanged. The mentor app is text-first; the Rust binary produces the
on-device text itself (Accessibility + Vision OCR), so unlike the earlier
v1 Rust path no app-side OCR stage is needed.

Platform-specific pieces (SIGINT stop on macOS, explicit ``stop --session <id>``
on Windows; ``check`` permission parsing for Screen Recording / Accessibility /
Input Monitoring) live here. Everything common to native providers --
launching ``<exe> start --session <id> --output <dir>``, reading stdout,
time-windowed get_activity/search, persisted-events replay -- lives in
native_common.

The Rust helper is cross-platform (scap: ScreenCaptureKit on macOS,
Windows.Graphics.Capture on Windows), so a single binary and a single provider
are intended to eventually replace the platform-specific Swift and C# helpers.
"""
import asyncio
import logging
import signal
import sys
from typing import Dict, Optional

from app.config import get_settings
from app.services.native_common import (
    NativeEvidenceProvider,
    _PROCESS_EXIT_TIMEOUT_SECONDS,
    _READER_DRAIN_TIMEOUT_SECONDS,
    _SIGINT_TIMEOUT_SECONDS,
    _STOP_COMMAND_TIMEOUT_SECONDS,
    _TERMINATE_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class RustCaptureEvidenceProvider(NativeEvidenceProvider):
    """Native evidence provider backed by the Rust ``cyberalfred-capture``
    binary (native_mac parity).

    Identical in shape to ``NativeMacEvidenceProvider``: it reuses the shared
    ``NativeEvidenceProvider`` base unchanged (the Rust binary emits native_mac's
    event schema, so the base's ``_ingest_line`` -> ``_translate_event`` path
    works as-is), and only implements the platform-specific pieces that differ
    -- health/permission checks via the Rust ``check`` subcommand (Screen
    Recording + Accessibility + Input Monitoring on macOS) and the stop path
    (SIGINT on macOS, ``stop --session`` on Windows, since Win32 asyncio can't
    deliver SIGINT -- the same split as native_mac / native_windows).
    """

    def __init__(self, executable: Optional[str] = None, output_dir: Optional[str] = None):
        settings = get_settings()
        super().__init__(
            executable or settings.native_rust_capture_executable,
            output_dir or settings.native_capture_output,
            log_prefix="native_rust",
        )

    # -- Health / permissions (Rust `check`) --------------------------------

    async def health(self) -> bool:
        """True if the executable exists and `check` exits 0. Like native_mac,
        deliberately independent of permission status -- a missing permission
        (Screen Recording / Accessibility / Input Monitoring) is a
        degraded-but-running state, not a health failure (see
        permission_status()).
        """
        if not self.executable.exists():
            return False
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.executable), "check",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=5)
            return process.returncode == 0
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning("native_rust: health check failed: %s", exc)
            return False

    async def permission_status(self) -> Dict[str, bool]:
        """Parses `cyberalfred-capture check` output into per-permission
        booleans. On macOS the Rust helper reports Screen Recording,
        Accessibility, and Input Monitoring (matching native_mac -- it now does
        AX text extraction, Vision OCR, and input capture, so all three are
        needed). On Windows it reports functional capabilities. The
        line-parsing mirrors native_mac's / native_windows' approach: lowercase
        the line and check the permission prefix plus the "granted" substring.
        """
        if sys.platform == "darwin":
            status = {
                "screen_recording": False,
                "accessibility": False,
                "input_monitoring": False,
            }
        else:
            status = {
                "screen_capture": False,
                "ui_automation": False,
                "input_hooks": False,
                "active_window": False,
            }
        if not self.executable.exists():
            return status
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.executable), "check",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning("native_rust: permission check failed: %s", exc)
            return status

        for line in stdout.decode("utf-8", errors="replace").splitlines():
            lower = line.strip().lower()
            if lower.startswith("screen recording"):
                status["screen_recording"] = "granted" in lower
            elif lower.startswith("accessibility"):
                status["accessibility"] = "granted" in lower
            elif lower.startswith("input monitoring"):
                status["input_monitoring"] = "granted" in lower
            elif lower.startswith("screen capture"):
                status["screen_capture"] = "granted" in lower
            elif lower.startswith("ui automation"):
                status["ui_automation"] = "granted" in lower
            elif lower.startswith("input hooks"):
                status["input_hooks"] = "granted" in lower
            elif lower.startswith("active window"):
                status["active_window"] = "granted" in lower
        return status

    # -- Stop (platform-specific) -------------------------------------------

    async def stop_session(self, session_id: str) -> None:
        """Stop the Rust capture process. On macOS, SIGINT (like native_mac);
        on Windows, the explicit `stop --session <id>` subcommand (like
        native_windows, because Win32 asyncio can't deliver SIGINT). After the
        process exits and readers drain, the in-memory events are ready for
        evaluation -- no OCR-task drain is needed (the Rust binary produces its
        own text events).
        """
        cs = self._sessions.get(session_id)
        if cs is None or cs.process is None:
            return

        process = cs.process

        if sys.platform == "darwin":
            await self._stop_via_sigint(process, session_id)
        else:
            await self._stop_via_command(process, session_id)

        await self._drain_readers(cs)
        logger.info(
            "native_rust: stopped capture for session %s (%d events collected)",
            session_id, len(cs.events),
        )

    async def _stop_via_sigint(self, process, session_id: str) -> None:
        if process.returncode is None:
            try:
                process.send_signal(signal.SIGINT)
                await asyncio.wait_for(process.wait(), timeout=_SIGINT_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("native_rust: session %s did not exit after SIGINT, terminating", session_id)
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=_TERMINATE_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    logger.warning("native_rust: session %s did not terminate, killing", session_id)
                    process.kill()
                    await process.wait()
            except ProcessLookupError:
                pass

    async def _stop_via_command(self, process, session_id: str) -> None:
        # Ask the helper to shut itself down gracefully via its sentinel-file
        # stop signal (see native_capture/rust/src/stop.rs). Win32 asyncio
        # can't deliver SIGINT, so this is the supported stop path.
        if process.returncode is None and self.executable.exists():
            try:
                stop_proc = await asyncio.create_subprocess_exec(
                    str(self.executable), "stop", "--session", session_id,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(stop_proc.communicate(), timeout=_STOP_COMMAND_TIMEOUT_SECONDS)
            except (OSError, asyncio.TimeoutError) as exc:
                logger.warning("native_rust: stop command failed for session %s: %s", session_id, exc)

        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("native_rust: session %s did not exit after stop, terminating", session_id)
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=_TERMINATE_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    logger.warning("native_rust: session %s did not terminate, killing", session_id)
                    try:
                        process.kill()
                        await process.wait()
                    except ProcessLookupError:
                        pass
                except (ProcessLookupError, OSError):
                    pass
