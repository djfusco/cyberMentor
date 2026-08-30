"""EvidenceProvider backed by the native Windows capture helper
(native_capture/windows mentor-capture.exe).

Mirror of NativeMacEvidenceProvider for Windows: the same session-bound
subprocess model and the same newline-delimited JSON event schema (emitted by
mentor-capture.exe -- see native_capture/windows/Program.cs), translated by
the shared NativeEvidenceProvider base in app.services.native_common.

The only platform-specific differences from the macOS provider:
  * Stop is an explicit ``mentor-capture.exe stop --session <id>`` command
    rather than SIGINT -- Win32 asyncio subprocesses cannot deliver SIGINT,
    so the helper exposes a named stop event instead (see StopSignal.cs).
  * Health/permission checks run ``mentor-capture.exe status`` and report
    Windows-relevant capabilities (screen capture, UI Automation, foreground
    window) instead of the macOS TCC permissions.

Everything downstream (EvidenceNormalizer, EvidenceService, mentor,
evaluator, Ollama) is unchanged.
"""
import asyncio
import logging
from typing import Dict, Optional

from app.config import get_settings
from app.services.native_common import (
    NativeEvidenceProvider,
    _PROCESS_EXIT_TIMEOUT_SECONDS,
    _READER_DRAIN_TIMEOUT_SECONDS,
    _STOP_COMMAND_TIMEOUT_SECONDS,
    _TERMINATE_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class NativeWindowsEvidenceProvider(NativeEvidenceProvider):
    def __init__(self, executable: Optional[str] = None, output_dir: Optional[str] = None):
        settings = get_settings()
        super().__init__(
            executable or settings.native_windows_capture_executable,
            output_dir or settings.native_capture_output,
            log_prefix="native_windows",
        )

    # -- Health / permissions (Windows) --------------------------------------

    async def health(self) -> bool:
        """True if mentor-capture.exe exists and `status` runs successfully.

        Independent of capability status -- a missing capability (e.g. UI
        Automation unavailable) is a degraded-but-running state, not a health
        failure (see permission_status()).
        """
        if not self.executable.exists():
            return False
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.executable), "status",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=5)
            return process.returncode == 0
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning("native_windows: health check failed: %s", exc)
            return False

    async def permission_status(self) -> Dict[str, bool]:
        """Parses `mentor-capture.exe status` output into per-capability booleans."""
        status = {"screen_capture": False, "ui_automation": False, "foreground_window": False}
        if not self.executable.exists():
            return status
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.executable), "status",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning("native_windows: status check failed: %s", exc)
            return status

        for line in stdout.decode("utf-8", errors="replace").splitlines():
            lower = line.strip().lower()
            if lower.startswith("screen capture"):
                status["screen_capture"] = "granted" in lower
            elif lower.startswith("ui automation"):
                status["ui_automation"] = "granted" in lower
            elif lower.startswith("foreground window"):
                status["foreground_window"] = "granted" in lower
        return status

    # -- Stop (Windows: explicit `stop` command) -----------------------------

    async def stop_session(self, session_id: str) -> None:
        cs = self._sessions.get(session_id)
        if cs is None or cs.process is None:
            return

        process = cs.process

        # Ask the helper to shut itself down gracefully via its named stop
        # event. Win32 asyncio can't deliver SIGINT to a subprocess, so this
        # is the supported stop path; terminate/kill below are the fallback.
        if process.returncode is None and self.executable.exists():
            try:
                stop_proc = await asyncio.create_subprocess_exec(
                    str(self.executable), "stop", "--session", session_id,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(stop_proc.communicate(), timeout=_STOP_COMMAND_TIMEOUT_SECONDS)
            except (OSError, asyncio.TimeoutError) as exc:
                logger.warning("native_windows: stop command failed for session %s: %s", session_id, exc)

        # Wait for the capture process to actually exit after being signaled.
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("native_windows: session %s did not exit after stop, terminating", session_id)
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=_TERMINATE_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    logger.warning("native_windows: session %s did not terminate, killing", session_id)
                    try:
                        process.kill()
                        await process.wait()
                    except ProcessLookupError:
                        pass
                except (ProcessLookupError, OSError):
                    pass

        await self._drain_readers(cs)
        logger.info("native_windows: stopped capture for session %s (%d events collected)", session_id, len(cs.events))
