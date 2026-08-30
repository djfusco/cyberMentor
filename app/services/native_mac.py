"""EvidenceProvider backed by the local native macOS capture helper
(native_capture/.build/debug/mentor-capture).

Unlike a continuously-running background capture system queried by time
window, this helper is session-bound: one subprocess is launched per
active exercise session and stopped (via SIGINT) when the exercise
finishes. Its stdout is a stream of newline-delimited JSON events (see
native_capture's own README for the schema), which the shared
NativeEvidenceProvider base (app.services.native_common) translates into the
exact raw-dict shape app.services.evidence.EvidenceNormalizer already knows
how to extract text/app/window/frame fields from -- so the normalizer,
EvidenceEvent, mentor, and evaluator all remain completely unchanged.

Platform-specific pieces (SIGINT stop, macOS permission checks) live here;
everything common to native providers lives in native_common. The Swift
helper itself is untouched.
"""
import asyncio
import logging
import signal
from typing import Dict, Optional

from app.config import get_settings
from app.services.native_common import (
    NativeEvidenceProvider,
    _NON_EVIDENCE_TYPES,
    _READER_DRAIN_TIMEOUT_SECONDS,
    _SIGINT_TIMEOUT_SECONDS,
    _TERMINATE_TIMEOUT_SECONDS,
    _resolve_path,
    _synthesize_text,
    _translate_event,
)

logger = logging.getLogger(__name__)

# Re-exported so existing tests and callers that import the translator (and
# the shared helpers) from this module keep working after the common layer
# was extracted into app.services.native_common.
__all__ = [
    "NativeMacEvidenceProvider",
    "_translate_event",
    "_synthesize_text",
    "_NON_EVIDENCE_TYPES",
    "_resolve_path",
]


class NativeMacEvidenceProvider(NativeEvidenceProvider):
    def __init__(self, executable: Optional[str] = None, output_dir: Optional[str] = None):
        settings = get_settings()
        super().__init__(
            executable or settings.native_capture_executable,
            output_dir or settings.native_capture_output,
            log_prefix="native_mac",
        )

    # -- Health / permissions (macOS) ----------------------------------------

    async def health(self) -> bool:
        """True if the executable exists and `check` runs successfully.

        Deliberately independent of permission status -- missing macOS
        permissions are a degraded-but-running state, not a health failure
        (see permission_status()).
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
            logger.warning("native_mac: health check failed: %s", exc)
            return False

    async def permission_status(self) -> Dict[str, bool]:
        """Parses `mentor-capture check` output into per-permission booleans."""
        status = {"screen_recording": False, "accessibility": False, "input_monitoring": False}
        if not self.executable.exists():
            return status
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.executable), "check",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning("native_mac: permission check failed: %s", exc)
            return status

        for line in stdout.decode("utf-8", errors="replace").splitlines():
            lower = line.strip().lower()
            if lower.startswith("screen recording"):
                status["screen_recording"] = "granted" in lower
            elif lower.startswith("accessibility"):
                status["accessibility"] = "granted" in lower
            elif lower.startswith("input monitoring"):
                status["input_monitoring"] = "granted" in lower
        return status

    # -- Stop (macOS: SIGINT) -------------------------------------------------

    async def stop_session(self, session_id: str) -> None:
        cs = self._sessions.get(session_id)
        if cs is None or cs.process is None:
            return

        process = cs.process
        if process.returncode is None:
            try:
                process.send_signal(signal.SIGINT)
                await asyncio.wait_for(process.wait(), timeout=_SIGINT_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("native_mac: session %s did not exit after SIGINT, terminating", session_id)
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=_TERMINATE_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    logger.warning("native_mac: session %s did not terminate, killing", session_id)
                    process.kill()
                    await process.wait()
            except ProcessLookupError:
                pass

        await self._drain_readers(cs)
        logger.info("native_mac: stopped capture for session %s (%d events collected)", session_id, len(cs.events))
