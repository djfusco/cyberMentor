"""Evidence provider abstraction.

Every capture backend (native macOS, native Windows, the cross-platform
Rust helper) implements this interface. Callers depend only on
EvidenceProvider so a new capture system can be added without changes
elsewhere in the app.
"""
import abc
from datetime import datetime
from typing import Any, Dict, List, Optional


class EvidenceProviderError(Exception):
    """Raised when an evidence provider cannot be reached or returns bad data."""


class EvidenceProvider(abc.ABC):
    @abc.abstractmethod
    async def health(self) -> bool:
        ...

    @abc.abstractmethod
    async def get_activity(
        self, start_time: datetime, end_time: datetime, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Return raw evidence records relevant to a session's timeframe."""

    @abc.abstractmethod
    async def search(
        self,
        query: str = "",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        content_type: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        ...

    # Optional session-lifecycle hooks. No-op by default -- providers that
    # capture continuously in the background never need to override these;
    # providers that are session-bound (e.g. a native helper process
    # launched per exercise) do. Routes call these unconditionally on
    # whatever provider is active, so the rest of the app never needs to
    # know which provider is in use.
    async def start_session(self, session_id: str) -> None:
        return None

    async def stop_session(self, session_id: str) -> None:
        return None

    async def load_persisted_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Persisted/on-disk evidence for one session, for historical queries
        after the capture process is gone.

        No-op by default: providers that capture a continuous background
        timeline have no per-session persisted file and return [], so
        callers fall back to the live time-windowed path. Session-bound
        native providers override this to read their
        capture_sessions/<id>/events.jsonl back from disk.
        """
        return []
