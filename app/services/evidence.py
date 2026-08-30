"""Normalizes raw evidence-provider records into a concise, deduplicated timeline.

Raw capture-provider JSON must never be sent directly to the LLM. This
module is the single place responsible for turning noisy, repetitive
capture records into a clean internal representation.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from app.services.evidence_provider import EvidenceProvider, EvidenceProviderError

logger = logging.getLogger(__name__)

TERMINAL_APP_HINTS = {"terminal", "iterm", "iterm2", "warp", "alacritty", "kitty", "hyper"}


@dataclass
class EvidenceEvent:
    timestamp: datetime
    application: str
    type: str
    text: str
    # Grounding context for GUI/web exercises -- which page/window was active.
    browser_url: Optional[str] = None
    window_title: Optional[str] = None
    # Path to the screenshot backing this record, if any (one per capture).
    # Preserved so vision-assisted evaluation can look at the actual image
    # instead of only OCR'd text -- see OllamaService.
    frame_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "application": self.application,
            "type": self.type,
            "text": self.text,
            "browser_url": self.browser_url,
            "window_title": self.window_title,
            "frame_path": self.frame_path,
        }


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _describe_input_event(content: Dict[str, Any]) -> str:
    """Input records (clicks/scrolls/keystrokes) have no natural text field --
    synthesize a short description so they aren't silently dropped."""
    event_type = content.get("event_type") or "input"
    text_content = content.get("text_content")
    if text_content:
        return f'typed "{text_content}"'
    key_code = content.get("key_code")
    if key_code:
        return f"pressed key {key_code}"
    element = content.get("element_name")
    role = content.get("element_role")
    if element:
        target = f"{element} ({role})" if role else element
    else:
        x, y = content.get("x"), content.get("y")
        target = f"({x},{y})" if x is not None and y is not None else "an element"
    return f"{event_type} on {target}"


def _extract_text(item: Dict[str, Any]) -> str:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    for source in (content, item):
        for key in ("text", "ocr_text", "transcription"):
            value = source.get(key)
            if value:
                return str(value).strip()
    if str(item.get("type", "")).lower() == "input":
        return _describe_input_event(content)
    return ""


def _extract_app(item: Dict[str, Any]) -> str:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    for source in (content, item):
        for key in ("app_name", "application", "window_name"):
            value = source.get(key)
            if value:
                return str(value)
    return "unknown"


def _extract_type(item: Dict[str, Any]) -> str:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    return str(item.get("type") or content.get("type") or "screen_text")


def _extract_timestamp_raw(item: Dict[str, Any]) -> Any:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    return item.get("timestamp") or content.get("timestamp")


def _extract_optional_str(item: Dict[str, Any], *keys: str) -> Optional[str]:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    for source in (content, item):
        for key in keys:
            value = source.get(key)
            if value:
                return str(value)
    return None


class EvidenceNormalizer:
    def __init__(self, similarity_threshold: float = 0.92):
        self.similarity_threshold = similarity_threshold

    def normalize(self, raw_items: List[Dict[str, Any]]) -> List[EvidenceEvent]:
        events: List[EvidenceEvent] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            text = _extract_text(item)
            if not text:
                continue
            timestamp = _parse_timestamp(_extract_timestamp_raw(item))
            if timestamp is None:
                continue
            events.append(
                EvidenceEvent(
                    timestamp=timestamp,
                    application=_extract_app(item),
                    type=_extract_type(item),
                    text=text,
                    browser_url=_extract_optional_str(item, "browser_url"),
                    window_title=_extract_optional_str(item, "window_title", "window_name"),
                    frame_path=_extract_optional_str(item, "file_path", "frame_name"),
                )
            )
        events.sort(key=lambda e: e.timestamp)
        return self._deduplicate(events)

    def _deduplicate(self, events: List[EvidenceEvent], lookback: int = 3) -> List[EvidenceEvent]:
        """Collapse near-identical observations from the same app.

        Native capture frequently re-observes the same screen state -- not
        only in immediately consecutive records, but also a few events
        later (e.g. a periodic safety-checkpoint screenshot, or a brief
        switch away and back). Compare each event against the last few
        *kept* events (not just the immediately preceding one) so those
        non-consecutive repeats are collapsed too, keeping the longest/most
        complete text.
        """
        deduped: List[EvidenceEvent] = []
        for event in events:
            match_index: Optional[int] = None
            for idx in range(len(deduped) - 1, max(-1, len(deduped) - 1 - lookback), -1):
                candidate = deduped[idx]
                if candidate.application == event.application and self._is_similar(candidate.text, event.text):
                    match_index = idx
                    break
            if match_index is not None:
                existing = deduped[match_index]
                if len(event.text) > len(existing.text):
                    deduped[match_index] = EvidenceEvent(
                        timestamp=existing.timestamp,
                        application=existing.application,
                        type=existing.type,
                        text=event.text,
                        browser_url=event.browser_url or existing.browser_url,
                        window_title=event.window_title or existing.window_title,
                        frame_path=event.frame_path or existing.frame_path,
                    )
                continue
            deduped.append(event)
        return deduped

    def _is_similar(self, a: str, b: str) -> bool:
        if a == b:
            return True
        if not a or not b:
            return False
        return SequenceMatcher(None, a, b).ratio() >= self.similarity_threshold


class EvidenceService:
    """High-level, normalized access to session evidence."""

    def __init__(self, provider: EvidenceProvider, normalizer: Optional[EvidenceNormalizer] = None):
        self.provider = provider
        self.normalizer = normalizer or EvidenceNormalizer()

    async def get_session_activity(
        self, started_at: datetime, ended_at: Optional[datetime] = None, limit: int = 200
    ) -> List[EvidenceEvent]:
        end_time = ended_at or datetime.now(timezone.utc)
        try:
            raw_items = await self.provider.get_activity(started_at, end_time, limit=limit)
        except EvidenceProviderError:
            logger.warning("Evidence retrieval failed for session window %s - %s", started_at, end_time)
            raise
        return self.normalizer.normalize(raw_items)

    async def get_persisted_session_activity(self, session_id: str) -> List[EvidenceEvent]:
        """Load a completed session's persisted on-disk evidence and normalize it.

        Used for historical queries after the capture process is gone (see
        SessionQueryService). Providers that don't persist per-session evidence
        return an empty list, so the caller can fall back to the live
        time-windowed path. The persisted file is already scoped to one
        session, so no time-window filter is applied -- the normalizer still
        sorts chronologically and de-duplicates near-identical states.
        """
        raw_items = await self.provider.load_persisted_events(session_id)
        return self.normalizer.normalize(raw_items)

    @staticmethod
    def filter_terminal_events(events: List[EvidenceEvent]) -> List[EvidenceEvent]:
        terminal_events = [
            e for e in events if any(hint in e.application.lower() for hint in TERMINAL_APP_HINTS)
        ]
        return terminal_events or events
