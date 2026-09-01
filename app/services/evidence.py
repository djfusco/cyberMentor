"""Normalizes raw evidence-provider records into a concise, deduplicated timeline.

Raw capture-provider JSON must never be sent directly to the LLM. This
module is the single place responsible for turning noisy, repetitive
capture records into a clean internal representation.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from app.services.evidence_provider import EvidenceProvider, EvidenceProviderError

logger = logging.getLogger(__name__)

TERMINAL_APP_HINTS = {"terminal", "iterm", "iterm2", "warp", "alacritty", "kitty", "hyper"}

# Each LineChange is (tag, line_text).  Tags:
#   'unchanged'           line present and identical in both screens
#   'added'               line present only in new screen (new terminal output)
#   'removed'             line present only in old screen (scrolled off / cleared)
#   'meaningful_modified' same line position, content genuinely changed
#                         (e.g. chmod 600 → chmod 644, Failed → Passed)
#   'ocr_jitter'          same line position, nearly identical text -- minor
#                         OCR noise, not a meaningful change
LineChange = Tuple[str, str]

# SequenceMatcher ratio >= this: treat a replaced-line pair as OCR noise only.
_OCR_JITTER_THRESHOLD: float = 0.95
# ratio >= this but below jitter: a real content change on an existing line.
_MEANINGFUL_MODIFIED_THRESHOLD: float = 0.60


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
    # Derived line-level delta versus the previous text_observed event from
    # the same application.  Populated during deduplication when at least one
    # added or meaningful_modified line is detected; None for all other event
    # types and for the first text_observed from each application (no prior
    # screen to diff against).  The raw .text field is always the complete
    # original OCR text -- this field is strictly additive.
    text_delta: Optional[List[LineChange]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "application": self.application,
            "type": self.type,
            "text": self.text,
            "browser_url": self.browser_url,
            "window_title": self.window_title,
            "frame_path": self.frame_path,
            "text_delta": (
                [{"tag": tag, "text": text} for tag, text in self.text_delta]
                if self.text_delta is not None
                else None
            ),
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


def _compute_line_delta(old_text: str, new_text: str) -> List[LineChange]:
    """Compare two consecutive full-screen text_observed captures line by line.

    Uses SequenceMatcher at the line level (autojunk=False so duplicate lines
    such as repeated shell prompts are not silently suppressed).

    For 'replace' opcodes where block lengths match, each line pair is
    classified individually by character-level similarity:
      ratio >= _OCR_JITTER_THRESHOLD  → 'ocr_jitter'   (minor OCR noise)
      ratio >= _MEANINGFUL_MODIFIED_THRESHOLD → 'meaningful_modified'
      ratio <  _MEANINGFUL_MODIFIED_THRESHOLD → 'removed' + 'added'

    For unequal-length 'replace' blocks (content too different to pair),
    all old lines are emitted as 'removed' and all new lines as 'added'.
    This preserves the newer event's content without making assumptions
    about which old line corresponds to which new line.
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    result: List[LineChange] = []
    matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in new_lines[j1:j2]:
                result.append(("unchanged", line))
        elif tag == "insert":
            for line in new_lines[j1:j2]:
                result.append(("added", line))
        elif tag == "delete":
            for line in old_lines[i1:i2]:
                result.append(("removed", line))
        elif tag == "replace":
            old_block = old_lines[i1:i2]
            new_block = new_lines[j1:j2]
            if len(old_block) == len(new_block):
                for old_l, new_l in zip(old_block, new_block):
                    ratio = SequenceMatcher(None, old_l, new_l).ratio()
                    if ratio >= _OCR_JITTER_THRESHOLD:
                        result.append(("ocr_jitter", new_l))
                    elif ratio >= _MEANINGFUL_MODIFIED_THRESHOLD:
                        result.append(("meaningful_modified", new_l))
                    else:
                        result.append(("removed", old_l))
                        result.append(("added", new_l))
            else:
                for line in old_block:
                    result.append(("removed", line))
                for line in new_block:
                    result.append(("added", line))
    return result


def _has_meaningful_change(delta: List[LineChange]) -> bool:
    """Return True when the delta contains at least one 'added' or
    'meaningful_modified' line -- i.e. the screen shows content the
    previous capture did not, and the event warrants being emitted."""
    return any(tag in ("added", "meaningful_modified") for tag, _ in delta)


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
        """Collapse near-duplicate observations using per-type strategies.

        text_observed events (full-screen OCR dumps):
            Line-level delta deduplication.  Each event is compared against
            the most recently seen screen for that application using
            _compute_line_delta().  An event is emitted only when at least
            one line is 'added' or 'meaningful_modified'; the event carries
            the full delta in .text_delta so downstream prompts can show
            exactly what became newly visible.  The baseline is updated on
            every event (kept or discarded) so it always tracks the real
            current screen state.

        All other event types:
            Whole-text similarity lookback (unchanged from prior behaviour).
            Their synthesised texts -- "Switched to Terminal", "Clicked at
            (x, y)", "Keyboard activity: typing x8" -- are short and
            application-specific, so the character-level SequenceMatcher
            threshold remains appropriate.

        The old "keep longer text on match" branch is intentionally removed:
        it was the mechanism by which a post-scroll screen that was shorter
        than its predecessor caused the prior (longer) text to be retained
        even when the new screen contained genuinely new bottom lines.
        """
        deduped: List[EvidenceEvent] = []
        # Per-application baseline for text_observed delta comparison.
        # Updated on every text_observed event, kept or discarded, so the
        # delta always measures against the actual current screen state.
        last_text_by_app: Dict[str, str] = {}

        for event in events:
            if event.type == "text_observed":
                prev = last_text_by_app.get(event.application)
                last_text_by_app[event.application] = event.text
                if prev is None:
                    # First observation for this app -- no prior screen to
                    # diff against; emit as-is with no delta.
                    deduped.append(event)
                else:
                    delta = _compute_line_delta(prev, event.text)
                    if _has_meaningful_change(delta):
                        deduped.append(
                            EvidenceEvent(
                                timestamp=event.timestamp,
                                application=event.application,
                                type=event.type,
                                text=event.text,
                                browser_url=event.browser_url,
                                window_title=event.window_title,
                                frame_path=event.frame_path,
                                text_delta=delta,
                            )
                        )
                    # No meaningful change (jitter / removed-only / unchanged):
                    # silently discard.  Baseline was already updated above.
            else:
                # Non-text_observed: whole-text lookback deduplication.
                match_index: Optional[int] = None
                for idx in range(len(deduped) - 1, max(-1, len(deduped) - 1 - lookback), -1):
                    candidate = deduped[idx]
                    if (
                        candidate.application == event.application
                        and self._is_similar(candidate.text, event.text)
                    ):
                        match_index = idx
                        break
                if match_index is None:
                    deduped.append(event)
                # else: discard identical non-text event.
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
