"""Anchor-first visual keyframe selection for non-terminal GUI exercises.

Replaces evenly-spread-only screenshot selection with a two-tier strategy:
  1. Anchor frames  (ANCHOR_FRAME_FRACTION of budget)
       Each significant event (application switch, window change, mouse click,
       scroll, visual screen change, and Slice-1-detected text addition) receives
       the nearest available frame at or before the event *and* the nearest frame
       strictly after it, giving the model a before/after view of each transition.
       If anchor-associated frames exceed the anchor budget, the set is sampled
       so early, middle, and late session activity are all represented.
  2. Spread frames  (remaining budget)
       Evenly distributed across periods not already covered by anchor frames,
       always sampling from the full span so no phase of the session is invisible.

Result is deduplicated by path and returned in chronological order with full
provenance metadata (timestamp, application, window_title, trigger_type, role)
so the model prompt can associate every image with its place in the session.

Security constraint: this module never captures raw keystrokes, terminal
input/output, browser DOM data, shell history, or application-specific private
data. It selects from already-captured frame paths only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from app.services.evidence import EvidenceEvent
from app.services.text_relevance import outcome_keyword_text, overlap_score, significant_words

# ---------------------------------------------------------------------------
# Named configuration constants
# ---------------------------------------------------------------------------

# Share of the frame budget reserved for anchor-associated before/after pairs.
ANCHOR_FRAME_FRACTION: float = 0.75
# Share reserved for evenly-distributed coverage frames (1 - ANCHOR_FRAME_FRACTION).
SPREAD_FRAME_FRACTION: float = 0.25

# Event types that always mark an anchor regardless of content.
_ANCHOR_EVENT_TYPES: frozenset = frozenset(
    {"app_change", "window_change", "mouse_click", "scroll", "screen_change"}
)

# Tags from Slice 1 that indicate a text_observed event is a meaningful anchor.
_MEANINGFUL_TAGS: frozenset = frozenset({"added", "meaningful_modified"})


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass
class SelectedFrame:
    """A single frame chosen for inclusion in a vision prompt.

    Carries enough provenance that the model prompt can describe every image
    with its context: when it was captured, which application was active, what
    it was selected to show, and whether it is the state just before or just
    after a significant event.
    """

    path: str
    timestamp: object  # datetime -- typed as object to avoid runtime import cost
    application: str
    window_title: Optional[str]
    trigger_type: str  # anchor event type or 'spread'
    role: str          # 'before' | 'after' | 'spread'
    # Timestamp of the anchor event that caused this frame to be selected.
    # None for spread frames (no anchor) and for frames selected before this
    # field was introduced. Populated for before/after anchor frames only.
    anchor_timestamp: object = None  # datetime or None
    # Populated only by the outcome-aware path (select_keyframes(..., outcomes=...)).
    # Zero/empty for the original generic path (mentor chat, session Q&A),
    # which never sets these -- purely additive, safe defaults.
    relevance_score: float = 0.0
    # Human-readable reason this frame was selected -- which outcome(s) its
    # nearby evidence text matched, or "generic coverage" -- surfaced in
    # diagnostics (capture_manifest) so a human can see WHY a frame was kept.
    matched_evidence: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_anchor_event(event: EvidenceEvent) -> bool:
    """True when the event represents a significant user action or visible change.

    Anchor event types are always significant (app switch, window change, click,
    scroll, and Rust-detected screen_change). text_observed events are anchors
    only when Slice 1 determined the screen contained new meaningful content.
    All other event types (session_start, session_stop, key_activity) are not
    anchors -- they carry no frame that the vision model can observe.
    """
    if event.type in _ANCHOR_EVENT_TYPES:
        return True
    if event.type == "text_observed" and event.text_delta is not None:
        return any(tag in _MEANINGFUL_TAGS for tag, _ in event.text_delta)
    return False


def _evenly_sample(items: list, budget: int) -> list:
    """Return up to `budget` items sampled evenly from `items`, always including
    the first and last.  Returns the full list when len(items) <= budget."""
    n = len(items)
    if n <= budget:
        return items
    if budget <= 1:
        return [items[0]]
    indices = sorted({round(i * (n - 1) / (budget - 1)) for i in range(budget)})
    return [items[i] for i in indices]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_keyframes(
    events: List[EvidenceEvent], max_frames: int, outcomes: Optional[Sequence] = None
) -> List[SelectedFrame]:
    """Return up to *max_frames* SelectedFrames using anchor-first selection.

    Algorithm
    ---------
    1. Collect all events that carry a frame_path; build an ordered list of
       (event_index, path, event) tuples (frame_events).
    2. Walk every event; for each anchor event find the closest frame at or
       before (role='before') and the closest frame strictly after (role='after').
       Deduplicate by path, keeping the first-seen metadata.
    3. If anchor frames exceed ``anchor_budget = int(max_frames * 0.75)`` sample
       them evenly across the session so early, middle, and late activity are
       all represented.
    4. Fill the remaining budget with spread frames: distinct paths not already
       claimed by anchors, sampled evenly across the remaining session span.
    5. Combine, deduplicate by path, sort chronologically, cap at max_frames.

    `outcomes`: when given (a sequence of ExpectedOutcome-like objects with
    `.description` / `.success_criteria` / `.evidence_requirements` /
    `.student_demonstration`), selection switches to a RELEVANCE-RANKED mode
    instead of step 3/4's blind even-temporal sampling -- see
    `_select_relevant_frames`. This is what EvaluatorService uses so decisive
    result evidence (a completed query result, a final confirmation screen)
    is never discarded just because it falls in a temporally dense cluster.
    Every other caller (mentor chat, session Q&A) omits `outcomes` and gets
    byte-for-byte the original generic behavior.
    """
    if max_frames <= 0:
        return []

    # --- Step 1: ordered index of frame-bearing events ---
    frame_events: List[Tuple[int, str, EvidenceEvent]] = [
        (i, e.frame_path, e)
        for i, e in enumerate(events)
        if e.frame_path
    ]
    if not frame_events:
        return []

    # Map from path to its first-seen position (used for chronological sort).
    path_order: Dict[str, int] = {}
    for seq, (_, path, _) in enumerate(frame_events):
        if path not in path_order:
            path_order[path] = seq

    # Budget split
    anchor_budget: int = max(1, int(max_frames * ANCHOR_FRAME_FRACTION))

    # --- Step 2: collect anchor-associated before/after frames ---
    anchor_frames: List[SelectedFrame] = []
    seen_anchor_paths: Set[str] = set()

    for anchor_idx, anchor_event in enumerate(events):
        if not _is_anchor_event(anchor_event):
            continue

        # Closest frame at or before the anchor (search backwards)
        before: Optional[Tuple[int, str, EvidenceEvent]] = None
        for fi, path, fe in reversed(frame_events):
            if fi <= anchor_idx:
                before = (fi, path, fe)
                break

        # Closest frame strictly after the anchor (search forwards)
        after: Optional[Tuple[int, str, EvidenceEvent]] = None
        for fi, path, fe in frame_events:
            if fi > anchor_idx:
                after = (fi, path, fe)
                break

        for role, pair in (("before", before), ("after", after)):
            if pair is None:
                continue
            _, path, fe = pair
            if path in seen_anchor_paths:
                continue  # already claimed; preserve first-seen metadata
            seen_anchor_paths.add(path)
            anchor_frames.append(
                SelectedFrame(
                    path=path,
                    timestamp=fe.timestamp,
                    application=fe.application,
                    window_title=fe.window_title,
                    trigger_type=anchor_event.type,
                    role=role,
                    anchor_timestamp=anchor_event.timestamp,
                )
            )

    if outcomes:
        all_frames = _select_relevant_frames(
            anchor_frames, frame_events, outcomes, max_frames,
        )
    else:
        # --- Step 3: distribute anchor frames across session if over budget ---
        if len(anchor_frames) > anchor_budget:
            anchor_frames = _evenly_sample(anchor_frames, anchor_budget)

        # --- Step 4: spread frames from uncovered periods ---
        anchor_paths_final: Set[str] = {sf.path for sf in anchor_frames}
        remaining_budget = max_frames - len(anchor_frames)

        spread_frames: List[SelectedFrame] = []
        if remaining_budget > 0:
            spread_seen: Set[str] = set()
            spread_candidates: List[Tuple[int, str, EvidenceEvent]] = []
            for idx, path, ev in frame_events:
                if path not in anchor_paths_final and path not in spread_seen:
                    spread_seen.add(path)
                    spread_candidates.append((idx, path, ev))

            for _, path, ev in _evenly_sample(spread_candidates, remaining_budget):
                spread_frames.append(
                    SelectedFrame(
                        path=path,
                        timestamp=ev.timestamp,
                        application=ev.application,
                        window_title=ev.window_title,
                        trigger_type="spread",
                        role="spread",
                    )
                )

        all_frames = anchor_frames + spread_frames

    # --- Step 5: combine, deduplicate, sort chronologically ---
    final: List[SelectedFrame] = []
    seen_final: Set[str] = set()
    for sf in all_frames:
        if sf.path not in seen_final:
            seen_final.add(sf.path)
            final.append(sf)

    final.sort(key=lambda sf: path_order.get(sf.path, 0))
    return final[:max_frames]


# ---------------------------------------------------------------------------
# Outcome-aware relevance selection
# ---------------------------------------------------------------------------


def _event_text_signal(ev: EvidenceEvent) -> str:
    """Every word of text plausibly associated with one captured frame:
    the window title (the dominant generic signal for GUI/web/SIEM
    exercises -- browser/app window titles reliably carry topic words, e.g.
    "Search | Splunk", "Add Data - Success"), the event's own synthesized
    text, and any newly-added/changed OCR/AX text on that same event.
    """
    parts = [ev.window_title or "", ev.text or ""]
    if ev.text_delta:
        parts.extend(line for tag, line in ev.text_delta if tag in ("added", "meaningful_modified"))
    return " ".join(parts)


def _select_relevant_frames(
    anchor_frames: List[SelectedFrame],
    frame_events: List[Tuple[int, str, EvidenceEvent]],
    outcomes: Sequence,
    max_frames: int,
) -> List[SelectedFrame]:
    """Rank every frame-bearing event by relevance to the exercise's own
    outcomes (description + success_criteria + evidence_requirements +
    student_demonstration -- whatever the author actually wrote, nothing
    invented), and fill the frame budget by relevance first, falling back to
    even temporal coverage only for whatever budget relevance can't fill.

    This is what keeps a temporally-clustered but decisive result frame (a
    query's own result screen, a final confirmation) from being discarded by
    blind even-sampling just because several other frames sit near it in
    time -- see the "even temporal sampling cannot discard more relevant
    evidence" regression requirement.
    """
    outcome_keywords = [
        (getattr(o, "id", str(i)), significant_words(outcome_keyword_text(o)))
        for i, o in enumerate(outcomes)
    ]
    outcome_keywords = [(oid, kw) for oid, kw in outcome_keywords if kw]

    # One candidate per distinct frame path, scored against every outcome's
    # keyword set. anchor_frames' metadata (role/trigger_type from the
    # before/after pairing) is preserved when available, since "after a
    # meaningful screen/query change" is itself a relevance signal (role
    # "after" outranks "before" on a tie -- see the sort key below);
    # anchor coverage is a superset check, not a replacement, so every
    # frame_event is still considered even if it wasn't anchor-paired.
    by_path_anchor: Dict[str, SelectedFrame] = {}
    for sf in anchor_frames:
        # Keep the first (highest-priority: earliest-discovered) anchor
        # metadata per path, matching the existing dedup rule.
        by_path_anchor.setdefault(sf.path, sf)

    seen_paths: Set[str] = set()
    candidates: List[Tuple[float, str, SelectedFrame]] = []
    for idx, path, ev in frame_events:
        if path in seen_paths:
            continue
        seen_paths.add(path)

        text_words = significant_words(_event_text_signal(ev))
        best_score = 0.0
        best_outcome_id: Optional[str] = None
        for oid, kw in outcome_keywords:
            score = float(overlap_score(text_words, kw))
            if score > best_score:
                best_score = score
                best_outcome_id = oid

        anchor = by_path_anchor.get(path)
        role = anchor.role if anchor is not None else "spread"
        trigger_type = anchor.trigger_type if anchor is not None else "spread"
        anchor_timestamp = anchor.anchor_timestamp if anchor is not None else None

        matched_evidence = (
            f"relevant to outcome '{best_outcome_id}' (matched: "
            f"{', '.join(sorted(text_words & dict(outcome_keywords)[best_outcome_id]))})"
            if best_outcome_id is not None
            else "generic timeline coverage (no outcome-keyword match)"
        )

        candidates.append((
            best_score,
            best_outcome_id or "",
            SelectedFrame(
                path=path,
                timestamp=ev.timestamp,
                application=ev.application,
                window_title=ev.window_title,
                trigger_type=trigger_type,
                role=role,
                anchor_timestamp=anchor_timestamp,
                relevance_score=best_score,
                matched_evidence=matched_evidence,
            ),
        ))

    if not candidates:
        return []

    # Round-robin: give each outcome a fair shot at its own best-scoring,
    # not-yet-claimed frame(s) before spending budget on a second frame for
    # any single outcome -- this is what makes "~2 relevant frames per
    # outcome" the natural spread rather than one outcome's evidence
    # crowding out another's.
    frames_per_outcome = max(1, -(-max_frames // max(1, len(outcome_keywords)))) if outcome_keywords else 0
    claimed: Set[str] = set()
    selected: List[SelectedFrame] = []

    def _sort_key(item):
        score, _, sf = item
        # Higher relevance first; among ties, prefer a post-change/result
        # frame ("after") over a pre-change one, then the LATER frame (a
        # late-session result state over an earlier, possibly-superseded one).
        role_rank = {"after": 0, "spread": 1, "before": 2}.get(sf.role, 1)
        ts = sf.timestamp
        recency = -(ts.timestamp()) if hasattr(ts, "timestamp") else 0
        return (-score, role_rank, recency)

    for oid, _ in outcome_keywords:
        if len(selected) >= max_frames:
            break
        own = sorted(
            (c for c in candidates if c[1] == oid and c[2].path not in claimed and c[0] > 0),
            key=_sort_key,
        )
        for _, _, sf in own[:frames_per_outcome]:
            if len(selected) >= max_frames:
                break
            claimed.add(sf.path)
            selected.append(sf)

    # A second pass lets any outcome with still-unused, still-relevant
    # frames take more of the remaining budget before falling back to
    # generic coverage, rather than immediately handing leftover slots to
    # zero-relevance filler.
    if len(selected) < max_frames:
        leftover_relevant = sorted(
            (c for c in candidates if c[0] > 0 and c[2].path not in claimed),
            key=_sort_key,
        )
        for _, _, sf in leftover_relevant:
            if len(selected) >= max_frames:
                break
            claimed.add(sf.path)
            selected.append(sf)

    # Fill any remaining budget with generic even-temporal coverage from
    # whatever is left, so an outcome with no textual signal at all (e.g. a
    # purely visual confirmation with no matching window-title words) still
    # gets some representation instead of zero frames.
    if len(selected) < max_frames:
        remaining = [c for c in candidates if c[2].path not in claimed]
        remaining.sort(key=lambda c: path_order_lookup(c[2].path, frame_events))
        fill = _evenly_sample(remaining, max_frames - len(selected))
        for _, _, sf in fill:
            claimed.add(sf.path)
            selected.append(sf)

    return selected


def path_order_lookup(path: str, frame_events: List[Tuple[int, str, EvidenceEvent]]) -> int:
    for idx, p, _ in frame_events:
        if p == path:
            return idx
    return 0


def format_frame_captions(frames: List[SelectedFrame]) -> str:
    """Return a human-readable numbered list of frame captions for the prompt.

    Each line associates the image number (as the model receives it) with a
    timestamp, application, window, and the reason the frame was selected.
    This lets the model reason "screenshot 3 was captured just before a
    mouse_click, so it shows the state the learner was looking at" without
    having to infer it from filename patterns.
    """
    if not frames:
        return ""
    lines = [
        "Attached screenshots (chronological; each shows when and why it was selected):"
    ]
    role_labels = {
        "before": "before {trigger}",
        "after": "after {trigger}",
        "spread": "periodic coverage",
    }
    for i, sf in enumerate(frames, 1):
        try:
            ts = sf.timestamp.strftime("%H:%M:%S")
        except AttributeError:
            ts = str(sf.timestamp)
        window = f" ({sf.window_title[:60]})" if sf.window_title else ""
        role_label = role_labels.get(sf.role, sf.role).format(trigger=sf.trigger_type)
        line = f"  {i}. [{ts}] {sf.application}{window} — {role_label}"
        if sf.matched_evidence:
            line += f" ({sf.matched_evidence})"
        lines.append(line)
    return "\n".join(lines)
