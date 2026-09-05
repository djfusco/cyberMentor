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
from app.services.text_relevance import (
    criterion_relevance_score,
    distinctive_word_sets,
    significant_words,
)

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

# Maximum frames the criterion-coverage allocator (_select_relevant_frames)
# gives any ONE criterion per round-robin pass, regardless of how many
# tied/positively-scoring candidates it has. Without this cap, a criterion
# with many identically-scoring candidates (e.g. a dozen frames sharing one
# generic window title) can consume the entire frame budget on its own,
# since a criterion with ZERO candidates never "progresses" and therefore
# never claims a turn -- observed live across an 8-frame budget and 8
# criteria, where 6 criteria had no lexical candidates at all.
MAX_FRAMES_PER_CRITERION: int = 2

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
    frame_events, anchor_frames, anchor_budget, path_order = _prepare_frame_candidates(events, max_frames)
    if not frame_events:
        return []

    if outcomes:
        all_frames, _uncovered = _select_relevant_frames(
            anchor_frames, frame_events, events, outcomes, max_frames,
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


def select_keyframes_with_coverage(
    events: List[EvidenceEvent], max_frames: int, outcomes: Sequence,
) -> "tuple[List[SelectedFrame], Dict[str, List[str]]]":
    """Like select_keyframes(events, max_frames, outcomes=outcomes), but also
    returns which specific success criteria received NO relevant candidate
    evidence at all (outcome_id -> list of criterion texts). Callers
    (EvaluatorService) use this to distinguish "evaluated and not
    demonstrated" from "no evidence was ever selected to check" -- the
    latter must never be scored as a learner failure (see
    EvaluatorService._evaluate_criteria's not_evaluated state).
    """
    if max_frames <= 0 or not outcomes:
        return [], {}
    frame_events, anchor_frames, _anchor_budget, path_order = _prepare_frame_candidates(events, max_frames)
    if not frame_events:
        return [], {o.id: list(o.get_success_criteria()) for o in outcomes}
    frames, uncovered = _select_relevant_frames(anchor_frames, frame_events, events, outcomes, max_frames)
    final: List[SelectedFrame] = []
    seen_final: Set[str] = set()
    for sf in frames:
        if sf.path not in seen_final:
            seen_final.add(sf.path)
            final.append(sf)
    final.sort(key=lambda sf: path_order.get(sf.path, 0))
    return final[:max_frames], uncovered


def _prepare_frame_candidates(
    events: List[EvidenceEvent], max_frames: int,
) -> "tuple[List[Tuple[int, str, EvidenceEvent]], List[SelectedFrame], int, Dict[str, int]]":
    """Shared prep for both selection modes: the ordered frame-bearing event
    index, the generic anchor before/after pairing, the anchor budget, and
    chronological path ordering. Extracted so select_keyframes() and
    select_keyframes_with_coverage() never compute this differently."""
    # --- Step 1: ordered index of frame-bearing events ---
    frame_events: List[Tuple[int, str, EvidenceEvent]] = [
        (i, e.frame_path, e)
        for i, e in enumerate(events)
        if e.frame_path
    ]
    if not frame_events:
        return [], [], 0, {}

    path_order: Dict[str, int] = {}
    for seq, (_, path, _) in enumerate(frame_events):
        if path not in path_order:
            path_order[path] = seq

    anchor_budget: int = max(1, int(max_frames * ANCHOR_FRAME_FRACTION))

    # --- Step 2: collect anchor-associated before/after frames ---
    anchor_frames: List[SelectedFrame] = []
    seen_anchor_paths: Set[str] = set()

    for anchor_idx, anchor_event in enumerate(events):
        if not _is_anchor_event(anchor_event):
            continue

        before: Optional[Tuple[int, str, EvidenceEvent]] = None
        for fi, path, fe in reversed(frame_events):
            if fi <= anchor_idx:
                before = (fi, path, fe)
                break

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
                continue
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

    return frame_events, anchor_frames, anchor_budget, path_order


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


def _build_result_propagation(
    events: List[EvidenceEvent], frame_events: List[Tuple[int, str, EvidenceEvent]],
) -> Dict[str, str]:
    """Maps a RESULT frame's path -> the immediately preceding distinct
    frame's path, for every pair of chronologically consecutive frames with
    a mouse_click or "return"-keypress action event between them.

    This is what lets a query/action's own (often lexically generic or
    identical-looking) result screen inherit relevance from whatever made
    the preceding frame relevant, instead of needing its OWN window title
    to carry distinguishing words -- "entering a query and viewing its
    results are separate evidence", and the result is what proves or
    disproves an outcome even when neither frame's title changes at all
    (e.g. a single-page search app where every screen shares one title).
    """
    ordered_unique: List[Tuple[int, str]] = []
    seen: Set[str] = set()
    for idx, path, _ in frame_events:
        if path not in seen:
            seen.add(path)
            ordered_unique.append((idx, path))

    propagation: Dict[str, str] = {}
    for k in range(len(ordered_unique) - 1):
        idx_a, path_a = ordered_unique[k]
        idx_b, path_b = ordered_unique[k + 1]
        has_action = False
        for m in range(idx_a + 1, idx_b):
            e = events[m]
            if e.type == "mouse_click":
                has_action = True
                break
            if e.type == "key_activity" and "return" in (e.text or "").lower():
                has_action = True
                break
        if has_action:
            propagation[path_b] = path_a
    return propagation


def _select_relevant_frames(
    anchor_frames: List[SelectedFrame],
    frame_events: List[Tuple[int, str, EvidenceEvent]],
    events: List[EvidenceEvent],
    outcomes: Sequence,
    max_frames: int,
) -> "tuple[List[SelectedFrame], Dict[str, List[str]]]":
    """Select evidence to MAXIMIZE COVERAGE OF INDIVIDUAL SUCCESS CRITERIA,
    not a fixed per-outcome quota: an outcome with 3 criteria that each need
    a different screen can end up with 3 selected frames, while a
    single-criterion outcome needs just 1 -- and a criterion with no
    available evidence at all is reported as UNCOVERED (second return
    value: outcome_id -> [uncovered criterion text]) rather than silently
    dropped, so the caller can treat "no evidence was selected to check
    this" as unverifiable/needing a follow-up rather than a failure.

    Relevance is scored per (frame, CRITERION) pair using ONLY that
    criterion's own text (via ExpectedOutcome.get_success_criteria(), which
    falls back to the description for outcomes with no explicit criteria) --
    never the outcome's full blended description+evidence_requirements blob.
    Scoring an outcome's loosely-worded description against frames let a
    single incidental word (e.g. the host application's own name, appearing
    in nearly every window title AND in one outcome's description) score
    that outcome as "relevant" almost everywhere in the session, starving
    every other outcome of budget; criteria text is authored to be specific,
    and criterion_relevance_score additionally requires 2+ shared words (or
    1 that is DISTINCTIVE to this criterion vs every other criterion in the
    exercise) before counting a match at all -- see text_relevance.py.

    Action/result pairing (_build_result_propagation) lets a result frame
    inherit relevance from the action that produced it, so entering a query
    and viewing its result are not scored as if only the query text
    mattered. This is what keeps a temporally-clustered but decisive result
    frame from being discarded by blind even-sampling just because several
    other frames sit near it in time.
    """
    criteria_keys: List[Tuple[str, int]] = []
    criterion_text: Dict[Tuple[str, int], str] = {}
    for o in outcomes:
        oid = getattr(o, "id", "")
        for cidx, ctext in enumerate(o.get_success_criteria()):
            key = (oid, cidx)
            criteria_keys.append(key)
            criterion_text[key] = ctext

    if not criteria_keys:
        return [], {}

    criterion_words = {key: significant_words(criterion_text[key]) for key in criteria_keys}
    distinctive_list = distinctive_word_sets([criterion_words[key] for key in criteria_keys])
    criterion_distinctive = dict(zip(criteria_keys, distinctive_list))

    # Per-frame metadata, computed once per distinct path. anchor_frames'
    # metadata (role/trigger_type from the before/after pairing) is
    # preserved when available; anchor coverage is a superset check, not a
    # replacement, so every frame_event is still considered even if it
    # wasn't anchor-paired.
    by_path_anchor: Dict[str, SelectedFrame] = {}
    for sf in anchor_frames:
        by_path_anchor.setdefault(sf.path, sf)

    frame_meta: Dict[str, dict] = {}
    for idx, path, ev in frame_events:
        if path in frame_meta:
            continue
        anchor = by_path_anchor.get(path)
        frame_meta[path] = {
            "timestamp": ev.timestamp,
            "application": ev.application,
            "window_title": ev.window_title,
            "role": anchor.role if anchor is not None else "spread",
            "trigger_type": anchor.trigger_type if anchor is not None else "spread",
            "anchor_timestamp": anchor.anchor_timestamp if anchor is not None else None,
            "text_words": significant_words(_event_text_signal(ev)),
        }

    if not frame_meta:
        uncovered: Dict[str, List[str]] = {}
        for (oid, _cidx), text in criterion_text.items():
            uncovered.setdefault(oid, []).append(text)
        return [], uncovered

    result_propagation = _build_result_propagation(events, frame_events)
    augmented_words: Dict[str, Set[str]] = {}
    for path, meta in frame_meta.items():
        words = set(meta["text_words"])
        source_path = result_propagation.get(path)
        if source_path and source_path in frame_meta:
            words |= frame_meta[source_path]["text_words"]
        augmented_words[path] = words

    def _sort_key(entry):
        score, path = entry
        meta = frame_meta[path]
        # Higher relevance first; among ties, prefer a post-change/result
        # frame ("after") over a pre-change one, then the LATER frame (a
        # late-session result state over an earlier, possibly-superseded one).
        role_rank = {"after": 0, "spread": 1, "before": 2}.get(meta["role"], 1)
        ts = meta["timestamp"]
        recency = -(ts.timestamp()) if hasattr(ts, "timestamp") else 0
        return (-score, role_rank, recency)

    # One independent candidate ranking PER CRITERION -- see docstring: this
    # is what lets one outcome receive as many (or as few) frames as its own
    # criteria genuinely need, instead of a fixed per-outcome quota.
    per_criterion_sorted: Dict[Tuple[str, int], List[Tuple[float, str]]] = {}
    uncovered: Dict[str, List[str]] = {}
    for key in criteria_keys:
        c_words = criterion_words[key]
        d_words = criterion_distinctive[key]
        scored = [
            (float(criterion_relevance_score(c_words, d_words, augmented_words[path])), path)
            for path in frame_meta
        ]
        scored = [(s, p) for s, p in scored if s > 0]
        scored.sort(key=_sort_key)
        per_criterion_sorted[key] = scored
        if not scored:
            uncovered.setdefault(key[0], []).append(criterion_text[key])

    # Interleaved round-robin ACROSS CRITERIA (not outcomes): cycle through
    # every criterion bucket repeatedly, each time taking its next-best
    # still-unclaimed candidate, until the budget is exhausted or no bucket
    # has any candidate left -- see _select_relevant_frames docstring for
    # why a single global sort (or a fixed per-outcome cap) is unfair here.
    claimed: Set[str] = set()
    selected: List[SelectedFrame] = []
    idx_by_bucket: Dict[Tuple[str, int], int] = {key: 0 for key in per_criterion_sorted}
    taken_by_bucket: Dict[Tuple[str, int], int] = {key: 0 for key in per_criterion_sorted}

    progressed = True
    while len(selected) < max_frames and progressed:
        progressed = False
        for key in per_criterion_sorted:
            if len(selected) >= max_frames:
                break
            # Cap how many frames any ONE criterion can absorb per round-robin
            # pass, so a criterion with many tied/generic-scoring candidates
            # cannot consume the whole budget while OTHER criteria (including
            # ones with zero candidates, which never "progress" on their own)
            # get nothing at all -- observed live: two criteria with dozens
            # of identically-titled candidate frames repeatedly refilled from
            # the same two buckets until the entire budget was spent, leaving
            # six other criteria with no chance at any remaining slots.
            if taken_by_bucket[key] >= MAX_FRAMES_PER_CRITERION:
                continue
            lst = per_criterion_sorted[key]
            idx = idx_by_bucket[key]
            while idx < len(lst) and lst[idx][1] in claimed:
                idx += 1
            if idx < len(lst):
                score, path = lst[idx]
                oid, _cidx = key
                meta = frame_meta[path]
                matched = sorted(criterion_words[key] & augmented_words[path])
                reason = (
                    f"relevant to outcome '{oid}' criterion \"{criterion_text[key][:70]}\" "
                    f"(matched: {', '.join(matched)})"
                )
                if path in result_propagation:
                    reason += " [paired result frame following a submitted action]"
                sf = SelectedFrame(
                    path=path,
                    timestamp=meta["timestamp"],
                    application=meta["application"],
                    window_title=meta["window_title"],
                    trigger_type=meta["trigger_type"],
                    role=meta["role"],
                    anchor_timestamp=meta["anchor_timestamp"],
                    relevance_score=score,
                    matched_evidence=reason,
                )
                claimed.add(path)
                selected.append(sf)
                idx_by_bucket[key] = idx + 1
                taken_by_bucket[key] += 1
                progressed = True
            else:
                idx_by_bucket[key] = idx

    # Fill any remaining budget with generic even-temporal coverage from
    # whatever is left, so a criterion with no textual signal at all (e.g. a
    # purely visual confirmation with no matching window-title words) still
    # contributes SOME frame to the packet instead of the packet being
    # smaller than the budget allows for no reason. This does NOT change
    # `uncovered` -- a criterion stays reported as uncovered even if a
    # generic filler frame happens to also get attached to the packet,
    # since that filler was not chosen because it evidences that criterion.
    if len(selected) < max_frames:
        remaining_paths = [p for p in frame_meta if p not in claimed]
        remaining_paths.sort(key=lambda p: path_order_lookup(p, frame_events))
        fill_paths = _evenly_sample(remaining_paths, max_frames - len(selected))
        for path in fill_paths:
            meta = frame_meta[path]
            sf = SelectedFrame(
                path=path,
                timestamp=meta["timestamp"],
                application=meta["application"],
                window_title=meta["window_title"],
                trigger_type=meta["trigger_type"],
                role=meta["role"],
                anchor_timestamp=meta["anchor_timestamp"],
                relevance_score=0.0,
                matched_evidence="generic timeline coverage (no criterion match)",
            )
            claimed.add(path)
            selected.append(sf)

    return selected, uncovered


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
