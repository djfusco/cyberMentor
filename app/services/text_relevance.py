"""Small, generic text-relevance helpers shared by keyframe selection and
evaluation scoring.

Deliberately exercise-agnostic: every function here operates on whatever
words appear in outcome descriptions/criteria/evidence-requirements text and
whatever words appear in captured evidence text. Nothing here knows about any
specific exercise, application, or domain vocabulary (Splunk, Linux, etc.).
"""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set, Tuple

# Deliberately small and generic -- English function words plus a handful of
# instructional-authoring words ("student", "learner", "session") that would
# otherwise dominate overlap scoring without carrying topic information.
_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "this", "that", "of", "to", "in", "on",
    "for", "and", "or", "with", "student", "learner", "their", "its", "it", "as", "by",
    "during", "session", "already", "e", "g", "be", "at", "from", "not", "if", "one",
    "then", "than", "so", "do", "does", "did", "has", "have", "had", "should", "would",
    "you", "your", "will", "can", "must", "into", "which", "when", "each", "using", "use",
}

_WORD_RE = re.compile(r"[a-z0-9_]+")


def significant_words(text: str) -> Set[str]:
    """Lowercase, tokenize, and drop stopwords/very short tokens.

    Underscore-joined identifiers (e.g. "src_ip") are kept intact as single
    tokens so field-name-shaped criteria match field-name-shaped evidence.
    """
    if not text:
        return set()
    return {
        w for w in _WORD_RE.findall(text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def backtick_literals(text: str) -> List[str]:
    """Extract literal values an author quoted in backticks, e.g. a criterion
    like "the fields `src_ip`, `dst_ip`, and `bytes` appear" -> ["src_ip",
    "dst_ip", "bytes"]. Used to flag exact-value claims that evaluation
    should not accept loosely -- see EVALUATION_SYSTEM_PROMPT's exact-value
    rule. Generic string parsing; no domain vocabulary.
    """
    if not text:
        return []
    return re.findall(r"`([^`]+)`", text)


def outcome_keyword_text(outcome) -> str:
    """Concatenate every author-provided text field that describes what an
    outcome is looking for, for relevance scoring / exact-value extraction.
    Works for both enriched outcomes (explicit criteria) and legacy ones
    (description only) -- never invents text beyond what was authored.
    """
    parts: List[str] = [outcome.description or ""]
    parts.extend(outcome.success_criteria or [])
    parts.extend(outcome.evidence_requirements or [])
    if outcome.student_demonstration:
        parts.append(outcome.student_demonstration)
    return " ".join(parts)


def overlap_score(text_words: Set[str], keyword_words: Set[str]) -> int:
    """Count of significant words shared between a piece of evidence text and
    an outcome's keyword set. A plain count (not a ratio) is deliberate here:
    unlike the narrative-reconciliation check in evaluator.py (which needs a
    ratio to avoid one common word triggering a false match on a short
    sentence), keyframe relevance ranking only needs a monotonic ordering
    across many candidate frames -- more shared, specific terms should always
    rank a frame higher, regardless of how long the outcome's own text is.
    """
    if not text_words or not keyword_words:
        return 0
    return len(text_words & keyword_words)


def text_corpus_words(texts: Iterable[str]) -> Set[str]:
    """Union of significant words across many evidence text fragments --
    used to sanity-check whether an exact-value claim actually appears
    anywhere in the available text evidence."""
    words: Set[str] = set()
    for t in texts:
        words |= significant_words(t)
    return words


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_literal(text: str) -> str:
    """Case- and whitespace-only normalization for exact-value comparison --
    deliberately NOT stemming, fuzzy matching, or synonym awareness: `dst_ip`
    and `dest_ip` must remain distinct, and `bytes` must not match
    `totalSourceBytes`. Used to compare a criterion's own backtick-quoted
    literal against the captured evidence text corpus (see
    EvaluatorService._evaluate_criteria) and against a judgment's
    self-reported quote.
    """
    return _WHITESPACE_RE.sub(" ", (text or "").strip().lower())


# Minimum shared significant words to treat a candidate text as relevant to
# a criterion when NONE of the shared words are DISTINCTIVE to that specific
# criterion (see distinctive_word_sets). Requiring 2+ generic shared words
# is what keeps a single incidental common term (an outcome's own name for
# the application it runs in, or a generic verb like "set"/"search") from
# matching almost every screen in a session -- observed live: an outcome
# whose description happened to mention the host application's name scored
# positively on nearly every window title in a 36-frame session purely via
# that one word, drowning out every other outcome's genuine signal.
RELEVANCE_MIN_GENERIC_OVERLAP = 2


def distinctive_word_sets(word_sets: Sequence[Set[str]]) -> List[Set[str]]:
    """Given the significant-word set for each of several criteria (or any
    other short texts being compared against each other), return, for each
    one, the words that do NOT also appear in any of the OTHERS. A word
    shared by two+ criteria/outcomes carries no discriminating power between
    them (e.g. the application's own name showing up in nearly every
    criterion's evidence_requirements) and must not single-handedly mark a
    piece of evidence as relevant to one specific criterion over another.
    """
    result: List[Set[str]] = []
    for i, words in enumerate(word_sets):
        others: Set[str] = set()
        for j, other in enumerate(word_sets):
            if j != i:
                others |= other
        result.append(words - others)
    return result


def criterion_relevance_score(
    criterion_words: Set[str], distinctive_words: Set[str], candidate_words: Set[str],
) -> int:
    """Relevance of `candidate_words` (e.g. a frame's window title, or a
    model's paraphrased claim) to one criterion. Returns the raw overlap
    count as a score (for ranking), but ONLY when the overlap clears the
    relevance bar: at least RELEVANCE_MIN_GENERIC_OVERLAP shared words, OR
    at least one shared word that is DISTINCTIVE to this criterion (not
    shared with sibling criteria/outcomes) -- see distinctive_word_sets.
    Returns 0 (not relevant) otherwise. Shared by keyframe selection and
    evaluator criterion-matching so both apply the identical discipline.
    """
    overlap = criterion_words & candidate_words
    if not overlap:
        return 0
    if len(overlap) >= RELEVANCE_MIN_GENERIC_OVERLAP or (overlap & distinctive_words):
        return len(overlap)
    return 0


def ocr_text_available(events: Iterable) -> bool:
    """True if the session captured at least one substantive text_observed
    (OCR/AX) event -- i.e. real on-screen text content was actually
    extracted at some point this session, as opposed to only synthesized
    descriptions (window titles, "Clicked at (x, y)", "Scrolled"). Used to
    decide whether ABSENCE of a claimed exact value from the text corpus is
    meaningful evidence (OCR ran and didn't see it) or simply uninformative
    (OCR never produced any real content to check against, e.g. a GUI
    session where on-screen text was never extracted) -- see
    EvaluatorService._evaluate_criteria's exact-value asymmetry handling.
    A short length threshold (20 chars) excludes trivial/empty OCR blips
    (a stray dialog title) from counting as "OCR is working".
    """
    for ev in events:
        if getattr(ev, "type", None) != "text_observed":
            continue
        text = (getattr(ev, "text", None) or "").strip()
        if len(text) >= 20:
            return True
    return False


def build_text_corpus(events: Iterable) -> str:
    """Concatenate every piece of CAPTURED (OCR/AX-sourced) text available
    for exact-value cross-checking: each event's own text, window title, and
    any newly-observed OCR/AX delta lines. Duck-typed (no EvidenceEvent
    import) so this stays a pure, dependency-free text utility.

    This is deliberately the FULL session's text, not scoped to one frame:
    native capture emits window titles/OCR text as separate events from the
    screenshot they describe, so there is no reliable per-frame boundary to
    restrict to -- and over-restricting risks false "not found" rejections
    for evidence that is genuinely present nearby. The corpus is lowercased
    once here; callers normalize their own search term the same way via
    normalize_literal before checking membership.
    """
    parts: List[str] = []
    for ev in events:
        text = getattr(ev, "text", None)
        if text:
            parts.append(text)
        window_title = getattr(ev, "window_title", None)
        if window_title:
            parts.append(window_title)
        delta = getattr(ev, "text_delta", None)
        if delta:
            parts.extend(line for tag, line in delta if tag in ("added", "meaningful_modified"))
    return normalize_literal(" ".join(parts))
