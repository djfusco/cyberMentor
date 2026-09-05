"""Small, generic text-relevance helpers shared by keyframe selection and
evaluation scoring.

Deliberately exercise-agnostic: every function here operates on whatever
words appear in outcome descriptions/criteria/evidence-requirements text and
whatever words appear in captured evidence text. Nothing here knows about any
specific exercise, application, or domain vocabulary (Splunk, Linux, etc.).
"""
from __future__ import annotations

import re
from typing import Iterable, List, Set

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
