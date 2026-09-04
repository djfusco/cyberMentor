"""Deterministic, read-only verification of objective exercise outcomes.

The LLM must never override these results for outcomes they cover. Verifiers
only ever read filesystem state -- they never create, modify, or delete
anything.

Two ways an outcome gets deterministic verification:
1. A hand-written Python `Verifier` class registered by exercise ID below
   (an escape hatch for verification too exotic for the declarative DSL).
2. A declarative `check` block on the outcome itself (see
   app/models/exercise.py:CheckSpec), executed generically by
   GenericDeclarativeVerifier. This is what lets exercises authored via the
   chat flow (app/services/exercise_author.py) get real verification without
   anyone writing Python for them.
"""
import glob
import json
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel

from app.models.evaluation import EvidenceBasis
from app.models.exercise import CheckKind, CheckSpec, Exercise, OutcomeType
from app.services.evidence import EvidenceEvent

logger = logging.getLogger(__name__)


class VerificationDetail(BaseModel):
    """The generic result shape every verifier must return per outcome ID."""

    passed: bool
    actual: Any = None
    expected: Any = None
    note: str = ""


class Verifier(Protocol):
    def verify(self, exercise: Exercise) -> Dict[str, VerificationDetail]:
        ...


def _mode_str(path: Path) -> str:
    mode = stat.S_IMODE(os.stat(path).st_mode)
    return format(mode, "03o")


def _glob_matches(path: Path) -> List[Path]:
    """Expand shell-style wildcards in a path to the entries that exist.

    Only called when glob.has_magic reports wildcards in the (already
    ~-expanded) path -- literal paths are checked directly via is_file()/
    is_dir() and never globbed, so a real filename that happens to contain
    glob metacharacters is never accidentally treated as a pattern.
    """
    return [Path(p) for p in glob.glob(str(path))]


class LinuxFilePermissionsVerifier:
    """Read-only verifier for the linux-file-permissions-001 exercise."""

    directory_name = "secure-data"
    file_name = "secrets.txt"
    expected_mode = "600"

    def verify(self, exercise: Exercise) -> Dict[str, VerificationDetail]:
        home = Path.home()
        directory = home / self.directory_name
        file_path = directory / self.file_name

        directory_created = directory.is_dir()
        file_created = file_path.is_file()

        actual_permissions: Optional[str] = None
        permissions_correct = False
        if file_created:
            try:
                actual_permissions = _mode_str(file_path)
                permissions_correct = actual_permissions == self.expected_mode
            except OSError:
                actual_permissions = None

        if permissions_correct:
            permissions_note = f"Mode is {actual_permissions}."
        elif actual_permissions is None:
            permissions_note = "File does not exist, so permissions could not be checked."
        else:
            permissions_note = f"Mode is {actual_permissions}, expected {self.expected_mode}."

        return {
            "directory_created": VerificationDetail(
                passed=directory_created,
                actual=directory_created,
                expected=True,
                note="Directory exists." if directory_created else "Directory does not exist.",
            ),
            "file_created": VerificationDetail(
                passed=file_created,
                actual=file_created,
                expected=True,
                note="File exists." if file_created else "File does not exist.",
            ),
            "permissions_correct": VerificationDetail(
                passed=permissions_correct,
                actual=actual_permissions,
                expected=self.expected_mode,
                note=permissions_note,
            ),
        }


# -- Structured evidence facts (terminal-evidence corroboration) -------------
#
# A deterministic Verifier reads filesystem state; it says nothing about
# captured activity. The complementary problem this section solves: some
# terminal-evidence facts (a permission-changing command ran, a long-format
# permission listing was produced) are reliably recognizable from their
# STRUCTURE, without needing an LLM to re-derive them from noisy OCR text
# every time. Extracting them once, deterministically, with provenance (what
# text produced the fact) means the LLM is no longer the *only* authority for
# facts this reliable -- see EvaluatorService._score_from_judgment, which
# treats a present, action-basis fact as sufficient corroborating evidence on
# its own, corroborating rather than replacing the LLM's judgment for
# everything else (screenshots, less common commands, GUI actions, ...).
#
# Deliberately NOT a shell parser: each fact is one narrow, explainable
# regex-based structural check, tolerant of realistic OCR noise the way a
# human skimming a blurry screenshot would be, never a rewrite of the
# underlying text (nothing here ever mutates captured evidence).


class EvidenceFact(BaseModel):
    """One structurally-derived fact about captured evidence, with the
    outcome it corroborates and the text it came from (provenance)."""

    key: str
    outcome_id: str
    present: bool
    detail: str
    basis: EvidenceBasis


class EvidenceFactExtractor(Protocol):
    def extract(
        self,
        exercise: Exercise,
        events: List[EvidenceEvent],
        verification: Dict[str, VerificationDetail],
    ) -> List[EvidenceFact]:
        ...


class LinuxPermissionEvidenceFacts:
    """Structural evidence extraction paired with LinuxFilePermissionsVerifier
    -- recognizes the SHAPE of a `chmod`-style mode-changing command and the
    SHAPE of a long-format permission listing (`ls -l`/`stat`/`getfacl`-style
    output) in captured terminal text, independent of exact OCR fidelity.

    Narrow by design: one command family (chmod) and one output shape
    (long-format listing), each a single small regex -- not a general shell
    parser. Only registered for linux-file-permissions-001-shaped exercises
    (see _EVIDENCE_FACT_EXTRACTORS below); every other exercise gets zero
    facts and is entirely unaffected (evaluation falls back to plain LLM
    judgment, exactly as before this module existed).
    """

    # A chmod invocation with an octal or symbolic mode argument. Matches the
    # command family, not any particular file/path -- works for any chmod
    # call, on any exercise that has one.
    _CHMOD_RE = re.compile(r"\bchmod\s+([0-7]{3,4}|[ugoa]*[+\-=][rwxXst]+)\b", re.IGNORECASE)

    # The first column of `ls -l`/similar output: a file-type character
    # followed by nine permission-bit characters. Tolerant of OCR noise in
    # the permission bits themselves (any letter or dash) since the
    # surrounding structural checks (total/date/size, in _has_listing_shape)
    # are what actually establish this is a listing, not the exact
    # characters of this one token. Restricted to the file-type characters
    # ('-', 'd', 'l' -- regular file, directory, symlink) that actually occur
    # in this lab's own realistic output, rather than the full ls(1) set --
    # the wider set (p/s/c/b) collided with ordinary words (e.g. "secure"
    # starts with 's', an ls(1) socket indicator).
    _MODE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[-dl][A-Za-z\-]{9}(?![A-Za-z0-9])")
    _TOTAL_RE = re.compile(r"\btotal\b", re.IGNORECASE)
    _MONTH_RE = re.compile(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", re.IGNORECASE
    )
    _TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
    _SIZE_RE = re.compile(r"(?<![\d.])\d+(?![\d.])")

    def extract(
        self,
        exercise: Exercise,
        events: List[EvidenceEvent],
        verification: Dict[str, VerificationDetail],
    ) -> List[EvidenceFact]:
        facts = [self._find_chmod_action(events)]
        # Only worth looking for a permission listing if this exercise
        # actually has a mode-correctness outcome to corroborate.
        if "permissions_correct" in verification:
            facts.append(self._find_permission_listing(events))
        return facts

    def _find_chmod_action(self, events: List[EvidenceEvent]) -> EvidenceFact:
        for event in events:
            match = self._CHMOD_RE.search(event.text)
            if match:
                return EvidenceFact(
                    key="permission_change_action_observed",
                    outcome_id="permissions_correct",
                    present=True,
                    detail=f'Captured text shows a chmod-style command: "{match.group(0).strip()}".',
                    basis=EvidenceBasis.action,
                )
        return EvidenceFact(
            key="permission_change_action_observed",
            outcome_id="permissions_correct",
            present=False,
            detail="No chmod-style (or equivalent mode-changing) command found in captured text.",
            basis=EvidenceBasis.unclear,
        )

    def _find_permission_listing(self, events: List[EvidenceEvent]) -> EvidenceFact:
        for event in events:
            if self._has_listing_shape(event.text):
                return EvidenceFact(
                    key="permission_listing_observed",
                    outcome_id="permissions_verified",
                    present=True,
                    detail=(
                        "Captured text has a 'total' header, a permission-mode-shaped token, and "
                        "size/date information together -- structurally a long-format permission "
                        "listing (ls -l, stat, getfacl, or equivalent), regardless of exact OCR "
                        "fidelity on the command or token text."
                    ),
                    basis=EvidenceBasis.action,
                )
        return EvidenceFact(
            key="permission_listing_observed",
            outcome_id="permissions_verified",
            present=False,
            detail="No long-format permission-listing structure found in captured text.",
            basis=EvidenceBasis.unclear,
        )

    def _has_listing_shape(self, text: str) -> bool:
        """A long-format listing needs a 'total' header, a permission-shaped
        token, AND (a date-like token or a size-like number) -- requiring
        several independent structural cues together is what keeps this from
        false-matching plain filename lists (e.g. genuine `ls -1` output,
        which has none of these) or unrelated text that merely contains one
        of these cues in isolation.
        """
        if not self._TOTAL_RE.search(text):
            return False
        if not self._MODE_TOKEN_RE.search(text):
            return False
        has_date = bool(self._MONTH_RE.search(text) or self._TIME_RE.search(text))
        has_size = bool(self._SIZE_RE.search(text))
        return has_date or has_size


_EVIDENCE_FACT_EXTRACTORS: Dict[str, EvidenceFactExtractor] = {
    "linux-file-permissions-001": LinuxPermissionEvidenceFacts(),
}


def extract_evidence_facts(
    exercise: Exercise,
    events: List[EvidenceEvent],
    verification: Dict[str, VerificationDetail],
) -> List[EvidenceFact]:
    """Structural evidence facts for this exercise, or [] when none are
    registered -- evaluation is unaffected either way (see
    EvaluatorService._score_from_judgment)."""
    extractor = _EVIDENCE_FACT_EXTRACTORS.get(exercise.id)
    if extractor is None:
        return []
    return extractor.extract(exercise, events, verification)


class GenericDeclarativeVerifier:
    """Executes each filesystem-type outcome's declarative `check` block.

    Used as the default verifier for any exercise that doesn't have a
    bespoke Python Verifier class registered below.
    """

    def verify(self, exercise: Exercise) -> Dict[str, VerificationDetail]:
        results: Dict[str, VerificationDetail] = {}
        # Exercise.get_all_outcomes() normalizes both single-task
        # (expected_outcomes) and multi-step (steps[].expected_outcomes)
        # exercises -- reading expected_outcomes directly here would silently
        # skip every outcome of a multi-step exercise, since steps and
        # expected_outcomes are mutually exclusive (see Exercise model_validator).
        for outcome in exercise.get_all_outcomes():
            if outcome.type != OutcomeType.filesystem or outcome.check is None:
                continue
            results[outcome.id] = self._run_check(outcome.check)
        return results

    def _run_check(self, check: CheckSpec) -> VerificationDetail:
        path = Path(check.path).expanduser()
        try:
            if check.kind == CheckKind.dir_exists:
                if glob.has_magic(str(path)):
                    matches = [p for p in _glob_matches(path) if p.is_dir()]
                    passed = bool(matches)
                    note = (
                        f"{path} matched {len(matches)} director{'y' if len(matches) == 1 else 'ies'}."
                        if passed
                        else f"{path} matched no directories."
                    )
                    return VerificationDetail(passed=passed, actual=passed, expected=True, note=note)
                passed = path.is_dir()
                return VerificationDetail(
                    passed=passed,
                    actual=passed,
                    expected=True,
                    note=f"{path} {'exists' if passed else 'does not exist'} as a directory.",
                )

            if check.kind == CheckKind.file_exists:
                if glob.has_magic(str(path)):
                    matches = [p for p in _glob_matches(path) if p.is_file()]
                    passed = bool(matches)
                    if passed:
                        names = ", ".join(p.name for p in matches[:5])
                        note = f"{path} matched {len(matches)} file(s): {names}."
                    else:
                        note = f"{path} matched no files."
                    return VerificationDetail(passed=passed, actual=passed, expected=True, note=note)
                passed = path.is_file()
                return VerificationDetail(
                    passed=passed,
                    actual=passed,
                    expected=True,
                    note=f"{path} {'exists' if passed else 'does not exist'} as a file.",
                )

            if check.kind == CheckKind.file_mode:
                if not path.is_file():
                    return VerificationDetail(
                        passed=False,
                        actual=None,
                        expected=check.expected_mode,
                        note=f"{path} does not exist, so its permissions could not be checked.",
                    )
                actual_mode = _mode_str(path)
                passed = actual_mode == check.expected_mode
                note = (
                    f"Mode is {actual_mode}."
                    if passed
                    else f"Mode is {actual_mode}, expected {check.expected_mode}."
                )
                return VerificationDetail(
                    passed=passed, actual=actual_mode, expected=check.expected_mode, note=note
                )

            if check.kind in (CheckKind.file_contains, CheckKind.file_not_contains):
                if not path.is_file():
                    return VerificationDetail(
                        passed=False,
                        actual=None,
                        expected=check.pattern,
                        note=f"{path} does not exist, so its contents could not be checked.",
                    )
                text = path.read_text(errors="replace")
                matched = bool(check.pattern) and re.search(check.pattern, text) is not None
                if check.kind == CheckKind.file_contains:
                    note = (
                        f"{path} contains a match for the expected pattern."
                        if matched
                        else f"{path} does not contain a match for the expected pattern."
                    )
                    return VerificationDetail(passed=matched, actual=matched, expected=True, note=note)
                passed = not matched
                note = (
                    f"{path} does not contain the prohibited pattern."
                    if passed
                    else f"{path} contains the prohibited pattern."
                )
                return VerificationDetail(passed=passed, actual=matched, expected=False, note=note)
        except OSError as exc:
            return VerificationDetail(passed=False, actual=None, expected=None, note=f"Could not check {path}: {exc}")

        return VerificationDetail(passed=False, actual=None, expected=None, note="Unsupported check kind.")


_VERIFIERS: Dict[str, Verifier] = {
    "linux-file-permissions-001": LinuxFilePermissionsVerifier(),
}
_GENERIC_VERIFIER = GenericDeclarativeVerifier()


def get_verifier(exercise_id: str) -> Optional[Verifier]:
    return _VERIFIERS.get(exercise_id)


def run_verification(exercise: Exercise) -> Dict[str, VerificationDetail]:
    """Return deterministic verification results, keyed by outcome ID.

    Prefers a bespoke registered Verifier if one exists for this exercise;
    otherwise falls back to the generic declarative check interpreter.
    Returns an empty dict (never None) when nothing is verifiable.
    """
    verifier = get_verifier(exercise.id)
    if verifier is not None:
        return verifier.verify(exercise)
    return _GENERIC_VERIFIER.verify(exercise)


# -- Session-start baseline (stale-state attribution) -------------------------
#
# A deterministic check only ever proves a final state exists; it says
# nothing about *when* that state was produced. Without a baseline, a
# student who leaves compliant files in place from a prior attempt gets full
# credit on a new session that did nothing at all. Capturing verification
# once at session start (see app.routes.sessions.create_session) and
# comparing it against the same check at Finish is what lets the evaluator
# tell "produced this session" apart from "already there before this
# session began" -- see EvaluatorService._score_filesystem_outcome.


def serialize_verification(verification: Dict[str, VerificationDetail]) -> str:
    """JSON-encode a verification result set for storage on ExerciseSession
    (baseline_verification_json). Inverse of deserialize_verification."""
    return json.dumps({key: detail.model_dump() for key, detail in verification.items()})


def deserialize_verification(raw: Optional[str]) -> Dict[str, VerificationDetail]:
    """Decode a verification result set previously stored by
    serialize_verification. Never raises: malformed or absent data (e.g. a
    session created before this field existed) yields {}, which callers
    treat as "no baseline available" -- see needs_llm_attribution and
    EvaluatorService._score_filesystem_outcome.
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Could not decode stored baseline verification: %s", exc)
        return {}
    result: Dict[str, VerificationDetail] = {}
    for key, value in data.items():
        try:
            result[key] = VerificationDetail(**value)
        except (TypeError, ValueError) as exc:
            logger.warning("Could not decode baseline verification entry %r: %s", key, exc)
    return result


def needs_llm_attribution(
    outcome_id: str,
    verification: Dict[str, VerificationDetail],
    baseline_verification: Optional[Dict[str, VerificationDetail]],
) -> bool:
    """True when a deterministically-checked outcome's current PASS does not,
    by itself, prove the student produced it *this* session -- because the
    same compliant state was already present in the baseline captured at
    session start. In that case only current-session evidence (an LLM
    judgment over captured activity, same as any non-deterministic outcome)
    can still earn credit.

    False whenever determinism alone already settles the outcome: no
    verification entry, a failing check (never overridable -- see
    EvaluatorService._score_filesystem_outcome), or a fail-at-baseline/no-baseline case
    where the pass represents a real transition (or no baseline was
    recorded at all, e.g. a session created before this field existed --
    treated permissively, matching prior behavior).
    """
    detail = verification.get(outcome_id)
    if detail is None or not detail.passed:
        return False
    baseline_detail = (baseline_verification or {}).get(outcome_id)
    return baseline_detail is not None and baseline_detail.passed
