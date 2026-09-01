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
import os
import re
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel

from app.models.exercise import CheckKind, CheckSpec, Exercise, OutcomeType


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
