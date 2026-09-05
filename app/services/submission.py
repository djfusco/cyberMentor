"""Signed submission export/import.

Best-effort tamper-evidence, not cryptographic proof: the HMAC is computed
using a signing_key embedded in the exercise file itself, on the student's
own machine. A student who is willing to modify their own local app (rather
than just hand-edit the exported file) could still fabricate a validly
signed submission. Real protection against that requires a server that
witnesses submission live, which is out of scope for this local-only app.
What this DOES catch: casual/accidental edits to an already-exported file
before it reaches the instructor.
"""
import hashlib
import hmac
import json
from datetime import datetime
from typing import Any, Dict, Optional

from app.models.evaluation import EvaluationResult, Submission
from app.models.exercise import Exercise
from app.models.session import ExerciseSession
from app.services.exercises import ExerciseService

SUBMISSION_FORMAT_VERSION = 1


def _canonical_payload(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sign(payload: Dict[str, Any], signing_key: str) -> str:
    mac = hmac.new(signing_key.encode("utf-8"), _canonical_payload(payload), hashlib.sha256)
    return mac.hexdigest()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def build_submission_export(
    session: ExerciseSession, evaluation: EvaluationResult, exercise: Exercise
) -> Dict[str, Any]:
    """Build the JSON bundle a student downloads to share with an instructor."""
    payload: Dict[str, Any] = {
        "format_version": SUBMISSION_FORMAT_VERSION,
        "exercise_id": exercise.id,
        "exercise_title": exercise.title,
        "student_name": session.student_name,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        # Top-level, easy-to-spot flag for an instructor scanning the raw
        # export -- the full detail is also in evaluation.ai_unavailable
        # (nested below), but a reviewer should not have to know that field
        # exists just to notice this attempt's score is not a real result.
        "evaluation_status": "unavailable" if evaluation.ai_unavailable else "complete",
        "evaluation": evaluation.model_dump(mode="json"),
    }

    bundle = dict(payload)
    if exercise.signing_key:
        bundle["signature"] = _sign(payload, exercise.signing_key)
        bundle["signature_note"] = "HMAC-SHA256, signed with this exercise's signing_key."
    else:
        bundle["signature"] = None
        bundle["signature_note"] = (
            "This exercise has no signing_key (hand-written exercises aren't signable). "
            "The instructor will not be able to verify this result was not edited after export."
        )
    return bundle


def import_submission(
    raw_bytes: bytes, exercise_service: ExerciseService, source_filename: Optional[str] = None
) -> Submission:
    """Parse an uploaded submission bundle and verify its signature if possible.

    Returns an unsaved Submission row -- the caller is responsible for
    adding/committing it to the database.
    """
    try:
        bundle = json.loads(raw_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Not a valid submission file: {exc}") from exc

    if not isinstance(bundle, dict) or "evaluation" not in bundle:
        raise ValueError("Not a recognized submission bundle format")

    exercise_id = bundle.get("exercise_id") or ""
    exercise = exercise_service.get_exercise(exercise_id) if exercise_id else None

    signature = bundle.get("signature")
    payload = {k: v for k, v in bundle.items() if k not in ("signature", "signature_note")}

    signature_valid: Optional[bool]
    if exercise is None:
        signature_valid = None
        signature_note = (
            f"Exercise '{exercise_id}' is not available locally, so the signature could not be verified."
        )
    elif not exercise.signing_key:
        signature_valid = None
        signature_note = "This exercise has no signing_key on this machine; signature could not be verified."
    elif not signature:
        signature_valid = False
        signature_note = "Submission has no signature."
    else:
        expected = _sign(payload, exercise.signing_key)
        signature_valid = hmac.compare_digest(expected, signature)
        signature_note = (
            "Signature verified -- matches this exercise's signing_key."
            if signature_valid
            else "Signature does NOT match -- this file may have been edited after export."
        )

    evaluation_data = bundle.get("evaluation") or {}
    try:
        evaluation = EvaluationResult.model_validate(evaluation_data)
    except Exception as exc:  # noqa: BLE001 -- surface any validation failure as an import error
        raise ValueError(f"Submission's evaluation data is malformed: {exc}") from exc

    return Submission(
        exercise_id=exercise_id,
        exercise_title=bundle.get("exercise_title") or (exercise.title if exercise else exercise_id),
        student_name=bundle.get("student_name"),
        started_at=_parse_dt(bundle.get("started_at")),
        ended_at=_parse_dt(bundle.get("ended_at")),
        score=evaluation.score,
        summary=evaluation.summary,
        evaluation_json=evaluation.model_dump_json(),
        signature_valid=signature_valid,
        signature_note=signature_note,
        source_filename=source_filename,
    )
