"""Local, single-row app settings -- currently just a display name and an
optional course label.

No accounts or auth: this is a convenience label stamped onto sessions and
authored exercises, not an identity/security boundary.
"""
import secrets
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Session

from app.models.session import AnonymousIdentity, AppSettings

SETTINGS_ROW_ID = 1
# Used whenever the student hasn't set an explicit course_id, so anonymous
# export works with zero setup (see get_or_create_anonymous_id).
DEFAULT_COURSE_ID = "default"


def get_settings_row(db: Session) -> AppSettings:
    row = db.get(AppSettings, SETTINGS_ROW_ID)
    if row is None:
        row = AppSettings(id=SETTINGS_ROW_ID)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def get_student_name(db: Session) -> Optional[str]:
    return get_settings_row(db).student_name


def set_student_name(db: Session, name: Optional[str]) -> AppSettings:
    row = get_settings_row(db)
    row.student_name = name.strip() if name else None
    row.updated_at = datetime.now(timezone.utc)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_course_id(db: Session) -> str:
    """Never returns empty -- falls back to DEFAULT_COURSE_ID so callers
    (e.g. anonymous export) never need their own default handling."""
    return get_settings_row(db).course_id or DEFAULT_COURSE_ID


def set_course_id(db: Session, course_id: Optional[str]) -> AppSettings:
    row = get_settings_row(db)
    row.course_id = course_id.strip() if course_id else None
    row.updated_at = datetime.now(timezone.utc)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_or_create_anonymous_id(db: Session, course_id: str) -> str:
    """A random identifier stable for this (student install, course_id) pair.

    Generated with `secrets` (never derived from name/email/OS user/machine),
    and scoped per course_id so the same student is not automatically
    correlatable across unrelated courses -- a different course_id always
    gets its own, independently-random id.
    """
    existing = db.get(AnonymousIdentity, course_id)
    if existing is not None:
        return existing.anonymous_id

    anonymous_id = secrets.token_hex(4).upper()
    row = AnonymousIdentity(course_id=course_id, anonymous_id=anonymous_id)
    db.add(row)
    db.commit()
    return anonymous_id
