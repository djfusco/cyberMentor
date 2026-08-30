"""Reusable reference library: authoritative documents/URLs an instructor
attaches to exercise authoring (software docs, process guidelines,
frameworks like MITRE ATT&CK).

References are files on disk (original + extracted text) with a DB row for
metadata/status -- the same pattern exercises use (files as the source of
truth, DB rows for runtime bookkeeping). See app/services/references.py.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel

from app.models.session import utcnow


class ReferenceCategory(str, Enum):
    software_documentation = "software_documentation"
    process_guideline = "process_guideline"
    framework = "framework"
    other = "other"


class ReferenceSourceType(str, Enum):
    uploaded_document = "uploaded_document"
    url_document = "url_document"
    website = "website"


class ExtractionStatus(str, Enum):
    pending = "pending"
    ok = "ok"
    failed = "failed"


class Reference(SQLModel, table=True):
    """One library entry. Original file lives at

    reference_library/<id>/original.<ext>

    and its extracted plain text at

    reference_library/<id>/extracted.txt
    """

    __tablename__ = "references"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    category: ReferenceCategory
    source_type: ReferenceSourceType
    original_filename: Optional[str] = None
    source_url: Optional[str] = None
    file_extension: Optional[str] = None
    content_hash: Optional[str] = None
    extraction_status: ExtractionStatus = Field(default=ExtractionStatus.pending)
    extraction_error: Optional[str] = None
    # Rough token estimate of the extracted text, used to decide whether a
    # summarization pass is needed before including it in authoring prompts.
    token_estimate: int = 0
    added_by: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class AuthoringSessionReference(SQLModel, table=True):
    """Join table: which library references are attached to a given
    in-progress authoring conversation. This -- not the library itself --
    is what actually feeds the authoring/finalize prompts."""

    __tablename__ = "authoring_session_references"

    id: Optional[int] = Field(default=None, primary_key=True)
    authoring_session_id: int = Field(index=True, foreign_key="exercise_authoring_sessions.id")
    reference_id: int = Field(index=True, foreign_key="references.id")
    created_at: datetime = Field(default_factory=utcnow)
