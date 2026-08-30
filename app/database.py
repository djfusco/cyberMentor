"""Database engine setup and per-request session dependency."""
import logging

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings

settings = get_settings()

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, echo=False, connect_args=_connect_args)

_logger = logging.getLogger(__name__)


def _quote(identifier: str) -> str:
    """Quote a SQL identifier so reserved words like `references` are safe."""
    return '"' + identifier.replace('"', '""') + '"'


def _existing_columns(conn, table: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({_quote(table)})")).fetchall()
    return {row[1] for row in rows}


def _migrate_schema() -> None:
    """Add columns present on the models but missing from the live DB.

    SQLModel.metadata.create_all only creates missing tables; it never adds
    columns to an existing table, so a column added to a model after the DB
    file already existed (e.g. ExerciseSession.student_difficulty,
    AppSettings.course_id) never gets applied and the ORM raises
    'no such column' on first query. This reflects every registered model
    against the live schema and ALTER TABLEs any missing column. Each column
    is added independently so one failure can't block the rest; a NOT NULL
    column with no default can't be back-filled this way and is logged.
    """
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            existing = _existing_columns(conn, table.name)
            if not existing:
                # create_all already made the whole table -- nothing to alter.
                continue
            for column in table.columns:
                if column.name in existing:
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                try:
                    conn.execute(
                        text(
                            f"ALTER TABLE {_quote(table.name)} "
                            f"ADD COLUMN {_quote(column.name)} {col_type}"
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    _logger.warning(
                        "could not auto-add column %s.%s (%s); "
                        "it may need a manual migration: %s",
                        table.name, column.name, col_type, exc,
                    )


def init_db() -> None:
    """Create tables for all registered SQLModel models."""
    # Import models so their tables are registered on SQLModel.metadata
    from app.models import authoring as _authoring_models  # noqa: F401
    from app.models import evaluation as _evaluation_models  # noqa: F401
    from app.models import mentor as _mentor_models  # noqa: F401
    from app.models import reference as _reference_models  # noqa: F401
    from app.models import session as _session_models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    _migrate_schema()


def get_session():
    with Session(engine) as session:
        yield session
