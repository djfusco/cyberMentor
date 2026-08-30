"""Pydantic models describing the YAML exercise definition schema.

These are distinct from the SQLModel database tables in session.py/evaluation.py:
exercises are static, file-based definitions, not runtime/session state.
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class OutcomeType(str, Enum):
    filesystem = "filesystem"
    observed_behavior = "observed_behavior"
    process = "process"


class CheckKind(str, Enum):
    """Supported declarative, read-only filesystem checks.

    Lets exercises created without a hand-written Python Verifier class
    (e.g. via the chat-based authoring flow) still get deterministic
    verification. See app/services/verifier.py:GenericDeclarativeVerifier.
    """

    dir_exists = "dir_exists"
    file_exists = "file_exists"
    file_mode = "file_mode"
    file_contains = "file_contains"
    file_not_contains = "file_not_contains"


class CheckSpec(BaseModel):
    kind: CheckKind
    path: str
    expected_mode: Optional[str] = None
    pattern: Optional[str] = None


class ExpectedOutcome(BaseModel):
    id: str
    description: str
    type: OutcomeType
    weight: float
    check: Optional[CheckSpec] = None


class EnvironmentType(str, Enum):
    terminal = "terminal"
    gui = "gui"
    siem = "siem"


class ExerciseEnvironment(BaseModel):
    type: EnvironmentType = EnvironmentType.terminal


class DifficultyLevel(str, Enum):
    """Instructor-selected difficulty for an exercise.

    "open" means the instructor did not fix a level -- the student picks
    beginner/intermediate/advanced for their own session instead (see
    ExerciseSession.student_difficulty and app.services.mentor._effective_difficulty).
    """

    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"
    open = "open"


class MentorConfig(BaseModel):
    allow_questions: bool = True
    default_help_level: int = 2
    reveal_answer: bool = False

    @field_validator("default_help_level")
    @classmethod
    def _validate_level(cls, v: int) -> int:
        if not 0 <= v <= 5:
            raise ValueError("default_help_level must be between 0 and 5")
        return v


class ExerciseStep(BaseModel):
    """One sub-task of a multi-step exercise (e.g. "log into Splunk and run a search").

    A "simple", single-task exercise is just an exercise with no `steps` at
    all -- see Exercise.get_steps(), which normalizes both shapes for the
    evaluator/UI so nothing downstream needs to special-case either form.
    """

    id: str
    title: str
    instructions: str
    expected_outcomes: List[ExpectedOutcome]

    @property
    def weight(self) -> float:
        return sum(o.weight for o in self.expected_outcomes)


class ExerciseSourceType(str, Enum):
    """What informed an exercise, recorded for instructor confidence/audit --
    not a runtime dependency of the finished exercise (see app/services/
    exercise_author.py and app/services/references.py).
    """

    frontier_research = "frontier_research"
    instructor_reference = "instructor_reference"


class ExerciseSource(BaseModel):
    type: ExerciseSourceType
    title: str
    detail: str = ""


class Exercise(BaseModel):
    id: str
    title: str
    description: str
    instructions: str
    environment: ExerciseEnvironment
    # Instructor-chosen difficulty. Defaults to intermediate for exercises
    # written before this field existed. Never inferred/calculated -- always
    # a manual instructor choice (or "open", deferring to the student).
    difficulty: DifficultyLevel = DifficultyLevel.intermediate
    # Legacy/simple form: a flat list of outcomes for a single-task exercise.
    # Mutually exclusive with `steps` -- use exactly one. Prefer `get_steps()`
    # / `get_all_outcomes()` over reading either field directly.
    expected_outcomes: List[ExpectedOutcome] = Field(default_factory=list)
    # Multi-step form: an ordered sequence of sub-tasks, each with its own
    # instructions and outcomes. Optional for backward compatibility with
    # every exercise written before this existed.
    steps: Optional[List[ExerciseStep]] = None
    acceptable_methods: List[str] = Field(default_factory=list)
    prohibited_behaviors: List[str] = Field(default_factory=list)
    mentor: MentorConfig = Field(default_factory=MentorConfig)
    # Present on exercises authored via the chat flow (app/services/exercise_author.py).
    # Used as the HMAC key when signing/verifying submission exports -- see
    # app/services/submission.py. Absent on hand-written exercises, in which
    # case their submissions simply can't be signed/verified.
    signing_key: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    # What informed this exercise -- frontier research queries and/or
    # instructor-provided reference materials -- so an instructor can see
    # later exactly what grounded it. Documents/content themselves are not
    # bundled; this is a citation record, not a runtime dependency.
    sources: List[ExerciseSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_outcomes_or_steps(self) -> "Exercise":
        if self.steps:
            if self.expected_outcomes:
                raise ValueError(
                    "exercise must not define both 'steps' and top-level 'expected_outcomes' -- use one or the other"
                )
            for step in self.steps:
                if not step.expected_outcomes:
                    raise ValueError(f"step '{step.id}' must define at least one expected outcome")
        elif not self.expected_outcomes:
            raise ValueError("exercise must define at least one expected outcome (or use 'steps')")
        return self

    def get_steps(self) -> List[ExerciseStep]:
        """Normalize both single-task and multi-step exercises into one uniform step list."""
        if self.steps:
            return self.steps
        return [
            ExerciseStep(
                id="_default",
                title=self.title,
                instructions=self.instructions,
                expected_outcomes=self.expected_outcomes,
            )
        ]

    def get_all_outcomes(self) -> List[ExpectedOutcome]:
        return [outcome for step in self.get_steps() for outcome in step.expected_outcomes]

    @property
    def total_weight(self) -> float:
        return sum(o.weight for o in self.get_all_outcomes())
