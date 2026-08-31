"""All LLM prompt construction lives here, kept out of route handlers."""
import logging
from typing import Dict, List, Optional, Tuple

from app.models.exercise import Exercise
from app.services.evidence import EvidenceEvent
from app.services.verifier import VerificationDetail

logger = logging.getLogger(__name__)

HELP_LEVEL_DESCRIPTIONS = {
    0: "observation only -- describe what you observe, offer no guidance",
    1: "confirm whether the learner's direction looks reasonable",
    2: "offer a conceptual hint",
    3: "offer a specific hint",
    4: "recommend a concrete next action",
    5: "reveal and explain the full solution",
}

MENTOR_SYSTEM_PROMPT_TEMPLATE = """You are an AI tutor observing a learner completing a technical exercise.

You know:
- the exercise instructions,
- the expected outcomes,
- deterministic verification results,
- observed activity from the learner's screen.

When screenshots of the learner's screen are attached to the student's
message, treat them as the primary source of truth for what the learner has
actually done. The text activity timeline is supplementary -- it records
window titles, click coordinates, and key-count summaries (no element names
or typed content), and OCR may be garbled. Do not tell the learner you
cannot see evidence of a step if a screenshot shows it.

Your goal is to help the learner reason through the exercise.

{difficulty_instructions}

Do not assume an exact sequence of steps is required. Multiple technically
correct approaches may exist.

Prefer hints over giving away the full solution. Current help level: {help_level}
({help_level_description}).

Do not claim the learner performed an action unless the evidence supports it.
Clearly distinguish between: observed, verified, inferred, and unknown.
If evidence is insufficient, say so plainly.

Do not execute actions for the learner, and do not write out a full command
solution unless "reveal full solution" is explicitly true below.
Reveal full solution: {reveal_answer}
"""

# The amount of assistance given depends on the effective difficulty (the
# instructor's chosen difficulty, or the student's own choice for an Open
# exercise -- see app.services.mentor._effective_difficulty). Keyed by
# DifficultyLevel value; "open" itself never reaches this dict since it is
# always resolved to one of the three below first.
DIFFICULTY_ASSISTANCE_INSTRUCTIONS = {
    "beginner": (
        "Difficulty level: Beginner. Be supportive and instructional. You may "
        "explain concepts, suggest the next step, provide specific commands or "
        "command examples when useful, explain what a command does, and help "
        "the student recover from mistakes."
    ),
    "intermediate": (
        "Difficulty level: Intermediate. Provide guidance and hints without "
        "immediately solving the task. Prefer explanations, hints, questions, "
        "and pointing the student toward the next step. Avoid giving the "
        "complete command or solution unless the student clearly needs "
        "additional help."
    ),
    "advanced": (
        "Difficulty level: Advanced. Provide minimal assistance. Prefer basic "
        "feedback, identifying that something appears incorrect, asking "
        "questions that make the student reason through the problem, and "
        "confirming whether their approach is reasonable. Do not normally "
        "provide exact commands, exact steps, or the solution. The goal is to "
        "make the student solve the exercise themselves."
    ),
}


# Capture allows up to 10,000 characters of raw AX/OCR text per event
# (native_capture/Sources/MentorCapture/Models.swift:CaptureConfig.
# maxTextLength). Sending that verbatim for dozens of events is what
# produced an oversized (~60k token) evaluation prompt in practice. Cap
# what actually reaches the prompt per event, keeping the head (usually the
# most identifying content, e.g. a command or page title) and tail
# (usually the latest/most relevant state) and dropping the noisy middle.
MAX_EVENT_TEXT_CHARS = 500
_EVENT_TEXT_HEAD_CHARS = 350
_EVENT_TEXT_TAIL_CHARS = 120


def _truncate_event_text(text: str, application: str, event_type: str) -> Tuple[str, bool]:
    if len(text) <= MAX_EVENT_TEXT_CHARS:
        return text, False
    omitted = len(text) - _EVENT_TEXT_HEAD_CHARS - _EVENT_TEXT_TAIL_CHARS
    truncated = (
        f"{text[:_EVENT_TEXT_HEAD_CHARS]}...[{omitted} chars omitted]...{text[-_EVENT_TEXT_TAIL_CHARS:]}"
    )
    logger.info(
        "Truncated %s event text for %s from %d to %d chars before prompting",
        event_type, application, len(text), len(truncated),
    )
    return truncated, True


def format_evidence_timeline(
    events: List[EvidenceEvent], max_events: int = 40, evidence_error: Optional[str] = None
) -> Tuple[str, bool]:
    if evidence_error:
        return (
            f"(EVIDENCE RETRIEVAL FAILED -- this is NOT evidence the learner did nothing, "
            f"it means the capture system could not be reached: {evidence_error})",
            False,
        )
    if not events:
        return (
            "(no activity was captured for this session -- the capture system was reachable but returned nothing)",
            False,
        )
    recent = events[-max_events:]
    lines = []
    truncated_any = False
    for e in recent:
        text, was_truncated = _truncate_event_text(e.text, e.application, e.type)
        truncated_any = truncated_any or was_truncated
        location = f"{e.application} @ {e.browser_url}" if e.browser_url else e.application
        lines.append(f"- [{e.timestamp.isoformat()}] ({location}) {text}")
    return "\n".join(lines), truncated_any


def format_verification(verification: Optional[Dict[str, VerificationDetail]]) -> str:
    if not verification:
        return "(no deterministic verification available)"
    lines = []
    for key, detail in verification.items():
        status = "PASS" if detail.passed else "FAIL"
        lines.append(f"- {key}: {status} -- {detail.note}")
    return "\n".join(lines)


def format_outcomes_by_step(exercise: Exercise) -> str:
    """Render expected outcomes grouped by step for multi-step exercises,
    or as a flat list for single-task exercises (Exercise.get_steps()
    normalizes both into the same shape).
    """
    steps = exercise.get_steps()
    is_multi_step = exercise.steps is not None
    blocks = []
    for step in steps:
        outcome_lines = "\n".join(
            f"  - {o.id} (weight {o.weight}, type {o.type.value}): {o.description}"
            for o in step.expected_outcomes
        )
        if is_multi_step:
            blocks.append(f"Step '{step.title}':\n{step.instructions}\n{outcome_lines}")
        else:
            blocks.append(outcome_lines)
    return "\n\n".join(blocks)


def format_judgeable_outcomes(exercise: Exercise, verification: Optional[Dict[str, VerificationDetail]]) -> str:
    """List outcome ids that need an LLM judgment (not covered by deterministic
    verification), so the evaluator prompt can require exactly one judgment
    per id rather than the model inventing or skipping ids.
    """
    verification = verification or {}
    lines = [
        f"- {o.id}: {o.description}"
        for o in exercise.get_all_outcomes()
        if o.id not in verification
    ]
    return "\n".join(lines) if lines else "(none -- every outcome is deterministically verified)"


def build_mentor_system_prompt(
    exercise: Exercise, help_level: int, effective_difficulty: str = "intermediate"
) -> str:
    description = HELP_LEVEL_DESCRIPTIONS.get(help_level, HELP_LEVEL_DESCRIPTIONS[2])
    reveal_answer = exercise.mentor.reveal_answer and help_level >= 5
    difficulty_instructions = DIFFICULTY_ASSISTANCE_INSTRUCTIONS.get(
        effective_difficulty, DIFFICULTY_ASSISTANCE_INSTRUCTIONS["intermediate"]
    )
    return MENTOR_SYSTEM_PROMPT_TEMPLATE.format(
        help_level=help_level,
        help_level_description=description,
        reveal_answer=reveal_answer,
        difficulty_instructions=difficulty_instructions,
    )


def build_mentor_user_prompt(
    exercise: Exercise,
    events: List[EvidenceEvent],
    verification: Optional[dict],
    question: str,
    evidence_error: Optional[str] = None,
    has_images: bool = False,
    max_events: int = 40,
) -> Tuple[str, bool]:
    outcomes = format_outcomes_by_step(exercise)
    timeline, truncated = format_evidence_timeline(events, max_events=max_events, evidence_error=evidence_error)
    image_note = (
        "\nScreenshots of the learner's screen are attached to this message. They are "
        "the PRIMARY source of truth for what the learner has actually done so far -- "
        "examine them directly for on-screen actions and UI state. The text activity "
        "timeline below is supplementary (window titles, click coordinates, and "
        "key-count summaries carry no element names or typed content, and OCR may be "
        "garbled). Do not tell the learner you cannot see evidence of a step if a "
        "screenshot shows it.\n"
        if has_images
        else ""
    )
    prompt = f"""Exercise: {exercise.title}
{image_note}
Instructions:
{exercise.instructions}

Expected outcomes:
{outcomes}

Deterministic verification (ground truth for filesystem outcomes):
{format_verification(verification)}

Observed activity timeline (most recent last):
{timeline}

The learner asks: "{question}"

Respond directly to the learner in 2-6 sentences. If evidence retrieval
failed, say so plainly and give general guidance instead of implying you
observed (or did not observe) anything.
"""
    return prompt, truncated


# ---------------------------------------------------------------------------
# Session Query -- natural-language Q&A about a completed capture session.
# Reuses format_evidence_timeline (which keeps chronological order and embeds
# each event's timestamp) so the model can count distinct actions and reason
# about sequence/repetition/app switching. Text evidence is always included;
# a small set of screenshots may be attached when the question is visual or
# text evidence is insufficient (see app/services/session_query.py).
# ---------------------------------------------------------------------------

SESSION_QUERY_SYSTEM_PROMPT = """You are analyzing evidence captured during one computer-use session. Answer the user's question using only the supplied evidence (the text timeline, and any attached screenshots). Distinguish repeated screenshots/states from distinct user actions. When asked for a count, count distinct observable actions, not frames. If the evidence is insufficient to determine the answer reliably, say so. Do not invent actions.

Always respond in this Markdown structure:

## Answer
Give the direct answer in 1-2 sentences. Put the key number or result in **bold**.

## Summary
Briefly explain what was counted or observed (1-2 sentences).

## Supporting Evidence
Present the strongest supporting evidence as a compact bullet list or a Markdown table. Prefer a table when there are multiple timestamped events, and preserve timestamps so the count can be verified. Keep all important evidence and supporting details -- do not drop them. Do not dump long paragraphs of step-by-step reasoning, and do not repeat the same evidence more than once.

## Notes / Confidence
Include this section only when needed: whether something was directly observed or inferred, ambiguity in the evidence, why a first or last event was excluded, or limitations of OCR/screenshots/event capture. Omit the section entirely when it is not needed.

Formatting rules:
- Give the answer first.
- Convert chronological evidence into tables or concise bullet lists.
- Preserve timestamps when they help validate the answer.
- Clearly distinguish observed facts from inferred actions; use "likely", "appears to", or "inferred" when the evidence does not directly prove the action.
- For count questions, show exactly which items/events were counted.
- For distinct-item questions, list the distinct items.
- For workflow questions, summarize the sequence clearly.
- For application-switch questions, show the transitions (From -> To).
- For website questions, group repeated visits by URL/site rather than repeating the full raw URL each time unless necessary.
- If the evidence is insufficient for a reliable answer, say so in the Answer section and keep the other sections short or omit them.
"""


def build_session_query_user_prompt(
    session_title: str,
    session_id,
    events: List[EvidenceEvent],
    question: str,
    evidence_error: Optional[str] = None,
    has_images: bool = False,
    max_events: int = 80,
) -> Tuple[str, bool]:
    timeline, truncated = format_evidence_timeline(
        events, max_events=max_events, evidence_error=evidence_error
    )
    image_note = (
        "\nOne or more screenshots from this session are attached to this message -- "
        "examine them directly for visual details (layout, colors, charts, on-screen "
        "confirmation/feedback state) that OCR/AX text alone might miss. Treat each "
        "attached screenshot as one captured observation; do not count repeated "
        "screenshots of the same state as separate actions.\n"
        if has_images
        else ""
    )
    prompt = f"""Session: {session_title} (session id {session_id})
{image_note}
The user asks: "{question}"

Captured evidence timeline for this session (chronological order, most recent
last). Each line is one captured observation with its timestamp; the timeline
has already been de-duplicated, so near-identical repeated screen states are
collapsed -- count distinct observable actions, not duplicate frames:

{timeline}

Answer the user's question using ONLY the supplied evidence (the text timeline
above and, if present, the attached screenshots). If the evidence is
insufficient to answer reliably, say so plainly rather than guessing.
"""
    return prompt, truncated


EVALUATION_SYSTEM_PROMPT = """You are an AI evaluator analyzing how a learner approached a technical
exercise, using observed screen activity as evidence. The exercise may be a
single task or a sequence of steps in one or more applications (terminal,
browser, or any other software) -- judge each outcome independently based on
its own description, not assumptions from any other exercise type.

When screenshots from the session are attached to the user message, treat
them as the PRIMARY source of truth for what the learner actually did on
screen. The text activity timeline is supplementary context only: window
titles, click coordinates, and key-count summaries carry no element names or
typed content, and OCR text is frequently garbled. Do NOT conclude a step did
not happen merely because the text timeline does not explicitly describe it --
if a screenshot shows the action or its on-screen result, that is sufficient
evidence it was observed.

STRICT EVIDENCE RULES for every outcome judgment (these prevent false credit):
- Mark an outcome observed=true ONLY when a screenshot or timeline event
  directly shows the learner PERFORMED the action, OR shows its direct
  on-screen result. Example: an "Upload complete. 1 file imported."
  confirmation message is direct evidence the upload happened.
- A UI element merely being VISIBLE is not evidence the learner interacted
  with it. Seeing a filter input field, a dropdown, or a button on screen
  does NOT prove the learner applied a filter, opened a menu, or clicked
  anything. The element existing is not the same as the learner using it.
- NEVER use the exercise instructions or outcome descriptions as evidence.
  The instructions state what the learner SHOULD do, not what they DID do.
  "The instructions mention taking annotated screenshots" is NEVER evidence
  that screenshots were taken. Judge only what the screenshots and timeline
  show, not what the lab expects.
- If no screenshot or timeline event directly shows the action or its result,
  mark it observed=false, even if the learner most likely did it as part of
  the workflow. Doubt goes to not-observed, not to credit.

You are NOT responsible for deciding pass/fail on objective, deterministically
verified outcomes (e.g. filesystem state) -- that has already been decided and
is provided to you as ground truth. Do not contradict it.

You ARE responsible for:
1. Judging the learner's process: whether their approach was reasonable,
   whether they took unnecessary or risky steps, whether they appeared to
   verify their own work, and producing constructive educational feedback.
2. Providing a per-outcome judgment for EVERY outcome id listed under
   "Outcomes requiring your judgment" below -- these are outcomes with no
   deterministic check, so your judgment is the only signal available for
   them. For each one, decide whether the evidence shows it was observed to
   happen, with a brief evidence-based justification. Do not guess -- if the
   evidence doesn't show it, mark it as not observed rather than assuming
   the learner probably did it.

Respond with ONLY a single JSON object, no prose before or after, and no
markdown code fences, matching exactly this schema:

{
  "summary": "string, 2-4 sentences",
  "strengths": ["string", ...],
  "improvements": ["string", ...],
  "observed_approach": ["string", ...],
  "alternative_approaches": ["string", ...],
  "risky_or_unnecessary_steps": ["string", ...],
  "outcome_judgments": [
    {
      "id": "the exact outcome id being judged",
      "observed": true,
      "evidence": "string describing what in the activity timeline supports this judgment"
    }
  ]
}

Include exactly one entry in "outcome_judgments" for every id listed under
"Outcomes requiring your judgment" -- no more, no fewer. If the observed
activity is insufficient to judge a particular field, use an empty
list/string or false rather than guessing, and say so in the summary.
"""


def build_evaluation_user_prompt(
    exercise: Exercise,
    events: List[EvidenceEvent],
    verification: Optional[dict],
    evidence_error: Optional[str] = None,
    has_images: bool = False,
    max_events: int = 80,
) -> Tuple[str, bool]:
    outcomes = format_outcomes_by_step(exercise)
    acceptable = "\n".join(f"- {m}" for m in exercise.acceptable_methods) or "(none specified)"
    prohibited = "\n".join(f"- {m}" for m in exercise.prohibited_behaviors) or "(none specified)"
    image_note = (
        "\nScreenshots from this session are attached to this message. They are the "
        "PRIMARY source of truth for what the learner did -- examine them directly for "
        "on-screen actions, UI state, filled-in fields, open panels, and results. The "
        "text timeline below is supplementary context (it records window titles, click "
        "coordinates, and key-count summaries -- no element names or typed content -- "
        "and OCR may be garbled). Do not treat absence from the text timeline as "
        "absence of the action if a screenshot shows it.\n"
        if has_images
        else ""
    )
    timeline, truncated = format_evidence_timeline(events, max_events=max_events, evidence_error=evidence_error)
    if evidence_error:
        closing = (
            "Analyze the learner's process and return the JSON object described in "
            "the system prompt. Evidence retrieval FAILED, so your summary must say "
            "so explicitly and must NOT claim or imply that the learner did anything "
            "-- say the learner's process could not be analyzed because activity "
            "capture was unavailable, and every judgment's \"observed\" must be false."
        )
    else:
        closing = (
            "Analyze the learner's process and return the JSON object described in "
            "the system prompt. Judge each outcome solely from the screenshots and "
            "timeline above, applying the strict evidence rules: mark observed=true "
            "only when the evidence directly shows the action or its on-screen result, "
            "and false otherwise. Do not claim an outcome did not happen unless you "
            "have examined the screenshots and found no evidence of it."
        )
    prompt = f"""Exercise: {exercise.title}
{image_note}
Instructions:
{exercise.instructions}

Expected outcomes:
{outcomes}

Outcomes requiring your judgment (respond with exactly one outcome_judgments entry per id):
{format_judgeable_outcomes(exercise, verification)}

Acceptable methods:
{acceptable}

Prohibited behaviors:
{prohibited}

Deterministic verification (ground truth, do not contradict):
{format_verification(verification)}

Observed activity timeline for the full session (most recent last):
{timeline}

{closing}
"""
    return prompt, truncated


def build_missing_judgments_prompt(
    exercise: Exercise,
    events: List[EvidenceEvent],
    missing_ids: List[str],
    has_images: bool = False,
    max_events: int = 80,
) -> Tuple[str, bool]:
    """A focused follow-up prompt used when the first evaluation returned a
    summary but skipped some (or all) required outcome judgments -- a common
    failure mode for smaller vision models, which describe what they see in
    `summary` but omit the structured `outcome_judgments` array.

    Asks for the SAME JSON schema the system prompt specifies (so the existing
    parse/repair path works unchanged), but narrows the request to ONLY the
    missing outcome ids and re-attaches the screenshots/timeline, so the model
    has a smaller, simpler ask with the same evidence in front of it.
    """
    missing_lines = "\n".join(
        f"- {oid}: {o.description}"
        for oid in missing_ids
        for o in [next((oc for oc in exercise.get_all_outcomes() if oc.id == oid), None)]
        if o is not None
    ) or "(none)"
    image_note = (
        "\nThe same screenshots from this session are attached again -- examine "
        "them directly as the primary source of truth for whether each outcome "
        "below was observed on screen.\n"
        if has_images
        else ""
    )
    timeline, truncated = format_evidence_timeline(events, max_events=max_events)
    prompt = f"""Your previous evaluation of this session did not include a judgment for every
required outcome id. You MUST now provide a judgment for EACH of the outcome ids
listed below -- no others, and do not omit any of them.
{image_note}
Outcomes requiring your judgment (respond with exactly one outcome_judgments entry per id below):
{missing_lines}

Observed activity timeline for the full session (most recent last):
{timeline}

Return ONLY the JSON object described in the system prompt, with an
"outcome_judgments" array containing exactly one entry for each id listed above.
For each, set "observed" to true only if the screenshots or timeline show the
action or its on-screen result; otherwise false. Provide a brief evidence-based
justification for each.
"""
    return prompt, truncated


REPAIR_PROMPT_TEMPLATE = """Your previous response could not be parsed as valid JSON matching the
required schema. Here is what you returned:

{previous_response}

Return ONLY a corrected, valid JSON object matching the required schema.
No prose, no markdown code fences.
"""


# ---------------------------------------------------------------------------
# Exercise authoring, review, and class-analysis prompts have moved to
# app/services/instructor_prompts.py -- they are not needed by the
# student-facing Practice flow (mentor chat, session Q&A, evaluation) that
# this module serves, and keeping them separate lets a trimmed export (see
# scripts/beta_manifest.txt) ship this module without also shipping
# instructor/authoring/review prompt content. See instructor_prompts.py for
# EXERCISE_AUTHOR_SYSTEM_PROMPT, EXERCISE_FINALIZE_SYSTEM_PROMPT,
# EXERCISE_SEED_SYSTEM_PROMPT, HELP_TOPIC_SYSTEM_PROMPT,
# MENTOR_REVIEW_SYSTEM_PROMPT, CLASS_MENTOR_REVIEW_SYSTEM_PROMPT, and their
# build_*_user_prompt functions.
# ---------------------------------------------------------------------------
