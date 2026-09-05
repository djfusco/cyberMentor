"""All LLM prompt construction lives here, kept out of route handlers."""
import logging
from typing import Dict, List, Optional, Tuple

from app.models.exercise import Exercise, LearningObjective
from app.services.evidence import EvidenceEvent
from app.services.verifier import EvidenceFact, VerificationDetail, needs_llm_attribution

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

Mouse-click coordinates in the timeline do NOT reveal which UI control,
row, field, or pane was clicked. You may acknowledge an observed click, but
do not infer the target from coordinates alone -- the target must remain
unknown unless a screenshot or accessibility evidence identifies it.

Your goal is to help the learner reason through the exercise.

{difficulty_instructions}

Do not assume an exact sequence of steps is required. Multiple technically
correct approaches may exist.

Prefer hints over giving away the full solution. Current help level: {help_level}
({help_level_description}).

Do not claim the learner performed an action unless the evidence supports it.
Clearly distinguish between: observed, verified, inferred, and unknown.
If evidence is insufficient, say so plainly.

When an action is absent from the captured evidence, do NOT state that the
learner "did not run" the command, "did not perform" the step, "never" did
something, or that it "was not entered" or "was not observed." Screen capture
is always incomplete: OCR may miss content, not every screen state is
captured, and the capture window may not cover all activity. If you cannot
confirm an action from the available evidence, say "I could not confirm that
from the available captured evidence" rather than asserting it did not occur.

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


# Number of 'unchanged' lines shown before the first change in each run,
# to give the model context for where on the screen the new content appeared.
_DELTA_CONTEXT_LINES: int = 2


def _format_text_observed_event(event: EvidenceEvent) -> str:
    """Format a text_observed event that carries a meaningful line delta.

    Produces a multi-line block clearly labelled as visual screen observation,
    not shell telemetry.  Only 'added' and 'meaningful_modified' lines are
    shown; up to _DELTA_CONTEXT_LINES immediately preceding 'unchanged' lines
    are included before each run of changes so the model can see what
    surrounded the new content.

    'ocr_jitter' and 'removed' lines are never shown -- they are not new
    information from the learner's perspective.  The raw .text field is
    intentionally not shown here; the delta is the communication medium.
    Returns an empty string if there are no visible changes (should not
    happen when called correctly, but guards against empty deltas).
    """
    delta = event.text_delta
    if not delta:
        return ""

    ts_str = event.timestamp.strftime("%H:%M:%S")
    loc = event.window_title or event.application
    header = f"[{ts_str}] {event.application} ({loc}) — newly observed on screen"

    output_lines: List[str] = []
    unchanged_buf: List[str] = []
    # Tracks whether we have already flushed context for the current run of
    # changes, so consecutive 'added' lines do not re-emit the same context.
    context_flushed = False

    for tag, line in delta:
        if tag == "unchanged":
            if line.strip():
                unchanged_buf.append(line)
            context_flushed = False  # next change block gets fresh context
        elif tag in ("added", "meaningful_modified"):
            if not context_flushed and unchanged_buf:
                for ctx in unchanged_buf[-_DELTA_CONTEXT_LINES:]:
                    if ctx.strip():
                        output_lines.append(f"  {ctx}")
                context_flushed = True
            unchanged_buf = []
            prefix = "+" if tag == "added" else "~"
            if line.strip():
                output_lines.append(f"{prefix} {line}")
        else:
            # 'removed' or 'ocr_jitter': not shown; reset context buffer.
            unchanged_buf = []
            context_flushed = False

    if not output_lines:
        return ""
    return header + ":\n" + "\n".join(output_lines)


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
        if e.type == "text_observed" and e.text_delta is not None:
            # This event has a meaningful delta -- use the delta-aware format,
            # which labels content as "newly observed on screen" rather than
            # dumping the raw OCR blob.
            formatted = _format_text_observed_event(e)
            if formatted:
                lines.append(formatted)
                continue
        # Fallback: raw text with existing truncation (first text_observed per
        # app before a baseline exists, and all non-text_observed event types).
        text, was_truncated = _truncate_event_text(e.text, e.application, e.type)
        truncated_any = truncated_any or was_truncated
        location = f"{e.application} @ {e.browser_url}" if e.browser_url else e.application
        lines.append(f"- [{e.timestamp.isoformat()}] ({location}) {text}")
    result = "\n".join(lines)
    if truncated_any:
        result += (
            "\n[Note: some captured screen text was truncated to fit available context. "
            "Absence of a specific action from this timeline does not confirm it was "
            "not performed -- the full screen content may not have been included.]"
        )
    return result, truncated_any


def format_verification(verification: Optional[Dict[str, VerificationDetail]]) -> str:
    if not verification:
        return "(no deterministic verification available)"
    lines = []
    for key, detail in verification.items():
        status = "PASS" if detail.passed else "FAIL"
        lines.append(f"- {key}: {status} -- {detail.note}")
    return "\n".join(lines)


def format_evidence_facts(evidence_facts: Optional[List[EvidenceFact]]) -> str:
    """Structural facts already established deterministically from captured
    text (see app.services.verifier) -- shown to the LLM as reliable
    corroborating context so it does not need to re-derive them from noisy
    OCR text itself. These are informational only here; scoring applies them
    independently (see EvaluatorService._score_from_judgment), so an absent
    or "not present" fact never blocks a judgment based on other evidence
    (e.g. a screenshot) the LLM finds instead.
    """
    if not evidence_facts:
        return "(none established for this exercise)"
    lines = []
    for fact in evidence_facts:
        status = "PRESENT" if fact.present else "not found"
        lines.append(f"- [{fact.outcome_id}] {status}: {fact.detail}")
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


def format_judgeable_outcomes(
    exercise: Exercise,
    verification: Optional[Dict[str, VerificationDetail]],
    baseline_verification: Optional[Dict[str, VerificationDetail]] = None,
) -> str:
    """List outcome ids that need an LLM judgment, so the evaluator prompt can
    require exactly one judgment per id rather than the model inventing or
    skipping ids.

    Two kinds of outcome end up here:
    1. Outcomes not covered by deterministic verification at all (as before).
    2. A deterministically-checked outcome whose current PASS doesn't, by
       itself, prove the student produced it THIS session -- the same
       compliant state was already present at session start (see
       needs_llm_attribution). Only current-session evidence can still earn
       these credit; a failing check or a real baseline->final transition
       never needs the LLM at all (see EvaluatorService._score_filesystem_outcome).
    """
    verification = verification or {}
    lines = [
        f"- {o.id}: {o.description}"
        for o in exercise.get_all_outcomes()
        if o.id not in verification or needs_llm_attribution(o.id, verification, baseline_verification)
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
    prompt = MENTOR_SYSTEM_PROMPT_TEMPLATE.format(
        help_level=help_level,
        help_level_description=description,
        reveal_answer=reveal_answer,
        difficulty_instructions=difficulty_instructions,
    )
    if _exercise_is_enriched(exercise):
        prompt += ENRICHED_MENTOR_ADDENDUM
    return prompt


def build_mentor_user_prompt(
    exercise: Exercise,
    events: List[EvidenceEvent],
    verification: Optional[dict],
    question: str,
    evidence_error: Optional[str] = None,
    has_images: bool = False,
    frame_captions: Optional[str] = None,
    max_events: int = 40,
) -> Tuple[str, bool]:
    outcomes = format_outcomes_for_mentor(exercise)
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
    if has_images and frame_captions:
        image_note += frame_captions + "\n"
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
    frame_captions: Optional[str] = None,
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
    if has_images and frame_captions:
        image_note += frame_captions + "\n"
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


def _exercise_is_enriched(exercise: Exercise) -> bool:
    """True when any outcome carries explicit success criteria.
    LOs alone are not sufficient — enrichment is measured at the outcome level."""
    return any(o.has_explicit_criteria for o in exercise.get_all_outcomes())


def format_outcomes_for_evaluation(exercise: Exercise) -> str:
    """Per-outcome structured blocks for the evaluator prompt.
    Enriched outcomes get full criteria/evidence sections;
    legacy outcomes get a compact implicit-criterion block.
    Multi-step exercises include a step header with instructions before each group."""
    blocks: List[str] = []
    sep = "─" * 57
    is_multi_step = exercise.steps is not None
    for step in exercise.get_steps():
        if is_multi_step:
            step_header = f"Step '{step.title}':"
            if step.instructions:
                step_header += f"\n  Instructions: {step.instructions}"
            blocks.append(step_header)
        for outcome in step.expected_outcomes:
            lines = [sep, f"OUTCOME: {outcome.id}"]
            lo_str = ", ".join(outcome.objective_ids) if outcome.objective_ids else ""
            weight_line = f"Weight: {outcome.weight} pts | Type: {outcome.type.value}"
            if lo_str:
                weight_line += f" | LOs: {lo_str}"
            lines.append(weight_line)

            if outcome.has_explicit_criteria:
                lines.append("Criteria source: [explicitly authored]")
                lines.append("")
                lines.append("Success criteria — evaluate EACH independently:")
                for i, c in enumerate(outcome.success_criteria, 1):
                    lines.append(f"  {i}. {c}")
                if outcome.evidence_requirements:
                    lines.append("")
                    lines.append("Evidence to look for (semantic, not UI-specific):")
                    for r in outcome.evidence_requirements:
                        lines.append(f"  - {r}")
                if outcome.feedback_if_missing:
                    lines.append("")
                    lines.append("Instructor guidance if not completed:")
                    lines.append(f"  {outcome.feedback_if_missing}")
            else:
                lines.append("Criteria source: [derived from description — no explicit criteria authored]")
                lines.append("")
                lines.append("Implicit criterion:")
                lines.append(f"  - {outcome.description}")
                lines.append("")
                lines.append("Note: Assess against the single implicit criterion only.")
                lines.append("Do not invent additional sub-criteria.")

            if outcome.student_demonstration:
                lines.append("")
                lines.append(
                    "What counts as demonstrating this DURING THIS SESSION "
                    "(action evidence -- see evidence_basis; a description "
                    "above may read as an end state, but this is what doing "
                    "it, not merely it being true, looks like):"
                )
                lines.append(f"  {outcome.student_demonstration}")

            lines.append(sep)
            blocks.append("\n".join(lines))
    return "\n".join(blocks)


def format_outcomes_for_mentor(exercise: Exercise) -> str:
    """Outcome context for the mentor prompt.
    Includes LO header when present; enriched outcomes show criteria and
    evidence; legacy outcomes use the existing compact format unchanged."""
    result_parts: List[str] = []

    if exercise.learning_objectives:
        lo_lines = ["Learning objectives for this exercise:"]
        for lo in exercise.learning_objectives:
            lo_lines.append(f"  {lo.id}: {lo.description}")
        result_parts.append("\n".join(lo_lines))

    steps = exercise.get_steps()
    is_multi_step = exercise.steps is not None

    for step in steps:
        outcome_lines: List[str] = []
        for outcome in step.expected_outcomes:
            lo_str = f" · {', '.join(outcome.objective_ids)}" if outcome.objective_ids else ""
            header = f"- {outcome.id} ({outcome.weight} pts{lo_str}): {outcome.description}"
            outcome_lines.append(header)

            if outcome.has_explicit_criteria:
                outcome_lines.append("  What success looks like:")
                for c in outcome.success_criteria:
                    outcome_lines.append(f"    • {c}")
                if outcome.evidence_requirements:
                    outcome_lines.append("  Evidence to look for:")
                    for r in outcome.evidence_requirements:
                        outcome_lines.append(f"    • {r}")
                if outcome.feedback_if_missing:
                    outcome_lines.append(
                        f"  [Instructor note if not completed: {outcome.feedback_if_missing}]"
                    )

        step_block = "\n".join(outcome_lines)
        if is_multi_step:
            step_header = f"Step '{step.title}':"
            if step.instructions:
                step_header += f"\n{step.instructions}"
            result_parts.append(f"{step_header}\n{step_block}")
        else:
            result_parts.append(step_block)

    return "\n\n".join(result_parts)


ENRICHED_MENTOR_ADDENDUM = """
When success criteria are listed for an expected outcome, use them internally
to understand what the learner still needs to demonstrate.
- If the student asks "am I doing this right?", "what am I missing?", "is this complete?",
  or a similar status question, compare the observed evidence to the relevant criteria explicitly.
- Otherwise, answer the student's actual question. Do not turn every response into a criteria checklist.

When describing evidence state, use language that matches what you actually know:
- "I can confirm..." — when evidence affirmatively supports something
- "I can see evidence that..." — when evidence strongly suggests something
- "I can't confirm yet..." — when evidence is absent or ambiguous
- "The evidence suggests..." — for inferences
- "This appears to..." — for partial support

Never convert absence of evidence into evidence of failure. These are three distinct situations:
A. Affirmative evidence of an error: state it directly.
B. Evidence shows partial progress: acknowledge what is confirmed.
C. Evidence is absent or ambiguous: say you cannot confirm yet, not that it failed.

Instructor notes on outcomes are background guidance. Adapt the content to the
learner's evidence state and difficulty level; do not quote them verbatim.
"""


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

When neither the text timeline nor any screenshot directly confirms an action,
do NOT state that the learner "did not run" the command, "did not perform" the
step, or "never" did it. Screen capture is always incomplete: OCR may miss
content, not every screen state is captured, and the capture window may not
cover all activity. When you cannot confirm an action, write "could not confirm
from the available captured evidence" and set observed=false -- never assert
the action did not occur.

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
- Mouse-click coordinates alone do NOT establish which UI control, row,
  field, or pane was clicked. A click at (x, y) may be reported as an
  observed click, but the target must remain unknown unless a screenshot or
  accessibility evidence (window title, AX label, OCR text) identifies it.
  Do not infer the target from coordinates.
- EXACT-VALUE CRITERIA: when a criterion or evidence requirement names an
  exact value -- a field name, index name, file name, or other literal
  string (often shown in backticks, e.g. "the field `src_ip` appears") --
  do not mark that criterion met unless that EXACT value is visible in a
  screenshot or present in the timeline text. A similarly-named, related, or
  conceptually-equivalent value does NOT satisfy an exact-value criterion.
  If what you actually observe differs from the value the criterion names
  (e.g. the criterion names `dst_ip` but the screen shows a field called
  `dest_ip`), report the ACTUAL value you saw in your evidence text and mark
  that specific criterion not met -- never state that the criterion's named
  value was visible when a different value is what actually appeared. Do
  not paraphrase or "round" an observed value to match what the criterion
  expected.
- A COMMAND OR QUERY BEING VISIBLE PROVES ONLY THAT IT WAS RUN, NOT THAT IT
  SUCCEEDED. Seeing a query/command entered and executed is evidence the
  learner attempted that step -- it is separate evidence from, and does not
  imply, that its result was correct or complete. Always evaluate the
  RESULT (the output, the returned values, an empty/error result) as its
  own piece of evidence, independent of whether the query/command text
  itself looks correct. An empty or null result must not be described as
  success merely because the query syntax was well-formed.

You are NOT responsible for deciding WHETHER the final state of an objective,
deterministically verified outcome (e.g. filesystem state) is correct -- that
fact has already been decided and is provided to you as ground truth. Do not
contradict it: never claim a state that verification marked PASS is actually
wrong, or vice versa.

That said, a small number of deterministically-verified outcomes may still
appear under "Outcomes requiring your judgment" below. This happens only
when the verified state already existed before the session began, so the
check passing proves the state exists but NOT that the learner produced it
during THIS session -- the state fact and the demonstration fact are
separate. For these, your judgment answers a different, narrower question
than the verification already answered: does the observed activity show the
learner actually PERFORMING the relevant action during this session (running
the command, making the change), not merely that the resulting state is
currently correct? If the timeline/screenshots show no such action this
session, mark observed=false even though verification says the state is
fine -- that is not a contradiction, it is answering a different question.

ACTION evidence vs. STATE evidence -- every outcome_judgments entry must
classify which kind of evidence it relied on, in the "evidence_basis" field:
- "action": the evidence shows the learner PERFORMING the relevant action
  during this session -- a command being typed/run, a change being made, or
  a direct on-screen result caused by that action (a confirmation message,
  a value changing in front of the learner). This is what a
  "student_demonstration" note on an outcome (when present, see the outcome
  block below) describes -- follow it exactly when it is present, since it
  is the authored, action-shaped criterion for that specific outcome.
- "state": the evidence shows only that a resulting state currently holds --
  a directory/file listing, a window title, a status display -- without
  showing who produced that state or when. A listing that shows a file
  exists is state evidence for "this file exists" but is NOT action
  evidence for "the learner created this file": the file could have existed
  before the session started.
- "unclear": you cannot tell which of the above applies.
For an outcome whose baseline state was already compliant (this is exactly
when it appears under "Outcomes requiring your judgment" despite having a
deterministic check -- see above), state evidence can NEVER earn credit, no
matter how convincing it looks -- only "action" evidence can. Do not mark
observed=true on state evidence for these; the scoring layer enforces this
regardless of what you set observed to, so an inaccurate evidence_basis will
produce a result that contradicts your own stated evidence -- classify it
honestly instead of picking whichever value yields a pass.

READING GARBLED / OCR-SOURCED TEXT -- do not require exact command spelling:
Captured terminal text is frequently OCR'd from a screenshot and contains
character-level errors (e.g. "l" read as "1" or "I", a "-" dropped, columns
of a table folded onto separate lines). Judge OCR'd text by its STRUCTURE
and CONTEXT, not by exact string matching:
- A block containing a "total N" header line, a permission-mode-shaped token
  (roughly ten characters of dashes and letters, even if some characters are
  clearly OCR noise -- e.g. "-IW-------" is recognizable as a corrupted
  "-rw-------"), an owner name, a file size, a date, and a filename
  TOGETHER constitute a long-format permission listing (the output of `ls
  -l`, `stat`, `getfacl`, or an equivalent command that visibly establishes
  permissions) -- this is real verification evidence, whatever the exact
  flag text reads as. Do not require the literal substring "ls -l"; accept
  any command whose OUTPUT visibly establishes the permission mode.
- Do NOT assume every stray "1" is a misread "l", or vice versa -- both are
  legitimate characters in real commands and output (e.g. "ls -1" -- one
  filename per line -- is a genuinely different, valid command from "ls
  -l"). Decide from context: if the surrounding output has no long-format
  columns (no permission token, owner, size, date), a lone "-1"-looking flag
  is plausibly a real "ls -1" and shows only filenames -- that is NOT
  permission-verification evidence. If the surrounding output DOES have
  those long-format columns, the same ambiguous flag text is part of a
  long listing and IS permission-verification evidence, regardless of
  whether it reads as "-l" or "-1".
- Apply this same tolerance consistently across every outcome you judge from
  the same piece of evidence. If you treat a garbled token as legible enough
  to establish one fact (e.g. reading a corrupted permission command as
  evidence the mode was set), you must extend the same reading to other
  facts visible in that same output (e.g. a permission listing appearing in
  the same or adjacent lines) for any other outcome it is also relevant to.
  Do not accept OCR noise as legible for one outcome while dismissing
  equally-garbled, equally-adjacent text as unintelligible for a related
  outcome -- if your two judgments disagree about the same underlying
  evidence, your evidence text for each must name the specific, real
  difference that justifies it (e.g. different actions, different outcomes
  covered), never mere inconsistency in how much OCR noise you tolerated.
- OCR quality is a capture limitation, never the learner's fault. When text
  is garbled, describe what the STRUCTURE shows (or say you cannot confirm
  it); do not phrase feedback as if the learner mistyped, misread, or did
  anything wrong -- the ambiguity is in the capture, not their actions.

FEEDBACK FOR A NOT-YET-DEMONSTRATED, ALREADY-EXISTING outcome: when a
filesystem outcome's state already existed before the session began and
nothing this session demonstrated producing it (see the baseline-attribution
guidance above), never suggest running mkdir/touch/recreating the object as
if that would "prove" creation -- the object already exists, so redoing it
would misrepresent what happened. If demonstrating creation genuinely
matters, suggest starting from a clean/reset lab workspace instead (a
setup/administration action, not something to fabricate mid-session).

You ARE responsible for:
1. Judging the learner's process: whether their approach was reasonable,
   whether they took unnecessary or risky steps, whether they appeared to
   verify their own work, and producing constructive educational feedback.
2. Providing a per-outcome judgment for EVERY outcome id listed under
   "Outcomes requiring your judgment" below -- for most of them there is no
   deterministic check at all, so your judgment is the only signal
   available; for the rest (see above), your judgment is what decides
   whether this session's activity demonstrates a state that was already
   verified. For each one, decide whether the evidence shows it was
   observed to happen, with a brief evidence-based justification and an
   honest evidence_basis (see ACTION vs. STATE evidence above). Do not
   guess -- if the evidence doesn't show it, mark it as not observed rather
   than assuming the learner probably did it.

REPORT FIELD RULES (applied when populating the JSON fields below):
- "observed_approach" and "strengths" must only reference outcomes or
  actions that were directly observed (or deterministically verified as
  passed). An outcome whose judgment is observed=false, or that is marked
  failed/unknown/insufficient evidence, must NOT be described there as
  something the learner completed, performed, or accomplished.
- Such outcomes may appear in "improvements" only, using cautious language
  ("it was not possible to confirm", "could not be verified from the
  available evidence", "may need attention").

VERIFICATION STATE RULES — must be consistent with criteria assessment:
- verified: all required success criteria are supported by sufficient evidence.
  Do NOT use verified if any required criterion is unsatisfied.
- partially_verified: at least one criterion is supported, but one or more are not.
- attempted: affirmative evidence of meaningful engagement, but no criterion is
  sufficiently supported.
- incorrect: affirmative evidence the task/result is wrong. Do NOT use merely
  because evidence is absent.
- not_observed: the task should have been observable but relevant evidence was absent.
- unverifiable: the capture mechanism cannot reliably determine completion.

THREE INDEPENDENT DIMENSIONS — do not conflate them:
- verification_state: what the evidence establishes about completion
- confidence: how certain you are about that assessment
- observed / score: credit under the exercise's scoring rules
A partially_verified outcome may still receive passing credit if the rubric allows it.
verified must not be used merely because an outcome received passing credit.

For outcomes with "Criteria source: [derived from description]", assess only the
single implicit criterion. Do not invent additional sub-criteria.
For enriched outcomes, verification_state is REQUIRED. For legacy outcomes, provide
it when determinable; omit only when genuinely uncertain.

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
      "id": "the exact outcome id",
      "verification_state": "verified | partially_verified | attempted | not_observed | incorrect | unverifiable",
      "criteria_met": ["criterion text that is supported by evidence"],
      "criteria_not_met": ["criterion text that is not supported"],
      "criterion_judgments": [
        {
          "criterion": "the exact criterion text being judged (copy it verbatim from the outcome block)",
          "supported": true,
          "evidence_basis": "action | state | unclear (same meaning as the outcome-level evidence_basis below, judged for THIS ONE criterion specifically)",
          "quote": "the EXACT text/value you read, ONLY when this criterion names a specific field name, command, value, count, or identifier -- copy precisely what you actually saw, character for character, even if it differs from what the criterion names; null/omit for criteria with no exact-value claim",
          "frame_reference": "which screenshot (its number) or timestamp this came from, or null",
          "note": "optional short note, or null"
        }
      ],
      "observed": true,
      "confidence": "verified | strongly_observed | inferred | unknown",
      "evidence": "string describing what in the activity supports this judgment",
      "evidence_basis": "action | state | unclear",
      "feedback": "concise student-facing note when the outcome fails or is partial, or null"
    }
  ]
}

REQUIRED: for every enriched outcome (one with authored success criteria,
listed under "Success criteria — evaluate EACH independently" in that
outcome's block), "criterion_judgments" must contain exactly one entry per
listed criterion -- do not omit any, do not merge several into one entry.
For any criterion naming an exact value (a field name, command, literal
string, count, or identifier -- typically shown in backticks, e.g. "the
field `src_ip` appears"), you MUST put your own verbatim reading of what
actually appears in "quote" -- do not copy the criterion's own expected
value into "quote" as if that proves it; state what you actually read, even
if it is a different value than the criterion names. Marking such a
criterion "supported": true without a matching, actually-observed "quote"
is treated as unsupported regardless of what you write elsewhere. This is
mechanically re-checked against the captured text evidence -- a claimed
quote that never actually appears in that evidence is discarded and the
criterion is scored not-met, independent of "observed" or
"verification_state" on the outcome as a whole.

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
    frame_captions: Optional[str] = None,
    max_events: int = 80,
    baseline_verification: Optional[dict] = None,
    evidence_facts: Optional[List[EvidenceFact]] = None,
) -> Tuple[str, bool]:
    outcomes = format_outcomes_for_evaluation(exercise)
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
    if has_images and frame_captions:
        image_note += frame_captions + "\n"
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
{format_judgeable_outcomes(exercise, verification, baseline_verification)}

Acceptable methods:
{acceptable}

Prohibited behaviors:
{prohibited}

Deterministic verification (ground truth, do not contradict):
{format_verification(verification)}

Structured evidence facts (already established mechanically from captured
text -- treat as reliable, do not second-guess a PRESENT fact, but you may
still find and cite additional evidence yourself for anything marked "not
found" here):
{format_evidence_facts(evidence_facts)}

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
    frame_captions: Optional[str] = None,
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
    if has_images and frame_captions:
        image_note += frame_captions + "\n"
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
# Targeted follow-up evidence check -- used by EvaluatorService for success
# criteria the main evaluation pass could not verify (no relevant screenshot
# was selected for them, or captured OCR/AX text was too sparse to trust an
# "absent" reading either way). Deliberately generic and deliberately does
# NOT quote the criterion's own expected exact value anywhere in the
# question -- the caller neutralizes each criterion (backtick content
# replaced with a placeholder) before it ever reaches this prompt, so a
# match can only come from the model independently reading the screenshots,
# never from parroting a value it was fed. See "Never award exact-text
# credit solely because the model repeated the expected field names from
# the prompt."
# ---------------------------------------------------------------------------

TARGETED_FOLLOWUP_SYSTEM_PROMPT = """You are answering narrow factual questions about a set of screenshots from one
learner's session. For each numbered question, report EXACTLY what text,
field name, or value is visible in the screenshots that is relevant to it --
quote precisely what appears on screen, character for character. Do not
guess, and do not assume any particular expected answer -- some questions
may have no expected answer in mind at all. If nothing relevant to a
question is visible in any of the attached screenshots, answer exactly
"not visible" for that question. Respond with ONLY a single JSON object, no
prose before or after, and no markdown code fences, matching exactly this
schema:

{
  "answers": ["answer to question 1", "answer to question 2", ...]
}

Include exactly one answer per numbered question, in order.
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
