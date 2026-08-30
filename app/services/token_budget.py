"""Lightweight, best-effort prompt/context budgeting for Ollama requests.

Ollama's HTTP API does not expose a tokenizer, so "estimated tokens" here is
a conservative heuristic (~4 characters per token, a common rule of thumb
for English/code text) used only to keep requests from growing far beyond a
model's context window before they are ever sent -- not an exact count.

The guiding rule for this module: when a prompt is estimated to be too big,
shrink the evidence included (fewer events), never grow the model's
num_ctx. Raising num_ctx just moves the failure to a bigger number and
costs more RAM/context-cache; it does not address the actual waste.
"""
import logging
from typing import Callable, Tuple

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
# Headroom reserved for the model's own response.
RESERVED_OUTPUT_TOKENS = 1024
# Rough, conservative per-image token cost, used only for budgeting --
# actual vision-tower token cost varies by model and image resolution.
ESTIMATED_TOKENS_PER_IMAGE = 600

# Renders a candidate prompt for a given evidence-event window size (the
# number of most-recent events to include), returning
# (system_prompt, user_prompt, per_event_text_was_truncated).
RenderFn = Callable[[int], Tuple[str, str, bool]]


def estimate_tokens(text: str) -> int:
    """Very rough token estimate from character count."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_prompt_tokens(system_prompt: str, user_prompt: str, image_count: int = 0) -> int:
    return (
        estimate_tokens(system_prompt)
        + estimate_tokens(user_prompt)
        + image_count * ESTIMATED_TOKENS_PER_IMAGE
    )


def budget_for_context(context_size: int) -> int:
    """Usable prompt-token budget for a context window, after reserving
    headroom for the model's own output."""
    return max(0, context_size - RESERVED_OUTPUT_TOKENS)


def fit_to_budget(
    render: RenderFn,
    max_count: int,
    context_size: int,
    image_count: int = 0,
    min_count: int = 1,
) -> Tuple[str, str, int, int, bool]:
    """Finds the largest recent-event window (<= max_count) whose rendered
    prompt fits the context budget, shrinking the window instead of raising
    num_ctx when it does not fit.

    Returns (system_prompt, user_prompt, event_count_used, estimated_tokens, truncated).
    `truncated` is True if either the event window had to be shrunk below
    `max_count`, or `render` reported that individual event text was
    truncated at the requested window size.
    """
    budget = budget_for_context(context_size)
    count = max(0, max_count)
    system_prompt, user_prompt, per_event_truncated = render(count)
    tokens = estimate_prompt_tokens(system_prompt, user_prompt, image_count)
    window_reduced = False

    while tokens > budget and count > min_count:
        window_reduced = True
        count = max(min_count, count // 2)
        system_prompt, user_prompt, per_event_truncated = render(count)
        tokens = estimate_prompt_tokens(system_prompt, user_prompt, image_count)

    if tokens > budget:
        logger.warning(
            "Estimated prompt (~%d tokens) still exceeds context budget (%d tokens of %d "
            "context) after reducing to %d events; sending as-is rather than raising num_ctx.",
            tokens, budget, context_size, count,
        )

    return system_prompt, user_prompt, count, tokens, (per_event_truncated or window_reduced)
