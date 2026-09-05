"""Client for the local Ollama chat API.

This is the only AI inference endpoint used by the application; no cloud
providers are contacted.
"""
import json
import logging
from typing import List, Optional

import httpx
from pydantic import ValidationError

from app.config import get_settings
from app.models.evaluation import LLMEvaluationInsights
from app.services.prompts import REPAIR_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised for any failure talking to Ollama; always carries a user-facing message."""


def _ensure_text(value) -> str:
    """Ollama's /api/chat requires messages[].content to be a JSON string --
    if it's ever a dict/list (e.g. structured evidence passed through
    without being formatted into prose first), Ollama's server-side
    validation rejects the whole request with a 400. Convert defensively
    rather than relying on every caller to have already done so."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def extract_json_object(text: str) -> Optional[str]:
    """Pull a JSON object out of a possibly markdown-fenced/prefixed LLM response.

    Shared by OllamaService.evaluate() and the exercise-authoring finalize
    step, since both need the same "the model didn't return pure JSON"
    tolerance.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


class OllamaService:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout_seconds

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError as exc:
            logger.info("Ollama health check failed: %s", exc)
            return False

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        images: Optional[List[str]] = None,
        model: Optional[str] = None,
        num_ctx: Optional[int] = None,
        timeout: Optional[float] = None,
        format: Optional[dict] = None,
        keep_alive: Optional[int] = None,
    ) -> str:
        """images: base64-encoded image strings (no data URL prefix), attached
        to the user message -- Ollama's /api/chat accepts this directly for
        multimodal models. Pass `model` to use a different (e.g. vision)
        model than this instance's default for a single call. Pass `num_ctx`
        to override that model's context window for this call -- some models
        (e.g. llava's default 32768) are too small for a full evaluation
        prompt plus attached screenshots, and Ollama supports raising this at
        request time rather than needing to trim the prompt. Pass `timeout`
        to override this instance's timeout for a single call (e.g. a slower
        session-query call) without creating a second OllamaService/client.

        `format`: a JSON Schema object (e.g. Pydantic's own
        `Model.model_json_schema()`) passed through to Ollama's real
        structured-output `format` request property, which grammar-
        constrains decoding so the response is guaranteed syntactically
        valid JSON matching the schema -- this is enforced by Ollama/
        llama.cpp itself, not merely requested in prompt text (verified
        directly against this Ollama build/model combination: see
        EvaluatorService._score_visual_outcome and the visual-evaluation
        repair notes). None (the default) preserves the exact prior
        behavior for every existing caller.

        `keep_alive`: passed through to Ollama's own `keep_alive` request
        property controlling how long the model stays loaded after this
        response. Pass 0 to force Ollama to fully unload the model
        immediately afterward -- reproducibly confirmed to clear a distinct
        server-side bug on this build where switching between DIFFERENT
        image sets on the SAME already-loaded multimodal model instance in
        rapid succession corrupts its internal image cache and fails with
        HTTP 500 "Chunk not found" (a fresh model load never exhibits it).
        A full reload costs real time, so this is used only as a bounded,
        last-resort recovery attempt (see EvaluatorService.
        VISUAL_OUTCOME_MAX_ATTEMPTS), never on every call. None (the
        default) omits the option entirely, preserving Ollama's own default
        residency behavior for every other caller.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        options: dict = {"temperature": temperature}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        if images:
            # A separate system-role message combined with an image in the
            # user turn reproducibly crashes this model/build with a
            # server-side "Chunk not found" 500 once the system prompt is
            # long (observed live with the ~17K-char evaluation system
            # prompt; a short system prompt + image did not trigger it, but
            # merging costs nothing when short either) -- folding both into
            # ONE user-role message with the image attached avoids it
            # entirely while sending the model the exact same instructions.
            # Scoped to images-only: the text-only path (including every
            # terminal-exercise call, which never attaches images) keeps the
            # original system+user message shape unchanged.
            combined = _ensure_text(system_prompt) + "\n\n---\n\n" + _ensure_text(user_prompt)
            messages = [{"role": "user", "content": combined, "images": images}]
        else:
            messages = [
                {"role": "system", "content": _ensure_text(system_prompt)},
                {"role": "user", "content": _ensure_text(user_prompt)},
            ]

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if format is not None:
            payload["format"] = format
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        try:
            async with httpx.AsyncClient(timeout=effective_timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"Ollama timed out after {effective_timeout:.0f}s. The model '{payload['model']}' may be slow "
                "to respond or still loading."
            ) from exc
        except httpx.HTTPError as exc:
            # Connection-level failure (refused, DNS, etc.) -- Ollama itself
            # never responded, as distinct from responding with an error status.
            raise OllamaError(
                f"Ollama is not reachable at {self.base_url}. Start it with: ollama serve"
            ) from exc

        if resp.status_code >= 400:
            # Reachable and responded -- just rejected or failed the request.
            # Log the real body so the actual cause is visible, and raise a
            # distinct error message so callers never conflate this with
            # "Ollama is not reachable".
            body_preview = resp.text[:2000]
            logger.error(
                "Ollama /api/chat returned HTTP %s for model '%s': %s",
                resp.status_code, payload["model"], body_preview,
            )
            raise OllamaError(
                f"Ollama is reachable but rejected the request (HTTP {resp.status_code}) for "
                f"model '{payload['model']}': {body_preview}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned a response that could not be parsed as JSON") from exc

        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise OllamaError(
                f"Ollama returned an empty response. Is the model '{payload['model']}' installed? "
                f"Try: ollama pull {payload['model']}"
            )
        return content

    async def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[str]] = None,
        model: Optional[str] = None,
        num_ctx: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> LLMEvaluationInsights:
        # Evaluation temperature is independently configurable (default 0 --
        # deterministic/greedy decoding) via Settings.evaluation_temperature,
        # separate from chat()'s own default (0.3) used by mentor chat /
        # session Q&A, which is left unchanged: scoring must be repeatable
        # for the SAME captured evidence, while mentoring can stay more
        # conversational. See app/config.py:evaluation_temperature.
        temperature = get_settings().evaluation_temperature
        raw = await self.chat(
            system_prompt, user_prompt, temperature=temperature, images=images,
            model=model, num_ctx=num_ctx, timeout=timeout,
        )
        parsed = self._try_parse(raw)
        if parsed is not None:
            return parsed

        logger.info("Ollama evaluation JSON malformed, retrying with a repair prompt (without re-sending images)")
        repair_prompt = REPAIR_PROMPT_TEMPLATE.format(previous_response=raw)
        # The repair retry only asks the model to reformat its own previous
        # text response as valid JSON -- the image evidence already informed
        # that response, so resending screenshots here would just double the
        # image/token cost for no benefit. Model/context stay the same.
        raw_retry = await self.chat(
            system_prompt, repair_prompt, temperature=0.0, model=model, num_ctx=num_ctx, timeout=timeout,
        )
        parsed_retry = self._try_parse(raw_retry)
        if parsed_retry is not None:
            return parsed_retry

        logger.warning("Ollama evaluation JSON still malformed after repair attempt")
        return LLMEvaluationInsights(
            summary=(
                "The AI evaluator returned a response that could not be parsed as valid JSON, "
                "even after a retry. Scoring below relies on deterministic verification and "
                "observed evidence only."
            ),
        )

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        return extract_json_object(text)

    def _try_parse(self, raw: str) -> Optional[LLMEvaluationInsights]:
        candidate = self._extract_json_object(raw)
        if not candidate:
            return None
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        try:
            return LLMEvaluationInsights.model_validate(data)
        except ValidationError:
            return None

    def _parse_json_dict(self, raw: str) -> Optional[dict]:
        """Like _try_parse, but returns a plain dict with no schema validation.

        Used by callers (e.g. exercise authoring) that need to inject fields
        before validating against their own model.
        """
        candidate = extract_json_object(raw)
        if not candidate:
            return None
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    async def chat_json(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2,
        images: Optional[List[str]] = None, model: Optional[str] = None,
        num_ctx: Optional[int] = None, timeout: Optional[float] = None,
    ) -> Optional[dict]:
        """Chat, extract a JSON object, and retry once with a repair prompt if malformed.

        Returns a plain dict (no schema validation) or None if unrecoverable.
        `images`/`model`/`num_ctx`/`timeout` are optional passthroughs (all
        None by default, matching every existing caller's behavior exactly)
        added so a lightweight vision-attached JSON call -- e.g. a targeted
        follow-up re-check of a few screenshots -- doesn't need the full
        LLMEvaluationInsights schema machinery in evaluate().
        """
        raw = await self.chat(
            system_prompt, user_prompt, temperature=temperature,
            images=images, model=model, num_ctx=num_ctx, timeout=timeout,
        )
        data = self._parse_json_dict(raw)
        if data is not None:
            return data

        logger.info("Chat JSON malformed, retrying with a repair prompt")
        repair_prompt = REPAIR_PROMPT_TEMPLATE.format(previous_response=raw)
        raw_retry = await self.chat(
            system_prompt, repair_prompt, temperature=0.0,
            model=model, num_ctx=num_ctx, timeout=timeout,
        )
        return self._parse_json_dict(raw_retry)

    async def chat_structured(
        self, system_prompt: str, user_prompt: str, schema: dict, temperature: float = 0.0,
        images: Optional[List[str]] = None, model: Optional[str] = None,
        num_ctx: Optional[int] = None, timeout: Optional[float] = None,
        keep_alive: Optional[int] = None,
    ) -> Optional[dict]:
        """Chat with Ollama's `format` request property set to `schema` (a
        JSON Schema object, e.g. from a Pydantic model's
        `.model_json_schema()`) so the response is grammar-constrained
        rather than merely requested in prompt wording -- see `chat()`'s
        `format` parameter. Deliberately does NOT run the markdown-fence/
        repair-prompt dance chat_json() does: a schema-constrained response
        is syntactically valid JSON by construction, so a parse failure here
        means the request itself failed (e.g. transport error) rather than
        the model choosing not to comply, and the caller (per-outcome/
        per-batch retry policy -- see EvaluatorService) decides whether and
        how to retry, typically with a SMALLER evidence packet rather than
        the same request again. Returns None on any parse failure; never
        raises OllamaError for a parse issue (only chat() itself can raise,
        for a genuine transport/HTTP failure).
        """
        raw = await self.chat(
            system_prompt, user_prompt, temperature=temperature, images=images,
            model=model, num_ctx=num_ctx, timeout=timeout, format=schema,
            keep_alive=keep_alive,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = self._parse_json_dict(raw)
            return data
        return data if isinstance(data, dict) else None
