"""Opt-in frontier-model backend for in-exercise mentor chat ONLY.

Used only when MENTOR_CHAT_PROVIDER=frontier (see app/config.py) -- for
people who can't or don't want to run Ollama locally and are comfortable
sending exercise/evidence context to a third-party model of their own
choosing instead. This reuses the exact same FRONTIER_PROVIDER/
FRONTIER_API_KEY/FRONTIER_MODEL settings the authoring-only "Research a
Task" feature (app/services/research.py) already uses -- deliberately not
a second provider integration to wire up separately.

Unlike research.py's OpenAIResearchClient (which uses OpenAI's Responses
API with the web_search tool), this makes a plain chat completion with no
tools -- mentor chat must only ever answer from the exercise + observed
evidence context it's given, never do a live web search.

FrontierChatService.chat() intentionally matches OllamaService.chat()'s
call signature (including accepting, but ignoring, `num_ctx`, an
Ollama-specific context-window override) so MentorService can hold either
backend interchangeably without any change to MentorService.ask() itself.
It also raises OllamaError on failure -- reusing that exception (rather
than introducing a parallel one) is what keeps MentorService.ask()'s
existing `except OllamaError` error handling working unchanged for
whichever backend is active. Every other AI-backed feature (evaluation,
session Q&A, authoring, mentor review) is untouched and stays Ollama-only.
"""
import logging
from typing import List, Optional

import httpx

from app.config import get_settings
from app.services.ollama import OllamaError

logger = logging.getLogger(__name__)

# Only OpenAI is wired up today, matching app/services/research.py.
SUPPORTED_FRONTIER_PROVIDERS = ("openai",)


class FrontierChatService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.frontier_api_key
        self.model = model or settings.frontier_model
        self.provider = provider or settings.frontier_provider
        self.timeout = timeout or 60.0
        self.base_url = "https://api.openai.com/v1"

    async def health(self) -> bool:
        return self.provider in SUPPORTED_FRONTIER_PROVIDERS and bool(self.api_key)

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        images: Optional[List[str]] = None,
        model: Optional[str] = None,
        num_ctx: Optional[int] = None,  # unused -- accepted only for interface parity with OllamaService.chat
        timeout: Optional[float] = None,
    ) -> str:
        if self.provider not in SUPPORTED_FRONTIER_PROVIDERS:
            raise OllamaError(
                f"FRONTIER_PROVIDER '{self.provider}' is not supported for mentor chat "
                f"(only {', '.join(SUPPORTED_FRONTIER_PROVIDERS)} currently)."
            )
        if not self.api_key:
            raise OllamaError(
                "MENTOR_CHAT_PROVIDER is set to 'frontier' but no FRONTIER_API_KEY is "
                "configured. Set it in .env, or set MENTOR_CHAT_PROVIDER=ollama to use "
                "local Ollama instead."
            )
        if images:
            raise OllamaError("Frontier mentor chat does not support image attachments.")

        effective_timeout = timeout if timeout is not None else self.timeout
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        try:
            async with httpx.AsyncClient(timeout=effective_timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"Frontier chat provider timed out after {effective_timeout:.0f}s."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach the frontier chat provider: {exc}") from exc

        if resp.status_code >= 400:
            body_preview = resp.text[:2000]
            logger.error(
                "Frontier chat provider returned HTTP %s: %s", resp.status_code, body_preview
            )
            raise OllamaError(
                f"Frontier chat provider rejected the request (HTTP {resp.status_code}): {body_preview}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError(
                "Frontier chat provider returned a response that could not be parsed as JSON"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OllamaError("Frontier chat provider response was not in the expected shape") from exc

        if not content:
            raise OllamaError("Frontier chat provider returned an empty response.")
        return content
