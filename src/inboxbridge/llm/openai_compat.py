"""OpenAI-compatible LLM provider (OpenCode Go / DeepSeek / OpenRouter).

Any endpoint speaking the OpenAI /chat/completions protocol works via
``base_url``. There is NO automatic fallback between providers.
"""

from __future__ import annotations

import logging

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai.types.chat import ChatCompletionMessageParam

from ..config import Settings, get_settings
from ..models import DraftReply, DraftRequest, ParsedEmail, ThreadContext
from . import prompts
from .base import LLMError, LLMInvalidResponse, LLMRateLimited, LLMUnavailable, call_with_retry

logger = logging.getLogger(__name__)

#: httpx timeouts for LLM calls (long reads: summaries/drafts can be slow).
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 120.0

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_PERMANENT = (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    BadRequestError,
    APIResponseValidationError,
)


class OpenAICompatLLM:
    """LLMProvider backed by any OpenAI-compatible /chat/completions endpoint.

    The SDK's built-in retries are disabled (``max_retries=0``) because
    retries are handled by :func:`call_with_retry` with jitter — otherwise
    transient errors would be retried twice.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        api_key = self._settings.llm_api_key.get_secret_value()
        if not api_key:
            raise ValueError("LLM_API_KEY is not configured")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self._settings.llm_base_url or None,
            timeout=httpx.Timeout(_READ_TIMEOUT, connect=_CONNECT_TIMEOUT),
            max_retries=0,
        )

    async def summarize_email(self, email: ParsedEmail) -> str:
        return await call_with_retry(
            lambda: self._complete(
                prompts.summary_messages(email), self._settings.llm_max_tokens_summary
            ),
            max_attempts=self._settings.llm_max_retries,
            base_backoff=self._settings.retry_backoff_base,
        )

    async def draft_reply(self, request: DraftRequest, thread: ThreadContext) -> DraftReply:
        body = await call_with_retry(
            lambda: self._complete(
                prompts.draft_messages(request, thread), self._settings.llm_max_tokens_draft
            ),
            max_attempts=self._settings.llm_max_retries,
            base_backoff=self._settings.retry_backoff_base,
        )
        # Recipients come from trusted Gmail data, never from LLM output.
        # The original sender is the first message of the thread; the
        # coordinator may override via the Gmail client if needed.
        recipients = [thread.messages[0].from_] if thread.messages else []
        return DraftReply(
            thread_id=request.thread_id,
            subject=thread.subject,
            to=recipients,
            cc=[],
            body=body,
            in_reply_to=thread.messages[-1].message_id if thread.messages else "",
            references="",
        )

    async def _complete(self, messages: list[ChatCompletionMessageParam], max_tokens: int) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=self._settings.llm_temperature,
                max_tokens=max_tokens,
            )
        except _RETRYABLE as exc:
            if isinstance(exc, RateLimitError):
                raise LLMRateLimited(f"LLM rate limited: {exc}") from exc
            raise LLMUnavailable(f"LLM unavailable: {exc}") from exc
        except _PERMANENT as exc:
            raise LLMError(f"LLM request rejected: {exc}") from exc
        except APIError as exc:
            raise LLMError(f"LLM API error: {exc}") from exc
        if not response.choices or not response.choices[0].message.content:
            raise LLMInvalidResponse("LLM returned an empty response")
        return response.choices[0].message.content.strip()

    async def close(self) -> None:
        await self._client.close()
