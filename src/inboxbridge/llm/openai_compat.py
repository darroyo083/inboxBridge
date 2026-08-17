"""OpenAI-compatible LLM provider (OpenCode Go / DeepSeek / OpenRouter).

Any endpoint speaking the OpenAI /chat/completions protocol works via
``base_url``. There is NO automatic fallback between providers.

The summary response is expected as JSON (``{"subject_es": ..., "summary_es": ...}``);
parsing is tolerant: when JSON is missing or malformed, the summary falls back
to the raw text and the Spanish subject is empty (the caller then shows the
original subject) — the inbound pipeline never fails because of it.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass

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
from ..models import DraftReply, DraftRequest, EmailSummary, ParsedEmail, ThreadContext
from . import prompts
from .base import (
    LLMEmptyResponse,
    LLMError,
    LLMIncompleteResponse,
    LLMRateLimited,
    LLMUnavailable,
    LLMUnsupportedModality,
    alternate_text_models,
    call_with_retry,
)
from .signature import finalize_draft_body

logger = logging.getLogger(__name__)

#: Tolerant JSON extraction for LLM output (fenced code blocks included).
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

#: Words that can never end a sentence naturally (articles, prepositions and
#: connectors in German — and Spanish equivalents for translated drafts).
#: A completion ending with one of these (without terminal punctuation) is
#: obviously truncated and must not become a sendable draft.
_DANGLING_WORDS = frozenset(
    {
        # German articles / prepositions / connectors.
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
        "einem", "einer", "für", "mit", "zu", "von", "auf", "bei", "nach",
        "über", "unter", "an", "in", "aus", "um", "gegen", "ohne", "durch",
        "und", "oder", "dass", "weil",
        # Spanish equivalents (for translated drafts).
        "el", "la", "los", "las", "un", "una", "de", "del", "con", "para",
        "por", "en", "y", "o", "que",
    }
)


def _looks_incomplete(text: str) -> bool:
    """Conservative check for OBVIOUSLY truncated completions.

    Only unambiguous signals: a trailing comma, or an ending that is a dangling
    article/preposition/connector with no terminal punctuation. Complete short
    replies such as "Danke, bis morgen." are never rejected.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if stripped[-1] in ".!?…":
        return False
    if stripped.endswith(","):
        return True
    last = stripped.rsplit(None, 1)[-1]
    return last.casefold() in _DANGLING_WORDS

#: httpx timeouts for LLM calls (long reads: summaries/drafts can be slow).
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 120.0

_PROVIDER = "opencode_go"


@dataclass
class CompletionMeta:
    """SAFE telemetry from one completion response.

    Never carries content: no visible text, no reasoning text — only counts
    and status the provider exposed. ``reasoning_available`` distinguishes
    "provider did not report reasoning tokens" from a real zero.
    """

    finish_reason: str = ""
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    max_tokens: int = 0
    reasoning_available: bool = False


def _capture_meta(meta: CompletionMeta, response: object, max_tokens: int) -> None:
    """Defensively copy response metadata into ``meta``.

    Provider response shapes differ (MiMo reports reasoning_tokens, DeepSeek
    may not); a shape difference must NEVER break the completion, so every
    access is guarded and the whole block is fail-open.
    """
    try:
        choice = getattr(response, "choices", [None])[0]
        meta.finish_reason = str(getattr(choice, "finish_reason", "") or "")
        meta.max_tokens = max_tokens
        usage = getattr(response, "usage", None)
        if usage is not None:
            meta.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", None)
            if reasoning is not None:
                meta.reasoning_tokens = int(reasoning or 0)
                meta.reasoning_available = True
    except Exception:
        # Response-shape variance must never break the completion.
        pass

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_PERMANENT = (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    APIResponseValidationError,
)

#: Provider-side audio container extensions by mime subtype (experimental).
_AUDIO_EXT = {"ogg": "ogg", "mp3": "mp3", "mpeg": "mp3", "wav": "wav", "webm": "webm", "m4a": "m4a"}


class OpenAICompatLLM:
    """LLMProvider backed by any OpenAI-compatible /chat/completions endpoint.

    The SDK's built-in retries are disabled (``max_retries=0``) because
    retries are handled by :func:`call_with_retry` with jitter — otherwise
    transient errors would be retried twice.

    A single instance talks to ONE model (``model`` overrides the configured
    text model); the :class:`AIService` facade owns routing between text /
    vision / audio models.
    """

    def __init__(self, settings: Settings | None = None, *, model: str | None = None) -> None:
        self._settings = settings or get_settings()
        api_key = self._settings.llm_api_key.get_secret_value()
        if not api_key:
            raise ValueError("LLM_API_KEY is not configured")
        self._model = model or self._settings.effective_text_model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self._settings.llm_base_url or None,
            timeout=httpx.Timeout(_READ_TIMEOUT, connect=_CONNECT_TIMEOUT),
            max_retries=0,
        )

    async def summarize_email(self, email: ParsedEmail) -> EmailSummary:
        models = alternate_text_models(
            self._model,
            self._settings.effective_text_fallback_model,
            task="incoming_summary",
        )
        content = await call_with_retry(
            lambda: self._complete(
                prompts.summary_messages(email),
                self._settings.llm_max_tokens_summary,
                model=models(),
            ),
            max_attempts=self._settings.llm_max_retries,
            base_backoff=self._settings.retry_backoff_base,
        )
        return _parse_summary(content)

    async def draft_reply(self, request: DraftRequest, thread: ThreadContext) -> DraftReply:
        models = alternate_text_models(
            self._model,
            self._settings.effective_text_fallback_model,
            task="reply",
        )
        body = await call_with_retry(
            lambda: self._complete(
                prompts.draft_messages(request, thread),
                self._settings.llm_max_tokens_draft,
                require_complete=True,
                model=models(),
            ),
            max_attempts=self._settings.llm_max_retries,
            base_backoff=self._settings.retry_backoff_base,
        )
        # Recipients come from trusted Gmail data, never from LLM output.
        # The original sender is the first message of the thread; the
        # coordinator may override via the Gmail client if needed.
        recipients = [thread.messages[0].from_] if thread.messages else []
        body = finalize_draft_body(body, self._settings.email_signature_name)
        return DraftReply(
            thread_id=request.thread_id,
            subject=thread.subject,
            to=recipients,
            cc=[],
            body=body,
            in_reply_to=thread.messages[-1].message_id if thread.messages else "",
            references="",
        )

    async def _complete(
        self,
        messages: list[ChatCompletionMessageParam],
        max_tokens: int,
        *,
        require_complete: bool = False,
        model: str | None = None,
    ) -> str:
        return await self.complete(
            messages,
            max_tokens=max_tokens,
            require_complete=require_complete,
            model=model,
        )

    async def translate_to_spanish(
        self, body: str, *, model: str | None = None
    ) -> str:
        """Translate a German draft body to Spanish (display-only, never sent)."""
        return await self.complete(
            prompts.translate_to_spanish_messages(body),
            max_tokens=self._settings.llm_max_tokens_translation,
            require_complete=True,
            model=model,
        )

    async def complete(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        max_tokens: int,
        require_complete: bool = False,
        model: str | None = None,
        meta: CompletionMeta | None = None,
    ) -> str:
        """One chat completion against THIS instance's provider.

        ``model`` overrides the model for this call (``None`` = the instance's
        configured model); the provider/gateway/credentials are always the
        SAME (no second API provider or key).

        ``meta`` receives SAFE response telemetry (finish_reason, token
        counts, max_tokens) — never content. When ``meta`` is None (direct
        provider users without an AIService), the same privacy-safe metadata
        line is logged here.

        ``require_complete`` (sendable content paths) additionally rejects
        truncated/incomplete outputs: ``finish_reason=length``, a content
        filter, or an obviously dangling ending raise a retryable/failing
        error so a cut-off draft can never become sendable. Incoming summaries
        keep the tolerant behavior (``require_complete=False``).
        """
        active = model or self._model
        local_meta = CompletionMeta() if meta is None else meta
        started = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=active,
                messages=messages,
                temperature=self._settings.llm_temperature,
                max_tokens=max_tokens,
            )
        except _RETRYABLE as exc:
            if isinstance(exc, RateLimitError):
                raise LLMRateLimited(f"LLM rate limited: {exc}") from exc
            raise LLMUnavailable(f"LLM unavailable: {exc}") from exc
        except (BadRequestError, UnprocessableEntityError) as exc:
            # Modality/format rejection (e.g. vision model refusing an image):
            # a technical, potentially fallback-worthy failure.
            raise LLMUnsupportedModality(f"LLM rejected the request: {exc}") from exc
        except _PERMANENT as exc:
            raise LLMError(f"LLM request rejected: {exc}") from exc
        except APIError as exc:
            raise LLMError(f"LLM API error: {exc}") from exc
        _capture_meta(local_meta, response, max_tokens)
        if meta is None:
            logger.info(
                "llm provider=%s model=%s finish_reason=%s completion_tokens=%d "
                "reasoning_tokens=%s max_tokens=%d duration_ms=%d",
                _PROVIDER,
                active,
                local_meta.finish_reason,
                local_meta.completion_tokens,
                (
                    str(local_meta.reasoning_tokens)
                    if local_meta.reasoning_available
                    else "unavailable"
                ),
                local_meta.max_tokens,
                int((time.monotonic() - started) * 1000),
            )
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            raise LLMEmptyResponse("LLM returned an empty response")
        if require_complete:
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                raise LLMIncompleteResponse(
                    "LLM output truncated (finish_reason=length)"
                )
            if finish_reason == "content_filter":
                raise LLMError("LLM output filtered (finish_reason=content_filter)")
            if _looks_incomplete(content):
                raise LLMIncompleteResponse(
                    "LLM output ends with an obviously dangling fragment"
                )
        return content.strip()

    async def complete_vision(
        self,
        prompt: str,
        images: list[tuple[str, bytes]],
        *,
        max_tokens: int,
    ) -> str:
        """Chat completion with inline images (data URLs) against THIS model.

        ``images`` is a list of (mime_type, bytes); mime must be an
        image/* type the provider can inline.
        """
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for mime, data in images:
            encoded = base64.b64encode(data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": content}  # type: ignore[list-item, misc]
        ]
        return await self.complete(messages, max_tokens=max_tokens)

    async def transcribe_audio(self, mime: str, data: bytes) -> str:
        """Experimental: transcribe audio via the provider's transcriptions API.

        Raises :class:`LLMError` on any failure (the caller falls back to
        asking the user to type). Uses the configured audio-capable model id;
        the feature is gated by ``ai_audio_enabled`` at the router level.
        """
        import io

        try:
            ext = _AUDIO_EXT.get(mime.split("/")[-1].lower(), "ogg")
            response = await self._client.audio.transcriptions.create(
                model=self._model,
                file=("voice." + ext, io.BytesIO(data), mime),
            )
        except Exception as exc:
            raise LLMError(f"audio transcription failed: {type(exc).__name__}") from exc
        text = getattr(response, "text", "")
        if not text or not text.strip():
            raise LLMEmptyResponse("audio transcription returned empty text")
        return text.strip()

    async def close(self) -> None:
        await self._client.close()


def _parse_summary(content: str) -> EmailSummary:
    """Parse the LLM JSON output into an EmailSummary, tolerantly.

    - valid JSON with both fields → used as-is;
    - valid JSON missing ``subject_es`` → empty subject (caller falls back);
    - not JSON at all → whole text becomes the summary, empty subject.
    Never raises: the pipeline must not fail because of a malformed subject.
    """
    match = _JSON_BLOCK_RE.search(content)
    if match is not None:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            summary_es = payload.get("summary_es")
            if isinstance(summary_es, str) and summary_es.strip():
                subject_es = payload.get("subject_es")
                return EmailSummary(
                    subject_es=subject_es if isinstance(subject_es, str) else "",
                    summary_es=summary_es.strip(),
                )
    logger.warning("LLM summary response was not valid JSON; using raw text")
    return EmailSummary(subject_es="", summary_es=content)
