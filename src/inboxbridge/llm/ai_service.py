"""Capability-aware AI routing facade.

Configuration-driven, zero hardcoded model IDs in business logic:

- ``text(...)``          → configured text model (DeepSeek)
- ``vision(...)``        → configured vision model (MiMo), bounded fallback
                           to the configured fallback model (Luna) ONLY on
                           technical/capability failures
- ``document_vision(...)``→ scanned-PDF: bounded page render → vision model
- ``audio(...)``         → experimental, gated by ``ai_audio_enabled``

Fallback policy (never on "I didn't like the answer"): provider unavailable,
rate limited, timeout, empty/malformed response, or unsupported modality.
Never falls back to local OCR (there is none).

Observability: safe metadata only (task, model ids, duration, success,
fallback_used) — never email content, images, audio or payloads.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessageParam

from ..config import Settings, get_settings
from . import prompts
from .base import (
    LLMEmptyResponse,
    LLMError,
    LLMRateLimited,
    LLMUnavailable,
    LLMUnsupportedModality,
)
from .openai_compat import OpenAICompatLLM
from .pdf_render import PdfRenderError, render_pdf_pages

logger = logging.getLogger(__name__)

_PROVIDER = "opencode_go"

#: Fallback-worthy technical failures. Content-quality preferences are NOT here.
_FALLBACK_EXCEPTIONS = (
    LLMUnavailable,
    LLMRateLimited,
    LLMEmptyResponse,
    LLMUnsupportedModality,
    TimeoutError,
    PdfRenderError,
)


@dataclass
class AiCall:
    """Observability record for one AI invocation (safe metadata only)."""

    task: str
    model: str
    provider: str = _PROVIDER
    duration_ms: int = 0
    success: bool = False
    fallback_used: bool = False
    fallback_model: str = ""


class AIService:
    """Routes tasks to the configured text/vision/audio models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._text_llm: OpenAICompatLLM | None = None
        self._vision_llm: OpenAICompatLLM | None = None
        self._vision_fallback_llm: OpenAICompatLLM | None = None
        self.calls: list[AiCall] = []  # test-visible observability log

    # ── clients ─────────────────────────────────────────────────────────────

    @property
    def text_model(self) -> str:
        return self._settings.effective_text_model

    @property
    def vision_model(self) -> str:
        return self._settings.ai_vision_model

    @property
    def vision_fallback_model(self) -> str:
        return self._settings.ai_vision_fallback_model

    def _text_client(self) -> OpenAICompatLLM:
        if self._text_llm is None:
            self._text_llm = OpenAICompatLLM(self._settings, model=self.text_model)
        return self._text_llm

    def _vision_client(self, fallback: bool = False) -> OpenAICompatLLM:
        model = self.vision_fallback_model if fallback else self.vision_model
        if fallback:
            if self._vision_fallback_llm is None:
                self._vision_fallback_llm = OpenAICompatLLM(self._settings, model=model)
            return self._vision_fallback_llm
        if self._vision_llm is None:
            self._vision_llm = OpenAICompatLLM(self._settings, model=model)
        return self._vision_llm

    # ── text ────────────────────────────────────────────────────────────────

    async def text(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        max_tokens: int = 1000,
        task: str = "text",
        require_complete: bool = False,
    ) -> str:
        """Text-only completion on the configured text model.

        ``require_complete`` (sendable content paths) rejects truncated/
        incomplete outputs (see :meth:`OpenAICompatLLM.complete`).
        """
        record = AiCall(task=task, model=self.text_model)
        started = time.monotonic()
        try:
            result = await self._text_client().complete(
                messages, max_tokens=max_tokens, require_complete=require_complete
            )
            record.success = True
            self._finish(record, started)
            return result
        except LLMError as exc:
            record.success = False
            self._finish(record, started, error=exc)
            raise

    async def translate_to_spanish(self, body: str) -> str:
        """Translate a German draft body to Spanish (display-only, never sent)."""
        return await self.text(
            prompts.translate_to_spanish_messages(body),
            max_tokens=self._settings.llm_max_tokens_draft,
            task="translate",
            require_complete=True,
        )

    # ── vision ──────────────────────────────────────────────────────────────

    async def vision(
        self,
        prompt: str,
        images: list[tuple[str, bytes]],
        *,
        max_tokens: int = 1000,
        task: str = "vision",
        allow_fallback: bool = True,
    ) -> str:
        """Vision completion: primary model, bounded technical fallback.

        ``images`` is (mime, bytes) with image/* mimes. The fallback model is
        used at most once, and only for real technical failures.
        """
        record = AiCall(task=task, model=self.vision_model)
        started = time.monotonic()
        try:
            result = await self._vision_client().complete_vision(
                prompt, images, max_tokens=max_tokens
            )
            record.success = True
            self._finish(record, started)
            return result
        except _FALLBACK_EXCEPTIONS as exc:
            if not allow_fallback or not self.vision_fallback_model:
                record.success = False
                self._finish(record, started, error=exc)
                raise
            logger.warning(
                "vision model %s failed (%s); falling back to %s",
                self.vision_model,
                type(exc).__name__,
                self.vision_fallback_model,
            )
            record.fallback_used = True
            record.fallback_model = self.vision_fallback_model
            try:
                result = await self._vision_client(fallback=True).complete_vision(
                    prompt, images, max_tokens=max_tokens
                )
                record.success = True
                record.model = self.vision_fallback_model
                self._finish(record, started)
                return result
            except _FALLBACK_EXCEPTIONS as exc2:
                record.success = False
                self._finish(record, started, error=exc2)
                raise
        except LLMError:
            # Non-technical rejection (auth, policy): do NOT burn a fallback.
            record.success = False
            self._finish(record, started)
            raise

    async def document_vision(
        self,
        prompt: str,
        pdf_bytes: bytes,
        *,
        max_tokens: int = 1500,
        task: str = "document_vision",
    ) -> str:
        """Scanned-PDF analysis: bounded render → external vision model.

        Raises :class:`PdfRenderPasswordError` / :class:`PdfRenderError`
        when the document cannot be rendered (caller reports gracefully).
        """
        pages = render_pdf_pages(
            pdf_bytes,
            max_pages=self._settings.ai_vision_max_pages,
            max_dimension=self._settings.ai_vision_max_dimension,
        )
        if not pages:
            raise PdfRenderError("PDF has no renderable pages")
        images = [("image/png", page) for page in pages]
        return await self.vision(
            prompt, images, max_tokens=max_tokens, task=task
        )

    # ── audio (experimental) ────────────────────────────────────────────────

    async def audio(self, mime: str, data: bytes, *, task: str = "audio") -> str:
        """Experimental transcription; gated by ``ai_audio_enabled``.

        Raises :class:`LLMError` on any failure — callers must fall back to
        asking the user to type the instruction.
        """
        if not self._settings.ai_audio_enabled:
            raise LLMError("audio is disabled (AI_AUDIO_ENABLED=false)")
        record = AiCall(task=task, model=self._settings.ai_text_model)
        started = time.monotonic()
        try:
            result = await self._text_client().transcribe_audio(mime, data)
            record.success = True
            self._finish(record, started)
            return result
        except LLMError as exc:
            record.success = False
            self._finish(record, started, error=exc)
            raise

    # ── helpers ─────────────────────────────────────────────────────────────

    def _finish(
        self, record: AiCall, started: float, *, error: Exception | None = None
    ) -> None:
        record.duration_ms = int((time.monotonic() - started) * 1000)
        self.calls.append(record)
        if record.success:
            logger.info(
                "ai provider=%s task=%s model=%s duration_ms=%d fallback_used=%s",
                record.provider, record.task, record.model,
                record.duration_ms, record.fallback_used,
            )
        else:
            logger.warning(
                "ai provider=%s task=%s model=%s duration_ms=%d fallback_used=%s error=%s",
                record.provider, record.task, record.model,
                record.duration_ms, record.fallback_used,
                type(error).__name__ if error else "unknown",
            )

    async def close(self) -> None:
        for client in (self._text_llm, self._vision_llm, self._vision_fallback_llm):
            if client is not None:
                await client.close()
