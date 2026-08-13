"""Inbound pipeline: Gmail event → fetch → summarize → Telegram.

Idempotency rules:
- A message is processed at most once (``messages.message_exists`` gate).
- If the LLM fails after exhausting retries, the message is marked ``failed``
  with a ``next_retry_at`` timestamp; ``retry_failed`` re-attempts later.
- The email is NEVER silently lost: it stays ``failed`` until processed.

Status transitions: RECEIVED → SUMMARIZING → SENT_TELEGRAM (or FAILED).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from .config import Settings
from .contracts import GmailClient, LLMProvider, TelegramNotifier
from .db import Storage
from .gmail.attachments import AttachmentPasswordError
from .gmail.history import HistoryDelta
from .gmail.watcher import META_HISTORY_ID
from .models import MessageStatus, PipelineResult, PubSubEvent

logger = logging.getLogger(__name__)


class InboundPipeline:
    """Process one incoming Gmail message end-to-end (dedup-safe)."""

    def __init__(
        self,
        settings: Settings,
        gmail: GmailClient,
        llm: LLMProvider,
        telegram: TelegramNotifier,
        storage: Storage,
        *,
        history_provider: Callable[[PubSubEvent], Awaitable[HistoryDelta]] | None = None,
    ) -> None:
        self._settings = settings
        self._gmail = gmail
        self._llm = llm
        self._telegram = telegram
        self._storage = storage
        self._history_provider = history_provider

    async def process_event(self, event: PubSubEvent) -> PipelineResult:
        """Process a Pub/Sub notification; safe to call repeatedly (dedup).

        Returns the result of the LAST message processed, or a no-op result
        when there was nothing new (e.g. duplicate event, non-Primary mail).
        """
        try:
            delta = await self._delta_for(event)
        except Exception as exc:
            logger.exception("history processing failed for historyId %d", event.history_id)
            return PipelineResult(message_id="", status=MessageStatus.FAILED, error=str(exc))
        results: list[PipelineResult] = []
        for message_id in delta.message_ids:
            results.append(await self.process_message(message_id))
        ok = all(r.status != MessageStatus.FAILED for r in results)
        # The baseline advances only when every candidate was resolved
        # (no UNKNOWN Primary status) AND every Primary message processed
        # cleanly. An unknown candidate must be re-examined on the next push —
        # advancing past it could silently lose a real Primary email.
        if ok and delta.unknown_count == 0:
            self._advance_baseline(max(delta.history_id, event.history_id))
        return (
            results[-1]
            if results
            else PipelineResult(message_id="", status=MessageStatus.SENT_TELEGRAM)
        )

    async def _delta_for(self, event: PubSubEvent) -> HistoryDelta:
        if self._history_provider is not None:
            return await self._history_provider(event)
        # Fallback (tests, minimal wiring): dedup on the push message id only.
        message_id = event.message_id
        if message_id and self._storage.message_exists(message_id):
            return HistoryDelta(history_id=event.history_id, message_ids=[])
        return HistoryDelta(
            history_id=event.history_id,
            message_ids=[message_id] if message_id else [],
        )

    def _advance_baseline(self, history_id: int) -> None:
        baseline = self._storage.get_meta(META_HISTORY_ID)
        if baseline is None:
            return
        try:
            new_baseline = max(int(baseline), history_id)
        except ValueError:
            return
        self._storage.set_meta(META_HISTORY_ID, str(new_baseline))

    async def process_message(self, message_id: str) -> PipelineResult:
        """Fetch → summarize → post to Telegram; idempotent by message id."""
        if self._storage.message_exists(message_id):
            status = self._storage.get_status(message_id)
            if status in (MessageStatus.SENT_TELEGRAM, MessageStatus.SUMMARIZING):
                return PipelineResult(message_id=message_id, status=status)
        try:
            self._storage.upsert_message(message_id, "", 0, MessageStatus.RECEIVED)
            email = await self._gmail.fetch_message(message_id)
            self._storage.upsert_message(
                email.message_id, email.thread_id, email.history_id, MessageStatus.SUMMARIZING
            )
            await self._telegram.send_typing()
            summary = await self._llm.summarize_email(email)
            telegram_message_id = await self._telegram.send_summary(email, summary)
            self._storage.mark_status(
                message_id, MessageStatus.SENT_TELEGRAM, telegram_message_id
            )
            return PipelineResult(
                message_id=message_id,
                status=MessageStatus.SENT_TELEGRAM,
                telegram_message_id=telegram_message_id,
            )
        except AttachmentPasswordError as exc:
            logger.error("attachment password failed for %s: %s", message_id, exc)
            self._mark_failed(message_id)
            try:
                await self._telegram.send_notice(
                    f"Adjunto protegido: no pude abrir un PDF (revisa PDF_PASSWORD). "
                    f"El correo {message_id} queda pendiente."
                )
            except Exception:
                logger.exception("could not notify attachment error for %s", message_id)
            return PipelineResult(message_id=message_id, status=MessageStatus.FAILED)
        except Exception as exc:
            logger.exception("pipeline failed for message %s", message_id)
            self._mark_failed(message_id)
            return PipelineResult(
                message_id=message_id, status=MessageStatus.FAILED, error=str(exc)
            )

    def _mark_failed(self, message_id: str) -> None:
        delay = self._settings.retry_backoff_base
        next_retry = (datetime.now(UTC) + timedelta(seconds=delay)).timestamp()
        self._storage.mark_status(message_id, MessageStatus.FAILED)
        self._storage.bump_retry(message_id, next_retry)

    async def retry_failed(self, limit: int = 50) -> int:
        """Re-process messages stuck in FAILED whose next_retry_at elapsed."""
        now = datetime.now(UTC).timestamp()
        pending = self._storage.pending_failures(now, limit=limit)
        for row in pending:
            message_id = row["message_id"]
            attempt = row["retry_count"]
            if attempt >= self._settings.retry_max_attempts:
                logger.warning("giving up on %s after %d attempts", message_id, attempt)
                continue
            logger.info("retrying failed message %s (attempt %d)", message_id, attempt + 1)
            try:
                await self.process_message(message_id)
            except Exception:
                logger.exception("retry failed for %s", message_id)
        return len(pending)


class RetryScheduler:
    """Periodic retry loop: re-process FAILED messages with backoff + jitter."""

    def __init__(self, pipeline: InboundPipeline, interval_seconds: float = 60.0) -> None:
        self._pipeline = pipeline
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        while True:
            try:
                await self._pipeline.retry_failed()
            except Exception:
                logger.exception("retry sweep failed")
            await asyncio.sleep(self._interval + random.uniform(0.0, 15.0))

    def start(self) -> None:
        self._task = asyncio.create_task(self.run(), name="retry-scheduler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
