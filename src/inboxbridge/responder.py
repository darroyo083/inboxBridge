"""Reply flow: Telegram request → thread context → draft → confirm → send.

Separation of concerns (MVP requirement):
- ``prepare_draft`` ONLY generates a draft and shows it for confirmation.
- ``send_draft`` ONLY sends an already-confirmed draft.
- Sending is technically impossible unless ``SEND_EMAILS=true`` (the Gmail
  client raises ``SendingDisabledError`` otherwise — kill switch).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from .config import Settings
from .contracts import GmailClient, LLMProvider
from .db import Storage
from .gmail.client import SendingDisabledError
from .models import DraftReply, DraftRequest, DraftStatus
from .telegram.bot import ReplyRequest, TelegramBot

logger = logging.getLogger(__name__)


class ReplyCoordinator:
    """Consumes Telegram reply requests and drives the draft/confirm/send cycle."""

    def __init__(
        self,
        settings: Settings,
        gmail: GmailClient,
        llm: LLMProvider,
        bot: TelegramBot,
        storage: Storage,
    ) -> None:
        self._settings = settings
        self._gmail = gmail
        self._llm = llm
        self._bot = bot
        self._storage = storage

    async def run_forever(self) -> None:
        """Process reply requests from the bot queue indefinitely."""
        async for request in self._bot.reply_requests():
            await self._handle_request(request)

    async def _handle_request(self, request: ReplyRequest) -> None:
        if not request.thread_id:
            await self._bot.send_notice(
                "No puedo asociar tu petición a ningún hilo. Responde directamente "
                "a un resumen de InboxBridge y escribe lo que quieres responder."
            )
            return
        try:
            await self._bot.send_typing()
            thread = await self._gmail.fetch_thread_context(request.thread_id)
            draft_request = DraftRequest(
                thread_id=request.thread_id,
                user_instructions=request.user_instructions,
                language="de",
            )
            draft: DraftReply = await self._llm.draft_reply(draft_request, thread)
            await self._present_draft(draft)
        except Exception:
            logger.exception("reply flow failed for thread %s", request.thread_id)
            await self._bot.send_notice("No pude preparar la respuesta. Inténtalo de nuevo.")

    async def _present_draft(self, draft: DraftReply) -> None:
        """Show recipients + body and wait for explicit confirmation.

        The draft row is persisted ONLY after confirmation, so a rejected or
        expired draft never leaves a trace in the DB.
        """
        message_id = await self._bot.send_draft_for_confirmation(draft)
        confirmed = await self._bot.wait_for_confirmation(message_id)
        if not confirmed:
            logger.info("draft for thread %s not confirmed; discarding", draft.thread_id)
            return
        draft_id = self._storage.create_draft(draft.thread_id, None, draft)
        self._storage.set_draft_status(draft_id, DraftStatus.CONFIRMED)
        await self._send_confirmed(draft_id, draft)

    async def _send_confirmed(self, draft_id: int, draft: DraftReply) -> None:
        try:
            new_message_id = await self._gmail.send_reply(draft)
        except SendingDisabledError:
            # Kill switch active: never attempt sending. Notify, keep pending.
            await self._bot.send_notice(
                "SEND_EMAILS=false: el envío está desactivado (kill switch). "
                "El borrador queda guardado; actívalo en la configuración para enviar."
            )
            return
        except Exception:
            logger.exception("sending draft %d failed", draft_id)
            await self._bot.send_notice("El envío falló. El borrador queda guardado.")
            return
        self._storage.set_draft_status(draft_id, DraftStatus.SENT)
        await self._bot.send_notice(
            f"Enviado ✓ (nuevo message-id {new_message_id})"
        )

    # ── programmatic send (tests / future automation) ──────────────────────

    async def send_confirmed_draft(self, draft: DraftReply) -> str:
        """Send a draft that has already been confirmed elsewhere.

        Returns the new Gmail message id. Raises ``SendingDisabledError`` when
        the kill switch is active.
        """
        return await self._gmail.send_reply(draft)


class ReplyWorker:
    """Task wrapper so the app can start/stop the reply loop cleanly."""

    def __init__(self, coordinator: ReplyCoordinator) -> None:
        self._coordinator = coordinator
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(
            self._coordinator.run_forever(), name="reply-coordinator"
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
