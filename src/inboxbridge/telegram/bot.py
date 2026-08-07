"""Telegram bot: group authorization, notifications, reply flow.

Only the configured group chat is ever touched. The bot reacts only to:
replies to its own messages, @mentions, and commands. Everything else in the
group is read and ignored.

Key design decisions (documented for the coordinator):

- A small ``Sender`` protocol (structurally satisfied by PTB's ``Bot``) keeps
  every send path testable without network; production wiring builds a real
  ``Bot`` from ``TELEGRAM_BOT_TOKEN``.
- ``process_update`` is the single entry point for ALL update kinds
  (messages, commands, callback queries); the PTB handlers are thin adapters.
- Draft confirmation uses an in-memory token registry. ``send_draft_for_
  confirmation`` posts the draft with inline buttons carrying
  ``confirm:<token>`` / ``cancel:<token>`` and returns the message_id. The
  coordinator awaits ``wait_for_confirmation(message_id)`` and persists the
  draft via ``db.Storage.create_draft`` ONLY after it returns True (the bot
  never writes draft rows itself). ``/cancel`` resolves all pending drafts to
  False.
- Summary → thread mapping: ``send_summary`` writes ``meta["tg:<message_id>"] =
  thread_id`` through ``db.Storage`` so a later reply to that message can be
  matched to the thread (survives restarts).
- Streaming is NOT enabled: the ``LLMProvider`` contract has no streaming
  surface. A typing indicator is shown while the LLM runs.
- Display text is scrubbed so URLs are never clickable (untrusted content).
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from telegram import (
    Bot,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
    MessageEntity,
    ReplyParameters,
    Update,
    User,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, MessageHandler, filters

from ..config import Settings
from ..contracts import TelegramNotifier
from ..db import Storage
from ..models import DraftReply, ParsedEmail

logger = logging.getLogger(__name__)

_CALLBACK_RE = re.compile(r"^(confirm|cancel):([A-Za-z0-9_-]{1,64})$")
_URL_SCHEME_RE = re.compile(r"https?://", re.IGNORECASE)
_TG_META_PREFIX = "tg:"
_CONFIRM_TIMEOUT_SECONDS = 900.0


class Sender(Protocol):
    """Minimal Telegram API surface used by the bot (satisfied by telegram.Bot)."""

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        parse_mode: ParseMode | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        link_preview_options: LinkPreviewOptions | None = None,
        reply_parameters: ReplyParameters | None = None,
    ) -> Message: ...

    async def send_chat_action(self, chat_id: int | str, action: ChatAction) -> bool: ...

    async def edit_message_text(
        self,
        text: str,
        *,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message | bool: ...

    async def answer_callback_query(
        self, callback_query_id: str, text: str | None = None, show_alert: bool = False
    ) -> bool: ...


@dataclass(frozen=True)
class ReplyRequest:
    """A group member asked for a reply to a thread (consumed by the responder)."""

    #: Empty string when the request carries no thread mapping (e.g. a plain
    #: mention); the coordinator decides how to handle it.
    thread_id: str
    user_instructions: str
    source_message_id: int


@dataclass
class _PendingDraft:
    token: str
    draft: DraftReply
    message_id: int
    decided: bool | None = None
    future: asyncio.Future[bool] | None = None

    def resolve(self, confirmed: bool) -> None:
        self.decided = confirmed
        if self.future is not None and not self.future.done():
            self.future.set_result(confirmed)


def neutralize_links(text: str) -> str:
    """Make URLs non-clickable in Telegram display (https:// → hxxps://)."""
    return _URL_SCHEME_RE.sub(lambda match: f"hxxp{match.group(0)[4:]}", text)


class TelegramBot(TelegramNotifier):
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        *,
        sender: Sender | None = None,
        bot_user_id: int | None = None,
        bot_username: str | None = None,
        status_provider: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._sender = sender
        self._bot_user_id = bot_user_id
        self._bot_username = bot_username
        self._status_provider = status_provider
        self._allowed_chat_id = settings.telegram_allowed_chat_id
        self._queue: asyncio.Queue[ReplyRequest] = asyncio.Queue()
        self._pending_drafts: dict[str, _PendingDraft] = {}
        self._application: Application[Any, Any, Any, Any, Any, Any] | None = None
        self._started = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _ensure_sender(self) -> Sender:
        if self._sender is None:
            token = self._settings.telegram_bot_token.get_secret_value()
            if not token:
                raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")
            self._sender = Bot(token=token)
        return self._sender

    async def start(self) -> None:
        """Build the PTB application, register handlers and start polling."""
        if self._started:
            return
        if self._allowed_chat_id == 0:
            raise RuntimeError("TELEGRAM_ALLOWED_CHAT_ID is not configured")
        sender = self._ensure_sender()
        if self._bot_user_id is None or self._bot_username is None:
            if not isinstance(sender, Bot):
                raise RuntimeError(
                    "bot_user_id/bot_username must be injected when sender is not a real Bot"
                )
            me: User = await sender.get_me()
            self._bot_user_id = me.id
            self._bot_username = me.username
        application = Application.builder().bot(cast(Bot, sender)).build()
        application.add_handler(MessageHandler(filters.TEXT, self._on_message))
        application.add_handler(CallbackQueryHandler(self._on_callback_query, pattern=_CALLBACK_RE))
        await application.initialize()
        await application.start()
        if application.updater is None:
            raise RuntimeError("polling updater unavailable")
        await application.updater.start_polling(drop_pending_updates=True)
        self._application = application
        self._started = True
        logger.info("Telegram bot started for chat %s", self._allowed_chat_id)

    async def stop(self) -> None:
        if self._application is not None:
            application = self._application
            if application.updater is not None:
                await application.updater.stop()
            await application.stop()
            await application.shutdown()
            self._application = None
        self._started = False

    # ── PTB handler adapters ───────────────────────────────────────────────

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.process_update(update)

    async def _on_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.process_update(update)

    # ── update entry point ─────────────────────────────────────────────────

    async def process_update(self, update: Update) -> None:
        """Single entry point for every update kind (used by handlers and tests)."""
        if update.callback_query is not None:
            await self._handle_callback_query(update.callback_query)
            return
        message = update.effective_message
        if message is None or message.chat is None or message.from_user is None:
            return
        if message.chat.id != self._allowed_chat_id:
            return
        if message.from_user.is_bot:
            return
        text = message.text
        if text is None:
            return
        if text.startswith("/"):
            await self._handle_command(message, text)
            return
        await self._handle_plain_message(message)

    # ── commands ───────────────────────────────────────────────────────────

    async def _handle_command(self, message: Message, text: str) -> None:
        command = text.split(maxsplit=1)[0].lower().split("@", maxsplit=1)[0]
        if command == "/status":
            await self._run_status(message)
        elif command == "/cancel":
            await self._run_cancel(message)

    async def _run_status(self, message: Message) -> None:
        if self._status_provider is None:
            text = "Estado: no hay checks configurados."
        else:
            text = await self._status_provider()
        await self._send(text, reply_to=message.message_id)

    async def _run_cancel(self, message: Message) -> None:
        cancelled = 0
        for pending in list(self._pending_drafts.values()):
            pending.resolve(False)
            cancelled += 1
        self._pending_drafts.clear()
        text = "Nada que cancelar." if cancelled == 0 else f"Borradores cancelados: {cancelled}."
        await self._send(text, reply_to=message.message_id)

    # ── plain messages ─────────────────────────────────────────────────────

    async def _handle_plain_message(self, message: Message) -> None:
        if not (self._is_reply_to_own(message) or self._is_mentioned(message)):
            return
        thread_id = ""
        reply = message.reply_to_message
        if reply is not None and self._is_own_message(reply):
            if any(p.message_id == reply.message_id for p in self._pending_drafts.values()):
                return  # draft confirmation message: use the inline buttons
            thread_id = self._storage.get_meta(f"{_TG_META_PREFIX}{reply.message_id}") or ""
        self._queue.put_nowait(
            ReplyRequest(
                thread_id=thread_id,
                user_instructions=message.text or "",
                source_message_id=message.message_id,
            )
        )

    def _is_own_message(self, message: Message) -> bool:
        return (
            self._bot_user_id is not None
            and message.from_user is not None
            and message.from_user.id == self._bot_user_id
        )

    def _is_reply_to_own(self, message: Message) -> bool:
        reply = message.reply_to_message
        return reply is not None and self._is_own_message(reply)

    def _is_mentioned(self, message: Message) -> bool:
        text = message.text or ""
        for entity in message.entities or ():
            if entity.type == MessageEntity.MENTION and self._bot_username is not None:
                mention = text[entity.offset : entity.offset + entity.length].lstrip("@")
                if mention.lower() == self._bot_username.lower():
                    return True
            elif entity.type == MessageEntity.TEXT_MENTION and entity.user is not None:
                if entity.user.id == self._bot_user_id:
                    return True
        return self._bot_username is not None and f"@{self._bot_username.lower()}" in text.lower()

    # ── callback queries (draft confirm/cancel buttons) ────────────────────

    async def _handle_callback_query(self, query: CallbackQuery) -> None:
        message = query.message
        if message is None or message.chat is None:
            return
        if message.chat.id != self._allowed_chat_id:
            return
        match = _CALLBACK_RE.fullmatch(query.data or "")
        if match is None:
            return
        action, token = match.group(1), match.group(2)
        pending = self._pending_drafts.get(token)
        if pending is None:
            await self._ensure_sender().answer_callback_query(
                query.id, "Este borrador ya no está disponible."
            )
            return
        confirmed = action == "confirm"
        pending.resolve(confirmed)
        sender = self._ensure_sender()
        await sender.answer_callback_query(query.id, "Enviando…" if confirmed else "Cancelado.")
        await sender.edit_message_text(
            "Confirmado ✓" if confirmed else "Cancelado.",
            chat_id=message.chat.id,
            message_id=message.message_id,
        )

    # ── TelegramNotifier implementation ────────────────────────────────────

    async def send_summary(self, email: ParsedEmail, summary: str) -> int:
        sender = self._ensure_sender()
        sender_name = email.sender.name or email.sender.email
        text = f"{email.subject}\nDe: {sender_name}\n\n{neutralize_links(summary)}"
        message = await sender.send_message(
            self._allowed_chat_id,
            text,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
        self._storage.set_meta(f"{_TG_META_PREFIX}{message.message_id}", email.thread_id)
        return message.message_id

    async def send_notice(self, text: str) -> int:
        return await self._send(neutralize_links(text))

    async def send_typing(self) -> None:
        await self._ensure_sender().send_chat_action(self._allowed_chat_id, ChatAction.TYPING)

    async def send_draft_for_confirmation(self, draft: DraftReply) -> int:
        token = secrets.token_urlsafe(8)
        to_line = (
            ", ".join(str(address) for address in draft.to) if draft.to else "(sin destinatario)"
        )
        text = (
            f"Borrador de respuesta\n"
            f"Para: {to_line}\n"
            f"Asunto: {draft.subject}\n\n"
            f"{neutralize_links(draft.body)}"
        )
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Confirmar", callback_data=f"confirm:{token}"),
                    InlineKeyboardButton("Cancelar", callback_data=f"cancel:{token}"),
                ]
            ]
        )
        message = await self._ensure_sender().send_message(
            self._allowed_chat_id,
            text,
            reply_markup=keyboard,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
        self._pending_drafts[token] = _PendingDraft(
            token=token, draft=draft, message_id=message.message_id
        )
        return message.message_id

    async def wait_for_confirmation(
        self, message_id: int, timeout_seconds: float = _CONFIRM_TIMEOUT_SECONDS
    ) -> bool:
        """Wait for the user's confirm/cancel decision on a draft message.

        The coordinator calls this right after ``send_draft_for_confirmation``
        and persists the draft (``db.Storage.create_draft``) only if True.
        Returns False on cancel, timeout, or an unknown message_id.
        """
        pending = next(
            (p for p in self._pending_drafts.values() if p.message_id == message_id), None
        )
        if pending is None:
            return False
        if pending.decided is not None:
            self._pending_drafts.pop(pending.token, None)
            return pending.decided
        if pending.future is None:
            pending.future = asyncio.get_running_loop().create_future()
        try:
            result = await asyncio.wait_for(pending.future, timeout=timeout_seconds)
        except TimeoutError:
            self._pending_drafts.pop(pending.token, None)
            return False
        self._pending_drafts.pop(pending.token, None)
        return result

    # ── events consumed by the coordinator's responder ─────────────────────

    async def reply_requests(self) -> AsyncIterator[ReplyRequest]:
        """Stream of reply requests; consume this forever with ``async for``."""
        while True:
            yield await self._queue.get()

    # ── helpers ────────────────────────────────────────────────────────────

    async def _send(self, text: str, *, reply_to: int | None = None) -> int:
        sender = self._ensure_sender()
        message = await sender.send_message(
            self._allowed_chat_id,
            text,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
            reply_parameters=ReplyParameters(message_id=reply_to) if reply_to is not None else None,
        )
        return message.message_id
