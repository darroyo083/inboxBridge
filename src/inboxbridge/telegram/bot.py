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
  matched to the thread (survives restarts). It also writes
  ``meta["tgm:<message_id>"] = gmail message_id`` (IDs only) so the
  "Ver original" button can fetch the exact original email on demand —
  Gmail stays the source of truth, bodies are never persisted.
- "Ver original": the summary is edited to carry a ``view:<tg_message_id>``
  button. On press, the original email is fetched from Gmail (no LLM call,
  no SQLite body) and posted as temporary message(s) with an
  ``Ocultar original`` button (``hide:<tg_message_id>``). The temporary
  message ids are kept in ``meta["tgo:<tg_message_id>"]`` so hiding works
  after restarts; hiding deletes them. Original content is untrusted:
  links are neutralized and text is plain (no markdown/HTML rendering).
- Streaming is NOT enabled: the ``LLMProvider`` contract has no streaming
  surface. A typing indicator is shown while the LLM runs.
- Display text is scrubbed so URLs are never clickable (untrusted content).
- Explicit memory V1 (``/remember``, ``/memory``, ``/forget``): per-member
  facts stored in SQLite via ``db.Storage``, keyed by the Telegram user id
  (never the chat id) and isolated between members. Facts are ONLY stored
  when a member explicitly asks; nothing is ever learned from emails or
  conversation. A fact is normalized into a matching key (first 4 words,
  lowercased, punctuation stripped — or the whole text when short) plus the
  original text as the displayed value. Secret-like content (api keys,
  tokens, passwords) is rejected and never persisted. The requesting
  member's facts ride along on ``ReplyRequest`` so the responder can feed
  them to the draft prompt as bounded, untrusted context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from telegram import (
    Bot,
    CallbackQuery,
    File,
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
from ..models import DraftReply, EmailSummary, OutgoingAttachment, ParsedEmail

logger = logging.getLogger(__name__)

_CALLBACK_RE = re.compile(r"^(confirm|cancel|view|hide|reply|resend):([A-Za-z0-9_-]{1,64})$")
_URL_SCHEME_RE = re.compile(r"https?://", re.IGNORECASE)
_TG_META_PREFIX = "tg:"  # tg message_id -> thread_id
_TG_GMAIL_PREFIX = "tgm:"  # tg message_id -> gmail message_id (IDs only)
_TG_SENDER_PREFIX = "tgs:"  # tg message_id -> sender display name
_TG_ORIG_PREFIX = "tgo:"  # tg message_id -> JSON list of temporary original message ids
_CONFIRM_TIMEOUT_SECONDS = 900.0

#: Telegram hard limit is 4096 chars per message; stay safely below.
_MAX_MSG_CHARS = 3900
#: Cap on temporary "original" messages to avoid flooding the group.
_MAX_ORIGINAL_MESSAGES = 8

#: Explicit memory V1: a fact's matching key is its first N normalized words.
_MEMORY_KEY_WORDS = 4

#: Reject credential-like content before persisting a memory. Words cover
#: Spanish and English ("clave" alone is intentionally NOT matched to avoid
#: false positives on non-secret usage); prefixes cover common API tokens.
_SECRET_RE = re.compile(
    r"("
    r"\b(api[ _-]?key|apikey|password|passwd|secret|contraseña|token|bearer"
    r"|client[ _-]?secret|credenciales)\b"
    r"|sk-[A-Za-z0-9]{4,}"
    r"|ghp_[A-Za-z0-9]{4,}"
    r"|xox[baprs]-[A-Za-z0-9-]{4,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r"|BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY"
    r")",
    re.IGNORECASE,
)


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

    async def delete_message(self, chat_id: int | str, message_id: int) -> bool: ...

    async def get_file(self, file_id: str) -> File: ...


@dataclass(frozen=True)
class ReplyRequest:
    """A group member asked for a reply to a thread (consumed by the responder)."""

    #: Empty string when the request carries no thread mapping (e.g. a plain
    #: mention); the coordinator decides how to handle it.
    thread_id: str
    user_instructions: str
    source_message_id: int
    #: Explicit facts the requesting member asked the bot to remember
    #: (bounded, untrusted context for the LLM).
    memory: tuple[str, ...] = ()
    #: Telegram user id of the member who initiated the request. Zero when
    #: unknown (older flows); used for draft ownership/isolation.
    user_id: int = 0
    #: Telegram-supplied files (documents/photos) downloaded to a temporary
    #: directory; the responder attaches them to the outgoing reply.
    attachments: tuple[OutgoingAttachment, ...] = ()


@dataclass
class _PendingDraft:
    token: str
    draft: DraftReply
    message_id: int
    #: Telegram user id that requested the draft; only this member may
    #: confirm/cancel it (group isolation).
    user_id: int = 0
    decided: bool | None = None
    future: asyncio.Future[bool] | None = None

    def resolve(self, confirmed: bool) -> None:
        self.decided = confirmed
        if self.future is not None and not self.future.done():
            self.future.set_result(confirmed)


def neutralize_links(text: str) -> str:
    """Make URLs non-clickable in Telegram display (https:// → hxxps://)."""
    return _URL_SCHEME_RE.sub(lambda match: f"hxxp{match.group(0)[4:]}", text)


def _normalize_memory_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace (matching keys)."""
    stripped = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(stripped.split())


def _split_memory_fact(text: str) -> tuple[str, str]:
    """Split a fact into (key, value) for storage.

    Key = the first ``_MEMORY_KEY_WORDS`` normalized words, or the whole
    normalized text when the fact is short (≤ 4 words). The key is what
    ``/memory`` and ``/forget`` match against. Value = the original text,
    kept for display (``/memory`` shows what the member actually wrote).

    Raises ValueError when the text has no meaningful content.
    """
    normalized = _normalize_memory_text(text)
    words = normalized.split()
    if not words:
        raise ValueError("memory fact is empty")
    key = (
        normalized
        if len(words) <= _MEMORY_KEY_WORDS
        else " ".join(words[:_MEMORY_KEY_WORDS])
    )
    return key, text.strip()


def _format_sender(email: ParsedEmail) -> str:
    """Human-readable sender display: "Name (email)", or just "email".

    The address is shown alone when the name is empty/whitespace or when it
    duplicates the address (avoids noise like "ana@example.com
    (ana@example.com)").
    """
    address = email.sender
    name = address.name.strip() if address.name else ""
    if name and name.casefold() != address.email.casefold():
        return f"{name} ({address.email})"
    return address.email


def sanitize_filename(filename: str) -> str:
    """Reduce an untrusted Telegram filename to safe display metadata.

    Keeps the basename only (no traversal), drops control characters and
    caps the length; never used as a filesystem path on its own.
    """
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]+", "", name).strip()
    if name in ("", ".", ".."):
        return "attachment"
    if len(name) > 180:
        stem, dot, suffix = name.rpartition(".")
        name = stem[: 180 - len(suffix) - 1] + dot + suffix
    return name


def _guess_mime(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024} KB"
    return f"{size_bytes // (1024 * 1024)} MB"


def _remove_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("could not remove temp file %s", path)


def _split_original(text: str) -> list[str]:
    """Split untrusted original text into Telegram-safe chunks.

    Splits on newlines when possible (prefer whole lines), truncates the last
    chunk with a clear marker when the cap is hit, and returns at most
    ``_MAX_ORIGINAL_MESSAGES`` chunks.
    """
    if len(text) <= _MAX_MSG_CHARS:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > _MAX_MSG_CHARS and len(chunks) < _MAX_ORIGINAL_MESSAGES:
        cut = remaining.rfind("\n", 0, _MAX_MSG_CHARS)
        if cut < _MAX_MSG_CHARS // 2:
            cut = _MAX_MSG_CHARS
        chunks.append(remaining[:cut].rstrip("\n"))
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        if len(chunks) >= _MAX_ORIGINAL_MESSAGES:
            chunks.append(remaining[:_MAX_MSG_CHARS - 30] + "\n[…original truncado…]")
        else:
            chunks.append(remaining)
    return chunks


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
        original_fetcher: Callable[[str], Awaitable[ParsedEmail]] | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._sender = sender
        self._bot_user_id = bot_user_id
        self._bot_username = bot_username
        self._status_provider = status_provider
        self._original_fetcher = original_fetcher
        self._resend_callback: Callable[[int], Awaitable[None]] | None = None
        self._allowed_chat_id = settings.telegram_allowed_chat_id
        self._queue: asyncio.Queue[ReplyRequest] = asyncio.Queue()
        self._pending_drafts: dict[str, _PendingDraft] = {}
        #: user_id -> (tg summary message_id, thread_id): a member pressed
        #: "Responder" and their next plain message is their reply intent.
        self._pending_replies: dict[int, tuple[int, str]] = {}
        self._application: Application[Any, Any, Any, Any, Any, Any] | None = None
        self._started = False

    def register_resend_callback(self, callback: Callable[[int], Awaitable[None]]) -> None:
        """Wire the "Reintentar envío" button to the reply coordinator."""
        self._resend_callback = callback

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
        if text is None and message.document is None and not message.photo:
            return
        if text is not None and text.startswith("/"):
            await self._handle_command(message, text)
            return
        await self._handle_plain_message(message)

    # ── commands ───────────────────────────────────────────────────────────

    async def _handle_command(self, message: Message, text: str) -> None:
        user = message.from_user
        if user is None:
            return
        parts = text.split(maxsplit=1)
        command = parts[0].lower().split("@", maxsplit=1)[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        if command == "/status":
            await self._run_status(message)
        elif command == "/cancel":
            await self._run_cancel(message)
        elif command == "/remember":
            await self._run_remember(message, user.id, arg)
        elif command == "/memory":
            await self._run_memory(message, user.id, arg)
        elif command == "/forget":
            await self._run_forget(message, user.id, arg)

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

    async def _run_remember(self, message: Message, user_id: int, arg: str) -> None:
        if not arg:
            await self._send(
                "Dame algo que recordar: /remember <dato>", reply_to=message.message_id
            )
            return
        if _SECRET_RE.search(arg):
            await self._send(
                "Eso tiene pinta de ser un secreto; no lo guardo.",
                reply_to=message.message_id,
            )
            return
        try:
            key, value = _split_memory_fact(arg)
        except ValueError:
            await self._send(
                "Dame algo que recordar: /remember <dato>", reply_to=message.message_id
            )
            return
        self._storage.set_memory(user_id, key, value)
        await self._send(
            f"Me guardo que {value.rstrip(' .!?')}.", reply_to=message.message_id
        )

    async def _run_memory(self, message: Message, user_id: int, arg: str) -> None:
        query = _normalize_memory_text(arg) or None
        memories = self._storage.list_memories(user_id, query)
        if not memories:
            if query is None:
                await self._send(
                    "No tengo nada guardado todavía. Cuéntame algo con /remember.",
                    reply_to=message.message_id,
                )
            else:
                await self._send(
                    "No tengo nada guardado sobre eso.", reply_to=message.message_id
                )
            return
        text = "Recuerdo:\n" + "\n".join(f"* {m['value']}" for m in memories)
        if len(text) > _MAX_MSG_CHARS:
            text = text[: _MAX_MSG_CHARS - 30] + "\n[…y más…]"
        await self._send(text, reply_to=message.message_id)

    async def _run_forget(self, message: Message, user_id: int, arg: str) -> None:
        query = _normalize_memory_text(arg)
        if not query:
            await self._send("Uso: /forget <dato>", reply_to=message.message_id)
            return
        deleted = self._storage.delete_memories(user_id, query)
        if deleted == 0:
            await self._send(
                "No tengo nada guardado de eso.", reply_to=message.message_id
            )
            return
        title = query[:1].upper() + query[1:]
        if deleted == 1:
            await self._send(f"He olvidado lo de {title}.", reply_to=message.message_id)
        else:
            await self._send(
                f"He olvidado {deleted} cosas de {title}.", reply_to=message.message_id
            )

    # ── plain messages ─────────────────────────────────────────────────────

    async def _handle_plain_message(self, message: Message) -> None:
        user = message.from_user
        if user is None:
            return
        # "Responder" flow: the member pressed the button; this message (with
        # any document/photo) is their reply intent.
        pending = self._pending_replies.pop(user.id, None)
        if pending is not None:
            attachments = await self._collect_outgoing_attachments(message)
            _tg_message_id, thread_id = pending
            text = message.text or message.caption or ""
            memory = tuple(m["value"] for m in self._storage.list_memories(user.id))
            self._queue.put_nowait(
                ReplyRequest(
                    thread_id=thread_id,
                    user_instructions=text,
                    source_message_id=message.message_id,
                    memory=memory,
                    user_id=user.id,
                    attachments=attachments,
                )
            )
            return
        if not (self._is_reply_to_own(message) or self._is_mentioned(message)):
            return
        # Reply intent confirmed: only now download any attachments (files are
        # never fetched for messages without a workflow purpose).
        attachments = await self._collect_outgoing_attachments(message)
        thread_id = ""
        reply = message.reply_to_message
        if reply is not None and self._is_own_message(reply):
            if any(p.message_id == reply.message_id for p in self._pending_drafts.values()):
                return  # draft confirmation message: use the inline buttons
            thread_id = self._storage.get_meta(f"{_TG_META_PREFIX}{reply.message_id}") or ""
        memory = tuple(m["value"] for m in self._storage.list_memories(user.id))
        self._queue.put_nowait(
            ReplyRequest(
                thread_id=thread_id,
                user_instructions=message.text or message.caption or "",
                source_message_id=message.message_id,
                memory=memory,
                user_id=user.id,
                attachments=attachments,
            )
        )

    async def _collect_outgoing_attachments(
        self, message: Message
    ) -> tuple[OutgoingAttachment, ...]:
        """Download Telegram documents/photos to the temp dir (bounded, safe).

        Limits: ``outgoing_attachment_max_count`` files and
        ``outgoing_attachment_max_bytes`` each. Violations reject the whole
        batch with a notice; filenames are sanitized to display metadata only.
        """
        entries: list[tuple[str, str, int, str]] = []
        if message.document is not None:
            doc = message.document
            size = doc.file_size or 0
            name = sanitize_filename(doc.file_name or f"documento_{doc.file_unique_id}")
            mime = doc.mime_type or _guess_mime(name)
            entries.append((doc.file_id, name, size, mime))
        if message.photo:
            largest = max(message.photo, key=lambda p: p.file_size or 0)
            size = largest.file_size or 0
            name = sanitize_filename(
                f"foto_{message.message_id}_{len(entries) + 1}.jpg"
            )
            entries.append((largest.file_id, name, size, "image/jpeg"))
        if not entries:
            return ()
        max_count = self._settings.outgoing_attachment_max_count
        max_bytes = self._settings.outgoing_attachment_max_bytes
        if len(entries) > max_count:
            await self.send_notice(
                f"Demasiados adjuntos (máximo {max_count}); no los añado. Vuelve a intentarlo."
            )
            return ()
        oversized = [name for name, _, size, _ in entries if size > max_bytes]
        if oversized:
            await self.send_notice(
                "Adjunto demasiado grande (máximo 10 MB); no los añado: "
                + ", ".join(oversized)
            )
            return ()
        results: list[OutgoingAttachment] = []
        for file_id, name, size, mime in entries:
            try:
                file = await self._ensure_sender().get_file(file_id)
                target = self._outgoing_tmp_path(name)
                # PTB File.download is the blocking variant; run it off the loop.
                await asyncio.to_thread(file.download, target)  # type: ignore[attr-defined]
            except Exception:
                logger.exception("downloading telegram attachment %s failed", name)
                for r in results:
                    _remove_file(r.path)
                await self.send_notice("No pude descargar un adjunto; vuelve a intentarlo.")
                return ()
            results.append(
                OutgoingAttachment(
                    filename=name, mime_type=mime, size_bytes=size, path=str(target)
                )
            )
        return tuple(results)

    def _outgoing_tmp_path(self, filename: str) -> Path:
        """Unique temp path for one outgoing attachment (safe, no traversal)."""
        base = Path(self._settings.tmp_dir) / "incoming"
        base.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(6)
        return base / f"{token}_{filename}"

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

    # ── callback queries (draft confirm/cancel + view original) ────────────

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
        if action == "reply":
            await self._handle_reply_callback(query, token)
            return
        if action in ("view", "hide"):
            await self._handle_view_callback(query, action, token)
            return
        if action == "resend":
            await self._handle_resend_callback(query, token)
            return
        pending = self._pending_drafts.get(token)
        if pending is None:
            await self._ensure_sender().answer_callback_query(
                query.id, "Este borrador ya no está disponible."
            )
            return
        # Group isolation: only the member who requested the draft may
        # confirm/cancel it.
        if pending.user_id and query.from_user.id != pending.user_id:
            await self._ensure_sender().answer_callback_query(
                query.id, "Ese borrador lo está gestionando otro miembro."
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

    # ── "Responder" button ─────────────────────────────────────────────────

    async def _handle_reply_callback(self, query: CallbackQuery, token: str) -> None:
        """User pressed "Responder": remember the target thread and ask for
        their intent; their next plain message becomes the reply instruction."""
        try:
            tg_message_id = int(token)
        except ValueError:
            return
        sender = self._ensure_sender()
        await sender.answer_callback_query(query.id)
        thread_id = self._storage.get_meta(f"{_TG_META_PREFIX}{tg_message_id}")
        if not thread_id:
            await self._send("No puedo asociar eso a ningún hilo.")
            return
        sender_name = self._storage.get_meta(f"{_TG_SENDER_PREFIX}{tg_message_id}") or ""
        self._pending_replies[query.from_user.id] = (tg_message_id, thread_id)
        if sender_name:
            await self._send(f"¿Qué quieres decirle a {sender_name}?")
        else:
            await self._send("¿Qué quieres decirle?")

    # ── "Reintentar envío" button ──────────────────────────────────────────

    async def _handle_resend_callback(self, query: CallbackQuery, token: str) -> None:
        """User pressed retry on a failed/uncertain draft.

        Chat authorization is enforced above; draft ownership is checked
        server-side via the persisted row. The coordinator re-verifies Gmail
        FIRST and only resends on definitive evidence the mail never left.
        """
        try:
            draft_id = int(token)
        except ValueError:
            return
        sender = self._ensure_sender()
        row = self._storage.get_draft(draft_id)
        if row is None:
            await sender.answer_callback_query(query.id, "Ese borrador ya no existe.")
            return
        owner = int(row.get("telegram_user_id") or 0)
        if owner and query.from_user.id != owner:
            await sender.answer_callback_query(
                query.id, "Ese borrador lo está gestionando otro miembro."
            )
            return
        if self._resend_callback is None:
            await sender.answer_callback_query(query.id, "Reintento no disponible.")
            return
        await sender.answer_callback_query(query.id, "Reintentando…")
        await self._resend_callback(draft_id)

    # ── "Ver original" / "Ocultar original" ────────────────────────────────

    async def _handle_view_callback(self, query: CallbackQuery, action: str, token: str) -> None:
        try:
            tg_message_id = int(token)
        except ValueError:
            return
        sender = self._ensure_sender()
        if action == "hide":
            await self._hide_original(query, sender, tg_message_id)
            return
        await self._show_original(query, sender, tg_message_id)

    async def _show_original(
        self, query: CallbackQuery, sender: Sender, tg_message_id: int
    ) -> None:
        """Fetch the original email from Gmail on demand and post it as
        temporary message(s). No LLM call, no SQLite body read."""
        await sender.answer_callback_query(query.id, "Cargando…")
        gmail_message_id = self._storage.get_meta(f"{_TG_GMAIL_PREFIX}{tg_message_id}")
        if not gmail_message_id or self._original_fetcher is None:
            await self._send("No pude cargar el original ahora mismo.")
            return
        try:
            original = await self._original_fetcher(gmail_message_id)
        except Exception:
            logger.exception("fetching original email %s failed", gmail_message_id)
            await self._send("No pude cargar el original ahora mismo.")
            return
        original_ids = await self._post_original(sender, tg_message_id, original)
        if original_ids:
            self._storage.set_meta(
                f"{_TG_ORIG_PREFIX}{tg_message_id}", json.dumps(original_ids)
            )

    async def _post_original(
        self, sender: Sender, tg_message_id: int, original: ParsedEmail
    ) -> list[int]:
        """Post the original as plain-text chunks with an 'Ocultar original'
        button on the first one. Returns the posted message ids."""
        header = (
            f"Original\n\n"
            f"Asunto: {neutralize_links(original.subject)}\n"
            f"De: {_format_sender(original)}\n\n"
        )
        attachments_line = ""
        if original.attachments:
            names = ", ".join(
                a.filename for a in original.attachments[:5]
            )
            attachments_line = f"\n\nAdjuntos: {names}"
        body = neutralize_links(original.body_text) + attachments_line
        hide_button = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Ocultar original", callback_data=f"hide:{tg_message_id}"
                    )
                ]
            ]
        )
        chunks = _split_original(body)
        posted: list[int] = []
        for index, chunk in enumerate(chunks):
            text = f"{header}{chunk}" if index == 0 else chunk
            message = await sender.send_message(
                self._allowed_chat_id,
                text,
                reply_markup=hide_button if index == 0 else None,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
            posted.append(message.message_id)
        return posted

    async def _hide_original(
        self, query: CallbackQuery, sender: Sender, tg_message_id: int
    ) -> None:
        """Delete the temporary original message(s) and clean up the mapping."""
        raw = self._storage.get_meta(f"{_TG_ORIG_PREFIX}{tg_message_id}")
        message_ids: list[int] = []
        if raw:
            try:
                message_ids = [int(m) for m in json.loads(raw)]
            except (ValueError, TypeError, json.JSONDecodeError):
                logger.warning("corrupt tgo mapping for %s", tg_message_id)
                message_ids = []
        for message_id in message_ids:
            try:
                await sender.delete_message(self._allowed_chat_id, message_id)
            except Exception:
                logger.exception("deleting original message %s failed", message_id)
        self._storage.delete_meta(f"{_TG_ORIG_PREFIX}{tg_message_id}")
        await sender.answer_callback_query(query.id, "Original oculto.")

    # ── TelegramNotifier implementation ────────────────────────────────────

    async def send_summary(self, email: ParsedEmail, summary: EmailSummary) -> int:
        sender = self._ensure_sender()
        # Spanish subject from the LLM; original subject as fallback. The
        # original email.subject stays the source of truth for threading.
        display_subject = summary.subject_es or email.subject
        text = (
            f"{display_subject}\nDe: {_format_sender(email)}\n\n"
            f"{neutralize_links(summary.summary_es)}"
        )
        message = await sender.send_message(
            self._allowed_chat_id,
            text,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
        self._storage.set_meta(f"{_TG_META_PREFIX}{message.message_id}", email.thread_id)
        self._storage.set_meta(f"{_TG_GMAIL_PREFIX}{message.message_id}", email.message_id)
        self._storage.set_meta(f"{_TG_SENDER_PREFIX}{message.message_id}", _format_sender(email))
        # Attach the action buttons now that the message_id is known. The
        # message_id doubles as the callback token (stable across restarts).
        buttons = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Ver original", callback_data=f"view:{message.message_id}"
                    ),
                    InlineKeyboardButton(
                        "Responder", callback_data=f"reply:{message.message_id}"
                    ),
                ]
            ]
        )
        await sender.edit_message_text(
            text,
            chat_id=self._allowed_chat_id,
            message_id=message.message_id,
            reply_markup=buttons,
        )
        return message.message_id

    async def send_notice(self, text: str) -> int:
        return await self._send(neutralize_links(text))

    async def send_typing(self) -> None:
        await self._ensure_sender().send_chat_action(self._allowed_chat_id, ChatAction.TYPING)

    async def send_draft_for_confirmation(
        self, draft: DraftReply, *, user_id: int = 0, draft_id: int = 0
    ) -> int:
        token = secrets.token_urlsafe(8)
        to_line = (
            ", ".join(str(address) for address in draft.to) if draft.to else "(sin destinatario)"
        )
        attachments_line = ""
        if draft.attachments:
            names = "\n".join(
                f"- {a.filename} ({_format_size(a.size_bytes)})" for a in draft.attachments
            )
            attachments_line = f"\nAdjuntos:\n{names}"
        text = (
            f"Borrador de respuesta\n"
            f"Para: {to_line}\n"
            f"Asunto: {draft.subject}\n"
            f"{attachments_line}\n\n"
            f"{neutralize_links(draft.body)}"
        )
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Enviar", callback_data=f"confirm:{token}"),
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
            token=token,
            draft=draft,
            message_id=message.message_id,
            user_id=user_id,
        )
        return message.message_id

    async def offer_resend(self, draft_id: int, user_id: int = 0) -> int:
        """Post the controlled-retry button for a failed/uncertain send.

        The button token is the persisted draft id (survives restarts); the
        coordinator re-verifies Gmail before resending — never a blind retry.
        """
        text = (
            "¿Reintento el envío? Primero compruebo Gmail para no enviar duplicados."
        )
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Reintentar envío", callback_data=f"resend:{draft_id}"
                    )
                ]
            ]
        )
        message = await self._ensure_sender().send_message(
            self._allowed_chat_id,
            text,
            reply_markup=keyboard,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
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
