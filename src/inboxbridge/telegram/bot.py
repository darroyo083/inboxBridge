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
import contextlib
import html
import json
import logging
import mimetypes
import re
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
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
from ..intents import (
    IntentAction,
    IntentClassifier,
    has_latest_reference,
    is_thread_summary_request,
    strip_latest_reference,
)
from ..llm.qa import CONTEXTUAL_EMOJIS, QaSection
from ..models import DraftReply, EmailSummary, OutgoingAttachment, ParsedEmail

logger = logging.getLogger(__name__)

_CALLBACK_RE = re.compile(r"^([A-Za-z]+):([A-Za-z0-9_-]{1,96})$")
_URL_SCHEME_RE = re.compile(r"https?://", re.IGNORECASE)
_TG_META_PREFIX = "tg:"  # tg message_id -> thread_id
_TG_GMAIL_PREFIX = "tgm:"  # tg message_id -> gmail message_id (IDs only)
_TG_SENDER_PREFIX = "tgs:"  # tg message_id -> sender display name
_TG_ORIG_PREFIX = "tgo:"  # tg message_id -> JSON list of temporary original message ids
_CONFIRM_TIMEOUT_SECONDS = 900.0

#: Bounded pending conversational slot-filling (reply target / compose) expires
#: after this many seconds. In-memory only (intentionally not persisted): a
#: restart clears it, which is safe — the user just re-issues the command.
_FLOW_TTL_SECONDS = 900.0

#: Explicit cancellation phrases that clear a pending conversational flow.
_FLOW_CANCEL = re.compile(
    r"^(cancela|cancelar|anula|anular|olv[ií]dalo|d[eé]jalo|nada|nada m[aá]s)$",
    re.IGNORECASE,
)

#: Intent actions that create a NEW draft. With an active draft these must ask
#: for explicit cancellation instead of silently replacing/mutating it.
_NEW_DRAFT_ACTIONS = frozenset(
    {
        IntentAction.COMPOSE_NEW_EMAIL,
        IntentAction.FORWARD_EMAIL,
        IntentAction.REPLY_TO_EMAIL,
    }
)

#: Telegram hard limit is 4096 chars per message; stay safely below.
_MAX_MSG_CHARS = 3900
#: Cap on temporary "original" messages to avoid flooding the group.
_MAX_ORIGINAL_MESSAGES = 8
#: Cap on attachment buttons in the delivery panel.
MAX_ATTACHMENTS_PANEL = 8

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

    async def send_document(
        self,
        chat_id: int | str,
        document: Any,
        *,
        filename: str | None = None,
        reply_parameters: ReplyParameters | None = None,
    ) -> Message: ...


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
    #: EXACT incoming Gmail message this reply targets, FROZEN at queue time
    #: (the Telegram summary the user replied to, or the resolved "al último"
    #: message). The recipient and in_reply_to come only from this message.
    target_message_id: str = ""


@dataclass
class _PendingDraft:
    token: str
    draft: DraftReply
    message_id: int
    #: Persisted draft row id (owner checks + coordinator state).
    draft_id: int = 0
    #: Telegram user id that requested the draft; only this member may
    #: confirm/cancel it (group isolation).
    user_id: int = 0
    #: Every edit bumps draft_version; the preview shown bumps preview_version.
    #: Sending requires preview_version == draft_version (no stale preview).
    draft_version: int = 1
    preview_version: int = 1
    decided: bool | None = None
    future: asyncio.Future[bool] | None = None

    def resolve(self, confirmed: bool) -> None:
        self.decided = confirmed
        if self.future is not None and not self.future.done():
            self.future.set_result(confirmed)


def neutralize_links(text: str) -> str:
    """Make URLs non-clickable in Telegram display (https:// → hxxps://)."""
    return _URL_SCHEME_RE.sub(lambda match: f"hxxp{match.group(0)[4:]}", text)


class TelegramAttachmentError(RuntimeError):
    """A Telegram file could not be downloaded/validated.

    Raised (after a user-facing notice) so attachment-bearing actions abort
    instead of silently presenting a draft that implies an attachment is
    included when it is not. The user is never left with a misleading draft.
    """


#: Neutral fallback emoji for sections whose model-provided emoji is not in
#: the shared allowlist (``llm.qa.CONTEXTUAL_EMOJIS``).
_NEUTRAL_EMOJI = "ℹ️"


def render_rich_text(text: str) -> str:
    """Render an AI answer with SAFE rich formatting (Telegram HTML).

    The model output is treated as untrusted data: every character is
    HTML-escaped, and the ONLY structure added is a bold heading on lines that
    start with a whitelisted contextual emoji (e.g. ``💰 125 CHF`` →
    ``💰 <b>125 CHF</b>``). Bullets and blank lines pass through as plain
    text. The result is always well-formed HTML because only this function
    emits tags.
    """
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped[0] in CONTEXTUAL_EMOJIS:
            heading = html.escape(stripped[1:].strip())
            lines.append(f"{stripped[0]} <b>{heading}</b>" if heading else stripped[0])
        else:
            lines.append(html.escape(line))
    return "\n".join(lines)


def _cap_field(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def render_qa_answer(answer: str, sections: list[QaSection]) -> str:
    """Deterministic, safe render of the structured Q&A contract.

    Application-controlled layout: one line per section — ``<emoji> <b>title``
    + items (a single item is rendered bare on the next line, multiple items
    as bullets). A single-section/single-item answer whose item appears in
    ``answer`` renders compact and inline: ``👤 El contacto es <b>Markus
    Schneider</b>.`` All dynamic values are HTML-escaped; emojis outside the
    allowlist fall back to a neutral one; only this function emits tags, so
    the output is always well-formed Telegram HTML.
    """
    answer = _cap_field(answer, 600)
    if not sections:
        return html.escape(answer)
    if (
        len(sections) == 1
        and len(sections[0].items) == 1
        and answer
    ):
        section = sections[0]
        item = section.items[0]
        escaped_answer = html.escape(answer)
        escaped_item = html.escape(_cap_field(item, 300))
        if escaped_item and escaped_item in escaped_answer:
            emoji = (
                section.emoji if section.emoji in CONTEXTUAL_EMOJIS else _NEUTRAL_EMOJI
            )
            bolded = escaped_answer.replace(escaped_item, f"<b>{escaped_item}</b>", 1)
            return f"{emoji} {bolded}"
    lines: list[str] = []
    if answer:
        lines.append(html.escape(answer))
    for section in sections:
        emoji = section.emoji if section.emoji in CONTEXTUAL_EMOJIS else _NEUTRAL_EMOJI
        lines.append(f"{emoji} <b>{html.escape(_cap_field(section.title, 60))}</b>")
        items = [_cap_field(item, 300) for item in section.items]
        if len(items) == 1:
            lines.append(html.escape(items[0]))
        else:
            lines.extend(f"• {html.escape(item)}" for item in items)
    return "\n".join(lines)


def render_qa_plain(answer: str, sections: list[QaSection]) -> str:
    """Plain-text variant of the structured answer (no markup at all).

    Used as the Telegram fallback when a formatted send fails: keeps every
    fact, loses only the bold layout.
    """
    lines: list[str] = []
    if answer:
        lines.append(answer)
    for section in sections:
        lines.append(f"{section.emoji} {section.title}")
        items = section.items
        if len(items) == 1:
            lines.append(items[0])
        else:
            lines.extend(f"• {item}" for item in items)
    return "\n".join(lines)


def render_summary(headline: str, sections: list[QaSection]) -> str:
    """Deterministic, safe render of the structured thread-summary contract.

    Layout: a fixed ``📬 <b>headline</b>`` header followed by the same
    application-controlled section blocks as Q&A (emoji + bold title, single
    item bare, multiple items as bullets). A single-section summary whose
    title equals the headline renders as the header plus bullets only — the
    compact form for simple threads, no repeated header. All values are
    HTML-escaped; only this function emits tags.
    """
    headline = _cap_field(headline, 60)
    header = f"📬 <b>{html.escape(headline)}</b>"
    if not sections:
        return header
    if len(sections) == 1 and sections[0].title == headline:
        items = [_cap_field(item, 300) for item in sections[0].items]
        lines = [header] + [f"• {html.escape(item)}" for item in items]
        return "\n".join(lines)
    body = render_qa_answer("", sections)
    return f"{header}\n{body}"


def render_summary_plain(headline: str, sections: list[QaSection]) -> str:
    """Plain-text variant of the structured summary (no markup at all)."""
    lines = [f"📬 {headline}"]
    if len(sections) == 1 and sections[0].title == headline:
        lines.extend(f"• {item}" for item in sections[0].items)
        return "\n".join(lines)
    body = render_qa_plain("", sections)
    if body:
        lines.append(body)
    return "\n".join(lines)


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


#: Production message-handler filter: every supported message type reaches the
#: SINGLE ``process_update`` entry point (text, documents, photos, voice).
_MESSAGE_FILTERS = (
    filters.TEXT | filters.Document.ALL | filters.PHOTO | filters.VOICE
)


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
        self._action_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
        self._intent_classifier: Any | None = None
        self._assistant: Any | None = None
        self._contacts: Any | None = None
        self._reminders: Any | None = None
        self._allowed_chat_id = settings.telegram_allowed_chat_id
        self._queue: asyncio.Queue[ReplyRequest] = asyncio.Queue()
        self._pending_drafts: dict[str, _PendingDraft] = {}
        #: user_id -> (tg summary message_id, thread_id, mode)
        #: mode: "reply" (Responder) | "question" (Preguntar)
        #: user_id -> (tg summary message_id, thread_id, mode, target Gmail
        #: message_id). mode: "reply" (Responder) | "question" (Preguntar).
        #: The target Gmail message is FROZEN here (summary -> Gmail mapping).
        self._pending_replies: dict[int, tuple[int, str, str, str]] = {}
        #: user_id -> pending multi-step UI state (compose, contact ops)
        self._pending_flows: dict[int, dict[str, Any]] = {}
        #: media_group_id -> seen-at timestamp: only the first member of a
        #: Telegram album is processed (bounded, in-memory).
        self._seen_media_groups: dict[str, float] = {}
        self._application: Application[Any, Any, Any, Any, Any, Any] | None = None
        self._started = False

    def register_resend_callback(self, callback: Callable[[int], Awaitable[None]]) -> None:
        """Wire the "Reintentar envío" button to the reply coordinator."""
        self._resend_callback = callback

    def register_action_callback(
        self, callback: Callable[[str, dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Wire NL/button actions (draft edit/send/cancel, contacts, reminders,
        attachments, mark/archive, compose, forward, Q&A) to the coordinator."""
        self._action_callback = callback

    def set_intent_classifier(self, classifier: Any) -> None:
        self._intent_classifier = classifier

    def register_assistant(self, assistant: Any) -> None:
        """Wire the V1.1 assistant (contacts/reminders/AI access for UI flows)."""
        self._assistant = assistant
        self._contacts = getattr(assistant, "contacts", None)
        self._reminders = getattr(assistant, "reminders", None)

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
        # Every supported message type must reach the SINGLE process_update
        # entry point (it already handles text, document, photo and voice).
        # Text-only registration meant documents/photos/voice never arrived.
        application.add_handler(MessageHandler(_MESSAGE_FILTERS, self._on_message))
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
        if (
            text is None
            and message.document is None
            and not message.photo
            and message.voice is None
        ):
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
        elif command == "/menu" or command == "/ayuda" or command == "/help":
            await self.show_menu()
        elif command == "/contactos":
            await self.show_contacts_panel()
        elif command == "/recordatorios":
            await self.show_reminders(user.id)
        elif command == "/nuevo":
            await self.prompt_compose_recipient(user.id)
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
        user = message.from_user
        user_id = user.id if user is not None else 0
        cancelled = 0
        for pending in list(self._pending_drafts.values()):
            if pending.user_id and pending.user_id != user_id:
                continue  # only the owner may cancel their own draft
            pending.resolve(False)
            cancelled += 1
            self._pending_drafts.pop(pending.token, None)
        # Also clear any pending conversational slot-fill flow for this member.
        self._pending_flows.pop(user_id, None)
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
        # Experimental voice: bounded download → transcription → intent flow.
        if message.voice is not None:
            await self._handle_voice(message, user.id)
            return

        # Media groups (albums) arrive as several separate updates. Only the
        # FIRST member of a group is processed: a multi-photo album can never
        # create several drafts/actions from one intended action. Single
        # documents/photos (the supported cases) have no media_group_id.
        media_group_id = getattr(message, "media_group_id", None)
        if media_group_id:
            if media_group_id in self._seen_media_groups:
                return  # already handled this album
            self._seen_media_groups[media_group_id] = time.time()
            self._prune_media_groups()

        text = message.text or message.caption or ""

        # Active owned draft + Telegram file(s): attach to THAT draft only —
        # never a second draft; recipient/subject/body/thread/attachments are
        # preserved; the preview is re-rendered with a bumped draft version.
        if (
            self._active_pending_for(user.id) is not None
            and (message.document is not None or message.photo)
            and await self._attach_to_active_draft(message, user.id)
        ):
            return

        # Multi-step UI flows (compose recipient/instruction, contact inputs,
        # candidate selection, draft edit mode) take precedence.
        flow = self._pending_flows.get(user.id)
        if flow is not None:
            handled = await self._handle_flow_input(message, user.id, flow, text)
            if handled:
                return

        # "Responder"/"Preguntar" button flows: next message is the intent.
        pending = self._pending_replies.pop(user.id, None)
        if pending is not None:
            tg_message_id, thread_id, mode, target_message_id = pending
            # An EXPLICIT forward instruction bound to a Gmail summary wins
            # over an active/pending reply slot: never let a pending reply
            # flow steal a forward (e.g. the user pressed "Responder" earlier
            # and then replies to a summary with "reenvía esto a …").
            if mode == "reply" and self._is_reply_to_own(message):
                intent = (self._intent_classifier or IntentClassifier()).classify_rule_only(text)
                if intent.action == IntentAction.FORWARD_EMAIL:
                    reply = message.reply_to_message
                    fwd_thread_id = (
                        self._storage.get_meta(f"{_TG_META_PREFIX}{reply.message_id}") or ""
                        if reply is not None
                        else ""
                    )
                    await self._dispatch_intent(
                        text,
                        thread_id=fwd_thread_id,
                        tg_message_id=reply.message_id if reply is not None else 0,
                        user_id=user.id,
                        fallback_to_reply=True,
                        message=message,
                    )
                    return
            if mode == "question":
                if is_thread_summary_request(text):
                    # Natural "resume este hilo" phrases route rules-first to
                    # SUMMARIZE_THREAD even when the user pressed "Preguntar",
                    # instead of being answered as a Q&A question.
                    await self._dispatch_intent(
                        text,
                        thread_id=thread_id,
                        tg_message_id=tg_message_id,
                        user_id=user.id,
                        force=IntentAction.SUMMARIZE_THREAD,
                    )
                    return
                await self._dispatch_intent(
                    text,
                    thread_id=thread_id,
                    tg_message_id=tg_message_id,
                    user_id=user.id,
                    force=IntentAction.ASK_ABOUT_EMAIL,
                )
                return
            try:
                attachments = await self._collect_outgoing_attachments(message)
            except TelegramAttachmentError:
                return  # notice already sent; never a misleading draft
            memory = tuple(m["value"] for m in self._storage.list_memories(user.id))
            self._queue.put_nowait(
                ReplyRequest(
                    thread_id=thread_id,
                    user_instructions=text,
                    source_message_id=message.message_id,
                    memory=memory,
                    user_id=user.id,
                    attachments=attachments,
                    target_message_id=target_message_id,
                )
            )
            return

        # Draft edit mode: an active draft for this user; the message is an edit
        # instruction (e.g. after pressing EDIT).
        if self._active_pending_for(user.id) is not None and text.strip():
            await self._dispatch_intent(
                text, thread_id="", tg_message_id=0, user_id=user.id
            )
            return

        if not (self._is_reply_to_own(message) or self._is_mentioned(message)):
            # Standalone natural-language commands are still handled (e.g.
            # "escribe a Roman…", "recuérdame…", contact management).
            if self._action_callback is not None and text.strip():
                await self._dispatch_intent(
                    text,
                    thread_id="",
                    tg_message_id=0,
                    user_id=user.id,
                    message=message,
                )
                return
            # A file without any instruction and without conversational
            # context must never guess a recipient/thread — ask instead.
            if message.document is not None or message.photo:
                await self.send_notice(
                    "Adjunto recibido. Dime qué quieres que haga con él, por "
                    "ejemplo: «Envía un correo a … adjuntando esto», responde a "
                    "un resumen con el adjunto, o «adjúntalo» si tienes un "
                    "borrador activo."
                )
            return

        # Reply-to-own message: resolve context, then intent-dispatch.
        thread_id = ""
        tg_message_id = 0
        reply = message.reply_to_message
        if reply is not None and self._is_own_message(reply):
            if self._pending_draft_for_message(reply.message_id) is not None:
                # A draft confirmation message: "ok" would be ambiguous — ask.
                await self._send(
                    "Usa los botones del borrador (o escribe «envíalo» / "
                    "«cancela el borrador»)."
                )
                return
            tg_message_id = reply.message_id
            thread_id = self._storage.get_meta(f"{_TG_META_PREFIX}{reply.message_id}") or ""
        await self._dispatch_intent(
            text,
            thread_id=thread_id,
            tg_message_id=tg_message_id,
            user_id=user.id,
            fallback_to_reply=True,
            message=message,
        )

    async def _dispatch_intent(
        self,
        text: str,
        *,
        thread_id: str,
        tg_message_id: int,
        user_id: int,
        force: IntentAction | None = None,
        fallback_to_reply: bool = False,
        message: Message | None = None,
    ) -> None:
        """Classify + execute one user message (validated, deterministic).

        ``fallback_to_reply``: when the message replies to one of our summary
        messages and the intent is unclear, treat it as a plain reply
        instruction (the classic flow) instead of asking.
        """
        classifier = self._intent_classifier or IntentClassifier()
        has_draft = self._active_pending_for(user_id) is not None
        context = f"hilo={thread_id or 'ninguno'}; borrador_activo={'sí' if has_draft else 'no'}"
        intent = await classifier.classify(text, context=context)
        action = force or intent.action
        payload: dict[str, Any] = dict(intent.payload)
        payload.setdefault("user_id", user_id)
        payload.setdefault("thread_id", thread_id)
        payload.setdefault("tg_message_id", tg_message_id)
        # The user's exact text must ALWAYS reach the intent handler: rules
        # without an instruction payload (Q&A, thread summary) must not fall
        # back to an internal default question.
        if not payload.get("instruction"):
            payload["instruction"] = text
        if tg_message_id:
            payload.setdefault(
                "message_id", self._storage.get_meta(f"{_TG_GMAIL_PREFIX}{tg_message_id}") or ""
            )

        # An active unsent draft must never be silently replaced by a NEW
        # compose/reply/forward flow. Require explicit cancellation first.
        if has_draft and action in _NEW_DRAFT_ACTIONS:
            await self._send(
                "Tienes un borrador pendiente. Cáncelo antes de empezar otro: "
                "«cancela el borrador»."
            )
            return

        # Edit language ("más largo", "más formal"...) only routes as an edit
        # when there IS an active draft. Outside a draft it must not mutate
        # anything: treat it as ambiguous so the normal fallback applies.
        if action in (IntentAction.MODIFY_DRAFT, IntentAction.REGENERATE_DRAFT) and not has_draft:
            action = IntentAction.UNKNOWN

        if action in (IntentAction.CLARIFY, IntentAction.UNKNOWN):
            if fallback_to_reply:
                # Classic reply flow: the member's message is the reply intent
                # (empty thread → the coordinator asks for context).
                if message is not None and (
                    message.document is not None or message.photo
                ):
                    try:
                        attachments = await self._collect_outgoing_attachments(message)
                    except TelegramAttachmentError:
                        return  # notice already sent; never a misleading draft
                else:
                    attachments = ()
                memory = tuple(m["value"] for m in self._storage.list_memories(user_id))
                target_message_id = ""
                if tg_message_id:
                    target_message_id = (
                        self._storage.get_meta(f"{_TG_GMAIL_PREFIX}{tg_message_id}") or ""
                    )
                self._queue.put_nowait(
                    ReplyRequest(
                        thread_id=thread_id,
                        user_instructions=text,
                        source_message_id=(
                            message.message_id
                            if message is not None
                            else tg_message_id
                        ),
                        memory=memory,
                        user_id=user_id,
                        attachments=attachments,
                        target_message_id=target_message_id,
                    )
                )
                return
            if has_draft:
                await self._send(
                    "Tienes un borrador pendiente. Puedes decir: «envíalo», "
                    "«cancela el borrador», «hazlo más corto»… o editar el texto."
                )
            elif message is not None and (message.document is not None or message.photo):
                # A file without a recognizable instruction must never guess a
                # recipient/thread — ask instead.
                await self.send_notice(
                    "Adjunto recibido. Dime qué quieres que haga con él, por "
                    "ejemplo: «Envía un correo a … adjuntando esto», responde a "
                    "un resumen con el adjunto, o «adjúntalo» si tienes un "
                    "borrador activo."
                )
            else:
                await self._send("No te he entendido del todo. ¿Qué quieres que haga?")
            return

        # High-impact text acts execute only on EXPLICIT deterministic verbs.
        if action == IntentAction.SEND_DRAFT:
            await self._send_draft_via_text(user_id)
            return
        if action == IntentAction.CANCEL_DRAFT:
            await self._cancel_draft_via_text(user_id)
            return

        if action in (IntentAction.MODIFY_DRAFT, IntentAction.REGENERATE_DRAFT):
            await self._edit_draft_via_text(user_id, text, action)
            return

        # Same-message compose with Telegram files: the caption is the
        # instruction and the attachments travel in the payload (single
        # download, single draft).
        if (
            action == IntentAction.COMPOSE_NEW_EMAIL
            and message is not None
            and (message.document is not None or message.photo)
        ):
            try:
                attachments = await self._collect_outgoing_attachments(message)
            except TelegramAttachmentError:
                return  # notice already sent; never a misleading draft
            if attachments:
                payload["attachments"] = attachments

        # Reply intent (rules-first, but the LLM may still classify it): the
        # classic reply flow when bound to a thread; otherwise resolve "al
        # último" directly or start a bounded reply-target slot fill.
        if action == IntentAction.REPLY_TO_EMAIL:
            if fallback_to_reply:
                await self._queue_reply(
                    user_id, text, thread_id, message=message, tg_message_id=tg_message_id
                )
                return
            if has_latest_reference(text):
                latest = self._latest_reply_target()
                if latest is None:
                    await self._send(
                        "No tengo ningún correo recibido reciente al que responder."
                    )
                    return
                # Freeze the exact incoming message id NOW (never re-resolved).
                await self._queue_reply(
                    user_id,
                    strip_latest_reference(text) or text,
                    latest["thread_id"],
                    message=message,
                    tg_message_id=tg_message_id,
                    target_message_id=latest["message_id"],
                )
                return
            # No target yet: remember the instruction and ask for the target.
            self._pending_flows[user_id] = {
                "flow": "reply_target",
                "instruction": text,
                "user_id": user_id,
                "ts": time.time(),
            }
            await self._send(
                "¿A qué correo quieres responder? Di «al último» para el correo "
                "recibido más reciente, o responde directamente a un resumen de "
                "InboxBridge."
            )
            return

        if self._action_callback is None:
            await self._send("No puedo hacer eso ahora mismo.")
            return
        await self._action_callback(action.value, payload)

    # ── reply target resolution ("al último") ────────────────────────────────

    def _latest_reply_target(self) -> dict[str, Any] | None:
        """Resolve "the latest incoming email" to a concrete thread target.

        Frozen at resolution time: the returned ``thread_id``/``message_id`` are
        immutable, so a draft bound to them can never silently switch to a newer
        email that arrives later.
        """
        row = self._storage.latest_incoming_message()
        if row is None or not str(row.get("thread_id") or "").strip():
            return None
        return {
            "thread_id": str(row["thread_id"]),
            "message_id": str(row.get("message_id") or ""),
        }

    async def _queue_reply(
        self,
        user_id: int,
        instruction: str,
        thread_id: str,
        *,
        message: Message | None = None,
        tg_message_id: int = 0,
        target_message_id: str = "",
    ) -> None:
        """Put a ReplyRequest on the coordinator queue (classic reply flow).

        A failed Telegram attachment download aborts the reply (notice already
        sent) — the queue never receives a request whose attachment silently
        vanished. The reply target Gmail message is FROZEN here: the exact
        message mapped to the Telegram summary (``tgm:`` meta) or the explicit
        resolved target ("al último").
        """
        if message is not None and (message.document is not None or message.photo):
            try:
                attachments = await self._collect_outgoing_attachments(message)
            except TelegramAttachmentError:
                return
        else:
            attachments = ()
        if not target_message_id and tg_message_id:
            target_message_id = (
                self._storage.get_meta(f"{_TG_GMAIL_PREFIX}{tg_message_id}") or ""
            )
        memory = tuple(m["value"] for m in self._storage.list_memories(user_id))
        self._queue.put_nowait(
            ReplyRequest(
                thread_id=thread_id,
                user_instructions=instruction,
                source_message_id=(
                    message.message_id if message is not None else tg_message_id
                ),
                memory=memory,
                user_id=user_id,
                attachments=attachments,
                target_message_id=target_message_id,
            )
        )

    # ── text draft actions (explicit only; never on "ok"/"sí") ─────────────

    def _active_pending_for(self, user_id: int) -> _PendingDraft | None:
        for pending in self._pending_drafts.values():
            if pending.user_id == user_id:
                return pending
        return None

    def _pending_draft_for_message(self, message_id: int) -> _PendingDraft | None:
        for pending in self._pending_drafts.values():
            if pending.message_id == message_id:
                return pending
        return None

    def pending_draft_for_owner(self, draft_id: int, user_id: int) -> _PendingDraft | None:
        for pending in self._pending_drafts.values():
            if pending.draft_id == draft_id and pending.user_id == user_id:
                return pending
        return None

    async def _send_draft_via_text(self, user_id: int) -> None:
        pending = self._active_pending_for(user_id)
        if pending is None:
            await self._send("No hay ningún borrador pendiente para enviar.")
            return
        if pending.preview_version != pending.draft_version:
            await self._send(
                "El borrador cambió y aún no te he mostrado la versión final. "
                "Revisa la vista previa actualizada antes de enviar."
            )
            return
        self._pending_drafts.pop(pending.token, None)
        pending.resolve(True)

    async def _cancel_draft_via_text(self, user_id: int) -> None:
        pending = self._active_pending_for(user_id)
        if pending is None:
            await self._send("No hay ningún borrador pendiente para cancelar.")
            return
        self._pending_drafts.pop(pending.token, None)
        pending.resolve(False)

    async def _edit_draft_via_text(
        self, user_id: int, text: str, action: IntentAction
    ) -> None:
        pending = self._active_pending_for(user_id)
        if pending is None:
            await self._send("No hay ningún borrador activo para editar.")
            return
        instruction = text if action == IntentAction.MODIFY_DRAFT else "reescribe por completo"
        if self._action_callback is None:
            await self._send("No puedo editar el borrador ahora mismo.")
            return
        await self._action_callback(
            "edit_draft",
            {"draft_id": pending.draft_id, "instruction": instruction, "user_id": user_id},
        )

    # ── pending flow inputs (compose steps, contact inputs, candidates) ─────

    async def _handle_flow_input(
        self, message: Message, user_id: int, flow: dict[str, Any], text: str
    ) -> bool:
        flow_name = str(flow.get("flow") or "")

        # Bounded: stale pending flows expire (in-memory only, no persistence).
        ts = flow.get("ts")
        if isinstance(ts, int | float) and (time.time() - ts) > _FLOW_TTL_SECONDS:
            self._pending_flows.pop(user_id, None)
            await self._send("Se acabó el tiempo de esta petición; empieza de nuevo.")
            return True

        # Explicit cancellation clears the pending flow (never a send).
        if _FLOW_CANCEL.match(text.strip()):
            self._pending_flows.pop(user_id, None)
            await self._send("Cancelado.")
            return True

        if flow_name == "reply_target":
            await self._reply_target_step(user_id, flow, text)
            return True
        if flow_name == "compose_recipient":
            await self._compose_recipient_step(user_id, text)
            return True
        if flow_name == "compose_instruction":
            await self._compose_instruction_step(user_id, text)
            return True
        if flow_name == "forward_recipient":
            await self._forward_recipient_step(user_id, text)
            return True
        if flow_name == "choose_candidate":
            await self._choose_candidate_step(user_id, flow, text)
            return True
        if flow_name == "contact_new_name":
            self._pending_flows[user_id] = {
                "flow": "contact_new_email",
                "name": text.strip(),
            }
            await self._send("¿Y cuál es su correo?")
            return True
        if flow_name == "contact_new_email":
            await self._confirm_contact_new(user_id, flow, text)
            return True
        if flow_name == "contact_email_wait":
            await self._confirm_flow(
                user_id,
                f"¿Cambio el correo del contacto a {text.strip()}?",
                "contact_update_email",
                {"contact_id": int(flow["contact_id"]), "email": text.strip()},
            )
            return True
        if flow_name == "contact_alias_wait":
            await self._confirm_flow(
                user_id,
                f"¿Guardo «{text.strip()}» como alias?",
                "contact_add_alias_confirm",
                {"contact_id": int(flow["contact_id"]), "alias": text.strip()},
            )
            return True
        if flow_name == "contact_aliasdel_wait":
            await self._confirm_flow(
                user_id,
                f"¿Elimino el alias «{text.strip()}»?",
                "contact_remove_alias_confirm",
                {"alias": text.strip()},
            )
            return True
        if flow_name == "contact_rename_wait":
            await self._confirm_flow(
                user_id,
                f"¿Renombro el contacto a «{text.strip()}»?",
                "contact_rename",
                {"contact_id": int(flow["contact_id"]), "name": text.strip()},
            )
            return True
        if flow_name == "ask_unknown_recipient":
            # User replied with an address (or a different name).
            self._pending_flows.pop(user_id, None)
            await self._compose_recipient_step(user_id, text, flow.get("fallback_instruction"))
            return True
        return False

    async def _reply_target_step(
        self, user_id: int, flow: dict[str, Any], text: str
    ) -> None:
        """Fill the missing reply target slot ("al último" → latest email)."""
        if has_latest_reference(text):
            latest = self._latest_reply_target()
            if latest is None:
                await self._send(
                    "No tengo ningún correo recibido reciente al que responder."
                )
                return
            self._pending_flows.pop(user_id, None)
            instruction = str(flow.get("instruction") or "")
            # Freeze the exact incoming message id NOW (never re-resolved).
            await self._queue_reply(
                user_id,
                instruction,
                latest["thread_id"],
                target_message_id=latest["message_id"],
            )
            return
        await self._send(
            "Dime «al último» para responder al correo recibido más reciente, "
            "o responde directamente a un resumen de InboxBridge."
        )

    async def _compose_recipient_step(
        self, user_id: int, text: str, instruction_hint: str | None = None
    ) -> None:
        from ..contacts import validate_email

        # Allow "a <recipient>" / "para <recipient>" as a natural slot fill.
        phrase = re.sub(r"^(a |para )", "", text.strip(), flags=re.IGNORECASE)
        phrase = phrase.strip(" .,;:!?¿¡")
        if validate_email(phrase):
            self._pending_flows[user_id] = {
                "flow": "compose_instruction",
                "recipient": phrase,
                "display_name": phrase.split("@")[0],
            }
        else:
            resolution = self._contacts.resolve(phrase) if self._contacts is not None else None
            if resolution is None or not resolution.resolved:
                if resolution is not None and resolution.ambiguous:
                    await self.choose_candidate(
                        user_id,
                        "Hay varios contactos que coinciden; dime cuál:",
                        resolution.candidates,
                        flow="compose",
                    )
                    return
                await self.ask_unknown_recipient(user_id, phrase, "No tengo a nadie guardado como")
                return
            assert resolution.contact is not None
            self._pending_flows[user_id] = {
                "flow": "compose_instruction",
                "recipient": resolution.contact["email"],
                "display_name": resolution.contact["display_name"],
            }
        if instruction_hint:
            await self._compose_instruction_step(user_id, instruction_hint)
        else:
            await self._send("¿Qué le digo?")

    async def _compose_instruction_step(self, user_id: int, text: str) -> None:
        flow = self._pending_flows.pop(user_id, None)
        if flow is None:
            return
        if self._action_callback is None:
            await self._send("No puedo redactar ahora mismo.")
            return
        await self._action_callback(
            "compose",
            {
                "recipient": f"{flow['display_name']} <{flow['recipient']}>",
                "instruction": text,
                "user_id": user_id,
            },
        )

    async def _forward_recipient_step(self, user_id: int, text: str) -> None:
        flow = self._pending_flows.pop(user_id, None)
        if flow is None:
            return
        if self._action_callback is None:
            await self._send("No puedo reenviar ahora mismo.")
            return
        await self._action_callback(
            "forward",
            {
                "recipient": text.strip(),
                "tg_message_id": int(flow.get("tg_message_id") or 0),
                "user_id": user_id,
            },
        )

    async def _choose_candidate_step(
        self, user_id: int, flow: dict[str, Any], text: str
    ) -> None:
        candidates = list(flow.get("candidates") or [])
        choice = text.strip()
        selected: dict[str, Any] | None = None
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(candidates):
                selected = candidates[index]
        else:
            for candidate in candidates:
                if str(candidate.get("display_name") or "").casefold() == choice.casefold():
                    selected = candidate
                    break
        if selected is None:
            await self._send("No reconocí esa opción. Escribe el número o el nombre.")
            return
        self._pending_flows.pop(user_id, None)
        flow_name = str(flow.get("target_flow") or "compose")
        if flow_name == "compose":
            self._pending_flows[user_id] = {
                "flow": "compose_instruction",
                "recipient": selected["email"],
                "display_name": selected["display_name"],
            }
            await self._send("¿Qué le digo?")
        else:
            callback = self._action_callback
            if callback is None:
                return
            await callback(
                "forward",
                {
                    "recipient": f"{selected['display_name']} <{selected['email']}>",
                    "tg_message_id": int(flow.get("tg_message_id") or 0),
                    "user_id": user_id,
                },
            )

    async def _confirm_contact_new(
        self, user_id: int, flow: dict[str, Any], email_text: str
    ) -> None:
        self._pending_flows.pop(user_id, None)
        await self._confirm_flow(
            user_id,
            f"¿Guardo el contacto {flow['name']} <{email_text.strip()}>?",
            "contact_create",
            {"name": flow["name"], "email": email_text.strip()},
        )

    async def request_confirmation(
        self, text: str, action: str, payload: dict[str, Any], *, user_id: int = 0
    ) -> None:
        """Public one-shot confirmation UI (assistant contact/reminder flows)."""
        await self._confirm_flow(user_id, text, action, payload)

    async def _confirm_flow(
        self,
        user_id: int,
        text: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        """One-shot confirmation UI for persistent/destructive changes.

        The token is unguessable; the flow records the requesting user so a
        (leaked/replayed) confirmation from someone else is inert.
        """
        token = secrets.token_urlsafe(8)
        self._pending_flows[user_id] = {
            "flow": "confirm",
            "confirm_token": token,
            "action": action,
            "payload": payload,
            "user_id": user_id,
        }
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Sí", callback_data=f"confyes:{token}"),
                    InlineKeyboardButton("No", callback_data=f"confno:{token}"),
                ]
            ]
        )
        await self._ensure_sender().send_message(
            self._allowed_chat_id,
            text,
            reply_markup=keyboard,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )

    # ── public UI helpers (used by the assistant) ───────────────────────────

    async def prompt_compose_recipient(self, user_id: int) -> None:
        self._pending_flows[user_id] = {"flow": "compose_recipient"}
        await self._send("¿A quién le escribo? (nombre, alias o correo)")

    async def choose_candidate(
        self,
        user_id: int,
        question: str,
        candidates: list[dict[str, Any]],
        *,
        flow: str = "compose",
    ) -> None:
        lines = []
        for index, candidate in enumerate(candidates, start=1):
            aliases = self._contacts.aliases_of(int(candidate["id"])) if self._contacts else []
            suffix = f" (alias: {', '.join(aliases[:2])})" if aliases else ""
            lines.append(f"{index}. {candidate['display_name']} <{candidate['email']}>{suffix}")
        self._pending_flows[user_id] = {
            "flow": "choose_candidate",
            "candidates": candidates,
            "target_flow": flow,
        }
        await self._send(question + "\n" + "\n".join(lines))

    async def ask_unknown_recipient(
        self, user_id: int, phrase: str, prefix: str
    ) -> None:
        self._pending_flows[user_id] = {
            "flow": "ask_unknown_recipient",
            "phrase": phrase,
        }
        await self._send(
            f"{prefix} «{phrase}». ¿Me das su correo? "
            "(o dime «escribe a otro nombre»)"
        )

    async def show_contacts_panel(self) -> None:
        contacts = self._contacts.list_contacts() if self._contacts else []
        if not contacts:
            await self._send("No tienes contactos guardados todavía.")
            return
        buttons = [
            [
                InlineKeyboardButton(
                    c["display_name"], callback_data=f"contact:{c['id']}"
                )
            ]
            for c in contacts[:10]
        ]
        buttons.append(
            [
                InlineKeyboardButton("➕ Añadir contacto", callback_data="cnew"),
                InlineKeyboardButton("✖️ Cerrar", callback_data="close"),
            ]
        )
        await self._send(
            "👥 Contactos",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def show_contact_detail(self, contact_id: int) -> None:
        contact = self._contacts.get(contact_id) if self._contacts else None
        if contact is None:
            await self._send("Ese contacto ya no existe.")
            return
        aliases = self._contacts.aliases_of(contact_id) if self._contacts else []
        alias_text = ", ".join(f"«{a}»" for a in aliases) or "(ninguno)"
        buttons = [
            [InlineKeyboardButton("✉️ Cambiar correo", callback_data=f"cemail:{contact_id}")],
            [InlineKeyboardButton("➕ Añadir alias", callback_data=f"caliasadd:{contact_id}")],
            [InlineKeyboardButton("➖ Quitar alias", callback_data=f"caliasdel:{contact_id}")],
            [InlineKeyboardButton("✏️ Renombrar", callback_data=f"crename:{contact_id}")],
            [InlineKeyboardButton("🗑 Borrar contacto", callback_data=f"cdel:{contact_id}")],
            [InlineKeyboardButton("↩️ Volver", callback_data="contacts")],
        ]
        await self._send(
            f"👤 {contact['display_name']}\n📧 {contact['email']}\nAlias: {alias_text}",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def show_menu(self) -> None:
        """Discoverability menu — buttons, no slash commands required."""
        buttons = [
            [
                InlineKeyboardButton("✉️ Nuevo correo", callback_data="nuevo"),
                InlineKeyboardButton("👥 Contactos", callback_data="contacts"),
            ],
            [
                InlineKeyboardButton("⏰ Recordatorios", callback_data="reminders"),
                InlineKeyboardButton("❓ Ayuda", callback_data="help"),
            ],
            [InlineKeyboardButton("✖️ Cerrar", callback_data="close")],
        ]
        await self._send(
            "¿Qué quieres hacer?",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def show_reminders(self, user_id: int) -> None:
        rows = self._reminders.list_pending(user_id) if self._reminders else []
        if not rows:
            await self._send("No tienes recordatorios pendientes.")
            return
        from ..reminders import format_due

        lines = []
        buttons = []
        for row in rows[:10]:
            lines.append(f"⏰ #{row['id']} — {format_due(float(row['due_at']))}")
            buttons.append(
                [
                    InlineKeyboardButton(
                        f"Cancelar #{row['id']}", callback_data=f"rcancel:{row['id']}"
                    )
                ]
            )
        buttons.append([InlineKeyboardButton("✖️ Cerrar", callback_data="close")])
        await self._send(
            "⏰ Recordatorios\n" + "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def show_help(self) -> None:
        await self._send(
            "Puedo ayudarte con el correo. Algunos ejemplos:\n\n"
            "• «respóndele que el viernes sí puedo» (respondiendo a un resumen)\n"
            "• «hazlo más corto» / «más formal» (mientras hay un borrador)\n"
            "• «envíalo» / «cancela el borrador»\n"
            "• «escribe a Roman y dile que mañana llego a las seis»\n"
            "• «reenvíaselo a Daniel»\n"
            "• «mándame el pdf» / «archívalo» / «márcalo como leído»\n"
            "• «recuérdamelo mañana» / «¿qué recordatorios tengo?»\n"
            "• «¿qué contactos tengo?» / «cuando diga Roman usa femo@femo.ch»\n\n"
            "Botones: ✉️ nuevo correo · 👥 contactos · ⏰ recordatorios · ❓ ayuda"
        )

    # ── voice (experimental) ────────────────────────────────────────────────

    async def _handle_voice(self, message: Message, user_id: int) -> None:
        voice = message.voice
        if voice is None or self._action_callback is None:
            return
        if not self._settings.ai_audio_enabled:
            await self._send(
                "Las notas de voz aún no están activadas en esta instalación. "
                "Escríbeme la instrucción por texto, por favor."
            )
            return
        duration = voice.duration or 0
        size = voice.file_size or 0
        if duration > self._settings.ai_audio_max_seconds:
            max_secs = self._settings.ai_audio_max_seconds
            await self._send(f"La nota es muy larga (máximo {max_secs}s).")
            return
        if size > self._settings.ai_audio_max_bytes:
            await self._send("La nota es demasiado grande.")
            return
        try:
            file = await self._ensure_sender().get_file(voice.file_id)
            target = self._outgoing_tmp_path(f"voice_{voice.file_unique_id}.ogg")
            await file.download_to_drive(custom_path=target)
        except Exception:
            logger.exception("voice download failed")
            await self._send("No pude descargar la nota de voz.")
            return
        try:
            await self._send("Escuchando… (experimental)")
            data = target.read_bytes()
            transcript = await self._ai_audio(mime="audio/ogg", data=data)
        except Exception:
            logger.exception("voice transcription failed")
            await self._send(
                "No pude entender la nota de voz (transcripción no disponible). "
                "Escríbeme la instrucción por texto, por favor."
            )
            return
        finally:
            _remove_file(str(target))
        await self._dispatch_intent(
            transcript, thread_id="", tg_message_id=0, user_id=user_id
        )

    async def _ai_audio(self, mime: str, data: bytes) -> str:
        """Transcription via the assistant's AI service (experimental)."""
        if self._assistant is None:
            raise RuntimeError("no assistant wired")
        result = await self._assistant.transcribe_audio(mime, data)
        return str(result)

    async def _collect_outgoing_attachments(
        self, message: Message
    ) -> tuple[OutgoingAttachment, ...]:
        """Download Telegram documents/photos to the temp dir (bounded, safe).

        Limits: ``outgoing_attachment_max_count`` files and
        ``outgoing_attachment_max_bytes`` each. Any rejection (too many,
        oversized, download failure) sends a user-facing notice and raises
        :class:`TelegramAttachmentError` so the attachment-bearing action
        ABORTS — a draft must never silently imply an attachment is included
        when it is not. Filenames are sanitized to display metadata only.
        """
        kind = "document" if message.document is not None else "photo" if message.photo else "none"
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
            logger.info("telegram_attachment type=%s count=%d outcome=rejected", kind, len(entries))
            await self.send_notice(
                f"Demasiados adjuntos (máximo {max_count}); no los añado. Vuelve a intentarlo."
            )
            raise TelegramAttachmentError("attachment count exceeds the limit")
        # Entries shape: (file_id, sanitized_name, reported_size, mime). The
        # rejection notice must name the SANITIZED FILENAME, never the
        # Telegram file id (the file id is an internal identifier).
        oversized = [n for _file_id, n, size, _mime in entries if size > max_bytes]
        if oversized:
            logger.info("telegram_attachment type=%s count=%d outcome=rejected", kind, len(entries))
            await self.send_notice(
                f"Adjunto demasiado grande (máximo {max_bytes // (1024 * 1024)} MB); "
                "no los añado: "
                + ", ".join(oversized)
            )
            raise TelegramAttachmentError("attachment reported oversized")
        results: list[OutgoingAttachment] = []
        for file_id, name, _reported_size, mime in entries:
            target: Path | None = None
            try:
                file = await self._ensure_sender().get_file(file_id)
                target = self._outgoing_tmp_path(name)
                # PTB >=20 async download API (File.download was removed).
                await file.download_to_drive(custom_path=target)
            except Exception:
                logger.exception("downloading telegram attachment %s failed", name)
                logger.info(
                    "telegram_attachment type=%s count=%d outcome=rejected",
                    kind,
                    len(entries),
                )
                for r in results:
                    _remove_file(r.path)
                if target is not None:
                    _remove_file(str(target))
                await self.send_notice("No pude descargar un adjunto; vuelve a intentarlo.")
                raise TelegramAttachmentError("telegram download failed") from None
            # Re-validate against the REAL downloaded size (Telegram's reported
            # size is client-supplied metadata and must not be trusted).
            actual_size = target.stat().st_size
            if actual_size > max_bytes:
                logger.warning("attachment %s exceeds %d bytes after download", name, max_bytes)
                _remove_file(str(target))
                await self.send_notice(
                    f"Adjunto demasiado grande (máximo {max_bytes // (1024 * 1024)} MB); "
                    "no lo añado."
                )
                for r in results:
                    _remove_file(r.path)
                raise TelegramAttachmentError("attachment oversized after download")
            results.append(
                OutgoingAttachment(
                    filename=name, mime_type=mime, size_bytes=actual_size, path=str(target)
                )
            )
        logger.info(
            "telegram_attachment type=%s count=%d outcome=accepted",
            kind,
            len(results),
        )
        return tuple(results)

    def _prune_media_groups(self) -> None:
        """Drop media-group bookkeeping older than the flow TTL (bounded)."""
        cutoff = time.time() - _FLOW_TTL_SECONDS
        stale = [g for g, seen in self._seen_media_groups.items() if seen < cutoff]
        for group in stale:
            self._seen_media_groups.pop(group, None)

    async def _attach_to_active_draft(self, message: Message, user_id: int) -> bool:
        """Attach a Telegram file to the user's ACTIVE draft (no second draft).

        Preserves recipient/subject/body/thread and existing attachments,
        deduplicates by sanitized filename, claims the file into the draft's
        temp dir (deterministic ``NN_name`` convention, same as the responder)
        and re-renders the preview with a bumped draft version. Ownership is
        per Telegram user; stale-preview protection is untouched.
        """
        pending = self._active_pending_for(user_id)
        if pending is None or pending.draft_id == 0:
            return False
        try:
            attachments = await self._collect_outgoing_attachments(message)
        except TelegramAttachmentError:
            return True  # notice already sent; draft untouched
        if not attachments:
            return False
        existing = list(pending.draft.attachments)
        existing_names = {a.filename for a in existing}
        added = [a for a in attachments if a.filename not in existing_names]
        skipped = len(attachments) - len(added)
        if added:
            target_dir = Path(self._settings.tmp_dir) / f"draft-{pending.draft_id}"
            target_dir.mkdir(parents=True, exist_ok=True)
            claimed: list[OutgoingAttachment] = []
            for index, attachment in enumerate(added, start=len(existing)):
                source = Path(attachment.path)
                target = target_dir / f"{index + 1:02d}_{attachment.filename}"
                try:
                    if source != target and source.is_file():
                        source.replace(target)
                except OSError:
                    _remove_file(str(source))
                    continue
                claimed.append(
                    OutgoingAttachment(
                        filename=attachment.filename,
                        mime_type=attachment.mime_type,
                        size_bytes=attachment.size_bytes,
                        path=str(target),
                    )
                )
            if claimed:
                updated = replace(
                    pending.draft,
                    attachments=tuple(existing) + tuple(claimed),
                )
                self._storage.set_draft_attachments(pending.draft_id, updated.attachments)
                await self.apply_draft_edit(pending.draft_id, updated)
                logger.info(
                    "telegram_attachment type=attach outcome=accepted draft=%d count=%d",
                    pending.draft_id,
                    len(claimed),
                )
                await self.send_notice(
                    "Adjunto añadido al borrador: "
                    + ", ".join(a.filename for a in claimed)
                )
            else:
                await self.send_notice("No pude añadir el adjunto al borrador.")
        if skipped:
            await self.send_notice("Adjunto duplicado; no lo añado de nuevo.")
        return True

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
        sender = self._ensure_sender()

        # Generic one-shot confirmations (contacts, destructive changes).
        if action == "confyes":
            await self._confirm_yes(query, token)
            return
        if action == "confno":
            await self._confirm_no(query, token)
            return
        if action == "close":
            await sender.answer_callback_query(query.id, "Cerrado.")
            return

        # Menu / panels.
        if action == "menu":
            await sender.answer_callback_query(query.id)
            await self.show_menu()
            return
        if action == "help":
            await sender.answer_callback_query(query.id)
            await self.show_help()
            return
        if action == "nuevo":
            await sender.answer_callback_query(query.id)
            await self.prompt_compose_recipient(query.from_user.id)
            return
        if action == "contacts":
            await sender.answer_callback_query(query.id)
            await self.show_contacts_panel()
            return
        if action == "cnew":
            await sender.answer_callback_query(query.id)
            self._pending_flows[query.from_user.id] = {"flow": "contact_new_name"}
            await self._send("¿Cómo se llama el contacto?")
            return
        if action == "reminders":
            await sender.answer_callback_query(query.id)
            await self.show_reminders(query.from_user.id)
            return
        if action == "rcancel":
            await self._reminder_cancel(query, token)
            return
        if action == "question":
            await self._handle_question_callback(query, token)
            return
        if action == "att":
            await self._handle_attachment_callback(query, token)
            return

        # Contact management (detail + ops).
        if action == "contact":
            await sender.answer_callback_query(query.id)
            await self._contact_id_or_panel(query, token)
            return
        if action == "cemail":
            await self._contact_flow_start(
                query, token, "contact_email_wait", "¿Cuál es el nuevo correo?"
            )
            return
        if action == "caliasadd":
            await self._contact_flow_start(
                query, token, "contact_alias_wait", "¿Qué alias añado?"
            )
            return
        if action == "caliasdel":
            await self._contact_flow_start(
                query, token, "contact_aliasdel_wait", "¿Qué alias elimino?"
            )
            return
        if action == "crename":
            await self._contact_flow_start(
                query, token, "contact_rename_wait", "¿Cuál es el nuevo nombre?"
            )
            return
        if action == "cdel":
            await self._contact_delete_confirm(query, token)
            return

        # Reply button (Responder).
        if action == "reply":
            await self._handle_reply_callback(query, token)
            return
        # View original.
        if action in ("view", "hide"):
            await self._handle_view_callback(query, action, token)
            return
        if action == "resend":
            await self._handle_resend_callback(query, token)
            return

        # Draft two-step confirmation flow.
        if action in ("sendyes", "cancelyes", "edit"):
            await self._draft_second_step(query, action, token)
            return
        if action == "sendback":
            await self._draft_back(query, token, "Envío cancelado, el borrador sigue pendiente.")
            return
        if action == "cancelback":
            await self._draft_back(query, token, "El borrador sigue pendiente.")
            return
        if action == "editback":
            await self._draft_back(query, token, "No se ha modificado nada.")
            return

        # First tap on SEND / CANCEL / EDIT: confirmation UI, no action yet.
        pending = self._pending_drafts.get(token)
        if pending is None:
            await sender.answer_callback_query(
                query.id, "Este borrador ya no está disponible."
            )
            return
        # Group isolation: only the member who requested the draft may
        # confirm/cancel it (owner 0 = no owner recorded → fail closed).
        if not pending.user_id or query.from_user.id != pending.user_id:
            await sender.answer_callback_query(
                query.id, "Ese borrador lo está gestionando otro miembro."
            )
            return
        if action == "confirm":
            await self._show_send_confirm(query, pending)
            return
        if action == "cancel":
            await self._show_cancel_confirm(query, pending)
            return
        await sender.answer_callback_query(query.id, "Acción no reconocida.")

    # ── generic confirmation ────────────────────────────────────────────────

    async def _confirm_yes(self, query: CallbackQuery, token: str) -> None:
        flow = self._pop_flow_by_token(token)
        if flow is None:
            await self._ensure_sender().answer_callback_query(
                query.id, "Esa confirmación ya no está disponible."
            )
            return
        owner = int(flow.get("user_id") or 0)
        if owner and query.from_user.id != owner:
            await self._ensure_sender().answer_callback_query(
                query.id, "Esa confirmación pertenece a otro miembro."
            )
            return
        await self._ensure_sender().answer_callback_query(query.id, "Hecho ✓")
        if self._action_callback is not None:
            await self._action_callback(str(flow["action"]), dict(flow["payload"]))

    async def _confirm_no(self, query: CallbackQuery, token: str) -> None:
        self._pop_flow_by_token(token)
        await self._ensure_sender().answer_callback_query(query.id, "Cancelado.")

    def _pop_flow_by_token(self, token: str) -> dict[str, Any] | None:
        for user_id, flow in list(self._pending_flows.items()):
            if flow.get("confirm_token") == token:
                self._pending_flows.pop(user_id, None)
                return flow
        return None

    # ── draft two-step confirmations ────────────────────────────────────────

    async def _show_send_confirm(self, query: CallbackQuery, pending: _PendingDraft) -> None:
        """SEND tap → ask "are you sure?" (no send yet)."""
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Sí, enviar", callback_data=f"sendyes:{pending.token}"),
                    InlineKeyboardButton("Volver", callback_data=f"sendback:{pending.token}"),
                ]
            ]
        )
        await self._ensure_sender().answer_callback_query(query.id)
        await self._ensure_sender().send_message(
            self._allowed_chat_id,
            "¿Seguro que quieres enviar este correo?",
            reply_markup=keyboard,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )

    async def _show_cancel_confirm(self, query: CallbackQuery, pending: _PendingDraft) -> None:
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Sí, cancelar",
                        callback_data=f"cancelyes:{pending.token}",
                    ),
                    InlineKeyboardButton("Volver", callback_data=f"cancelback:{pending.token}"),
                ]
            ]
        )
        await self._ensure_sender().answer_callback_query(query.id)
        await self._ensure_sender().send_message(
            self._allowed_chat_id,
            "¿Cancelar este borrador?",
            reply_markup=keyboard,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )

    async def _draft_second_step(self, query: CallbackQuery, action: str, token: str) -> None:
        pending = self._pending_drafts.get(token)
        if pending is None:
            await self._ensure_sender().answer_callback_query(
                query.id, "Este borrador ya no está disponible."
            )
            return
        if not pending.user_id or query.from_user.id != pending.user_id:
            await self._ensure_sender().answer_callback_query(
                query.id, "Ese borrador lo está gestionando otro miembro."
            )
            return
        sender = self._ensure_sender()
        if action == "sendyes":
            if pending.preview_version != pending.draft_version:
                await sender.answer_callback_query(
                    query.id, "El borrador cambió; revisa la vista previa actualizada."
                )
                return
            self._pending_drafts.pop(token, None)
            pending.resolve(True)
            await sender.answer_callback_query(query.id, "Enviando…")
            return
        if action == "cancelyes":
            self._pending_drafts.pop(token, None)
            pending.resolve(False)
            await sender.answer_callback_query(query.id, "Cancelado.")
            return
        if action == "edit":
            await sender.answer_callback_query(query.id)
            await self._send(
                "Dime qué cambio: «hazlo más corto», «más formal», "
                "«cambia las 18:00 por las 19:00»…"
            )

    async def _draft_back(self, query: CallbackQuery, token: str, notice: str) -> None:
        pending = self._pending_drafts.get(token)
        if pending is None:
            await self._ensure_sender().answer_callback_query(
                query.id, "Este borrador ya no está disponible."
            )
            return
        await self._ensure_sender().answer_callback_query(query.id, notice)

    # ── contact UI helpers ──────────────────────────────────────────────────

    async def _contact_id_or_panel(self, query: CallbackQuery, token: str) -> None:
        try:
            contact_id = int(token)
        except ValueError:
            await self.show_contacts_panel()
            return
        await self.show_contact_detail(contact_id)

    async def _contact_flow_start(
        self, query: CallbackQuery, token: str, flow_name: str, prompt: str
    ) -> None:
        try:
            contact_id = int(token)
        except ValueError:
            return
        self._pending_flows[query.from_user.id] = {
            "flow": flow_name,
            "contact_id": contact_id,
        }
        await self._ensure_sender().answer_callback_query(query.id)
        await self._send(prompt)

    async def _contact_delete_confirm(self, query: CallbackQuery, token: str) -> None:
        try:
            contact_id = int(token)
        except ValueError:
            return
        await self._ensure_sender().answer_callback_query(query.id)
        await self._confirm_flow(
            query.from_user.id,
            "¿Borro este contacto y sus alias?",
            "contact_delete_confirm",
            {"contact_id": contact_id},
        )

    async def _reminder_cancel(self, query: CallbackQuery, token: str) -> None:
        try:
            reminder_id = int(token)
        except ValueError:
            return
        if self._action_callback is None:
            return
        await self._ensure_sender().answer_callback_query(query.id, "Cancelando…")
        await self._action_callback(
            "cancel_reminder",
            {"reminder_id": reminder_id, "user_id": query.from_user.id},
        )

    # ── "Responder" button ─────────────────────────────────────────────────

    async def _handle_reply_callback(self, query: CallbackQuery, token: str) -> None:
        """User pressed "Responder": remember the target thread and ask for
        their intent; their next plain message becomes the reply instruction."""
        await self._set_reply_mode(query, token, mode="reply")

    async def _handle_question_callback(self, query: CallbackQuery, token: str) -> None:
        """User pressed "Preguntar": next message is a question about the email."""
        await self._set_reply_mode(query, token, mode="question")

    async def _set_reply_mode(
        self, query: CallbackQuery, token: str, *, mode: str
    ) -> None:
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
        # Freeze the exact Gmail message the summary maps to (reply target).
        target_message_id = self._storage.get_meta(
            f"{_TG_GMAIL_PREFIX}{tg_message_id}"
        ) or ""
        self._pending_replies[query.from_user.id] = (
            tg_message_id,
            thread_id,
            mode,
            target_message_id,
        )
        if mode == "question":
            await self._send("¿Qué quieres saber sobre este correo?")
        elif sender_name:
            await self._send(f"¿Qué quieres decirle a {sender_name}?")
        else:
            await self._send("¿Qué quieres decirle?")

    # ── attachments panel (Gmail → Telegram) ────────────────────────────────

    async def _handle_attachment_callback(self, query: CallbackQuery, token: str) -> None:
        """Token format "<tg_message_id>-<index>"; index -1 lists the panel."""
        parts = token.split("-", maxsplit=1)
        try:
            tg_message_id = int(parts[0])
            index = int(parts[1]) if len(parts) > 1 else -1
        except ValueError:
            return
        await self._ensure_sender().answer_callback_query(query.id, "Preparando…")
        if self._action_callback is None:
            return
        await self._action_callback(
            "get_attachment",
            {"tg_message_id": tg_message_id, "index": index, "user_id": query.from_user.id},
        )

    async def show_attachments_panel(self, tg_message_id: int, email: ParsedEmail) -> None:
        buttons = [
            [
                InlineKeyboardButton(
                    f"📎 {a.filename} ({_format_size(a.size_bytes)})",
                    callback_data=f"att:{tg_message_id}-{index}",
                )
            ]
            for index, a in enumerate(email.attachments[:MAX_ATTACHMENTS_PANEL])
        ]
        buttons.append([InlineKeyboardButton("✖️ Cerrar", callback_data="close")])
        await self._send(
            f"Adjuntos de «{email.subject}»:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def send_document_file(self, path: str, filename: str) -> None:
        """Send a temp file to the group; the caller removes it afterwards."""
        data = await asyncio.to_thread(Path(path).read_bytes)
        await self._ensure_sender().send_document(
            self._allowed_chat_id,
            data,
            filename=filename,
        )

    def write_temp_file(self, filename: str, data: bytes) -> str:
        """Safe temp file for attachment delivery (random token prefix)."""
        base = Path(self._settings.tmp_dir) / "delivery"
        base.mkdir(parents=True, exist_ok=True)
        safe = sanitize_filename(filename)
        token = secrets.token_hex(6)
        target = base / f"{token}_{safe}"
        target.write_bytes(data)
        return str(target)

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
        if not owner or query.from_user.id != owner:
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
        temporary message(s). No LLM call, no SQLite body read.

        Re-pressing hides the previously posted set first, so repeated presses
        never flood the group or orphan messages.
        """
        await sender.answer_callback_query(query.id, "Cargando…")
        raw_prev = self._storage.get_meta(f"{_TG_ORIG_PREFIX}{tg_message_id}")
        if raw_prev:
            await self._delete_original_messages(sender, raw_prev)
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
        await self._delete_original_messages(sender, raw)
        self._storage.delete_meta(f"{_TG_ORIG_PREFIX}{tg_message_id}")
        await sender.answer_callback_query(query.id, "Original oculto.")

    async def _delete_original_messages(self, sender: Sender, raw: str | None) -> None:
        message_ids: list[int] = []
        if raw:
            try:
                message_ids = [int(m) for m in json.loads(raw)]
            except (ValueError, TypeError, json.JSONDecodeError):
                logger.warning("corrupt tgo mapping")
                message_ids = []
        for message_id in message_ids:
            try:
                await sender.delete_message(self._allowed_chat_id, message_id)
            except Exception:
                logger.exception("deleting original message %s failed", message_id)

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
                        "Preguntar", callback_data=f"question:{message.message_id}"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Responder", callback_data=f"reply:{message.message_id}"
                    ),
                    InlineKeyboardButton(
                        "📎 Adjuntos", callback_data=f"att:{message.message_id}"
                    ),
                ],
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

    async def send_rich_notice(self, text: str) -> int:
        """Send an informational answer (Q&A / summary) with SAFE rich
        formatting: only emoji-led section headings are bolded, everything else
        is HTML-escaped plain text. Falls back to plain text if the formatted
        send fails (the answer is never lost)."""
        rendered = render_rich_text(text)
        try:
            sender = self._ensure_sender()
            message = await sender.send_message(
                self._allowed_chat_id,
                rendered,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
            return message.message_id
        except Exception:
            logger.warning("rich-format send failed; falling back to plain text")
            return await self._send(neutralize_links(text))

    async def send_qa_answer(self, answer: str, sections: list[QaSection]) -> int:
        """Send the structured Q&A contract with deterministic safe formatting
        (application-controlled sections, escaped values, allowlisted emojis).
        Falls back to a plain-text rendering of the SAME content when the
        formatted send fails — the answer is never lost."""
        rendered = render_qa_answer(answer, sections)
        try:
            sender = self._ensure_sender()
            message = await sender.send_message(
                self._allowed_chat_id,
                rendered,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
            return message.message_id
        except Exception:
            logger.warning("rich-format send failed; falling back to plain text")
            return await self._send(neutralize_links(render_qa_plain(answer, sections)))

    async def send_summary_answer(
        self, headline: str, sections: list[QaSection]
    ) -> int:
        """Send the structured thread-summary contract with the same safe,
        deterministic formatting as Q&A (fixed 📬 header, section blocks).
        Falls back to a plain-text rendering of the SAME content when the
        formatted send fails — the summary is never lost."""
        rendered = render_summary(headline, sections)
        try:
            sender = self._ensure_sender()
            message = await sender.send_message(
                self._allowed_chat_id,
                rendered,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
            return message.message_id
        except Exception:
            logger.warning("rich-format send failed; falling back to plain text")
            return await self._send(
                neutralize_links(render_summary_plain(headline, sections))
            )

    async def send_typing(self) -> None:
        await self._ensure_sender().send_chat_action(self._allowed_chat_id, ChatAction.TYPING)

    async def send_draft_for_confirmation(
        self, draft: DraftReply, *, user_id: int = 0, draft_id: int = 0
    ) -> int:
        """Show the COMPLETE final draft (recipients, subject, attachments,
        type) with SEND/EDIT/CANCEL buttons. SEND/CANCEL need a second tap.

        The preview is versioned: any edit re-renders it and bumps the preview
        version — a stale preview can never authorize a send.
        """
        pending = _PendingDraft(
            token=secrets.token_urlsafe(8),
            draft=draft,
            message_id=0,
            draft_id=draft_id,
            user_id=user_id,
        )
        message_id = await self._post_draft_preview(pending)
        pending.message_id = message_id
        self._pending_drafts[pending.token] = pending
        return message_id

    async def _post_draft_preview(self, pending: _PendingDraft) -> int:
        """Post (or re-post) the draft preview; returns the new message id."""
        draft = pending.draft
        to_line = (
            ", ".join(str(address) for address in draft.to)
            if draft.to
            else "(sin destinatario)"
        )
        cc_line = ""
        if draft.cc:
            cc_line = f"\nCC: {', '.join(str(a) for a in draft.cc)}"
        attachments_line = ""
        if draft.attachments:
            names = "\n".join(
                f"- {a.filename} ({_format_size(a.size_bytes)})" for a in draft.attachments
            )
            attachments_line = f"\nAdjuntos:\n{names}"
        kind = (
            "Reenvío"
            if draft.forward_of
            else ("Respuesta" if draft.thread_id else "Nuevo correo")
        )
        spanish_block = ""
        if draft.body_es:
            spanish_block = (
                f"\n\n🇪🇸 Español · traducción\n{neutralize_links(draft.body_es)}"
            )
        elif draft.translation_failed:
            spanish_block = (
                "\n\n🇪🇸 Español · traducción\n⚠️ No pude generar la traducción ahora."
            )
        text = (
            f"Borrador ({kind})\n"
            f"Para: {to_line}{cc_line}\n"
            f"Asunto: {draft.subject}\n"
            f"{attachments_line}\n\n"
            f"🇩🇪 Alemán · se enviará\n{neutralize_links(draft.body)}"
            f"{spanish_block}"
        )
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Enviar", callback_data=f"confirm:{pending.token}"),
                    InlineKeyboardButton("Editar", callback_data=f"edit:{pending.token}"),
                    InlineKeyboardButton("Cancelar", callback_data=f"cancel:{pending.token}"),
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

    async def apply_draft_edit(self, draft_id: int, new_draft: DraftReply) -> None:
        """Coordinator hook: replace the pending draft, bump versions, and
        RE-RENDER the complete preview (old token invalidated → replay-safe)."""
        pending = next(
            (p for p in self._pending_drafts.values() if p.draft_id == draft_id), None
        )
        if pending is None:
            return
        old_token = pending.token
        old_message_id = pending.message_id
        pending.draft = new_draft
        pending.draft_version += 1
        pending.token = secrets.token_urlsafe(8)
        new_message_id = await self._post_draft_preview(pending)
        pending.message_id = new_message_id
        pending.preview_version = pending.draft_version
        # Invalidate the old preview + buttons (replay of old callbacks is a no-op).
        self._pending_drafts.pop(old_token, None)
        self._pending_drafts[pending.token] = pending
        with contextlib.suppress(Exception):
            await self._ensure_sender().delete_message(
                self._allowed_chat_id, old_message_id
            )

    async def wait_for_confirmation(
        self, draft_id: int, timeout_seconds: float = _CONFIRM_TIMEOUT_SECONDS
    ) -> bool:
        """Wait for the user's send/cancel decision on a draft (by draft id).

        The coordinator calls this right after ``send_draft_for_confirmation``
        and persists the draft only if True. Edits re-render the preview
        without disturbing the pending future. Returns False on cancel,
        timeout, or an unknown draft id.
        """
        pending = next(
            (p for p in self._pending_drafts.values() if p.draft_id == draft_id), None
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

    # ── events consumed by the coordinator's responder ─────────────────────

    async def reply_requests(self) -> AsyncIterator[ReplyRequest]:
        """Stream of reply requests; consume this forever with ``async for``."""
        while True:
            yield await self._queue.get()

    # ── helpers ────────────────────────────────────────────────────────────

    async def _send(
        self,
        text: str,
        *,
        reply_to: int | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> int:
        sender = self._ensure_sender()
        message = await sender.send_message(
            self._allowed_chat_id,
            text,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
            reply_parameters=ReplyParameters(message_id=reply_to) if reply_to is not None else None,
            reply_markup=reply_markup,
        )
        return message.message_id
