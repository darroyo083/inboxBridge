"""Telegram bot tests: chat authorization, triggers, commands, draft flow.

No network: a FakeSender stands in for PTB's Bot and updates are built as
plain objects. The real ``db.Storage`` is used (tmp SQLite file).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from telegram import (
    CallbackQuery,
    Chat,
    Document,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
    MessageEntity,
    PhotoSize,
    ReplyParameters,
    Update,
    User,
)
from telegram.constants import ChatAction, ChatType, ParseMode

from inboxbridge.config import Settings
from inboxbridge.db import Storage
from inboxbridge.models import (
    DraftReply,
    EmailAddress,
    EmailSummary,
    OutgoingAttachment,
    ParsedEmail,
)
from inboxbridge.telegram.bot import TelegramBot

CHAT_ID = -100123456789
BOT_ID = 42
BOT_USERNAME = "inboxbridge_bot"


class FakeFile:
    """Minimal Telegram File stand-in with the blocking ``download`` API."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def download(self, custom_path: Any = None) -> Any:
        path = Path(custom_path)
        path.write_bytes(self._data)
        return path


class FakeSender:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.chat_actions: list[Any] = []
        self.edited: list[tuple[int | None, str, InlineKeyboardMarkup | None]] = []
        self.answered: list[str] = []
        self.deleted: list[int] = []
        self.files: dict[str, FakeFile] = {}
        self.downloaded: list[Any] = []
        self.files_delivered: list[tuple[str, bytes]] = []
        self.parse_modes: list[ParseMode | None] = []
        self.fail_html: bool = False  # True: formatted (HTML) sends raise
        self._next_id = 1

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        parse_mode: ParseMode | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        link_preview_options: LinkPreviewOptions | None = None,
        reply_parameters: ReplyParameters | None = None,
    ) -> Message:
        if self.fail_html and parse_mode == ParseMode.HTML:
            raise RuntimeError("simulated formatted send failure")
        self.parse_modes.append(parse_mode)
        message = Message(
            message_id=self._next_id,
            date=datetime.now(UTC),
            chat=Chat(id=cast(int, chat_id), type=ChatType.GROUP),
            text=text,
            reply_markup=reply_markup,
        )
        self._next_id += 1
        self.messages.append(message)
        return message

    async def send_chat_action(self, chat_id: int | str, action: ChatAction) -> bool:
        self.chat_actions.append(action)
        return True

    async def edit_message_text(
        self,
        text: str,
        *,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message:
        self.edited.append((message_id, text, reply_markup))
        return Message(
            message_id=message_id or 0,
            date=datetime.now(UTC),
            chat=Chat(id=cast(int, chat_id or 0), type=ChatType.GROUP),
            text=text,
        )

    async def answer_callback_query(
        self, callback_query_id: str, text: str | None = None, show_alert: bool = False
    ) -> bool:
        self.answered.append(text or "")
        return True

    async def delete_message(self, chat_id: int | str, message_id: int) -> bool:
        self.deleted.append(message_id)
        return True

    async def get_file(self, file_id: str) -> FakeFile:
        if file_id not in self.files:
            raise RuntimeError(f"unknown file id {file_id}")
        file = self.files[file_id]
        self.downloaded.append(file_id)
        return file

    async def send_document(
        self,
        chat_id: int | str,
        document: Any,
        *,
        filename: str | None = None,
        reply_parameters: ReplyParameters | None = None,
    ) -> Message:
        data = document if isinstance(document, bytes) else document.read()
        self.files_delivered.append((filename or "file", bytes(data)))
        message = Message(
            message_id=self._next_id,
            date=datetime.now(UTC),
            chat=Chat(id=cast(int, chat_id), type=ChatType.GROUP),
            text="",
        )
        self._next_id += 1
        return message


@pytest.fixture
def make_env(tmp_path: Any) -> Callable[..., tuple[TelegramBot, FakeSender, Storage]]:
    def _make(**bot_kwargs: Any) -> tuple[TelegramBot, FakeSender, Storage]:
        settings = Settings(
            _env_file=None,
            TELEGRAM_BOT_TOKEN="test-token",
            TELEGRAM_ALLOWED_CHAT_ID=CHAT_ID,
            LLM_API_KEY="test-key",
            LLM_BASE_URL="https://api.test/v1",
        )
        storage = Storage(tmp_path / "state.sqlite")
        storage.connect()
        sender = FakeSender()
        bot = TelegramBot(
            settings,
            storage,
            sender=sender,
            bot_user_id=BOT_ID,
            bot_username=BOT_USERNAME,
            **bot_kwargs,
        )
        return bot, sender, storage

    return _make


def _user(user_id: int, *, is_bot: bool = False) -> User:
    return User(id=user_id, is_bot=is_bot, first_name="Tester" if not is_bot else "Bot")


def _message(
    message_id: int,
    chat_id: int,
    text: str | None,
    user_id: int,
    *,
    is_bot: bool = False,
    reply_to: Message | None = None,
    entities: list[MessageEntity] | None = None,
) -> Message:
    return Message(
        message_id=message_id,
        date=datetime.now(UTC),
        chat=Chat(id=chat_id, type=ChatType.GROUP),
        from_user=_user(user_id, is_bot=is_bot),
        text=text,
        entities=entities,
        reply_to_message=reply_to,
    )


def _update(message: Message) -> Update:
    return Update(update_id=1, message=message)


def _callback_update(
    chat_id: int,
    data: str,
    *,
    callback_id: str = "cq-1",
    message_id: int = 50,
    from_user_id: int = 7,
) -> Update:
    message = _message(message_id, chat_id, "Borrador", from_user_id)
    query = CallbackQuery(
        id=callback_id,
        from_user=_user(from_user_id),
        chat_instance="instance",
        data=data,
        message=message,
    )
    return Update(update_id=2, callback_query=query)


async def _drain(bot: TelegramBot) -> list[Any]:
    out: list[Any] = []
    try:
        while True:
            out.append(await asyncio.wait_for(bot._queue.get(), timeout=0.05))
    except TimeoutError:
        return out


def _draft() -> DraftReply:
    return DraftReply(
        thread_id="t1",
        subject="Re: Proyecto",
        to=[EmailAddress("Ana", "ana@example.com")],
        cc=[],
        body="Sehr geehrte Frau Ana,\n\ndanke.\n\nMit freundlichen Grüßen",
    )


def _email() -> ParsedEmail:
    return ParsedEmail(
        message_id="m1",
        thread_id="t1",
        history_id=1,
        subject="Presupuesto",
        sender=EmailAddress("Ana", "ana@example.com"),
        recipients=[EmailAddress("Bob", "bob@example.com")],
        date_iso="2026-08-07T10:00:00+00:00",
        body_text="Hola, aquí va el presupuesto: https://example.com/track?id=1",
    )


# ── authorization ─────────────────────────────────────────────────────────


async def test_ignores_private_chat(make_env: Any) -> None:
    bot, sender, storage = make_env()
    message = _message(1, 555, "Hola", 7, reply_to=_message(2, 555, "Hola", BOT_ID, is_bot=True))
    await bot.process_update(_update(message))
    assert sender.messages == []
    assert await _drain(bot) == []


async def test_ignores_other_group_even_with_command(make_env: Any) -> None:
    bot, sender, storage = make_env()
    message = _message(1, -999, "/status", 7)
    await bot.process_update(_update(message))
    assert sender.messages == []
    assert await _drain(bot) == []


async def test_ignores_other_group_callback(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.send_draft_for_confirmation(_draft(), user_id=7, draft_id=1)
    task = asyncio.create_task(bot.wait_for_confirmation(1))
    await asyncio.sleep(0)
    await bot.process_update(_callback_update(-777, "cancel:whatever"))
    await asyncio.sleep(0.02)
    assert not task.done()
    task.cancel()


async def test_ignores_plain_message_in_authorized_chat(make_env: Any) -> None:
    bot, sender, storage = make_env()
    message = _message(1, CHAT_ID, "hola a todos", 7)
    await bot.process_update(_update(message))
    assert sender.messages == []
    assert await _drain(bot) == []


async def test_ignores_bot_messages(make_env: Any) -> None:
    bot, sender, storage = make_env()
    message = _message(1, CHAT_ID, "hola", BOT_ID, is_bot=True)
    await bot.process_update(_update(message))
    assert await _drain(bot) == []


# ── reply to bot's own message ────────────────────────────────────────────


async def test_reply_to_bot_message_emits_reply_request(make_env: Any) -> None:
    bot, sender, storage = make_env()
    storage.set_meta("tg:10", "thread-1")
    bot_message = _message(10, CHAT_ID, "Presupuesto", BOT_ID, is_bot=True)
    message = _message(11, CHAT_ID, "responde que sí", 7, reply_to=bot_message)
    await bot.process_update(_update(message))
    (request,) = await _drain(bot)
    assert request.thread_id == "thread-1"
    assert request.user_instructions == "responde que sí"
    assert request.source_message_id == 11


async def test_reply_to_bot_message_without_mapping_has_empty_thread(make_env: Any) -> None:
    bot, sender, storage = make_env()
    bot_message = _message(10, CHAT_ID, "aviso", BOT_ID, is_bot=True)
    message = _message(11, CHAT_ID, "gracias", 7, reply_to=bot_message)
    await bot.process_update(_update(message))
    (request,) = await _drain(bot)
    assert request.thread_id == ""


async def test_reply_to_draft_message_is_ignored(make_env: Any) -> None:
    bot, sender, storage = make_env()
    draft_message_id = await bot.send_draft_for_confirmation(
        _draft(), user_id=7, draft_id=1
    )
    draft_message = _message(draft_message_id, CHAT_ID, "Borrador", BOT_ID, is_bot=True)
    message = _message(99, CHAT_ID, "confirmo por texto", 7, reply_to=draft_message)
    await bot.process_update(_update(message))
    assert await _drain(bot) == []


# ── mentions ──────────────────────────────────────────────────────────────


async def test_mention_entity_emits_reply_request(make_env: Any) -> None:
    bot, sender, storage = make_env()
    text = f"@{BOT_USERNAME} redacta una respuesta"
    entity = MessageEntity(type=MessageEntity.MENTION, offset=0, length=len(text.split(" ")[0]))
    message = _message(5, CHAT_ID, text, 7, entities=[entity])
    await bot.process_update(_update(message))
    (request,) = await _drain(bot)
    assert request.user_instructions == text
    assert request.thread_id == ""


async def test_mention_via_text_mention_entity(make_env: Any) -> None:
    bot, sender, storage = make_env()
    text = "@InboxBridge responde"
    entity = MessageEntity(
        type=MessageEntity.TEXT_MENTION,
        offset=0,
        length=len("@InboxBridge"),
        user=_user(BOT_ID, is_bot=True),
    )
    message = _message(6, CHAT_ID, text, 7, entities=[entity])
    await bot.process_update(_update(message))
    (request,) = await _drain(bot)
    assert request.user_instructions == text


async def test_mention_without_entity_fallback(make_env: Any) -> None:
    bot, sender, storage = make_env()
    message = _message(7, CHAT_ID, f"hola @{BOT_USERNAME}", 7)
    await bot.process_update(_update(message))
    (request,) = await _drain(bot)
    assert request.user_instructions == f"hola @{BOT_USERNAME}"


# ── commands ──────────────────────────────────────────────────────────────


async def test_status_command_calls_provider(make_env: Any) -> None:
    calls: list[bool] = []

    async def provider() -> str:
        calls.append(True)
        return "llm: ok\ntelegram: ok"

    bot, sender, storage = make_env(status_provider=provider)
    message = _message(20, CHAT_ID, "/status", 7)
    await bot.process_update(_update(message))
    assert calls == [True]
    assert sender.messages[-1].text == "llm: ok\ntelegram: ok"


async def test_status_command_without_provider(make_env: Any) -> None:
    bot, sender, storage = make_env()
    message = _message(20, CHAT_ID, "/status", 7)
    await bot.process_update(_update(message))
    assert "no hay checks" in sender.messages[-1].text


async def test_unknown_command_ignored(make_env: Any) -> None:
    bot, sender, storage = make_env()
    message = _message(21, CHAT_ID, "/hack", 7)
    await bot.process_update(_update(message))
    assert sender.messages == []
    assert await _drain(bot) == []


async def test_cancel_command_resolves_pending_draft(make_env: Any) -> None:
    bot, sender, storage = make_env()
    draft_message_id = await bot.send_draft_for_confirmation(_draft())
    task = asyncio.create_task(bot.wait_for_confirmation(draft_message_id))
    await asyncio.sleep(0)
    message = _message(22, CHAT_ID, "/cancel", 7)
    await bot.process_update(_update(message))
    assert await task is False
    assert "cancelados" in sender.messages[-1].text


# ── explicit memory commands (/remember /memory /forget) ──────────────────


async def test_remember_stores_fact_and_replies(make_env: Any) -> None:
    bot, sender, storage = make_env()
    message = _message(30, CHAT_ID, "/remember Roman es tu jefe", 7)
    await bot.process_update(_update(message))
    assert sender.messages[-1].text == "Me guardo que Roman es tu jefe."
    facts = storage.list_memories(7)
    assert [f["value"] for f in facts] == ["Roman es tu jefe"]
    assert [f["key"] for f in facts] == ["roman es tu jefe"]


async def test_remember_long_fact_derives_key_from_first_four_words(make_env: Any) -> None:
    bot, sender, storage = make_env()
    fact = "La reunión de presupuesto es el martes a las 10"
    await bot.process_update(_update(_message(30, CHAT_ID, f"/remember {fact}", 7)))
    (saved,) = storage.list_memories(7)
    assert saved["key"] == "la reunión de presupuesto"
    assert saved["value"] == fact
    assert sender.messages[-1].text == f"Me guardo que {fact}."


async def test_remember_same_fact_is_upserted(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/remember Roman es tu jefe", 7)))
    await bot.process_update(_update(_message(31, CHAT_ID, "/remember Roman es tu jefe", 7)))
    assert len(storage.list_memories(7)) == 1


async def test_remember_rejects_secret_like_facts(make_env: Any) -> None:
    bot, sender, storage = make_env()
    for text in ("/remember sk-abcdefgh1234", "/remember password=123"):
        await bot.process_update(_update(_message(30, CHAT_ID, text, 7)))
    assert "no lo guardo" in sender.messages[-1].text
    assert storage.list_memories(7) == []


async def test_remember_without_argument_shows_usage(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/remember", 7)))
    assert "/remember" in sender.messages[-1].text
    assert storage.list_memories(7) == []


async def test_memory_command_lists_facts_ordered_by_key(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/remember Roman es tu jefe", 7)))
    await bot.process_update(_update(_message(31, CHAT_ID, "/remember Trabajas en FEMO", 7)))
    await bot.process_update(_update(_message(32, CHAT_ID, "/memory", 7)))
    assert sender.messages[-1].text == "Recuerdo:\n* Roman es tu jefe\n* Trabajas en FEMO"


async def test_memory_command_empty_state(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/memory", 7)))
    assert "No tengo nada guardado" in sender.messages[-1].text


async def test_memory_command_filters_by_query(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/remember Roman es tu jefe", 7)))
    await bot.process_update(_update(_message(31, CHAT_ID, "/memory roman", 7)))
    assert sender.messages[-1].text == "Recuerdo:\n* Roman es tu jefe"
    await bot.process_update(_update(_message(32, CHAT_ID, "/memory inexistente", 7)))
    assert sender.messages[-1].text == "No tengo nada guardado sobre eso."


async def test_forget_removes_matching_fact_and_replies(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/remember Roman es tu jefe", 7)))
    await bot.process_update(_update(_message(31, CHAT_ID, "/forget roman", 7)))
    assert sender.messages[-1].text == "He olvidado lo de Roman."
    assert storage.list_memories(7) == []


async def test_forget_removes_multiple_matching_facts(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/remember Roman es mi jefe", 7)))
    await bot.process_update(_update(_message(31, CHAT_ID, "/remember Roman es mi colega", 7)))
    await bot.process_update(_update(_message(32, CHAT_ID, "/forget roman es mi", 7)))
    assert sender.messages[-1].text == "He olvidado 2 cosas de Roman es mi."
    assert storage.list_memories(7) == []


async def test_forget_not_found_replies_naturally(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/forget roman", 7)))
    assert sender.messages[-1].text == "No tengo nada guardado de eso."


async def test_memory_isolated_between_telegram_users(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/remember Roman es tu jefe", 7)))
    await bot.process_update(_update(_message(31, CHAT_ID, "/memory", 8)))
    assert "No tengo nada guardado" in sender.messages[-1].text
    await bot.process_update(_update(_message(32, CHAT_ID, "/forget roman", 8)))
    assert sender.messages[-1].text == "No tengo nada guardado de eso."
    assert len(storage.list_memories(7)) == 1


async def test_memory_commands_ignored_outside_allowed_chat(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, -999, "/remember Roman es tu jefe", 7)))
    assert sender.messages == []
    assert storage.list_memories(7) == []


async def test_memory_persists_after_restart(make_env: Any, tmp_path: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/remember Roman es tu jefe", 7)))
    storage.close()
    storage2 = Storage(tmp_path / "state.sqlite")
    storage2.connect()
    assert [f["value"] for f in storage2.list_memories(7)] == ["Roman es tu jefe"]
    storage2.close()


async def test_email_content_is_never_auto_stored(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.send_summary(
        _email(), EmailSummary(subject_es="Asunto", summary_es="Resumen secreto")
    )
    assert storage.list_memories(7) == []
    assert storage.list_memories(BOT_ID) == []
    assert storage.get_meta("tg:1") == "t1"  # only ID mappings, never email content


async def test_reply_request_carries_only_requesting_users_memory(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_update(_message(30, CHAT_ID, "/remember Roman es tu jefe", 7)))
    await bot.process_update(_update(_message(31, CHAT_ID, "/remember Trabajas en FEMO", 7)))
    await bot.process_update(_update(_message(32, CHAT_ID, "/remember Datos del banco", 8)))
    bot_message = _message(40, CHAT_ID, "Presupuesto", BOT_ID, is_bot=True)
    message = _message(41, CHAT_ID, "responde que sí", 7, reply_to=bot_message)
    await bot.process_update(_update(message))
    (request,) = await _drain(bot)
    assert request.memory == ("Roman es tu jefe", "Trabajas en FEMO")


def _button_data(markup: InlineKeyboardMarkup, row: int, col: int) -> str:
    """Callback data of a button (PTB types the attribute as object)."""
    return cast(str, markup.inline_keyboard[row][col].callback_data or "")


# ── draft confirmation buttons (two-step) ──────────────────────────────────


async def test_send_tap_requires_second_confirmation(make_env: Any) -> None:
    """Button misclick protection: SEND tap shows a confirm dialog; the draft
    is only resolved after 'Sí, enviar'."""
    bot, sender, storage = make_env()
    await bot.send_draft_for_confirmation(_draft(), user_id=7, draft_id=1)
    markup = sender.messages[-1].reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    token = _button_data(markup, 0, 0).split(":", 1)[1]
    task = asyncio.create_task(bot.wait_for_confirmation(1))
    await asyncio.sleep(0)
    # First tap: confirmation UI only, NO resolution.
    await bot.process_update(_callback_update(CHAT_ID, f"confirm:{token}", from_user_id=7))
    await asyncio.sleep(0.02)
    assert not task.done()
    assert any("Seguro" in (m.text or "") for m in sender.messages)
    # Second tap (the actual confirmation): resolve True.
    await bot.process_update(_callback_update(CHAT_ID, f"sendyes:{token}", from_user_id=7))
    assert await task is True


async def test_send_back_keeps_draft_pending(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.send_draft_for_confirmation(_draft(), user_id=7, draft_id=1)
    markup = sender.messages[-1].reply_markup
    token = _button_data(markup, 0, 0).split(":", 1)[1]
    task = asyncio.create_task(bot.wait_for_confirmation(1))
    await asyncio.sleep(0)
    await bot.process_update(_callback_update(CHAT_ID, f"confirm:{token}", from_user_id=7))
    await bot.process_update(_callback_update(CHAT_ID, f"sendback:{token}", from_user_id=7))
    await asyncio.sleep(0.02)
    assert not task.done()  # still pending
    task.cancel()


async def test_cancel_tap_requires_second_confirmation(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.send_draft_for_confirmation(_draft(), user_id=7, draft_id=1)
    markup = sender.messages[-1].reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    token = _button_data(markup, 0, 2).split(":", 1)[1]
    task = asyncio.create_task(bot.wait_for_confirmation(1))
    await asyncio.sleep(0)
    await bot.process_update(_callback_update(CHAT_ID, f"cancel:{token}", from_user_id=7))
    await asyncio.sleep(0.02)
    assert not task.done()  # first tap does not cancel
    await bot.process_update(_callback_update(CHAT_ID, f"cancelyes:{token}", from_user_id=7))
    assert await task is False


async def test_ownerless_draft_confirm_is_fail_closed(make_env: Any) -> None:
    """A draft with no recorded owner can never be confirmed (fail closed)."""
    bot, sender, storage = make_env()
    await bot.send_draft_for_confirmation(_draft(), draft_id=1)  # user_id=0
    markup = sender.messages[-1].reply_markup
    token = _button_data(markup, 0, 0).split(":", 1)[1]
    task = asyncio.create_task(bot.wait_for_confirmation(1))
    await asyncio.sleep(0)
    await bot.process_update(_callback_update(CHAT_ID, f"confirm:{token}"))
    assert "otro miembro" in sender.answered[-1]
    assert not task.done()  # not confirmed
    task.cancel()


async def test_stale_callback_is_answered_and_ignored(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_callback_update(CHAT_ID, "confirm:unknown-token"))
    assert "ya no está disponible" in sender.answered[-1]


async def test_wait_for_confirmation_unknown_draft_returns_false(make_env: Any) -> None:
    bot, sender, storage = make_env()
    assert await bot.wait_for_confirmation(99999) is False


async def test_wait_for_confirmation_timeout(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.send_draft_for_confirmation(_draft(), user_id=7, draft_id=1)
    assert await bot.wait_for_confirmation(1, timeout_seconds=0.05) is False


# ── notifier methods ──────────────────────────────────────────────────────


async def test_send_summary_stores_thread_mapping_and_shows_spanish_subject(
    make_env: Any,
) -> None:
    bot, sender, storage = make_env()
    summary = EmailSummary(
        subject_es="Plan de trabajo de la próxima semana",
        summary_es="Resumen con enlace https://evil.example.com/x",
    )
    message_id = await bot.send_summary(_email(), summary)
    assert message_id == 1
    assert storage.get_meta(f"tg:{message_id}") == "t1"
    text = sender.messages[0].text
    assert "Plan de trabajo de la próxima semana" in text
    assert "Presupuesto" not in text  # original German subject is not shown
    assert "De: Ana (ana@example.com)" in text
    assert "Resumen con enlace hxxps://evil.example.com/x" in text


async def test_send_summary_falls_back_to_original_subject_when_missing(
    make_env: Any,
) -> None:
    bot, sender, storage = make_env()
    summary = EmailSummary(subject_es="", summary_es="Resumen de prueba.")
    message_id = await bot.send_summary(_email(), summary)
    assert storage.get_meta(f"tg:{message_id}") == "t1"  # thread mapping intact
    assert "Presupuesto" in sender.messages[0].text  # original subject fallback
    assert "De: Ana (ana@example.com)" in sender.messages[0].text


@pytest.mark.parametrize("name", ["", "   "])
async def test_send_summary_shows_email_only_when_sender_has_no_useful_name(
    make_env: Any, name: str
) -> None:
    bot, sender, storage = make_env()
    email = _email()
    email = ParsedEmail(
        message_id=email.message_id,
        thread_id=email.thread_id,
        history_id=email.history_id,
        subject=email.subject,
        sender=EmailAddress(name, email.sender.email),
        recipients=email.recipients,
        date_iso=email.date_iso,
        body_text=email.body_text,
        attachments=email.attachments,
    )
    summary = EmailSummary(subject_es="Asunto traducido", summary_es="Resumen de prueba.")
    message_id = await bot.send_summary(email, summary)
    assert storage.get_meta(f"tg:{message_id}") == "t1"  # thread mapping intact
    text = sender.messages[0].text
    assert "Asunto traducido" in text
    assert "De: ana@example.com" in text
    assert "ana@example.com (ana@example.com)" not in text


async def test_send_summary_does_not_duplicate_email_as_sender_name(
    make_env: Any,
) -> None:
    bot, sender, storage = make_env()
    email = _email()
    email = ParsedEmail(
        message_id=email.message_id,
        thread_id=email.thread_id,
        history_id=email.history_id,
        subject=email.subject,
        sender=EmailAddress("ana@example.com", email.sender.email),
        recipients=email.recipients,
        date_iso=email.date_iso,
        body_text=email.body_text,
        attachments=email.attachments,
    )
    summary = EmailSummary(subject_es="Asunto traducido", summary_es="Resumen de prueba.")
    message_id = await bot.send_summary(email, summary)
    assert storage.get_meta(f"tg:{message_id}") == "t1"  # thread mapping intact
    text = sender.messages[0].text
    assert "Asunto traducido" in text
    assert "De: ana@example.com" in text
    assert "ana@example.com (ana@example.com)" not in text


async def test_send_notice_neutralizes_links(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.send_notice("Mira http://evil.example.com ahora")
    assert "hxxp://evil.example.com" in sender.messages[-1].text


async def test_send_typing_sends_chat_action(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.send_typing()
    assert sender.chat_actions == ["typing"]


async def test_send_draft_for_confirmation_posts_buttons(make_env: Any) -> None:
    bot, sender, storage = make_env()
    draft = _draft()
    message_id = await bot.send_draft_for_confirmation(draft, draft_id=1)
    assert message_id == 1
    text = sender.messages[0].text
    assert "Borrador (Respuesta)" in text  # type labeled; new emails say "Nuevo correo"
    assert "Para: Ana <ana@example.com>" in text
    assert "Re: Proyecto" in text
    assert draft.body in text
    markup = sender.messages[0].reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    assert _button_data(markup, 0, 0).startswith("confirm:")
    assert _button_data(markup, 0, 1).startswith("edit:")
    assert _button_data(markup, 0, 2).startswith("cancel:")
    buttons = markup.inline_keyboard[0]
    assert [b.text for b in buttons] == ["Enviar", "Editar", "Cancelar"]


def test_reply_requests_is_async_generator(make_env: Any) -> None:
    import inspect

    bot, sender, storage = make_env()
    assert inspect.isasyncgenfunction(bot.reply_requests)
    iterator = bot.reply_requests()
    assert hasattr(iterator, "__anext__")


# ── "Ver original" / "Ocultar original" ────────────────────────────────────


def _button_data(markup: InlineKeyboardMarkup, row: int, col: int) -> str:
    return markup.inline_keyboard[row][col].callback_data or ""


def _original_email(
    *,
    message_id: str = "gm-orig-1",
    subject: str = "Arbeitsplan für nächste Woche",
    body: str = "Hallo Daniel,\n\ndies ist der Originaltext.",
    attachments: list[Any] | None = None,
) -> ParsedEmail:
    return ParsedEmail(
        message_id=message_id,
        thread_id="t1",
        history_id=1,
        subject=subject,
        sender=EmailAddress("Daniel Arroyo", "darroyo083@gmail.com"),
        recipients=[EmailAddress("Ana", "ana@example.com")],
        date_iso="2026-08-07T10:00:00+00:00",
        body_text=body,
        attachments=attachments or [],
    )


def _view_summary_email(message_id: str = "gm-orig-1") -> ParsedEmail:
    """Summary-side email: in production the summarized ParsedEmail carries
    the SAME gmail message_id as the original (fetch_message returns it)."""
    return ParsedEmail(
        message_id=message_id,
        thread_id="t1",
        history_id=1,
        subject="Presupuesto",
        sender=EmailAddress("Ana", "ana@example.com"),
        recipients=[EmailAddress("Bob", "bob@example.com")],
        date_iso="2026-08-07T10:00:00+00:00",
        body_text="resumen body",
    )


def _summary_original_fetcher(email: ParsedEmail) -> Callable[..., Any]:
    async def fetcher(message_id: str) -> ParsedEmail:
        assert message_id == email.message_id
        return email

    return fetcher


async def test_send_summary_has_action_buttons(make_env: Any) -> None:
    bot, sender, storage = make_env()
    message_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="Asunto ES"))
    # Buttons attached via edit after the message_id is known.
    assert sender.edited[-1][0] == message_id
    markup = sender.edited[-1][2]
    assert isinstance(markup, InlineKeyboardMarkup)
    row = markup.inline_keyboard[0]
    assert [b.text for b in row] == ["Ver original", "Preguntar"]
    row2 = markup.inline_keyboard[1]
    assert [b.text for b in row2] == ["Responder", "📎 Adjuntos"]
    assert _button_data(markup, 0, 0) == f"view:{message_id}"
    assert _button_data(markup, 0, 1) == f"question:{message_id}"
    assert _button_data(markup, 1, 0) == f"reply:{message_id}"
    assert _button_data(markup, 1, 1) == f"att:{message_id}"
    # Mapping persisted: tg -> thread, tgm -> gmail message id, tgs -> sender.
    assert storage.get_meta(f"tg:{message_id}") == "t1"
    assert storage.get_meta(f"tgm:{message_id}") == "gm-orig-1"
    assert storage.get_meta(f"tgs:{message_id}") == "Ana (ana@example.com)"


async def test_view_original_fetches_gmail_on_demand_no_llm_no_persist(
    make_env: Any,
) -> None:
    original = _original_email()
    bot, sender, storage = make_env(original_fetcher=_summary_original_fetcher(original))
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="Asunto ES"))

    await bot.process_update(_callback_update(CHAT_ID, f"view:{summary_id}"))

    # Original shown as a NEW message, verbatim (no translation/summary).
    original_msg = next(
        m for m in sender.messages if m.message_id != summary_id
    )
    assert "Original" in original_msg.text
    assert "Asunto: Arbeitsplan für nächste Woche" in original_msg.text
    assert "dies ist der Originaltext" in original_msg.text
    # Not persisted: no meta key holds body content.
    assert storage.get_meta(f"tgm:{summary_id}") == original.message_id
    assert storage.get_meta(f"tg:{summary_id}") == "t1"
    # No LLM surface is involved at all (bot has no LLM dependency).


async def test_view_original_uses_correct_gmail_message(make_env: Any) -> None:
    """Pressing Ver original on summary 1 fetches gmail msg gm-orig-1."""
    fetched: list[str] = []

    async def fetcher(message_id: str) -> ParsedEmail:
        fetched.append(message_id)
        return _original_email(message_id=message_id)

    bot, sender, storage = make_env(original_fetcher=fetcher)
    await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))

    await bot.process_update(_callback_update(CHAT_ID, "view:1"))
    assert fetched == ["gm-orig-1"]


async def test_view_original_restart_safe_mapping(make_env: Any, tmp_path: Any) -> None:
    """The tgm mapping survives a Storage reconnect (same file)."""
    bot, sender, storage = make_env()
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))
    assert storage.get_meta(f"tgm:{summary_id}") == "gm-orig-1"

    storage.close()
    storage2 = Storage(tmp_path / "state.sqlite")
    storage2.connect()
    assert storage2.get_meta(f"tgm:{summary_id}") == "gm-orig-1"
    assert storage2.get_meta(f"tg:{summary_id}") == "t1"
    storage2.close()


async def test_view_original_neutralizes_urls(make_env: Any) -> None:
    original = _original_email(body="Ver esto: https://evil.example.com/x?track=1")
    bot, sender, storage = make_env(original_fetcher=_summary_original_fetcher(original))
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))

    await bot.process_update(_callback_update(CHAT_ID, f"view:{summary_id}"))

    original_msg = next(m for m in sender.messages if m.message_id != summary_id)
    assert "https://evil.example.com" not in original_msg.text
    assert "hxxps://evil.example.com" in original_msg.text


async def test_view_original_long_body_is_split_safely(make_env: Any) -> None:
    long_body = "\n".join(f"Línea de relleno número {i} con algo de contenido" for i in range(200))
    original = _original_email(body=long_body)
    bot, sender, storage = make_env(original_fetcher=_summary_original_fetcher(original))
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))

    await bot.process_update(_callback_update(CHAT_ID, f"view:{summary_id}"))

    original_msgs = [m for m in sender.messages if m.message_id != summary_id]
    assert len(original_msgs) > 1  # split into multiple messages
    assert all(len(m.text or "") <= 4096 for m in original_msgs)
    # Only the first chunk carries the hide button.
    assert original_msgs[0].reply_markup is not None
    assert all(m.reply_markup is None for m in original_msgs[1:])
    # All temporary ids recorded for hiding.
    raw = storage.get_meta(f"tgo:{summary_id}")
    assert raw is not None
    assert len(json.loads(raw)) == len(original_msgs)


async def test_view_original_shows_attachment_line(make_env: Any) -> None:
    from inboxbridge.models import AttachmentMeta

    original = _original_email(attachments=[AttachmentMeta("factura.pdf", "application/pdf", 100)])
    bot, sender, storage = make_env(original_fetcher=_summary_original_fetcher(original))
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))

    await bot.process_update(_callback_update(CHAT_ID, f"view:{summary_id}"))

    original_msg = next(m for m in sender.messages if m.message_id != summary_id)
    assert "Adjuntos: factura.pdf" in original_msg.text


async def test_hide_original_deletes_temporary_messages(make_env: Any) -> None:
    original = _original_email()
    bot, sender, storage = make_env(original_fetcher=_summary_original_fetcher(original))
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))

    await bot.process_update(_callback_update(CHAT_ID, f"view:{summary_id}"))
    original_msgs = [m for m in sender.messages if m.message_id != summary_id]
    temp_ids = [m.message_id for m in original_msgs]
    assert temp_ids

    await bot.process_update(_callback_update(CHAT_ID, f"hide:{summary_id}"))
    assert sender.deleted == temp_ids
    assert storage.get_meta(f"tgo:{summary_id}") is None
    assert "Original oculto." in sender.answered


async def test_view_original_fetch_failure_shows_natural_message(make_env: Any) -> None:
    async def broken_fetcher(message_id: str) -> ParsedEmail:
        raise RuntimeError("gmail api down")

    bot, sender, storage = make_env(original_fetcher=broken_fetcher)
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))

    await bot.process_update(_callback_update(CHAT_ID, f"view:{summary_id}"))

    texts = [m.text for m in sender.messages]
    assert any("No pude cargar el original ahora mismo." in t for t in texts)
    # No internal ids or error details leaked.
    assert not any("gm-orig" in t for t in texts)
    assert not any("RuntimeError" in t for t in texts)


async def test_view_original_unknown_mapping_is_ignored(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_callback_update(CHAT_ID, "view:999"))
    texts = [m.text for m in sender.messages]
    assert any("No pude cargar el original ahora mismo." in t for t in texts)


async def test_view_callback_unauthorized_chat_ignored(make_env: Any) -> None:
    bot, sender, storage = make_env(original_fetcher=_summary_original_fetcher(_original_email()))
    await bot.process_update(_callback_update(555, "view:1"))
    assert sender.messages == []
    assert sender.answered == []


async def test_draft_confirmation_callbacks_still_work_with_view_regex(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.send_draft_for_confirmation(_draft(), user_id=7, draft_id=1)
    markup = sender.messages[0].reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    token = _button_data(markup, 0, 0).split(":", 1)[1]
    task = asyncio.create_task(bot.wait_for_confirmation(1))
    await asyncio.sleep(0)
    await bot.process_update(_callback_update(CHAT_ID, f"confirm:{token}"))
    await bot.process_update(_callback_update(CHAT_ID, f"sendyes:{token}"))
    assert await task is True


# ── "Responder" button ─────────────────────────────────────────────────────


async def test_responder_button_asks_for_intent_and_uses_thread(make_env: Any) -> None:
    bot, sender, storage = make_env()
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))

    await bot.process_update(_callback_update(CHAT_ID, f"reply:{summary_id}"))

    texts = [m.text for m in sender.messages]
    assert any("¿Qué quieres decirle a Ana (ana@example.com)?" in t for t in texts)


async def test_responder_followup_message_creates_reply_request(make_env: Any) -> None:
    bot, sender, storage = make_env()
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))

    await bot.process_update(_callback_update(CHAT_ID, f"reply:{summary_id}"))
    # The member's next plain message (not a reply/mention) is the intent.
    message = _message(60, CHAT_ID, "Dile que sí puedo cubrir el viernes.", 7)
    await bot.process_update(_update(message))

    requests = await _drain(bot)
    assert len(requests) == 1
    assert requests[0].thread_id == "t1"
    assert requests[0].user_instructions == "Dile que sí puedo cubrir el viernes."
    assert requests[0].user_id == 7


async def test_responder_followup_isolated_between_users(make_env: Any) -> None:
    """User 7's pending reply is not consumed by user 8's message."""
    bot, sender, storage = make_env()
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))

    await bot.process_update(_callback_update(CHAT_ID, f"reply:{summary_id}"))
    # User 8 messages first: pending reply belongs to user 7, so nothing queued.
    message8 = _message(60, CHAT_ID, "Mensaje de otro miembro.", 8)
    await bot.process_update(_update(message8))
    assert await _drain(bot) == []
    # User 7 then sends their intent.
    message7 = _message(61, CHAT_ID, "Dile que voy el viernes.", 7)
    await bot.process_update(_update(message7))
    requests = await _drain(bot)
    assert len(requests) == 1
    assert requests[0].user_id == 7
    assert requests[0].user_instructions == "Dile que voy el viernes."


async def test_responder_callback_unauthorized_chat_ignored(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_callback_update(555, "reply:1"))
    assert sender.messages == []
    assert await _drain(bot) == []


async def test_responder_draft_owned_by_requesting_user(make_env: Any) -> None:
    """Another member pressing Enviar on someone else's draft is rejected."""
    bot, sender, storage = make_env()
    await bot.send_draft_for_confirmation(_draft(), user_id=7, draft_id=1)
    markup = sender.messages[0].reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    token = _button_data(markup, 0, 0).split(":", 1)[1]

    # User 8 tries to confirm user 7's draft (first tap: confirm dialog).
    update8 = _callback_update(CHAT_ID, f"confirm:{token}", from_user_id=8)
    await bot.process_update(update8)
    assert "otro miembro" in sender.answered[-1]

    # User 7 proceeds: confirm tap then "Sí, enviar".
    update7 = _callback_update(CHAT_ID, f"confirm:{token}")
    await bot.process_update(update7)
    task = asyncio.create_task(bot.wait_for_confirmation(1))
    await asyncio.sleep(0)
    await bot.process_update(_callback_update(CHAT_ID, f"sendyes:{token}"))
    assert await task is True


async def test_responder_flow_end_to_end_with_kill_switch(make_env: Any) -> None:
    """Full loop contract: button -> intent -> ReplyRequest with thread +
    instructions + owner; the draft confirmation shows Enviar/Editar/Cancelar
    (actual sending is guarded by SEND_EMAILS=false in GmailClient)."""
    bot, sender, storage = make_env()
    summary_id = await bot.send_summary(_view_summary_email(), EmailSummary(subject_es="A"))
    await bot.process_update(_callback_update(CHAT_ID, f"reply:{summary_id}"))
    message = _message(60, CHAT_ID, "Dile que confirmo el viernes.", 7)
    await bot.process_update(_update(message))

    requests = await _drain(bot)
    assert len(requests) == 1
    assert requests[0].thread_id == "t1"
    assert requests[0].user_instructions == "Dile que confirmo el viernes."
    assert requests[0].user_id == 7

    # Draft confirmation message shows recipients and Enviar/Editar/Cancelar
    # (responder.py calls send_draft_for_confirmation with the owner user_id).
    await bot.send_draft_for_confirmation(_draft(), user_id=7, draft_id=1)
    draft_msg = sender.messages[-1]
    assert "Para: Ana <ana@example.com>" in (draft_msg.text or "")
    markup = draft_msg.reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    assert [b.text for b in markup.inline_keyboard[0]] == ["Enviar", "Editar", "Cancelar"]
    token = _button_data(markup, 0, 2).split(":", 1)[1]
    # Cancel requires the second tap.
    task = asyncio.create_task(bot.wait_for_confirmation(1))
    await asyncio.sleep(0)
    await bot.process_update(_callback_update(CHAT_ID, f"cancel:{token}", from_user_id=7))
    assert not task.done()
    await bot.process_update(_callback_update(CHAT_ID, f"cancelyes:{token}", from_user_id=7))
    assert await task is False
    assert "Cancelado." in sender.answered


# ── outgoing attachments (Telegram documents/photos) ─────────────────────


def _document_message(
    message_id: int,
    *,
    file_id: str = "file-1",
    file_name: str = "factura.pdf",
    mime_type: str = "application/pdf",
    file_size: int = 1024,
    caption: str | None = None,
    chat_id: int = CHAT_ID,
    user_id: int = 7,
    reply_to: Message | None = None,
) -> Message:
    doc = Document(
        file_id=file_id,
        file_unique_id=f"uniq-{file_id}",
        file_name=file_name,
        mime_type=mime_type,
        file_size=file_size,
    )
    return Message(
        message_id=message_id,
        date=datetime.now(UTC),
        chat=Chat(id=chat_id, type=ChatType.GROUP),
        from_user=_user(user_id),
        document=doc,
        caption=caption,
        reply_to_message=reply_to,
    )


def _photo_message(
    message_id: int,
    *,
    chat_id: int = CHAT_ID,
    user_id: int = 7,
    reply_to: Message | None = None,
) -> Message:
    photo = PhotoSize(
        file_id="photo-1",
        file_unique_id="uniq-photo",
        width=100,
        height=100,
        file_size=2048,
    )
    return Message(
        message_id=message_id,
        date=datetime.now(UTC),
        chat=Chat(id=chat_id, type=ChatType.GROUP),
        from_user=_user(user_id),
        photo=[photo],
        reply_to_message=reply_to,
    )


async def test_document_attachment_downloads_and_rides_reply_request(
    make_env: Any, tmp_path: Path
) -> None:
    bot, sender, storage = make_env()
    bot._settings.tmp_dir = str(tmp_path)  # isolate temp storage
    sender.files["file-1"] = FakeFile(b"%PDF-1.4 fake")
    storage.set_meta("tg:10", "thread-1")
    bot_message = _message(10, CHAT_ID, "Presupuesto", BOT_ID, is_bot=True)
    message = _document_message(11, reply_to=bot_message, caption="Adjunto el contrato")
    await bot.process_update(_update(message))

    requests = await _drain(bot)
    assert len(requests) == 1
    assert requests[0].user_instructions == "Adjunto el contrato"
    assert len(requests[0].attachments) == 1
    attachment = requests[0].attachments[0]
    assert attachment.filename == "factura.pdf"
    assert attachment.mime_type == "application/pdf"
    assert Path(attachment.path).is_file()
    assert Path(attachment.path).read_bytes() == b"%PDF-1.4 fake"


async def test_photo_attachment_is_supported(make_env: Any, tmp_path: Path) -> None:
    bot, sender, storage = make_env()
    bot._settings.tmp_dir = str(tmp_path)
    sender.files["photo-1"] = FakeFile(b"jpeg-bytes")
    storage.set_meta("tg:10", "thread-1")
    bot_message = _message(10, CHAT_ID, "Presupuesto", BOT_ID, is_bot=True)
    message = _photo_message(11, reply_to=bot_message)
    await bot.process_update(_update(message))

    requests = await _drain(bot)
    assert len(requests) == 1
    assert len(requests[0].attachments) == 1
    attachment = requests[0].attachments[0]
    assert attachment.filename.endswith(".jpg")
    assert attachment.mime_type == "image/jpeg"


async def test_path_traversal_filename_is_sanitized(make_env: Any, tmp_path: Path) -> None:
    bot, sender, storage = make_env()
    bot._settings.tmp_dir = str(tmp_path)
    sender.files["file-1"] = FakeFile(b"data")
    storage.set_meta("tg:10", "thread-1")
    bot_message = _message(10, CHAT_ID, "Presupuesto", BOT_ID, is_bot=True)
    message = _document_message(11, file_name="../../etc/evil.pdf", reply_to=bot_message)
    await bot.process_update(_update(message))

    requests = await _drain(bot)
    attachment = requests[0].attachments[0]
    assert attachment.filename == "evil.pdf"  # display name sanitized
    assert ".." not in attachment.path  # temp path stays inside tmp_dir


async def test_oversized_attachment_rejected_with_notice(make_env: Any, tmp_path: Path) -> None:
    bot, sender, storage = make_env()
    bot._settings.tmp_dir = str(tmp_path)
    bot._settings.outgoing_attachment_max_bytes = 10
    sender.files["file-1"] = FakeFile(b"x" * 100)
    storage.set_meta("tg:10", "thread-1")
    bot_message = _message(10, CHAT_ID, "Presupuesto", BOT_ID, is_bot=True)
    message = _document_message(11, reply_to=bot_message)
    await bot.process_update(_update(message))

    requests = await _drain(bot)
    assert requests[0].attachments == ()  # rejected
    assert "grande" in (sender.messages[-1].text or "")  # notice posted


async def test_too_many_attachments_rejected(make_env: Any, tmp_path: Path) -> None:
    bot, sender, storage = make_env()
    bot._settings.tmp_dir = str(tmp_path)
    bot._settings.outgoing_attachment_max_count = 1
    sender.files["file-1"] = FakeFile(b"a")
    sender.files["photo-1"] = FakeFile(b"b")
    storage.set_meta("tg:10", "thread-1")
    bot_message = _message(10, CHAT_ID, "Presupuesto", BOT_ID, is_bot=True)
    # ONE message carrying both a document and a photo → 2 files > max 1.
    doc = Document(
        file_id="file-1", file_unique_id="uniq-file", file_name="a.pdf",
        mime_type="application/pdf", file_size=5,
    )
    photo = PhotoSize(
        file_id="photo-1", file_unique_id="uniq-photo", width=10, height=10, file_size=5
    )
    message = Message(
        message_id=11,
        date=datetime.now(UTC),
        chat=Chat(id=CHAT_ID, type=ChatType.GROUP),
        from_user=_user(7),
        document=doc,
        photo=[photo],
        caption="dos adjuntos",
        reply_to_message=bot_message,
    )
    await bot.process_update(_update(message))

    requests = await _drain(bot)
    assert len(requests) == 1
    assert requests[0].attachments == ()  # batch rejected
    assert any("Demasiados adjuntos" in (m.text or "") for m in sender.messages)


async def test_attachment_without_reply_intent_is_ignored(make_env: Any, tmp_path: Path) -> None:
    bot, sender, storage = make_env()
    bot._settings.tmp_dir = str(tmp_path)
    sender.files["file-1"] = FakeFile(b"data")
    message = _document_message(11)  # no reply-to-own, no mention, no pending
    await bot.process_update(_update(message))
    assert await _drain(bot) == []
    assert sender.downloaded == []


async def test_draft_confirmation_message_lists_attachments(make_env: Any) -> None:
    bot, sender, storage = make_env()
    draft = DraftReply(
        thread_id="t1",
        subject="Re: Proyecto",
        to=[EmailAddress("Ana", "ana@example.com")],
        cc=[],
        body="Sehr geehrte Frau Ana,\n\ndanke.",
        attachments=(
            OutgoingAttachment(
                filename="factura.pdf",
                mime_type="application/pdf",
                size_bytes=2048,
                path="C:/tmp/factura.pdf",
            ),
        ),
    )
    await bot.send_draft_for_confirmation(draft, draft_id=1)
    text = sender.messages[-1].text or ""
    assert "Adjuntos:" in text
    assert "factura.pdf" in text


# ── "Reintentar envío" (resend) button ────────────────────────────────────


async def test_resend_callback_invokes_coordinator_with_owner_validation(
    make_env: Any, tmp_path: Path
) -> None:
    bot, sender, storage = make_env()
    called: list[int] = []

    async def resend(draft_id: int) -> None:
        called.append(draft_id)

    bot.register_resend_callback(resend)
    draft = DraftReply(
        thread_id="t1",
        subject="Re: Proyecto",
        to=[EmailAddress("Ana", "ana@example.com")],
        cc=[],
        body="danke",
    )
    draft_id = storage.create_draft("t1", None, draft, telegram_user_id=7)

    # Unknown draft id → ignored.
    await bot.process_update(_callback_update(CHAT_ID, "resend:999", from_user_id=7))
    assert called == []

    # Wrong member → rejected without invoking the coordinator.
    await bot.process_update(_callback_update(CHAT_ID, f"resend:{draft_id}", from_user_id=8))
    assert called == []
    assert "otro miembro" in sender.answered[-1]

    # Owner → coordinator invoked.
    await bot.process_update(_callback_update(CHAT_ID, f"resend:{draft_id}", from_user_id=7))
    assert called == [draft_id]


async def test_resend_callback_ignored_from_unauthorized_chat(make_env: Any) -> None:
    bot, sender, storage = make_env()
    called: list[int] = []

    async def resend(draft_id: int) -> None:
        called.append(draft_id)

    bot.register_resend_callback(resend)
    draft = DraftReply(
        thread_id="t1",
        subject="Re: Proyecto",
        to=[EmailAddress("Ana", "ana@example.com")],
        cc=[],
        body="danke",
    )
    draft_id = storage.create_draft("t1", None, draft)
    await bot.process_update(_callback_update(-777, f"resend:{draft_id}", from_user_id=7))
    assert called == []


async def test_double_confirm_click_resolves_once(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.send_draft_for_confirmation(_draft(), user_id=7, draft_id=1)
    markup = sender.messages[-1].reply_markup
    token = _button_data(markup, 0, 0).split(":", 1)[1]

    task = asyncio.create_task(bot.wait_for_confirmation(1))
    await asyncio.sleep(0)
    # Two "Sí, enviar" taps: the first resolves and pops; the second is
    # refused as unavailable. Exactly one decision.
    await bot.process_update(_callback_update(CHAT_ID, f"sendyes:{token}", from_user_id=7))
    await bot.process_update(_callback_update(CHAT_ID, f"sendyes:{token}", from_user_id=7))
    result = await asyncio.wait_for(task, timeout=2)
    assert result is True  # exactly one decision; no double-send downstream
