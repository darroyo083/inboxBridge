"""Telegram bot tests: chat authorization, triggers, commands, draft flow.

No network: a FakeSender stands in for PTB's Bot and updates are built as
plain objects. The real ``db.Storage`` is used (tmp SQLite file).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from telegram import (
    CallbackQuery,
    Chat,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
    MessageEntity,
    ReplyParameters,
    Update,
    User,
)
from telegram.constants import ChatAction, ChatType, ParseMode

from inboxbridge.config import Settings
from inboxbridge.db import Storage
from inboxbridge.models import DraftReply, EmailAddress, ParsedEmail
from inboxbridge.telegram.bot import TelegramBot

CHAT_ID = -100123456789
BOT_ID = 42
BOT_USERNAME = "inboxbridge_bot"


class FakeSender:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.chat_actions: list[Any] = []
        self.edited: list[tuple[int | None, str]] = []
        self.answered: list[str] = []
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
        self.edited.append((message_id, text))
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
    chat_id: int, data: str, *, callback_id: str = "cq-1", message_id: int = 50
) -> Update:
    message = _message(message_id, chat_id, "Borrador", 7)
    query = CallbackQuery(
        id=callback_id,
        from_user=_user(7),
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
    draft_message_id = await bot.send_draft_for_confirmation(_draft())
    task = asyncio.create_task(bot.wait_for_confirmation(draft_message_id))
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
    draft_message_id = await bot.send_draft_for_confirmation(_draft())
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


def _button_data(markup: InlineKeyboardMarkup, row: int, col: int) -> str:
    """Callback data of a button (PTB types the attribute as object)."""
    return cast(str, markup.inline_keyboard[row][col].callback_data or "")


# ── draft confirmation buttons ────────────────────────────────────────────


async def test_confirm_button_resolves_draft(make_env: Any) -> None:
    bot, sender, storage = make_env()
    draft_message_id = await bot.send_draft_for_confirmation(_draft())
    markup = sender.messages[-1].reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    token = _button_data(markup, 0, 0).split(":", 1)[1]
    task = asyncio.create_task(bot.wait_for_confirmation(draft_message_id))
    await asyncio.sleep(0)
    await bot.process_update(_callback_update(CHAT_ID, f"confirm:{token}"))
    assert await task is True
    assert sender.answered[-1] == "Enviando…"
    assert sender.edited[-1][1] == "Confirmado ✓"


async def test_cancel_button_resolves_draft_false(make_env: Any) -> None:
    bot, sender, storage = make_env()
    draft_message_id = await bot.send_draft_for_confirmation(_draft())
    markup = sender.messages[-1].reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    token = _button_data(markup, 0, 1).split(":", 1)[1]
    task = asyncio.create_task(bot.wait_for_confirmation(draft_message_id))
    await asyncio.sleep(0)
    await bot.process_update(_callback_update(CHAT_ID, f"cancel:{token}"))
    assert await task is False


async def test_stale_callback_is_answered_and_ignored(make_env: Any) -> None:
    bot, sender, storage = make_env()
    await bot.process_update(_callback_update(CHAT_ID, "confirm:unknown-token"))
    assert "ya no está disponible" in sender.answered[-1]


async def test_wait_for_confirmation_unknown_message_returns_false(make_env: Any) -> None:
    bot, sender, storage = make_env()
    assert await bot.wait_for_confirmation(99999) is False


async def test_wait_for_confirmation_timeout(make_env: Any) -> None:
    bot, sender, storage = make_env()
    draft_message_id = await bot.send_draft_for_confirmation(_draft())
    assert await bot.wait_for_confirmation(draft_message_id, timeout_seconds=0.05) is False


# ── notifier methods ──────────────────────────────────────────────────────


async def test_send_summary_stores_thread_mapping(make_env: Any) -> None:
    bot, sender, storage = make_env()
    message_id = await bot.send_summary(_email(), "Resumen con enlace https://evil.example.com/x")
    assert message_id == 1
    assert storage.get_meta(f"tg:{message_id}") == "t1"
    text = sender.messages[0].text
    assert "Presupuesto" in text
    assert "Resumen con enlace hxxps://evil.example.com/x" in text


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
    message_id = await bot.send_draft_for_confirmation(draft)
    assert message_id == 1
    text = sender.messages[0].text
    assert "Borrador de respuesta" in text
    assert "Para: Ana <ana@example.com>" in text
    assert "Re: Proyecto" in text
    assert draft.body in text
    markup = sender.messages[0].reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    assert _button_data(markup, 0, 0).startswith("confirm:")
    assert _button_data(markup, 0, 1).startswith("cancel:")
    buttons = markup.inline_keyboard[0]
    assert buttons[0].text == "Confirmar"
    assert buttons[1].text == "Cancelar"


def test_reply_requests_is_async_generator(make_env: Any) -> None:
    import inspect

    bot, sender, storage = make_env()
    assert inspect.isasyncgenfunction(bot.reply_requests)
    iterator = bot.reply_requests()
    assert hasattr(iterator, "__anext__")
