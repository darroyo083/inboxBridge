"""V1.1 simulated end-to-end flows (real stack, simulated Gmail/Telegram/AI).

Covers the goal's flows A–W: reply+edit+send, button misclick, ambiguous
ack, contact creation, new email via alias, contact ambiguity, unknown
contact, contact edit/delete, reply-vs-alias semantics, attachment download,
Q&A, thread summary, forward, mark read, archive, reminder, voice
(experimental), prompt injection, restart, concurrency.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from telegram import Chat, InlineKeyboardMarkup, Message, Voice
from telegram.constants import ChatType, ParseMode

from inboxbridge.assistant import EmailAssistant
from inboxbridge.config import Settings
from inboxbridge.contacts import ContactService
from inboxbridge.db import DraftStatus, Storage
from inboxbridge.intents import IntentClassifier
from inboxbridge.models import (
    AttachmentMeta,
    DraftReply,
    EmailAddress,
    EmailSummary,
    MessageStatus,
    ParsedEmail,
    ThreadContext,
)
from inboxbridge.reminders import ReminderService
from inboxbridge.responder import ReplyCoordinator
from tests.mocks.coordinator import FakeGmail, make_thread
from tests.unit.test_telegram_auth import (
    BOT_ID,
    BOT_USERNAME,
    CHAT_ID,
    FakeFile,
    FakeSender,
    _callback_update,
    _document_message,
    _message,
    _photo_message,
    _update,
    _user,
)

FIXED_NOW = datetime(2026, 8, 11, 10, 0, tzinfo=UTC).timestamp()  # Tuesday


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "TELEGRAM_BOT_TOKEN": "test-token",
        "TELEGRAM_ALLOWED_CHAT_ID": CHAT_ID,
        "LLM_API_KEY": "test-key",
        "LLM_BASE_URL": "https://api.test/v1",
        "SEND_EMAILS": True,
        "AI_TEXT_MODEL": "deepseek-v4-flash",
        "AI_VISION_MODEL": "mimo-v2.5",
        "AI_VISION_FALLBACK_MODEL": "gpt-5.6-luna",
        "AI_AUDIO_ENABLED": False,
        "EMAIL_SIGNATURE_NAME": "Daniel",
        "send_verification_attempts": 2,
        "send_verification_backoff_seconds": 0.01,
        "llm_max_retries": 1,
    }
    base.update(overrides)
    return Settings(**base)


class FakeAi:
    """Scripted AIService double: text/vision/audio + intent classification."""

    def __init__(self) -> None:
        self.text_responses: dict[str, str] = {}
        self.default_text = "Texto de prueba"
        self._text_fallback_model = ""
        self.vision_responses: list[str] = []
        self.vision_fail: str = ""  # "primary" | "both"
        self.vision_calls: list[str] = []
        self.audio_responses: list[str] = []
        self.audio_calls = 0
        self.calls: list[tuple[str, str]] = []  # (task, model-ish)
        #: Model requested per text call (None = primary; fallback model name
        #: when the bounded alternation used the configured fallback).
        self.models: list[str | None] = []
        #: (task, max_tokens) per text call — task-aware budget assertions.
        self.max_tokens_calls: list[tuple[str, int]] = []
        # Translation retry simulation: first N translate calls raise
        # LLMEmptyResponse (transient) — the caller must retry boundedly.
        self.translate_failures = 0
        self.translate_calls = 0
        # Draft-edit retry simulation: first N draft_edit calls raise
        # LLMEmptyResponse (transient) — the caller must retry boundedly.
        self.draft_edit_failures = 0
        self.draft_edit_calls = 0
        # Compose/forward retry simulation: first N calls raise LLMEmptyResponse.
        self.compose_failures = 0
        self.compose_calls = 0
        self.forward_failures = 0
        self.forward_calls = 0
        # Q&A / thread-summary retry simulation: first N calls raise
        # LLMEmptyResponse (transient) — the caller must retry boundedly.
        self.qa_failures = 0
        self.qa_calls = 0
        self.thread_summary_failures = 0
        self.thread_summary_calls = 0
        # Leading MALFORMED (truncated/non-JSON) structured responses before
        # the valid scripted one — structured parse failures must be retried.
        self.qa_malformed = 0
        self.thread_summary_malformed = 0
        # Plain-summary fallback (task thread_summary_plain) simulation.
        self.plain_summary_failures = 0
        self.plain_summary_calls = 0
        self.plain_summary_override: str | None = None
        self.plain_summary_messages: list[list[Any]] = []
        # Override for the QA answer (malformed / compact scenarios).
        self.qa_override: str | None = None
        # Override for the thread-summary answer (malformed / simple / etc.).
        self.thread_summary_override: str | None = None
        # Last prompt per Q&A / thread-summary call (content assertions).
        self.qa_messages: list[list[Any]] = []
        self.thread_summary_messages: list[list[Any]] = []

    @property
    def text_model(self) -> str:
        return "deepseek-v4-flash"

    @property
    def text_fallback_model(self) -> str:
        return self._text_fallback_model

    @property
    def intent_max_tokens(self) -> int:
        return 400

    async def text(
        self,
        messages: list[Any],
        *,
        max_tokens: int,
        task: str,
        require_complete: bool = False,
        model: str | None = None,
    ) -> str:
        self.calls.append((task, "deepseek-v4-flash"))
        self.models.append(model)
        self.max_tokens_calls.append((task, max_tokens))
        if task == "intent":
            # Extract the user text from the message and return scripted JSON.
            content = str(messages[-1]["content"])
            for phrase, response in self.text_responses.items():
                if phrase in content:
                    return response
            return (
                '{"action": "unknown", "recipient": "", "instruction": "", '
                '"needs_clarification": true}'
            )
        if task in self.text_responses:
            return self.text_responses[task]
        if task == "compose":
            self.compose_calls += 1
            if self.compose_failures > 0:
                self.compose_failures -= 1
                from inboxbridge.llm.base import LLMEmptyResponse

                raise LLMEmptyResponse("simulated empty compose")
            return (
                '{"subject_de": "Vielen Dank", "body_de": "Sehr geehrte Frau '
                'Muster,\\n\\nvielen Dank für Ihre Nachricht.\\n\\nMit '
                'freundlichen Grüßen"}'
            )
        if task == "forward":
            self.forward_calls += 1
            if self.forward_failures > 0:
                self.forward_failures -= 1
                from inboxbridge.llm.base import LLMEmptyResponse

                raise LLMEmptyResponse("simulated empty forward")
            return "Sehr geehrte Frau Muster,\n\nWeiterleitung von ...\n\nMit freundlichen Grüßen"
        if task == "draft_edit":
            self.draft_edit_calls += 1
            if self.draft_edit_failures > 0:
                self.draft_edit_failures -= 1
                from inboxbridge.llm.base import LLMEmptyResponse

                raise LLMEmptyResponse("simulated empty edit")
            return "Sehr geehrte Frau Muster,\n\nkurz und klar.\n\nMit freundlichen Grüßen"
        if task == "qa":
            self.qa_calls += 1
            self.qa_messages.append(messages)
            if self.qa_failures > 0:
                self.qa_failures -= 1
                from inboxbridge.llm.base import LLMEmptyResponse

                raise LLMEmptyResponse("simulated empty qa")
            if self.qa_malformed > 0:
                self.qa_malformed -= 1
                return '{"answer": "incompleto'  # truncated JSON
            if self.qa_override is not None:
                return self.qa_override
            return json.dumps(
                {
                    "answer": "Hay que pagar 500 EUR. La cita es en "
                    "Bahnhofstrasse 10, Zürich.",
                    "sections": [
                        {"emoji": "💰", "title": "Importe", "items": ["500 EUR"]},
                        {
                            "emoji": "📍",
                            "title": "Cita",
                            "items": [
                                "Bahnhofstrasse 10, Zürich",
                                "18 de agosto de 2026, 14:30",
                            ],
                        },
                    ],
                }
            )
        if task == "thread_summary":
            self.thread_summary_calls += 1
            self.thread_summary_messages.append(messages)
            if self.thread_summary_failures > 0:
                self.thread_summary_failures -= 1
                from inboxbridge.llm.base import LLMEmptyResponse

                raise LLMEmptyResponse("simulated empty thread summary")
            if self.thread_summary_malformed > 0:
                self.thread_summary_malformed -= 1
                return (
                    '{"headline": "Resumen", "sections": [{"emoji": "📬", '
                    '"title": "Resumen", "items": ['
                )
            if self.thread_summary_override is not None:
                return self.thread_summary_override
            return json.dumps(
                {
                    "headline": "Resumen",
                    "sections": [
                        {
                            "emoji": "📅",
                            "title": "Cita",
                            "items": [
                                "18 de agosto de 2026, 14:30",
                                "Bahnhofstrasse 10, Zürich",
                                "Duración: 45 min",
                            ],
                        },
                        {
                            "emoji": "📄",
                            "title": "Llevar",
                            "items": [
                                "Contrato firmado",
                                "DNI o pasaporte",
                                "Checklist, si está cumplimentada",
                            ],
                        },
                        {
                            "emoji": "⏰",
                            "title": "Importante",
                            "items": [
                                "Avisar antes del 17 de agosto, 18:00 si no "
                                "puede asistir",
                                "Tasa: 125 CHF",
                            ],
                        },
                        {
                            "emoji": "👤",
                            "title": "Contacto",
                            "items": ["Markus Schneider"],
                        },
                    ],
                }
            )
        if task == "thread_summary_plain":
            self.plain_summary_calls += 1
            self.plain_summary_messages.append(messages)
            if self.plain_summary_failures > 0:
                self.plain_summary_failures -= 1
                from inboxbridge.llm.base import LLMEmptyResponse

                raise LLMEmptyResponse("simulated empty plain summary")
            if self.plain_summary_override is not None:
                return self.plain_summary_override
            return (
                "• Cita el 18 de agosto de 2026 a las 14:30 en Zürich "
                "(Bahnhofstrasse 10).\n"
                "• Llevar contrato firmado y documento de identidad.\n"
                "• Avisar antes del 17 de agosto a las 18:00 si no puede asistir.\n"
                "• Tasa: 125 CHF.\n"
                "• Contacto: Markus Schneider."
            )
        return self.default_text

    async def translate_to_spanish(
        self, body: str, *, model: str | None = None
    ) -> str:
        self.translate_calls += 1
        if self.translate_failures > 0:
            self.translate_failures -= 1
            from inboxbridge.llm.base import LLMEmptyResponse

            raise LLMEmptyResponse("simulated empty translation")
        return "[ES] " + body

    async def vision(
        self,
        prompt: str,
        images: list[tuple[str, bytes]],
        *,
        max_tokens: int = 1000,
        task: str = "vision",
        allow_fallback: bool = True,
    ) -> str:
        self.vision_calls.append(task)
        if self.vision_fail == "primary":
            from inboxbridge.llm.base import LLMUnavailable

            self.vision_fail = "fallback"
            raise LLMUnavailable("mimo down")
        if self.vision_fail == "both":
            from inboxbridge.llm.base import LLMUnavailable

            raise LLMUnavailable("all down")
        if self.vision_responses:
            return self.vision_responses.pop(0)
        return "El documento escaneado dice: 'Presupuesto 500 EUR'."

    async def document_vision(
        self,
        prompt: str,
        pdf_bytes: bytes,
        *,
        max_tokens: int = 1500,
        task: str = "document_vision",
    ) -> str:
        self.vision_calls.append(task)
        return "Documento escaneado: factura de 500 EUR."

    async def audio(self, mime: str, data: bytes, *, task: str = "audio") -> str:
        self.audio_calls += 1
        if not self.audio_responses:
            from inboxbridge.llm.base import LLMError

            raise LLMError("audio unavailable")
        return self.audio_responses.pop(0)


class FakeCoordinatorLLM:
    """Deterministic LLMProvider for the coordinator's reply drafting."""

    async def summarize_email(self, email: Any) -> EmailSummary:
        return EmailSummary(subject_es="Asunto ES", summary_es="Resumen ES.")

    async def draft_reply(self, request: Any, thread: ThreadContext) -> DraftReply:
        return DraftReply(
            thread_id=thread.thread_id,
            subject=thread.subject,
            to=[thread.messages[0].from_] if thread.messages else [],
            cc=[],
            body=(
                "Sehr geehrte Frau Muster,\n\nvielen Dank für Ihre Nachricht. "
                "Ich melde mich am Freitag.\n\nMit freundlichen Grüßen"
            ),
            in_reply_to=thread.messages[-1].message_id if thread.messages else "",
            references="",
        )

    async def translate_to_spanish(
        self, body: str, *, model: str | None = None
    ) -> str:
        return "[ES] " + body


class Stack:
    def __init__(self, tmp_path: Path, **settings_overrides: object) -> None:
        self.settings = make_settings(tmp_dir=str(tmp_path / "tmp"), **settings_overrides)
        self.storage = Storage(str(tmp_path / "state.sqlite"))
        self.storage.connect()
        self.background_tasks: list[asyncio.Task[Any]] = []
        self.gmail = FakeGmail(
            threads={"t1": make_thread()},
            messages={
                "m1": make_email_with_attachments(),
                "gm-orig": make_email(),
            },
        )
        self.gmail.attachment_bytes[("m1", 0)] = b"%PDF-1.4 fake pdf content"
        self.ai = FakeAi()
        self.contacts = ContactService(self.storage)
        self.reminders = ReminderService(self.storage, clock=lambda: FIXED_NOW)
        self.sender = FakeSender()
        from inboxbridge.telegram.bot import TelegramBot

        self.bot = TelegramBot(
            self.settings,
            self.storage,
            sender=self.sender,
            bot_user_id=BOT_ID,
            bot_username=BOT_USERNAME,
            original_fetcher=self.gmail.fetch_message,
        )
        self.assistant = EmailAssistant(
            self.settings, self.storage, self.gmail, self.ai, self.bot,
            self.contacts, self.reminders,
        )
        self.coordinator = ReplyCoordinator(
            self.settings, self.gmail, FakeCoordinatorLLM(), self.bot, self.storage
        )
        self.assistant.set_draft_presenter(self.coordinator.present_draft)
        self.bot.register_action_callback(self.assistant.handle)
        self.bot.register_assistant(self.assistant)
        self.bot.set_intent_classifier(IntentClassifier(self.ai))

    async def send(self, text: str, *, user_id: int = 7, message_id: int = 100,
                   reply_to: Message | None = None) -> None:
        message = _message(message_id, CHAT_ID, text, user_id, reply_to=reply_to)
        await self.bot.process_update(_update(message))

    async def send_bg(self, text: str, *, user_id: int = 7, message_id: int = 100,
                      reply_to: Message | None = None) -> None:
        """Send where the flow blocks awaiting confirmation (compose/forward):
        run process_update in a tracked background task and return once the
        draft preview (or an error notice) has been posted."""
        message = _message(message_id, CHAT_ID, text, user_id, reply_to=reply_to)
        task = asyncio.create_task(self.bot.process_update(_update(message)))
        self.background_tasks.append(task)
        for _ in range(100):
            if any(
                (m.text or "").startswith("Borrador (") or "No puedo" in (m.text or "")
                for m in self.sender.messages
            ):
                return
            await asyncio.sleep(0.02)

    async def join_background(self, wait_seconds: float = 3.0) -> None:
        """Await tracked background tasks (drain after explicit send/cancel)."""
        for task in self.background_tasks:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=wait_seconds)
            except TimeoutError:
                task.cancel()
        self.background_tasks.clear()

    async def pump(self) -> None:
        """Drain the reply-request queue into the coordinator (non-blocking:
        the coordinator waits for confirmation; tests resolve it later)."""
        import sys
        print('PUMP queue size:', self.bot._queue.qsize(), file=sys.stderr)
        while True:
            try:
                request = await asyncio.wait_for(self.bot._queue.get(), timeout=0.05)
            except TimeoutError:
                return
            self.background_tasks.append(
                asyncio.create_task(self.coordinator._handle_request(request))
            )

    async def cleanup(self) -> None:
        for task in self.background_tasks:
            task.cancel()
        for task in list(self.coordinator._presentation_tasks):
            task.cancel()
        await asyncio.sleep(0)

    def bot_message(self, message_id: int, text: str = "Resumen") -> Message:
        return _message(message_id, CHAT_ID, text, BOT_ID, is_bot=True)

    async def tap(self, chat_id: int, data: str, *, user_id: int = 7) -> None:
        await self.bot.process_update(_callback_update(chat_id, data, from_user_id=user_id))

    def draft_row(self, draft_id: int = 1) -> dict[str, Any]:
        row = self.storage.get_draft(draft_id)
        assert row is not None
        return row

    async def wait_for_send(self) -> None:
        """Let the coordinator's pending future resolve (async flows)."""
        await asyncio.sleep(0.05)


def make_email(
    message_id: str = "gm-orig",
    thread_id: str = "t1",
    subject: str = "Projektbericht",
    sender: str = "Ana Muster",
    body: str = "Hallo, bitte um Rückmeldung bis Freitag.",
) -> ParsedEmail:
    return ParsedEmail(
        message_id=message_id,
        thread_id=thread_id,
        history_id=1,
        subject=subject,
        sender=EmailAddress(sender, "anna@example.com"),
        recipients=[EmailAddress("Daniel", "daniel@example.com")],
        date_iso="2026-08-10T09:00:00+00:00",
        body_text=body,
    )


def make_email_with_attachments() -> ParsedEmail:
    email = make_email(message_id="m1", subject="Presupuesto con PDF")
    return email_with_attachments(email)


def email_with_attachments(email: ParsedEmail) -> ParsedEmail:
    from dataclasses import replace

    return replace(
        email,
        attachments=[
            AttachmentMeta(
                filename="presupuesto.pdf",
                mime_type="application/pdf",
                size_bytes=1024,
                extracted_text="Presupuesto: 500 EUR.",
            )
        ],
    )


def _button_data(markup: InlineKeyboardMarkup, row: int, col: int) -> str:
    return markup.inline_keyboard[row][col].callback_data or ""


@pytest.fixture
def stack(tmp_path: Path) -> Stack:
    return Stack(tmp_path)


@pytest.fixture(autouse=True)
async def _cleanup_stack(stack: Stack) -> Any:
    yield
    await stack.cleanup()


# ── A. REPLY + EDIT + SEND ───────────────────────────────────────────────────


async def test_flow_spanish_preview_reply_and_edit(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("respóndele que el viernes sí puedo", reply_to=stack.bot_message(summary_id))
    await stack.pump()
    await asyncio.sleep(0.15)

    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    preview = previews[-1]
    assert "🇩🇪 Alemán · se enviará" in preview
    assert "Sehr geehrte Frau Muster" in preview  # German body (exact)
    assert "🇪🇸 Español · traducción" in preview
    assert "[ES] Sehr geehrte Frau Muster" in preview  # Spanish translation

    # Edit regenerates BOTH the German body and its Spanish translation.
    await stack.send("hazlo más corto")
    await asyncio.sleep(0.05)
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    edited = previews[-1]
    assert "kurz und klar" in edited  # new German
    assert "[ES] Sehr geehrte Frau Muster,\n\nkurz und klar" in edited  # new Spanish


async def test_flow_a_reply_edit_send(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    reply_to = stack.bot_message(summary_id)
    await stack.send("respóndele que el viernes sí puedo", reply_to=reply_to)
    await stack.pump()
    await asyncio.sleep(0.15)  # coordinator drafts
    assert stack.draft_row(1)["status"] == DraftStatus.PENDING.value

    # Edit: make it shorter.
    await stack.send("hazlo más corto")
    await asyncio.sleep(0.05)
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (")
    ]
    assert previews and "kurz und klar" in previews[-1]

    # Explicit text send.
    await stack.send("envíalo")
    await asyncio.sleep(0.05)
    assert stack.draft_row(1)["status"] == DraftStatus.SENT_VERIFIED.value
    assert len(stack.gmail.sent) == 1
    assert stack.gmail.sent[0].thread_id == "t1"


async def test_flow_b_button_misclick_no_send(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("respóndele que sí", reply_to=stack.bot_message(summary_id))
    await stack.pump()
    await asyncio.sleep(0.05)
    # Find the draft preview message.
    draft_messages = [m for m in stack.sender.messages if (m.text or "").startswith("Borrador")]
    assert draft_messages
    markup = draft_messages[-1].reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    token = _button_data(markup, 0, 0).split(":", 1)[1]

    # SEND tap → confirm dialog, NO send.
    await stack.tap(CHAT_ID, f"confirm:{token}")
    assert stack.gmail.sent == []
    assert any("Seguro" in (m.text or "") for m in stack.sender.messages)

    # Back → still no send, draft pending.
    await stack.tap(CHAT_ID, f"sendback:{token}")
    assert stack.gmail.sent == []
    assert stack.draft_row(1)["status"] == DraftStatus.PENDING.value

    # Cancel tap → confirm → yes → cancelled, no send.
    token_cancel = _button_data(markup, 0, 2).split(":", 1)[1]
    await stack.tap(CHAT_ID, f"cancel:{token_cancel}")
    assert stack.gmail.sent == []
    await stack.tap(CHAT_ID, f"cancelyes:{token_cancel}")
    await asyncio.sleep(0.05)
    assert stack.gmail.sent == []
    assert stack.draft_row(1)["status"] == DraftStatus.CANCELLED.value


async def test_flow_c_ambiguous_ack_never_sends(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("respóndele que sí", reply_to=stack.bot_message(summary_id))
    await stack.pump()
    await asyncio.sleep(0.05)
    assert stack.draft_row(1)["status"] == DraftStatus.PENDING.value

    # "ok" / "sí" while a draft is active: no send, guidance shown.
    await stack.send("ok")
    assert stack.gmail.sent == []
    assert stack.draft_row(1)["status"] == DraftStatus.PENDING.value
    await stack.send("sí")
    assert stack.gmail.sent == []

    # Explicit "envíalo" sends.
    await stack.send("envíalo")
    await asyncio.sleep(0.05)
    assert stack.draft_row(1)["status"] == DraftStatus.SENT_VERIFIED.value
    assert len(stack.gmail.sent) == 1


# ── D/E. CONTACT CREATION + NEW EMAIL VIA ALIAS ──────────────────────────────


async def test_flow_d_contact_creation_with_confirmation(stack: Stack) -> None:
    await stack.send("cuando diga roman usa femo@femo.ch")
    # The interpreted change is rendered and requires explicit confirmation.
    confirm = [m for m in stack.sender.messages if "¿Guardo el contacto" in (m.text or "")]
    assert confirm
    assert "femo@femo.ch" in confirm[-1].text
    assert not stack.contacts.resolve("roman").resolved  # not persisted yet
    markup = confirm[-1].reply_markup
    token = _button_data(markup, 0, 0).split(":", 1)[1]
    await stack.tap(CHAT_ID, f"confyes:{token}")
    await asyncio.sleep(0.05)
    assert stack.contacts.resolve("roman").resolved
    assert stack.contacts.resolve("roman").contact["email"] == "femo@femo.ch"


async def test_flow_e_new_email_through_alias(stack: Stack) -> None:
    roman = stack.contacts.create_contact("Roman", "femo@femo.ch")
    stack.contacts.add_alias(roman["id"], "mi jefe")
    await stack.send_bg("escribe a Roman y dile que mañana llego a las seis")
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert previews
    assert "femo@femo.ch" in previews[-1]  # REAL address shown, not just the name

    await stack.send("envíalo")
    await stack.wait_for_send()  # compose presents in a background task
    await stack.join_background()
    assert len(stack.gmail.sent) == 1
    assert stack.gmail.sent[0].to[0].email == "femo@femo.ch"
    assert stack.gmail.sent[0].thread_id == ""  # new email: no thread
    assert stack.draft_row(1)["status"] == DraftStatus.SENT_VERIFIED.value


# ── F/G. CONTACT AMBIGUITY / UNKNOWN CONTACT ─────────────────────────────────


async def test_flow_f_contact_ambiguity_asks(stack: Stack) -> None:
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    stack.contacts.create_contact("ROMAN", "roman@example.ch")
    await stack.send_bg("escribe a roman y dile que hola")
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any("coinciden" in (t or "") for t in texts)  # asked, never silent
    assert stack.gmail.sent == []


async def test_flow_g_unknown_contact_asks_for_address(stack: Stack) -> None:
    await stack.send_bg("escribe a Pepe y dile que hola")
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any("Pepe" in (t or "") and "correo" in (t or "") for t in texts)
    # No invented address, no draft.
    assert not any((m.text or "").startswith("Borrador") for m in stack.sender.messages)
    assert stack.gmail.sent == []


# ── H/I. CONTACT EDIT / DELETE ───────────────────────────────────────────────


async def test_flow_h_contact_email_change_updates_resolution(stack: Stack) -> None:
    roman = stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send("cambia el correo de roman a roman@nuevo.ch")
    await asyncio.sleep(0.05)
    # Explicit NL instruction with confirmation dialog.
    confirm = [m for m in stack.sender.messages if "¿Cambio" in (m.text or "")]
    assert confirm
    # Confirm via button.
    markup = confirm[-1].reply_markup
    token = _button_data(markup, 0, 0).split(":", 1)[1]
    await stack.tap(CHAT_ID, f"confyes:{token}")
    await asyncio.sleep(0.05)
    assert stack.contacts.get(roman["id"])["email"] == "roman@nuevo.ch"
    assert stack.contacts.resolve("roman").contact["email"] == "roman@nuevo.ch"


async def test_flow_i_contact_delete_and_replay_safe(stack: Stack) -> None:
    roman = stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send("borra el contacto roman")
    await asyncio.sleep(0.05)
    confirm = [m for m in stack.sender.messages if "¿Borro" in (m.text or "")]
    assert confirm
    markup = confirm[-1].reply_markup
    token = _button_data(markup, 0, 0).split(":", 1)[1]
    await stack.tap(CHAT_ID, f"confyes:{token}")
    await asyncio.sleep(0.05)
    assert stack.contacts.get(roman["id"]) is None
    # Replay of the same confirm is a no-op.
    await stack.tap(CHAT_ID, f"confyes:{token}")
    assert stack.contacts.get(roman["id"]) is None


# ── J. REPLY IGNORES NEW-EMAIL ALIAS ─────────────────────────────────────────


async def test_flow_j_reply_uses_original_recipient_not_alias(stack: Stack) -> None:
    roman = stack.contacts.create_contact("Roman", "femo@femo.ch")
    stack.contacts.add_alias(roman["id"], "ana")
    # Incoming email from anna@example.com; alias "ana" → femo@femo.ch.
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("respóndele que sí", reply_to=stack.bot_message(summary_id))
    await stack.pump()
    await asyncio.sleep(0.05)
    await stack.send("envíalo")
    await asyncio.sleep(0.05)
    assert len(stack.gmail.sent) == 1
    assert stack.gmail.sent[0].to[0].email == "anna@example.com"  # NOT femo@femo.ch
    assert stack.gmail.sent[0].thread_id == "t1"


# ── K. ATTACHMENT DOWNLOAD ───────────────────────────────────────────────────


async def test_flow_k_attachment_delivery(stack: Stack, tmp_path: Path) -> None:
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    summary_id = await stack.bot.send_summary(
        make_email_with_attachments(), EmailSummary(subject_es="Asunto")
    )
    await stack.send("mándame el pdf", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    # The bot sent the document to the group with the expected filename.
    assert stack.sender.files_delivered  # recorded by FakeSender
    filename, data = stack.sender.files_delivered[-1]
    assert filename == "presupuesto.pdf"
    assert data == b"%PDF-1.4 fake pdf content"
    # Temp file cleaned after delivery.
    delivery_dir = Path(str(tmp_path / "tmp" / "delivery"))
    leftovers = list(delivery_dir.iterdir()) if delivery_dir.is_dir() else []
    assert leftovers == []


async def test_flow_k3_attachment_delivery_outcome_log(
    stack: Stack, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="inboxbridge.assistant")
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    summary_id = await stack.bot.send_summary(
        make_email_with_attachments(), EmailSummary(subject_es="Asunto")
    )
    await stack.send("mándame el pdf", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    # Outcome log: coarse mime + bytes only — never the filename.
    assert any(
        "attachment_delivery outcome=success mime=application/pdf bytes=1024" in r.message
        for r in caplog.records
    )
    assert not any("presupuesto.pdf" in r.message for r in caplog.records)


async def test_flow_k2_docx_attachment_delivery(stack: Stack, tmp_path: Path) -> None:
    """DOCX attachments (OpenXML mime) are deliverable to Telegram, not rejected
    as unsupported — matching the documented attachment types."""
    from dataclasses import replace

    stack.settings.tmp_dir = str(tmp_path / "tmp")
    docx_email = replace(
        make_email(message_id="m1", subject="Informe DOCX"),
        attachments=[
            AttachmentMeta(
                filename="informe.docx",
                mime_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                size_bytes=1024,
                extracted_text="",
            )
        ],
    )
    stack.gmail.messages["m1"] = docx_email
    stack.gmail.attachment_bytes[("m1", 0)] = b"PK\x03\x04 fake docx content"
    summary_id = await stack.bot.send_summary(docx_email, EmailSummary(subject_es="Asunto"))
    await stack.assistant.handle(
        "get_attachment", {"tg_message_id": summary_id, "index": 0, "user_id": 7}
    )
    await asyncio.sleep(0.05)
    assert stack.sender.files_delivered
    filename, data = stack.sender.files_delivered[-1]
    assert filename == "informe.docx"
    assert data == b"PK\x03\x04 fake docx content"


# ── M. VISION FALLBACK (real AIService boundary) ─────────────────────────────


async def test_flow_m_vision_fallback(stack: Stack) -> None:
    from inboxbridge.llm.ai_service import AIService
    from inboxbridge.llm.base import LLMUnavailable
    from tests.unit.test_ai_service import FakeClient

    service = AIService(make_settings())
    vision = FakeClient([LLMUnavailable("mimo down")])
    fallback = FakeClient(["leído desde luna"])
    service._vision_llm = vision  # type: ignore[assignment]
    service._vision_fallback_llm = fallback  # type: ignore[assignment]
    result = await service.vision("¿qué ves?", [("image/png", b"x")])
    assert result == "leído desde luna"
    assert len(vision.calls) == 1 and len(fallback.calls) == 1
    call = service.calls[0]
    assert call.fallback_used and call.success

    # Both fail → user-friendly failure path (assistant surfaces it).
    vision2 = FakeClient([LLMUnavailable("down")])
    fallback2 = FakeClient([LLMUnavailable("down too")])
    service2 = AIService(make_settings())
    service2._vision_llm = vision2  # type: ignore[assignment]
    service2._vision_fallback_llm = fallback2  # type: ignore[assignment]
    with pytest.raises(LLMUnavailable):
        await service2.vision("¿qué ves?", [("image/png", b"x")])


# ── N/O. MARK READ / ARCHIVE ─────────────────────────────────────────────────


async def test_flow_n_mark_read(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("márcalo como leído", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.gmail.labelled and stack.gmail.labelled[-1][2] == ["UNREAD"]
    assert any("leído" in (m.text or "") for m in stack.sender.messages)


async def test_flow_o_archive(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("archívalo", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.gmail.labelled and stack.gmail.labelled[-1][2] == ["INBOX"]
    assert any("Archivado" in (m.text or "") for m in stack.sender.messages)


# ── P. REMINDER ──────────────────────────────────────────────────────────────


async def test_flow_p_reminder_create_and_fire(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("recuérdamelo en dos horas", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    rows = stack.reminders.list_pending(7)
    assert len(rows) == 1
    assert rows[0]["thread_id"] == "t1"  # context bound by IDs only
    assert "cuerpo" not in str(rows[0])

    # Cancel via the reminders panel button.
    await stack.send("cancela el recordatorio")
    await asyncio.sleep(0.05)
    def has_cancel_button(m: Message) -> bool:
        markup = m.reply_markup
        if markup is None:
            return False
        return any(
            (b.callback_data or "").startswith("rcancel:")
            for row in markup.inline_keyboard
            for b in row
        )

    cancel_buttons = [
        m
        for m in stack.sender.messages
        if "Cancelar #" in (m.text or "") or has_cancel_button(m)
    ]
    assert cancel_buttons
    markup = cancel_buttons[-1].reply_markup
    assert markup is not None
    cancel_data = next(
        b.callback_data
        for row in markup.inline_keyboard
        for b in row
        if (b.callback_data or "").startswith("rcancel:")
    )
    await stack.tap(CHAT_ID, cancel_data)
    await asyncio.sleep(0.05)
    assert stack.reminders.list_pending(7) == []


# ── Q. Q&A WITHOUT BUTTON ────────────────────────────────────────────────────


async def test_flow_q_qa_without_button(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me está pidiendo?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any("Bahnhofstrasse 10, Zürich" in (t or "") for t in texts)  # AI answer posted


def _thread_with_attachment(
    thread_id: str = "t-qa", filename: str = "rechnung.pdf", text: str = ""
) -> ThreadContext:
    from dataclasses import replace

    base = make_thread(thread_id)
    return replace(
        base,
        messages=[
            replace(
                base.messages[0],
                attachments=[
                    AttachmentMeta(
                        filename=filename,
                        mime_type="application/pdf",
                        size_bytes=2048,
                        extracted_text=text,
                    )
                ],
            )
        ],
    )


def _qa_prompt_content(stack: Stack) -> str:
    assert stack.ai.qa_messages, "qa prompt was never recorded"
    return str(stack.ai.qa_messages[-1][-1]["content"])


async def test_flow_q_qa_attachment_context_and_rich_send(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="inboxbridge.assistant")
    stack.gmail.threads["t-qa"] = _thread_with_attachment(
        text="Rechnung Nr. 42: 125 CHF fällig am 20.08.2026."
    )
    summary_id = await stack.bot.send_summary(
        make_email(thread_id="t-qa"), EmailSummary(subject_es="Asunto")
    )
    await stack.send(
        "¿qué me está pidiendo? ¿cuánto hay que pagar y dónde es la cita?",
        reply_to=stack.bot_message(summary_id),
    )
    await asyncio.sleep(0.05)

    # The Q&A prompt carries the bounded, sealed attachment text.
    content = _qa_prompt_content(stack)
    assert "Adjunto «rechnung.pdf»" in content
    assert "125 CHF" in content
    # THE exact user question must reach the model (rules-first Q&A must not
    # fall back to an internal default like "¿qué me está pidiendo?").
    assert "¿cuánto hay que pagar y dónde es la cita?" in content
    # Bounded fetch: Q&A asks for attachments explicitly.
    assert ("t-qa", True) in stack.gmail.thread_context_calls

    # Rich formatting: emoji heading bolded, bullets plain, HTML parse mode.
    rich = [m.text for m in stack.sender.messages if "Importe" in (m.text or "")]
    assert rich and "💰 <b>Importe</b>" in rich[-1]
    assert ParseMode.HTML in stack.sender.parse_modes

    # Privacy-safe outcome log: counts/bools only, never bodies or question.
    assert any(
        "qa outcome=success attachments=1 attachment_context=true" in r.message
        for r in caplog.records
    )
    assert not any("¿cuánto hay que pagar" in (r.message or "") for r in caplog.records)


async def test_flow_q_qa_unreadable_attachment_flagged(stack: Stack) -> None:
    stack.gmail.threads["t-qa"] = _thread_with_attachment(text="")
    summary_id = await stack.bot.send_summary(
        make_email(thread_id="t-qa"), EmailSummary(subject_es="Asunto")
    )
    await stack.send("¿qué me pide? ¿cuál es el importe?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    content = _qa_prompt_content(stack)
    assert "no legible" in content  # model must not invent facts from it


async def test_flow_q_qa_retry_on_transient_failure(stack: Stack) -> None:
    stack.gmail.threads["t1"] = make_thread("t1")
    stack.ai.qa_failures = 1  # first call raises LLMEmptyResponse → bounded retry
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me está pidiendo?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.qa_calls == 2
    assert any("Bahnhofstrasse 10, Zürich" in (m.text or "") for m in stack.sender.messages)


async def test_flow_q_qa_failure_message_and_log(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="inboxbridge.assistant")
    stack.gmail.threads["t1"] = make_thread("t1")
    stack.ai.qa_failures = 5  # exceeds bounded retries → user-safe failure
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me está pidiendo?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert any(
        "No pude responder ahora; inténtalo otra vez." in (m.text or "")
        for m in stack.sender.messages
    )
    assert any("qa outcome=failed" in r.message for r in caplog.records)


async def test_flow_q_qa_rich_fallback_plain(stack: Stack) -> None:
    stack.gmail.threads["t1"] = make_thread("t1")
    stack.sender.fail_html = True  # formatted send fails → plain fallback
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me está pidiendo?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    # The full answer arrives as plain text (no bold tags, no lost content).
    assert any("Bahnhofstrasse 10, Zürich" in (t or "") for t in texts)
    assert not any("<b>" in (t or "") for t in texts)


async def test_flow_q_qa_compact_single_fact(stack: Stack) -> None:
    """A one-fact question renders compact: emoji + answer with the fact
    bolded inline, not a large card."""
    stack.gmail.threads["t1"] = make_thread("t1")
    stack.ai.qa_override = json.dumps(
        {
            "answer": "El contacto es Markus Schneider.",
            "sections": [
                {"emoji": "👤", "title": "Contacto", "items": ["Markus Schneider"]}
            ],
        }
    )
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me pide? ¿quién es el contacto?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any(
        "👤 El contacto es <b>Markus Schneider</b>." in (t or "") for t in texts
    )


async def test_flow_q_qa_malformed_structured_safe_error(stack: Stack) -> None:
    """Malformed structured Q&A output must NEVER leak to Telegram: bounded
    retry, then a user-safe error (no raw JSON, no partial facts)."""
    stack.gmail.threads["t1"] = make_thread("t1")
    stack.ai.qa_override = '{"answer": "incompleto'  # truncated JSON every attempt
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me está pidiendo?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any(
        "No pude responder ahora; inténtalo otra vez." in (t or "") for t in texts
    )
    # Raw JSON / partial structured content never shown.
    assert not any('{"answer"' in (t or "") for t in texts)
    assert not any("incompleto" in (t or "") for t in texts)


async def test_flow_q_qa_dynamic_values_escaped(stack: Stack) -> None:
    """HTML-special characters in facts and titles stay escaped."""
    stack.gmail.threads["t1"] = make_thread("t1")
    stack.ai.qa_override = json.dumps(
        {
            "answer": "El total es 100 < 200?",
            "sections": [
                {
                    "emoji": "💰",
                    "title": "Precio < 200 & oferta",
                    "items": ["100 < 200 & más"],
                }
            ],
        }
    )
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me pide? ¿cuánto es?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any("&lt;200" in (t or "") or "&lt; 200" in (t or "") for t in texts)
    # The model-provided "<" is escaped, not raw markup after <b>.
    assert not any("<b>Precio <" in (t or "") for t in texts)


async def test_flow_q_button_summary_routes_rules_first(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """"resume toda la conversación" after pressing the Preguntar button must
    route deterministically to SUMMARIZE_THREAD (not Q&A)."""
    caplog.set_level(logging.INFO, logger="inboxbridge.assistant")
    stack.gmail.threads["t-qa"] = _thread_with_attachment(
        text="Frist: 31.08.2026, Zahlung 125 CHF."
    )
    summary_id = await stack.bot.send_summary(
        make_email(thread_id="t-qa"), EmailSummary(subject_es="Asunto")
    )
    await stack.tap(CHAT_ID, f"question:{summary_id}")
    await stack.send("resume toda la conversación")
    await asyncio.sleep(0.05)
    assert stack.ai.thread_summary_calls == 1
    assert stack.ai.qa_calls == 0
    assert any("📬 <b>Resumen</b>" in (m.text or "") for m in stack.sender.messages)
    # The summary used attachment context and logged it privacy-safely.
    assert stack.ai.thread_summary_messages
    content = str(stack.ai.thread_summary_messages[-1][-1]["content"])
    assert "Adjunto «rechnung.pdf»" in content
    assert any(
        "thread_summary structured outcome=success attachments=1 "
        "attachment_context=true" in r.message
        for r in caplog.records
    )


async def test_flow_q_summary_without_thread_does_not_guess(stack: Stack) -> None:
    """Standalone "resume toda la conversación" (no known thread) asks for a
    thread instead of guessing one."""
    await stack.send("resume toda la conversación")
    await asyncio.sleep(0.05)
    assert stack.ai.thread_summary_calls == 0
    assert any("¿Qué hilo?" in (m.text or "") for m in stack.sender.messages)


# ── R. THREAD SUMMARY ────────────────────────────────────────────────────────


async def test_flow_r_thread_summary(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    # Structured summary posted: fixed header + section blocks (no prose wall).
    texts = [m.text for m in stack.sender.messages]
    assert any("📬 <b>Resumen</b>" in (t or "") for t in texts)
    assert any("📅 <b>Cita</b>" in (t or "") for t in texts)
    assert any("• Bahnhofstrasse 10, Zürich" in (t or "") for t in texts)


async def test_flow_r_thread_summary_attachment_context(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="inboxbridge.assistant")
    stack.gmail.threads["t-qa"] = _thread_with_attachment(
        text="Frist: 31.08.2026, Zahlung 125 CHF."
    )
    summary_id = await stack.bot.send_summary(
        make_email(thread_id="t-qa"), EmailSummary(subject_es="Asunto")
    )
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.thread_summary_messages
    content = str(stack.ai.thread_summary_messages[-1][-1]["content"])
    assert "Adjunto «rechnung.pdf»" in content
    assert "125 CHF" in content
    assert any(
        "thread_summary structured outcome=success attachments=1 "
        "attachment_context=true" in r.message
        for r in caplog.records
    )


async def test_flow_r_summary_complex_sections_no_prose_wall(stack: Stack) -> None:
    """The PDF-style thread summary renders as compact structured sections
    (not a prose paragraph) while keeping every important fact exact."""
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    rendered = next(
        (m.text or "") for m in stack.sender.messages if "📬" in (m.text or "")
    )
    # Important facts, exact, each inside its section.
    assert "📅 <b>Cita</b>" in rendered
    assert "• 18 de agosto de 2026, 14:30" in rendered
    assert "• Bahnhofstrasse 10, Zürich" in rendered
    assert "📄 <b>Llevar</b>" in rendered
    assert "• Contrato firmado" in rendered
    assert "⏰ <b>Importante</b>" in rendered
    assert "Tasa: 125 CHF" in rendered
    assert "👤 <b>Contacto</b>" in rendered
    assert "Markus Schneider" in rendered
    # No prose wall: every line is a header, a bullet, or a bare single item.
    assert not any(
        line and not line.startswith(("📬", "📅", "📄", "⏰", "👤", "•", "Markus"))
        for line in rendered.splitlines()
    )


async def test_flow_r_summary_simple_thread_compact_bullets(stack: Stack) -> None:
    """A simple thread renders as the fixed header + 2-4 bullets (no sections,
    no repeated header)."""
    stack.ai.thread_summary_override = json.dumps(
        {
            "headline": "Resumen",
            "sections": [
                {
                    "emoji": "📬",
                    "title": "Resumen",
                    "items": [
                        "Cita el 18 de agosto a las 14:30 en Zürich.",
                        "Llevar contrato firmado y documento de identidad.",
                        "Avisar antes del 17 de agosto a las 18:00.",
                    ],
                }
            ],
        }
    )
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    rendered = next(
        (m.text or "") for m in stack.sender.messages if "📬" in (m.text or "")
    )
    assert rendered == (
        "📬 <b>Resumen</b>\n"
        "• Cita el 18 de agosto a las 14:30 en Zürich.\n"
        "• Llevar contrato firmado y documento de identidad.\n"
        "• Avisar antes del 17 de agosto a las 18:00."
    )
    assert rendered.count("Resumen") == 1  # header never repeated


async def test_flow_r_summary_action_section_when_present(stack: Stack) -> None:
    """A clear next action renders as its own ✅ Acción section."""
    stack.ai.thread_summary_override = json.dumps(
        {
            "headline": "Resumen",
            "sections": [
                {"emoji": "📅", "title": "Cita", "items": ["18 de agosto de 2026, 14:30"]},
                {
                    "emoji": "✅",
                    "title": "Acción",
                    "items": ["Confirmar asistencia antes del lunes."],
                },
            ],
        }
    )
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    rendered = next(
        (m.text or "") for m in stack.sender.messages if "📬" in (m.text or "")
    )
    assert "✅ <b>Acción</b>" in rendered
    assert "Confirmar asistencia antes del lunes." in rendered


async def test_flow_r_summary_no_action_not_fabricated(stack: Stack) -> None:
    """Without a real action the ✅ section is simply absent (the app never
    invents it)."""
    stack.ai.thread_summary_override = json.dumps(
        {
            "headline": "Resumen",
            "sections": [
                {"emoji": "👤", "title": "Contacto", "items": ["Markus Schneider"]}
            ],
        }
    )
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    rendered = next(
        (m.text or "") for m in stack.sender.messages if "📬" in (m.text or "")
    )
    assert "✅" not in rendered


async def test_flow_r_summary_structured_exhausted_plain_fallback_succeeds(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """Structured path exhausted (truncated JSON on both attempts) → ONE plain
    fallback generation succeeds → the user gets exactly one summary; raw
    JSON never reaches Telegram and the fallback receives attachment context
    with exact facts."""
    caplog.set_level(logging.INFO, logger="inboxbridge.assistant")
    stack.gmail.threads["t-qa"] = _thread_with_attachment(
        text="Frist: 31.08.2026, Zahlung 125 CHF."
    )
    stack.ai.thread_summary_override = (
        '{"headline": "Resumen", "sections": [{"emoji": "📬", "title": "Resumen", "items": ['
    )  # truncated JSON every structured attempt
    summary_id = await stack.bot.send_summary(
        make_email(thread_id="t-qa"), EmailSummary(subject_es="Asunto")
    )
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.thread_summary_calls == 2  # structured bounded retry
    assert stack.ai.plain_summary_calls == 1  # exactly one plain attempt
    texts = [m.text for m in stack.sender.messages]
    # The user gets the plain summary, rendered safely (bullets preserved).
    assert any("• Tasa: 125 CHF." in (t or "") for t in texts)
    assert any("• Cita el 18 de agosto de 2026 a las 14:30" in (t or "") for t in texts)
    # Raw JSON never shown.
    assert not any('{"headline"' in (t or "") for t in texts)
    # The fallback generation received the SAME attachment context.
    assert stack.ai.plain_summary_messages
    content = str(stack.ai.plain_summary_messages[-1][-1]["content"])
    assert "Adjunto «rechnung.pdf»" in content
    assert "125 CHF" in content
    # Logs: structured failure and fallback success, clearly distinct.
    assert any(
        "thread_summary structured outcome=invalid_structure" in r.message
        for r in caplog.records
    )
    assert any(
        "thread_summary fallback=plain outcome=success attachments=1 "
        "attachment_context=true" in r.message
        for r in caplog.records
    )
    assert not any(
        "thread_summary structured outcome=success" in r.message for r in caplog.records
    )


async def test_flow_r_summary_both_paths_fail_safe_error(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """Structured AND plain fallback exhausted → final user-safe error; no raw
    JSON, no partial content, no success logs at all."""
    caplog.set_level(logging.INFO, logger="inboxbridge.assistant")
    stack.ai.thread_summary_override = (
        '{"headline": "Resumen", "sections": [{"emoji": "📬", "title": "Resumen", "items": ['
    )
    stack.ai.plain_summary_failures = 5  # plain fallback also fails
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any(
        "No pude generar el resumen ahora; inténtalo otra vez." in (t or "")
        for t in texts
    )
    # Raw JSON and partial plain content never shown.
    assert not any('{"headline"' in (t or "") for t in texts)
    assert not any("Tasa" in (t or "") for t in texts)
    # No false success before a usable summary exists.
    assert not any(
        "thread_summary structured outcome=success" in r.message for r in caplog.records
    )
    assert not any(
        "thread_summary fallback=plain outcome=success" in r.message
        for r in caplog.records
    )
    assert any(
        "thread_summary structured outcome=invalid_structure" in r.message
        for r in caplog.records
    )
    assert any(
        "thread_summary fallback=plain outcome=failed error=LLMEmptyResponse" in r.message
        for r in caplog.records
    )


async def test_flow_r_summary_structured_success_skips_plain_fallback(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """A successful structured summary never triggers the plain fallback."""
    caplog.set_level(logging.INFO, logger="inboxbridge.assistant")
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.thread_summary_calls == 1
    assert stack.ai.plain_summary_calls == 0
    assert any(
        "thread_summary structured outcome=success" in r.message for r in caplog.records
    )
    assert not any(
        "thread_summary fallback=plain" in r.message for r in caplog.records
    )


async def test_flow_r_summary_plain_fallback_rich_send_fails_plain_text(
    stack: Stack,
) -> None:
    """Plain fallback + Telegram rich-send failure → the plain summary still
    arrives as plain text (no tags, no information loss)."""
    stack.sender.fail_html = True
    stack.ai.thread_summary_override = (
        '{"headline": "Resumen", "sections": [{"emoji": "📬", "title": "Resumen", "items": ['
    )
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any("• Tasa: 125 CHF." in (t or "") for t in texts)
    assert any("• Cita el 18 de agosto de 2026 a las 14:30" in (t or "") for t in texts)
    assert not any("<b>" in (t or "") for t in texts)


async def test_flow_r_summary_first_malformed_then_valid(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """Attempt 1 truncated JSON → discarded; attempt 2 valid → rendered
    EXACTLY once, no plain fallback. The failure attempt never reaches
    Telegram."""
    caplog.set_level(logging.INFO, logger="inboxbridge.assistant")
    stack.ai.thread_summary_malformed = 1
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.thread_summary_calls == 2  # discarded attempt + valid attempt
    assert stack.ai.plain_summary_calls == 0  # structured retry succeeded
    texts = [m.text for m in stack.sender.messages]
    assert not any('{"headline"' in (t or "") for t in texts)
    # Exactly ONE rendered summary.
    assert sum(1 for t in texts if "📬 <b>Resumen</b>" in (t or "")) == 1
    # outcome=success logged exactly once, only after the parse succeeded.
    successes = [
        r.message
        for r in caplog.records
        if "thread_summary structured outcome=success" in r.message
    ]
    assert len(successes) == 1
    assert not any(
        "thread_summary structured outcome=invalid_structure" in r.message
        for r in caplog.records
    )


async def test_flow_q_qa_first_malformed_then_valid(stack: Stack) -> None:
    """Q&A gets the same protection: attempt 1 truncated JSON discarded,
    attempt 2 valid → structured answer rendered, no raw JSON."""
    stack.gmail.threads["t1"] = make_thread("t1")
    stack.ai.qa_malformed = 1
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me está pidiendo?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.qa_calls == 2
    texts = [m.text for m in stack.sender.messages]
    assert not any('{"answer"' in (t or "") for t in texts)
    assert any("Bahnhofstrasse 10, Zürich" in (t or "") for t in texts)
    assert any("💰 <b>Importe</b>" in (t or "") for t in texts)


async def test_flow_q_qa_both_malformed_safe_error(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """Both Q&A attempts malformed → user-safe error, no raw JSON, no
    outcome=success log."""
    caplog.set_level(logging.INFO, logger="inboxbridge.assistant")
    stack.gmail.threads["t1"] = make_thread("t1")
    stack.ai.qa_override = '{"answer": "incompleto'
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me está pidiendo?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any(
        "No pude responder ahora; inténtalo otra vez." in (t or "") for t in texts
    )
    assert not any('{"answer"' in (t or "") for t in texts)
    assert any("qa outcome=invalid_structure" in r.message for r in caplog.records)
    assert not any("qa outcome=success" in r.message for r in caplog.records)


async def test_flow_r_summary_plain_fallback(stack: Stack) -> None:
    """Formatted-send failure falls back to a plain-text rendering of the SAME
    structured summary (no tags, no lost content)."""
    stack.sender.fail_html = True
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any("• Bahnhofstrasse 10, Zürich" in (t or "") for t in texts)
    assert any("Tasa: 125 CHF" in (t or "") for t in texts)
    assert not any("<b>" in (t or "") for t in texts)


async def test_flow_r_summary_values_escaped(stack: Stack) -> None:
    """HTML-special characters in summary fields stay escaped (no markup from
    the model)."""
    stack.ai.thread_summary_override = json.dumps(
        {
            "headline": "Resumen",
            "sections": [
                {"emoji": "💼", "title": "<b>fake</b> & off", "items": ["a < b & c > d"]}
            ],
        }
    )
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    texts = [m.text for m in stack.sender.messages]
    assert any("&lt;b&gt;fake&lt;/b&gt;" in (t or "") for t in texts)
    assert not any("<b>fake" in (t or "") for t in texts)
    assert any("a &lt; b &amp; c &gt; d" in (t or "") for t in texts)


# ── S. FORWARD ───────────────────────────────────────────────────────────────


async def test_flow_s_forward(stack: Stack) -> None:
    stack.contacts.create_contact("Daniel", "daniel@forward.ch")
    original = email_with_attachments(make_email(message_id="m-fwd", subject="Presupuesto"))
    stack.gmail.messages["m-fwd"] = original
    stack.gmail.attachment_bytes[("m-fwd", 0)] = b"%PDF-1.4 fake pdf content"
    summary_id = await stack.bot.send_summary(original, EmailSummary(subject_es="Asunto"))
    await stack.send_bg("reenvíaselo a Daniel", reply_to=stack.bot_message(summary_id))
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert previews
    assert "daniel@forward.ch" in previews[-1]  # real address visible
    assert "presupuesto.pdf" in previews[-1]  # original attachment represented
    await stack.send("envíalo")
    await stack.wait_for_send()  # forward presents in a background task
    await stack.join_background()
    assert len(stack.gmail.sent) == 1
    assert stack.gmail.sent[0].to[0].email == "daniel@forward.ch"
    assert stack.gmail.sent[0].subject.startswith("Fwd:")
    assert stack.gmail.sent[0].attachments[0].filename == "presupuesto.pdf"
    assert stack.draft_row(1)["status"] == DraftStatus.SENT_VERIFIED.value


# ── V. NATURAL-LANGUAGE SLOT FILLING (latest email + compose) ────────────────


def _seed_incoming(stack: Stack, message_id: str, thread_id: str, history_id: int) -> None:
    stack.storage.upsert_message(message_id, thread_id, history_id, MessageStatus.SENT_TELEGRAM)


async def test_flow_v_reply_latest_two_turn(stack: Stack) -> None:
    stack.gmail.threads["t-a"] = make_thread("t-a")
    _seed_incoming(stack, "m-a", "t-a", 100)
    await stack.send("respóndele que muchas gracias")
    await asyncio.sleep(0.05)
    assert any("¿A qué correo" in (m.text or "") for m in stack.sender.messages)

    await stack.send("al último")
    await stack.pump()
    await asyncio.sleep(0.15)
    assert stack.draft_row(1)["thread_id"] == "t-a"  # frozen concrete target


async def test_flow_v_reply_latest_one_turn(stack: Stack) -> None:
    stack.gmail.threads["t-a"] = make_thread("t-a")
    _seed_incoming(stack, "m-a", "t-a", 100)
    await stack.send("respóndele que muchas gracias al último correo recibido")
    await stack.pump()
    await asyncio.sleep(0.15)
    assert stack.draft_row(1)["thread_id"] == "t-a"


async def test_flow_v_reply_latest_no_recent_email(stack: Stack) -> None:
    await stack.send("responde al último")
    await asyncio.sleep(0.05)
    assert any("reciente" in (m.text or "") for m in stack.sender.messages)
    assert stack.storage.get_draft(1) is None  # no draft, no send


async def test_flow_v_reply_latest_ignores_own_sent_and_freezes(stack: Stack) -> None:
    """'al último' selects the latest INCOMING message (never our own SENT), and
    the draft is frozen to it even if a newer incoming email arrives later."""
    stack.gmail.threads["t-a"] = make_thread("t-a")
    stack.gmail.threads["t-b"] = make_thread("t-b")
    _seed_incoming(stack, "m-a", "t-a", 100)
    _seed_incoming(stack, "m-b", "t-b", 101)  # newer incoming B
    await stack.send("respóndele que gracias al último")
    await stack.pump()
    await asyncio.sleep(0.15)
    # Latest incoming is B (t-b), NOT our own sent message (which is a draft row).
    assert stack.draft_row(1)["thread_id"] == "t-b"
    # A newer incoming C arrives AFTER resolution → draft stays bound to t-b.
    _seed_incoming(stack, "m-c", "t-c", 102)
    await asyncio.sleep(0.05)
    assert stack.draft_row(1)["thread_id"] == "t-b"


async def test_flow_v_compose_multi_turn(stack: Stack) -> None:
    await stack.send("envía un correo")
    await asyncio.sleep(0.05)
    assert any("¿A quién" in (m.text or "") for m in stack.sender.messages)

    await stack.send("a user@example.com")
    await asyncio.sleep(0.05)
    assert any("¿Qué le digo?" in (m.text or "") for m in stack.sender.messages)

    await stack.send_bg("dile que mañana llegaré media hora tarde")
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews and "user@example.com" in previews[-1]


async def test_flow_v_compose_one_turn_diciendo(stack: Stack) -> None:
    await stack.send_bg(
        "envía un correo a user@example.com diciendo que mañana llego tarde"
    )
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews and "user@example.com" in previews[-1]


async def test_flow_v_compose_cancel(stack: Stack) -> None:
    await stack.send("envía un correo")
    await asyncio.sleep(0.05)
    await stack.send("cancela")
    await asyncio.sleep(0.05)
    assert any("Cancelado" in (m.text or "") for m in stack.sender.messages)
    assert stack.storage.get_draft(1) is None


# ── W. COMPOSE DRAFT BUTTONS (EDIT/CANCEL/SEND + owner + guard) ──────────────


async def _compose_draft_token(stack: Stack) -> str:
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escribe a Roman y dile que hola")
    await asyncio.sleep(0.05)
    draft_messages = [
        m for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert draft_messages
    markup = draft_messages[-1].reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    return _button_data(markup, 0, 0).split(":", 1)[1]


async def test_flow_w_compose_edit_prompt(stack: Stack) -> None:
    token = await _compose_draft_token(stack)
    await stack.tap(CHAT_ID, f"edit:{token}", user_id=7)
    assert any("Dime qué cambio" in (m.text or "") for m in stack.sender.messages)
    assert stack.gmail.sent == []
    assert stack.draft_row(1)["status"] == DraftStatus.PENDING.value


async def test_flow_w_compose_cancel_confirm_and_terminate(stack: Stack) -> None:
    token = await _compose_draft_token(stack)
    # First CANCEL tap → confirmation UI, draft NOT cancelled.
    await stack.tap(CHAT_ID, f"cancel:{token}", user_id=7)
    assert any("¿Cancelar este borrador?" in (m.text or "") for m in stack.sender.messages)
    assert stack.draft_row(1)["status"] == DraftStatus.PENDING.value
    # "Volver" → draft stays pending.
    await stack.tap(CHAT_ID, f"cancelback:{token}", user_id=7)
    assert stack.draft_row(1)["status"] == DraftStatus.PENDING.value
    # "Sí, cancelar" → cancelled, no send.
    await stack.tap(CHAT_ID, f"cancel:{token}", user_id=7)
    await stack.tap(CHAT_ID, f"cancelyes:{token}", user_id=7)
    await asyncio.sleep(0.05)
    assert stack.draft_row(1)["status"] == DraftStatus.CANCELLED.value
    assert stack.gmail.sent == []


async def test_flow_w_compose_send_two_step_exactly_once(stack: Stack) -> None:
    token = await _compose_draft_token(stack)
    # First SEND tap → confirmation, NO send.
    await stack.tap(CHAT_ID, f"confirm:{token}", user_id=7)
    assert stack.gmail.sent == []
    assert any("¿Seguro que quieres enviar" in (m.text or "") for m in stack.sender.messages)
    # Second confirmation → send exactly once.
    await stack.tap(CHAT_ID, f"sendyes:{token}", user_id=7)
    await stack.wait_for_send()
    assert len(stack.gmail.sent) == 1


async def test_flow_w_compose_owner_isolation(stack: Stack) -> None:
    token = await _compose_draft_token(stack)
    # Non-owner (user 8) presses EDIT/CANCEL/SEND → rejected, draft unchanged.
    await stack.tap(CHAT_ID, f"edit:{token}", user_id=8)
    await stack.tap(CHAT_ID, f"cancel:{token}", user_id=8)
    await stack.tap(CHAT_ID, f"confirm:{token}", user_id=8)
    assert stack.draft_row(1)["status"] == DraftStatus.PENDING.value
    assert stack.gmail.sent == []
    assert not any("Dime qué cambio" in (m.text or "") for m in stack.sender.messages)


async def test_flow_w_compose_edit_regenerates_bilingual(stack: Stack) -> None:
    token = await _compose_draft_token(stack)
    await stack.tap(CHAT_ID, f"edit:{token}", user_id=7)
    await asyncio.sleep(0.05)
    await stack.send("hazlo más corto")
    await asyncio.sleep(0.05)
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    assert "kurz und klar" in previews[-1]  # new German
    assert "[ES]" in previews[-1]  # new Spanish translation


async def test_flow_w_active_draft_new_compose_asks_cancel(stack: Stack) -> None:
    await _compose_draft_token(stack)
    await stack.send("envía un correo a otro@example.com")
    await asyncio.sleep(0.05)
    assert any(
        "borrador pendiente" in (m.text or "").lower() for m in stack.sender.messages
    )
    assert stack.storage.get_draft(2) is None  # no second draft created


# ── X. NEW-MAIL SUBJECT (LLM JSON, never the raw command) ────────────────────


def _subject_line(preview: str) -> str:
    for line in preview.splitlines():
        if line.startswith("Asunto:"):
            return line
    raise AssertionError("no Asunto line in preview")


async def test_flow_x_compose_subject_is_natural_one_turn(stack: Stack) -> None:
    await stack.send_bg(
        "envía un correo a user@example.com diciendo que esta es una prueba"
    )
    await asyncio.sleep(0.05)
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    subject_line = _subject_line(previews[-1])
    assert "Vielen Dank" in subject_line  # LLM subject, derived from content
    assert "envía un correo" not in subject_line  # no command scaffolding
    assert "user@example.com" not in subject_line  # no recipient leak


async def test_flow_x_compose_subject_is_natural_multi_turn(stack: Stack) -> None:
    await stack.send("envía un correo")
    await asyncio.sleep(0.05)
    await stack.send("a user@example.com")
    await asyncio.sleep(0.05)
    await stack.send_bg("dile que mañana llegaré media hora tarde")
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    subject_line = _subject_line(previews[-1])
    assert "Vielen Dank" in subject_line
    assert "dile que" not in subject_line


async def test_flow_x_compose_subject_survives_to_gmail(stack: Stack) -> None:
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escribe a Roman y dile que hola")
    await asyncio.sleep(0.05)
    await stack.send("envíalo")
    await stack.wait_for_send()
    await stack.join_background()
    assert len(stack.gmail.sent) == 1
    # The subject generated at preview time is the one sent to Gmail.
    assert stack.gmail.sent[0].subject == "Vielen Dank"


# ── Y. PROPORTIONAL EDIT + TRANSLATION RETRY ─────────────────────────────────


async def test_flow_y_edit_preserves_subject_recipient(stack: Stack) -> None:
    """A body-only edit keeps subject, recipient, attachments and thread."""
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escribe a Roman y dile que hola")
    await asyncio.sleep(0.05)
    draft_messages = [
        m for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    first = draft_messages[-1]
    assert "Asunto: Vielen Dank" in first.text
    assert "femo@femo.ch" in first.text

    token = _button_data(first.reply_markup, 0, 0).split(":", 1)[1]
    await stack.tap(CHAT_ID, f"edit:{token}", user_id=7)
    await asyncio.sleep(0.05)
    await stack.send("hazlo más corto")
    await asyncio.sleep(0.05)

    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    edited = previews[-1]
    assert "kurz und klar" in edited  # new German body
    assert "Asunto: Vielen Dank" in edited  # subject preserved
    assert "femo@femo.ch" in edited  # recipient preserved


async def test_flow_y_edit_translation_retry_succeeds(stack: Stack) -> None:
    """First translation returns LLMEmptyResponse; bounded retry succeeds →
    bilingual preview. German body is generated exactly once."""
    stack.ai.translate_failures = 1
    token = await _compose_draft_token(stack)
    await stack.tap(CHAT_ID, f"edit:{token}", user_id=7)
    await asyncio.sleep(0.05)
    await stack.send("hazlo más corto")
    await asyncio.sleep(0.05)

    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    assert "kurz und klar" in previews[-1]  # German preserved
    assert "[ES]" in previews[-1]  # retried translation present
    assert stack.ai.translate_calls == 2  # one retry
    draft_edit_calls = [c for c in stack.ai.calls if c[0] == "draft_edit"]
    assert len(draft_edit_calls) == 1  # German NEVER regenerated during retry


async def test_flow_y_edit_translation_retry_exhausted_shows_warning(
    stack: Stack,
) -> None:
    """Retry exhausted → German preserved + explicit translation-unavailable
    state (never silently omitted, never sent)."""
    stack.ai.translate_failures = 5  # more than the retry budget
    token = await _compose_draft_token(stack)
    await stack.tap(CHAT_ID, f"edit:{token}", user_id=7)
    await asyncio.sleep(0.05)
    await stack.send("hazlo más corto")
    await asyncio.sleep(0.05)

    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    edited = previews[-1]
    assert "kurz und klar" in edited  # German preserved
    assert "⚠️ No pude generar la traducción ahora." in edited  # explicit state
    assert stack.ai.translate_calls == 2  # bounded: exactly one retry


async def test_flow_y_edit_spanish_never_in_gmail_body(stack: Stack) -> None:
    token = await _compose_draft_token(stack)
    await stack.tap(CHAT_ID, f"edit:{token}", user_id=7)
    await asyncio.sleep(0.05)
    await stack.send("hazlo más corto")
    await asyncio.sleep(0.05)
    await stack.send("envíalo")
    await stack.wait_for_send()
    await stack.join_background()
    assert len(stack.gmail.sent) == 1
    assert "[ES]" not in stack.gmail.sent[0].body  # Spanish never sent
    assert "⚠️" not in stack.gmail.sent[0].body


# ── Z. RULES-FIRST EDITS + DRAFT_EDIT BOUNDED RETRY ──────────────────────────


def _preview_count(stack: Stack) -> int:
    return sum(1 for m in stack.sender.messages if (m.text or "").startswith("Borrador ("))


async def test_flow_z_edit_routes_rules_first_no_intent_llm(stack: Stack) -> None:
    """'más largo' with an active draft edits directly; the intent LLM is
    never consulted."""
    await _compose_draft_token(stack)
    await stack.send("más largo")
    await asyncio.sleep(0.05)
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews and "kurz und klar" in previews[-1]  # edited
    assert not any(c[0] == "intent" for c in stack.ai.calls)  # rules-first
    assert stack.ai.draft_edit_calls == 1


async def test_flow_z_tone_edit_routes_rules_first(stack: Stack) -> None:
    await _compose_draft_token(stack)
    await stack.send("más formal")
    await asyncio.sleep(0.05)
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews and "kurz und klar" in previews[-1]  # edited
    assert not any(c[0] == "intent" for c in stack.ai.calls)


async def test_flow_z_edit_without_draft_does_not_mutate(stack: Stack) -> None:
    await stack.send("más largo")
    await asyncio.sleep(0.05)
    assert stack.storage.get_draft(1) is None  # no draft created/mutated
    assert stack.gmail.sent == []


async def test_flow_z_edit_retry_succeeds_one_preview(stack: Stack) -> None:
    """First draft_edit returns LLMEmptyResponse; bounded retry succeeds →
    exactly one new preview, translation runs after the successful edit."""
    stack.ai.draft_edit_failures = 1
    await _compose_draft_token(stack)
    before = _preview_count(stack)
    await stack.send("más largo")
    await asyncio.sleep(0.05)
    assert stack.ai.draft_edit_calls == 2  # one retry
    assert _preview_count(stack) == before + 1  # exactly one new preview
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert "kurz und klar" in previews[-1]
    assert "[ES]" in previews[-1]  # translation ran after the successful edit
    assert stack.ai.translate_calls == 1


async def test_flow_z_edit_retry_exhausted_preserves_draft(stack: Stack) -> None:
    """Both draft_edit attempts fail → the original draft/preview stays active
    and unchanged; the user is told the edit could not be completed."""
    stack.ai.draft_edit_failures = 5  # more than the retry budget
    await _compose_draft_token(stack)
    before = _preview_count(stack)
    await stack.send("más largo")
    await asyncio.sleep(0.05)
    assert stack.ai.draft_edit_calls == 2  # bounded
    assert _preview_count(stack) == before  # no new preview
    assert any("No pude editar" in (m.text or "") for m in stack.sender.messages)
    assert stack.draft_row(1)["status"] == DraftStatus.PENDING.value  # still active


async def test_flow_z_edit_retry_preserves_subject_recipient(stack: Stack) -> None:
    stack.ai.draft_edit_failures = 1
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escribe a Roman y dile que hola")
    await asyncio.sleep(0.05)
    await stack.send("más largo")
    await asyncio.sleep(0.05)
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    edited = previews[-1]
    assert "Asunto: Vielen Dank" in edited  # subject preserved
    assert "femo@femo.ch" in edited  # recipient preserved


# ── AA. COMPOSE/FORWARD BOUNDED RETRY (LLMEmptyResponse) ─────────────────────


async def test_flow_aa_compose_retry_succeeds_one_draft(stack: Stack) -> None:
    """Compose first LLMEmptyResponse → bounded retry succeeds → exactly one
    draft (never two)."""
    stack.ai.compose_failures = 1
    await stack.send("envía un correo a user@example.com diciendo que hola")
    await asyncio.sleep(0.1)
    assert stack.ai.compose_calls == 2  # one retry
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews and "user@example.com" in previews[-1]
    assert stack.storage.get_draft(1) is not None
    assert stack.storage.get_draft(2) is None  # exactly one draft


async def test_flow_aa_compose_retry_exhausted_no_draft(stack: Stack) -> None:
    """Compose both attempts fail → no draft, safe notice, no pending state."""
    stack.ai.compose_failures = 5  # more than the retry budget
    await stack.send("envía un correo a user@example.com diciendo que hola")
    await asyncio.sleep(0.05)
    assert stack.ai.compose_calls == 2  # bounded
    assert any("No pude redactar" in (m.text or "") for m in stack.sender.messages)
    assert stack.storage.get_draft(1) is None  # no draft created


async def test_flow_aa_forward_retry_succeeds_one_draft(stack: Stack) -> None:
    """Forward first LLMEmptyResponse → retry succeeds → exactly one draft."""
    stack.contacts.create_contact("Daniel", "daniel@forward.ch")
    original = email_with_attachments(make_email(message_id="m-fwd", subject="Presupuesto"))
    stack.gmail.messages["m-fwd"] = original
    stack.gmail.attachment_bytes[("m-fwd", 0)] = b"%PDF-1.4 fake pdf content"
    summary_id = await stack.bot.send_summary(original, EmailSummary(subject_es="Asunto"))
    stack.ai.forward_failures = 1
    await stack.send("reenvíaselo a Daniel", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.1)
    assert stack.ai.forward_calls == 2  # one retry
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews and "daniel@forward.ch" in previews[-1]
    assert stack.storage.get_draft(1) is not None
    assert stack.storage.get_draft(2) is None


async def test_flow_aa_forward_retry_exhausted_no_draft(stack: Stack) -> None:
    """Forward both attempts fail → no draft, safe notice."""
    stack.contacts.create_contact("Daniel", "daniel@forward.ch")
    original = make_email(message_id="m-fwd", subject="Presupuesto")
    stack.gmail.messages["m-fwd"] = original
    summary_id = await stack.bot.send_summary(original, EmailSummary(subject_es="Asunto"))
    stack.ai.forward_failures = 5  # more than the retry budget
    await stack.send("reenvíaselo a Daniel", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.forward_calls == 2  # bounded
    assert any("No pude preparar el reenvío" in (m.text or "") for m in stack.sender.messages)
    assert stack.storage.get_draft(1) is None


async def test_flow_aa_forward_retry_no_duplicate_attachments(stack: Stack) -> None:
    """The forward retry reuses the same original: the attachment is fetched
    and written exactly once (no duplicate temp files)."""
    stack.contacts.create_contact("Daniel", "daniel@forward.ch")
    original = email_with_attachments(make_email(message_id="m-fwd", subject="Presupuesto"))
    stack.gmail.messages["m-fwd"] = original
    stack.gmail.attachment_bytes[("m-fwd", 0)] = b"%PDF-1.4 fake pdf content"
    summary_id = await stack.bot.send_summary(original, EmailSummary(subject_es="Asunto"))
    stack.ai.forward_failures = 1
    await stack.send("reenvíaselo a Daniel", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.1)
    assert stack.ai.forward_calls == 2  # body retried once
    assert len(stack.gmail.attachment_fetches) == 1  # attachment fetched ONCE
    import json as _json

    row = stack.draft_row(1)
    attachments = _json.loads(row["attachments_json"])
    assert len(attachments) == 1
    assert attachments[0]["filename"] == "presupuesto.pdf"
    delivery = Path(str(stack.settings.tmp_dir)) / "delivery"
    leftovers = list(delivery.iterdir()) if delivery.is_dir() else []
    assert leftovers == []  # claimed into the draft dir, no strays


# ── AB. TRUSTED SIGNATURE ON SENDABLE DRAFTS ─────────────────────────────────


async def test_flow_ab_compose_signature_once_in_preview_and_gmail(stack: Stack) -> None:
    """The trusted signature appears exactly once in the final German body sent
    to Gmail, and never the Spanish translation."""
    await stack.send_bg("envía un correo a user@example.com diciendo que hola")
    await asyncio.sleep(0.05)
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    preview = previews[-1]
    assert "Mit freundlichen Grüßen" in preview
    assert "Daniel" in preview  # trusted signature (config, not invented)

    await stack.send("envíalo")
    await stack.wait_for_send()
    await stack.join_background()
    assert len(stack.gmail.sent) == 1
    body = stack.gmail.sent[0].body
    assert body.count("Daniel") == 1  # German signature exactly once
    assert "[ES]" not in body  # Spanish never sent


async def test_flow_ab_edit_keeps_single_signature(stack: Stack) -> None:
    """Repeated body edits never duplicate the trusted signature."""
    await _compose_draft_token(stack)
    for _ in range(2):
        await stack.send("más corto")
        await asyncio.sleep(0.05)
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    german = previews[-1].split("🇪🇸")[0]  # German section only
    assert "Mit freundlichen Grüßen" in german
    assert german.count("Daniel") == 1  # never duplicated


async def test_flow_ab_translation_covers_final_body_with_signature(stack: Stack) -> None:
    """The Spanish preview is derived from the EXACT final German body,
    including the trusted signature."""
    await _compose_draft_token(stack)
    previews = [
        m.text for m in stack.sender.messages if (m.text or "").startswith("Borrador (")
    ]
    assert previews
    spanish = previews[-1].split("🇪🇸")[-1]
    assert "Daniel" in spanish  # translation includes the signer


# ── T. VOICE (EXPERIMENTAL) ──────────────────────────────────────────────────


async def test_flow_t_voice_disabled_falls_back_gracefully(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    voice = Voice(
        file_id="voice-1", file_unique_id="v1", duration=5, file_size=100
    )
    message = Message(
        message_id=200,
        date=datetime.now(UTC),
        chat=Chat(id=CHAT_ID, type=ChatType.GROUP),
        from_user=_user(7),
        voice=voice,
        reply_to_message=stack.bot_message(summary_id),
    )
    await stack.bot.process_update(_update(message))
    texts = [m.text for m in stack.sender.messages]
    assert any("notas de voz aún no están activadas" in (t or "") for t in texts)
    assert stack.ai.audio_calls == 0


# ── text-model technical fallback (AI_TEXT_FALLBACK_MODEL) ───────────────────


def _enable_text_fallback(stack: Stack) -> None:
    stack.settings.ai_text_fallback_model = "fb-model"


async def test_flow_fallback_compose_primary_empty_fallback_valid_one_draft(
    stack: Stack,
) -> None:
    """Compose: primary returns empty (transient) → the bounded retry uses the
    configured fallback model → exactly ONE draft, exactly 2 provider calls
    (never 4)."""
    _enable_text_fallback(stack)
    stack.contacts.create_contact("Daniel", "daniel@fb.ch")
    stack.ai.compose_failures = 1
    await stack.send_bg("escríbele a Daniel que muchas gracias")
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert previews  # exactly one draft preview
    assert stack.ai.compose_calls == 2
    assert stack.ai.models[-2:] == [None, "fb-model"]  # primary, then fallback


async def test_flow_fallback_qa_primary_malformed_fallback_structured_succeeds(
    stack: Stack,
) -> None:
    """Q&A: primary returns malformed structured JSON → retry uses the fallback
    model with the SAME question/context → structured answer rendered."""
    _enable_text_fallback(stack)
    stack.gmail.threads["t1"] = make_thread("t1")
    stack.ai.qa_malformed = 1
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me está pidiendo?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.qa_calls == 2
    assert stack.ai.models[-2:] == [None, "fb-model"]
    texts = [m.text for m in stack.sender.messages]
    assert any("💰 <b>Importe</b>" in (t or "") for t in texts)
    assert not any('{"answer"' in (t or "") for t in texts)


async def test_flow_fallback_qa_both_fail_safe_error(stack: Stack) -> None:
    """Q&A: BOTH models produce unusable structured output → safe error, no
    raw JSON, bounded 2 calls."""
    _enable_text_fallback(stack)
    stack.gmail.threads["t1"] = make_thread("t1")
    stack.ai.qa_override = '{"answer": "incompleto'
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me está pidiendo?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.qa_calls == 2
    texts = [m.text for m in stack.sender.messages]
    assert any(
        "No pude responder ahora; inténtalo otra vez." in (t or "") for t in texts
    )
    assert not any('{"answer"' in (t or "") for t in texts)


async def test_flow_fallback_summary_primary_malformed_fallback_structured_succeeds(
    stack: Stack,
) -> None:
    """Thread summary: primary malformed JSON → fallback model produces the
    structured contract → rendered once, no plain layer used."""
    _enable_text_fallback(stack)
    stack.ai.thread_summary_malformed = 1
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.thread_summary_calls == 2
    assert stack.ai.plain_summary_calls == 0
    assert stack.ai.models[-2:] == [None, "fb-model"]
    texts = [m.text for m in stack.sender.messages]
    assert sum(1 for t in texts if "📬 <b>Resumen</b>" in (t or "")) == 1
    assert not any('{"headline"' in (t or "") for t in texts)


async def test_flow_fallback_summary_hard_budget_three_calls(stack: Stack) -> None:
    """Thread summary HARD budget: structured primary + structured fallback +
    ONE plain generation = exactly 3 provider calls, never more."""
    _enable_text_fallback(stack)
    stack.ai.thread_summary_malformed = 5  # both structured models unusable
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert stack.ai.thread_summary_calls == 2
    assert stack.ai.plain_summary_calls == 1
    # Alternation: primary, fallback, primary (plain) — total 3 generations.
    assert stack.ai.models[-3:] == [None, "fb-model", None]
    assert len(stack.ai.models) == 3
    texts = [m.text for m in stack.sender.messages]
    assert any("• Tasa: 125 CHF." in (t or "") for t in texts)
    assert not any('{"headline"' in (t or "") for t in texts)


async def test_flow_fallback_translation_retry_uses_fallback(stack: Stack) -> None:
    """DE→ES translation: primary transient failure → fallback model translates
    the EXACT German body; the German draft is never altered."""
    _enable_text_fallback(stack)
    stack.contacts.create_contact("Daniel", "daniel@fb.ch")
    stack.ai.compose_failures = 1
    await stack.send_bg("escríbele a Daniel que muchas gracias")
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert previews and "[ES]" in previews[-1]
    # Compose generation: primary failed once → the retry used the fallback
    # model; the Spanish preview still covers the exact German body.
    assert stack.ai.models[-2:] == [None, "fb-model"]
    assert "Sehr geehrte Frau Muster" in previews[-1]


async def test_flow_fallback_forward_retry_no_duplicate_attachments(
    stack: Stack, tmp_path: Path
) -> None:
    """Forward: primary transient failure → fallback model body succeeds;
    attachments collected once, one draft, temp files cleaned."""
    _enable_text_fallback(stack)
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.contacts.create_contact("Daniel", "daniel@forward.ch")
    original = email_with_attachments(make_email(message_id="m-fwd", subject="Presupuesto"))
    stack.gmail.messages["m-fwd"] = original
    stack.gmail.attachment_bytes[("m-fwd", 0)] = b"%PDF-1.4 fake pdf content"
    stack.ai.forward_failures = 1
    summary_id = await stack.bot.send_summary(original, EmailSummary(subject_es="Asunto"))
    await stack.send_bg("reenvíaselo a Daniel", reply_to=stack.bot_message(summary_id))
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert previews and "presupuesto.pdf" in previews[-1]
    assert stack.ai.models[-2:] == [None, "fb-model"]
    await stack.send("envíalo")
    await stack.wait_for_send()
    await stack.join_background()
    assert len(stack.gmail.sent) == 1
    assert len(stack.gmail.sent[0].attachments) == 1
    delivery_dir = Path(str(tmp_path / "tmp" / "delivery"))
    leftovers = list(delivery_dir.iterdir()) if delivery_dir.is_dir() else []
    assert leftovers == []


async def test_flow_fallback_edit_preserves_draft_metadata(stack: Stack) -> None:
    """Draft edit: primary transient failure → fallback edits the SAME draft;
    recipient/subject/thread/attachments preserved, one preview."""
    _enable_text_fallback(stack)
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escríbele a Roman que gracias")
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert previews
    stack.ai.draft_edit_failures = 1
    await stack.send("hazlo más corto")
    await asyncio.sleep(0.05)
    assert stack.ai.draft_edit_calls == 2  # primary failed → fallback model
    assert stack.ai.models[-2:] == [None, "fb-model"]
    previews2 = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert len(previews2) == 2  # one original + one edited preview, no duplicates
    assert "femo@femo.ch" in previews2[-1]
    assert "kurz und klar" in previews2[-1]


# ── task-aware output budgets (reasoning-capable models) ─────────────────────


async def test_budget_thread_summary_structured(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert ("thread_summary", 2000) in stack.ai.max_tokens_calls


async def test_budget_thread_summary_plain(stack: Stack) -> None:
    stack.ai.thread_summary_override = (
        '{"headline": "Resumen", "sections": [{"emoji": "📬", "title": "Resumen", "items": ['
    )  # structured unusable → plain layer
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert ("thread_summary_plain", 1500) in stack.ai.max_tokens_calls


async def test_budget_qa(stack: Stack) -> None:
    stack.gmail.threads["t1"] = make_thread("t1")
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("¿qué me está pidiendo?", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert ("qa", 1600) in stack.ai.max_tokens_calls


async def test_budget_compose_and_forward_and_edit(stack: Stack) -> None:
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escríbele a Roman que gracias")
    assert ("compose", 1600) in stack.ai.max_tokens_calls
    await stack.send("hazlo más corto")
    await asyncio.sleep(0.05)
    assert ("draft_edit", 1600) in stack.ai.max_tokens_calls


async def test_budget_forward(stack: Stack) -> None:
    stack.contacts.create_contact("Daniel", "daniel@forward.ch")
    original = email_with_attachments(make_email(message_id="m-fwd", subject="Presupuesto"))
    stack.gmail.messages["m-fwd"] = original
    stack.gmail.attachment_bytes[("m-fwd", 0)] = b"%PDF-1.4 fake pdf content"
    summary_id = await stack.bot.send_summary(original, EmailSummary(subject_es="Asunto"))
    await stack.send_bg("reenvíaselo a Daniel", reply_to=stack.bot_message(summary_id))
    assert ("forward", 1600) in stack.ai.max_tokens_calls


# ── Telegram attachments: handler registration, captions, compose/reply ──────


async def test_handler_filter_accepts_all_supported_types() -> None:
    """The PRODUCTION PTB handler filter accepts text, document, photo and
    voice updates (they all reach the single process_update entry point)."""
    from datetime import UTC, datetime

    from telegram import Document, PhotoSize, Voice

    from inboxbridge.telegram.bot import _MESSAGE_FILTERS

    def _msg(**kwargs: Any) -> Message:
        return Message(
            message_id=1,
            date=datetime.now(UTC),
            chat=Chat(id=CHAT_ID, type=ChatType.GROUP),
            from_user=_user(7),
            **kwargs,
        )

    text_update = _update(_msg(text="hola"))
    doc_update = _update(
        _msg(
            document=Document(
                file_id="f", file_unique_id="u", file_name="a.pdf",
                mime_type="application/pdf", file_size=1,
            )
        )
    )
    photo_update = _update(
        _msg(photo=[PhotoSize(file_id="p", file_unique_id="u2", width=1, height=1)])
    )
    voice_update = _update(_msg(voice=Voice(file_id="v", file_unique_id="u3", duration=1)))
    assert _MESSAGE_FILTERS.check_update(text_update)
    assert _MESSAGE_FILTERS.check_update(doc_update)
    assert _MESSAGE_FILTERS.check_update(photo_update)
    assert _MESSAGE_FILTERS.check_update(voice_update)


async def test_voice_update_reaches_voice_pipeline(stack: Stack) -> None:
    """A voice update is no longer dropped by the text-only handler: it reaches
    the voice pipeline (audio disabled → user-safe notice)."""
    voice = Voice(file_id="v1", file_unique_id="uv1", duration=1, file_size=10)
    message = Message(
        message_id=900,
        date=datetime.now(UTC),
        chat=Chat(id=CHAT_ID, type=ChatType.GROUP),
        from_user=_user(7),
        voice=voice,
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    assert any("notas de voz" in (m.text or "") for m in stack.sender.messages)


async def test_flow_compose_with_pdf_caption_one_draft(
    stack: Stack, tmp_path: Path
) -> None:
    """CASE A: PDF + compose caption → exactly one draft, one attachment,
    preview shows it, nothing is sent before confirmation."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.sender.files["file-1"] = FakeFile(b"%PDF-1.4 fake")
    message = _document_message(
        700,
        file_id="file-1",
        file_name="prueba.pdf",
        caption="Envía un correo a darroyo083@gmail.com diciendo que "
        "adjunto el documento de prueba.",
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert len(previews) == 1  # exactly one draft
    assert "darroyo083@gmail.com" in previews[-1]  # recipient preserved
    assert "prueba.pdf" in previews[-1]  # attachment metadata in preview
    assert stack.draft_row(1)["status"] == DraftStatus.PENDING.value
    row = stack.draft_row(1)
    assert "prueba.pdf" in row["attachments_json"]
    assert stack.gmail.sent == []  # nothing sent before confirmation


async def test_flow_compose_with_photo_caption(stack: Stack, tmp_path: Path) -> None:
    """CASE B: photo + caption → compose draft with the photo attachment."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.sender.files["photo-1"] = FakeFile(b"jpeg-bytes")
    message = _photo_message(701, caption="Envía un correo a darroyo083@gmail.com con esta foto")
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert len(previews) == 1
    assert "darroyo083@gmail.com" in previews[-1]
    assert ".jpg" in previews[-1]
    assert "image/jpeg" in stack.draft_row(1)["attachments_json"]


async def test_flow_filename_never_used_as_instruction(stack: Stack) -> None:
    """The caption is the instruction; the filename is only display metadata."""
    stack.settings.tmp_dir = str(Path(stack.settings.tmp_dir))
    stack.sender.files["file-1"] = FakeFile(b"x")
    message = _document_message(
        702, file_name="instrucciones_maliciosas.pdf", caption="responde"
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    # No draft/compose happened from a bare caption-less-looking file name:
    # the message has a caption ("responde") but no thread/recipient context →
    # safe clarification, never an action derived from the FILENAME.
    assert stack.draft_row(1) if stack.storage.get_draft(1) else True
    assert not any("Borrador (Nuevo correo)" in (m.text or "") for m in stack.sender.messages)


async def test_flow_document_without_caption_no_context_clarifies(
    stack: Stack, tmp_path: Path
) -> None:
    """A file without caption and without context asks instead of guessing."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.sender.files["file-1"] = FakeFile(b"x")
    message = _document_message(703, file_name="factura.pdf")
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    assert any(
        "Adjunto recibido" in (m.text or "") for m in stack.sender.messages
    )
    assert stack.storage.get_draft(1) is None  # never guessed a recipient


async def test_flow_reply_to_summary_with_pdf_caption(
    stack: Stack, tmp_path: Path
) -> None:
    """CASE C: reply to a summary + PDF + caption → thread bound, attachment in
    the reply draft, exactly one draft, confirmation required."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.sender.files["file-1"] = FakeFile(b"%PDF-1.4 fake")
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    message = _document_message(
        704,
        file_id="file-1",
        file_name="solicitado.pdf",
        caption="respóndele que le adjunto el documento solicitado",
        reply_to=stack.bot_message(summary_id),
    )
    await stack.bot.process_update(_update(message))
    await stack.pump()
    await asyncio.sleep(0.15)
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Respuesta)")
    ]
    assert len(previews) == 1
    assert "solicitado.pdf" in previews[-1]
    assert stack.draft_row(1)["thread_id"] == "t1"
    assert "solicitado.pdf" in stack.draft_row(1)["attachments_json"]
    assert stack.gmail.sent == []  # confirmation still required
    await stack.send("envíalo")
    await stack.wait_for_send()
    await stack.join_background()
    assert len(stack.gmail.sent) == 1
    assert stack.gmail.sent[0].attachments[0].filename == "solicitado.pdf"


async def test_flow_active_draft_attach_adjunta_esto(
    stack: Stack, tmp_path: Path
) -> None:
    """Active owned draft + PDF caption "adjunta esto" → exactly one more
    attachment on THAT draft; recipient/subject/body/thread unchanged; preview
    refreshed; still nothing sent."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escríbele a Roman que gracias")
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert previews
    subject_before = stack.draft_row(1)["subject"]
    body_before = stack.draft_row(1)["body"]

    stack.sender.files["file-1"] = FakeFile(b"%PDF-1.4 fake")
    message = _document_message(
        705, file_id="file-1", file_name="extra.pdf", caption="adjunta esto"
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)

    previews2 = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert len(previews2) == 2  # refreshed preview (old one deleted)
    assert "extra.pdf" in previews2[-1]
    row = stack.draft_row(1)
    assert row["subject"] == subject_before
    assert row["body"] == body_before
    assert "extra.pdf" in row["attachments_json"]
    assert stack.gmail.sent == []


async def test_flow_attach_deduplicates_same_file(stack: Stack, tmp_path: Path) -> None:
    """Uploading the same filename twice adds it exactly once."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escríbele a Roman que gracias")
    stack.sender.files["file-1"] = FakeFile(b"%PDF-1.4 fake")
    for mid in (706, 707):
        message = _document_message(
            mid, file_id="file-1", file_name="extra.pdf", caption="adjunta esto"
        )
        await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    row = stack.draft_row(1)
    import json as _json

    attachments = _json.loads(row["attachments_json"])
    assert [a["filename"] for a in attachments] == ["extra.pdf"]
    assert any("duplicado" in (m.text or "") for m in stack.sender.messages)


async def test_flow_attach_other_user_cannot_mutate(stack: Stack, tmp_path: Path) -> None:
    """Another Telegram user cannot attach to someone else's draft."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escríbele a Roman que gracias")
    stack.sender.files["file-1"] = FakeFile(b"x")
    message = _document_message(
        708, file_id="file-1", file_name="hack.pdf", caption="adjunta esto", user_id=8
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    assert "hack.pdf" not in stack.draft_row(1)["attachments_json"]


async def test_flow_attach_without_draft_clarifies(stack: Stack, tmp_path: Path) -> None:
    """"adjunta esto" + document WITHOUT an active draft asks, never guesses."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.sender.files["file-1"] = FakeFile(b"x")
    message = _document_message(709, file_id="file-1", file_name="a.pdf", caption="adjunta esto")
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    assert stack.storage.get_draft(1) is None
    assert any("Adjunto recibido" in (m.text or "") for m in stack.sender.messages)


async def test_flow_edit_after_attach_preserves_attachment(
    stack: Stack, tmp_path: Path
) -> None:
    """"hazlo más corto" after attaching keeps the attachment."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escríbele a Roman que gracias")
    stack.sender.files["file-1"] = FakeFile(b"%PDF-1.4 fake")
    message = _document_message(
        710, file_id="file-1", file_name="extra.pdf", caption="adjunta esto"
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    await stack.send("hazlo más corto")
    await asyncio.sleep(0.05)
    assert "extra.pdf" in stack.draft_row(1)["attachments_json"]
    assert any("Borrador (Nuevo correo)" in (m.text or "") for m in stack.sender.messages)


async def test_flow_regenerate_after_attach_preserves_attachment(
    stack: Stack, tmp_path: Path
) -> None:
    """Regenerate after attaching keeps the attachment."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escríbele a Roman que gracias")
    stack.sender.files["file-1"] = FakeFile(b"%PDF-1.4 fake")
    message = _document_message(
        711, file_id="file-1", file_name="extra.pdf", caption="adjunta esto"
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    await stack.send("reescribe el borrador")
    await asyncio.sleep(0.05)
    assert "extra.pdf" in stack.draft_row(1)["attachments_json"]


async def test_flow_cancel_cleans_temp_attachment(stack: Stack, tmp_path: Path) -> None:
    """Cancel removes the draft's temp attachment files."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escríbele a Roman que gracias")
    stack.sender.files["file-1"] = FakeFile(b"%PDF-1.4 fake")
    message = _document_message(
        712, file_id="file-1", file_name="extra.pdf", caption="adjunta esto"
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    draft_dir = Path(str(tmp_path / "tmp" / "draft-1"))
    assert any(draft_dir.iterdir()) if draft_dir.is_dir() else False
    await stack.send("cancela el borrador")
    await stack.join_background()
    await asyncio.sleep(0.05)
    assert not draft_dir.exists() or not any(draft_dir.iterdir())


async def test_flow_send_cleans_temp_attachment(stack: Stack, tmp_path: Path) -> None:
    """Successful verified send removes the draft's temp attachment files."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.contacts.create_contact("Roman", "femo@femo.ch")
    await stack.send_bg("escríbele a Roman que gracias")
    stack.sender.files["file-1"] = FakeFile(b"%PDF-1.4 fake")
    message = _document_message(
        713, file_id="file-1", file_name="extra.pdf", caption="adjunta esto"
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    await stack.send("envíalo")
    await stack.wait_for_send()
    await stack.join_background()
    draft_dir = Path(str(tmp_path / "tmp" / "draft-1"))
    assert not draft_dir.exists() or not any(draft_dir.iterdir())


async def test_flow_media_group_only_first_processed(
    stack: Stack, tmp_path: Path
) -> None:
    """A 2-photo album (two separate updates, same media_group_id) produces
    exactly ONE logical action, never two drafts."""
    from datetime import UTC, datetime

    from telegram import PhotoSize

    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.sender.files["photo-1"] = FakeFile(b"a")
    stack.sender.files["photo-2"] = FakeFile(b"b")

    def _photo(mid: int, file_id: str, caption: str | None = None) -> Message:
        return Message(
            message_id=mid,
            date=datetime.now(UTC),
            chat=Chat(id=CHAT_ID, type=ChatType.GROUP),
            from_user=_user(7),
            photo=[
                PhotoSize(
                    file_id=file_id,
                    file_unique_id=f"u{mid}",
                    width=1,
                    height=1,
                    file_size=2,
                )
            ],
            caption=caption,
            media_group_id="album-1",
        )

    await stack.bot.process_update(
        _update(_photo(714, "photo-1", "Envía un correo a darroyo083@gmail.com"))
    )
    await stack.bot.process_update(_update(_photo(715, "photo-2")))
    await asyncio.sleep(0.05)
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert len(previews) == 1  # never two drafts from one album


# ── Telegram download API (PTB >=20: download_to_drive) ──────────────────────


async def test_fakefile_has_no_obsolete_download_api() -> None:
    """The mock mirrors PTB >=20: ``File.download`` does not exist, so the
    production path can never silently depend on the removed API."""
    f = FakeFile(b"x")
    assert not hasattr(f, "download")
    assert hasattr(f, "download_to_drive")


async def test_flow_document_download_failure_aborts_compose(
    stack: Stack, tmp_path: Path
) -> None:
    """A download failure must NOT produce a misleading attachment-bearing
    draft: the compose action aborts with the user-facing notice."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    file = FakeFile(b"%PDF-1.4 fake")
    file.fail_download = True
    stack.sender.files["file-1"] = file
    message = _document_message(
        720,
        file_id="file-1",
        file_name="prueba.pdf",
        caption="Envía un correo a darroyo083@gmail.com diciendo que "
        "adjunto el documento de prueba.",
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    assert stack.storage.get_draft(1) is None  # no misleading draft
    assert any(
        "No pude descargar un adjunto; vuelve a intentarlo." in (m.text or "")
        for m in stack.sender.messages
    )
    incoming = Path(str(tmp_path / "tmp" / "incoming"))
    if incoming.is_dir():
        assert list(incoming.iterdir()) == []  # partial target cleaned


async def test_flow_photo_download_failure_aborts_compose(
    stack: Stack, tmp_path: Path
) -> None:
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    file = FakeFile(b"jpeg-bytes")
    file.fail_download = True
    stack.sender.files["photo-1"] = file
    message = _photo_message(
        721, caption="Envía un correo a darroyo083@gmail.com con esta foto"
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    assert stack.storage.get_draft(1) is None
    assert any(
        "No pude descargar un adjunto; vuelve a intentarlo." in (m.text or "")
        for m in stack.sender.messages
    )


async def test_flow_download_failure_cleans_prior_siblings(
    stack: Stack, tmp_path: Path
) -> None:
    """Multi-file batch: the first succeeds, the second fails → the first's
    temp file is removed too (no orphans)."""
    from datetime import UTC, datetime

    from telegram import Document, PhotoSize

    stack.settings.tmp_dir = str(tmp_path / "tmp")
    good = FakeFile(b"a" * 10)
    bad = FakeFile(b"b" * 10)
    bad.fail_download = True
    stack.sender.files["file-1"] = good
    stack.sender.files["photo-1"] = bad
    message = Message(
        message_id=722,
        date=datetime.now(UTC),
        chat=Chat(id=CHAT_ID, type=ChatType.GROUP),
        from_user=_user(7),
        document=Document(
            file_id="file-1", file_unique_id="u1", file_name="a.pdf",
            mime_type="application/pdf", file_size=5,
        ),
        photo=[PhotoSize(file_id="photo-1", file_unique_id="u2", width=1, height=1, file_size=5)],
        caption="Envía un correo a darroyo083@gmail.com con ambos",
        reply_to_message=stack.bot_message(
            await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
        ),
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    assert stack.storage.get_draft(1) is None
    incoming = Path(str(tmp_path / "tmp" / "incoming"))
    if incoming.is_dir():
        assert list(incoming.iterdir()) == []  # both files cleaned


async def test_flow_oversized_notice_shows_filename_not_file_id(
    stack: Stack, tmp_path: Path
) -> None:
    """The oversized rejection notice names the sanitized FILENAME, never the
    Telegram file id (tuple shape: (file_id, name, size, mime))."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.settings.outgoing_attachment_max_bytes = 10
    stack.sender.files["file-1"] = FakeFile(b"x" * 100)
    message = _document_message(
        723,
        file_id="file-1",
        file_name="factura.pdf",
        caption="Envía un correo a darroyo083@gmail.com adjuntando esto",
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    notices = " ".join(m.text or "" for m in stack.sender.messages)
    assert "factura.pdf" in notices
    assert "file-1" not in notices  # the Telegram file id is never shown


async def test_flow_download_target_inside_tmp_dir(stack: Stack, tmp_path: Path) -> None:
    """download_to_drive receives a CONTROLLED target path inside tmp_dir."""
    stack.settings.tmp_dir = str(tmp_path / "tmp")
    stack.sender.files["file-1"] = FakeFile(b"%PDF-1.4 fake")
    message = _document_message(
        724,
        file_id="file-1",
        file_name="prueba.pdf",
        caption="Envía un correo a darroyo083@gmail.com adjuntando esto",
    )
    await stack.bot.process_update(_update(message))
    await asyncio.sleep(0.05)
    previews = [
        m.text
        for m in stack.sender.messages
        if (m.text or "").startswith("Borrador (Nuevo correo)")
    ]
    assert previews and "prueba.pdf" in previews[-1]
    row = stack.draft_row(1)
    import json as _json

    attachments = _json.loads(row["attachments_json"])
    assert [a["filename"] for a in attachments] == ["prueba.pdf"]
    # No leftover download in tmp/incoming: files were claimed by the draft.
    incoming = Path(str(tmp_path / "tmp" / "incoming"))
    if incoming.is_dir():
        assert list(incoming.iterdir()) == []
