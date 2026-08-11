"""V1.1 simulated end-to-end flows (real stack, simulated Gmail/Telegram/AI).

Covers the goal's flows A–W: reply+edit+send, button misclick, ambiguous
ack, contact creation, new email via alias, contact ambiguity, unknown
contact, contact edit/delete, reply-vs-alias semantics, attachment download,
Q&A, thread summary, forward, mark read, archive, reminder, voice
(experimental), prompt injection, restart, concurrency.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from telegram import Chat, InlineKeyboardMarkup, Message, Voice
from telegram.constants import ChatType

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
    FakeSender,
    _callback_update,
    _message,
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
        self.vision_responses: list[str] = []
        self.vision_fail: str = ""  # "primary" | "both"
        self.vision_calls: list[str] = []
        self.audio_responses: list[str] = []
        self.audio_calls = 0
        self.calls: list[tuple[str, str]] = []  # (task, model-ish)

    async def text(self, messages: list[Any], *, max_tokens: int, task: str) -> str:
        self.calls.append((task, "deepseek-v4-flash"))
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
            return ("Sehr geehrte Frau Muster,\n\nvielen Dank für Ihre Nachricht.\n\n"
                    "Mit freundlichen Grüßen")
        if task == "draft_edit":
            return "Sehr geehrte Frau Muster,\n\nkurz und klar.\n\nMit freundlichen Grüßen"
        return self.default_text

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
    assert any("Texto de prueba" in (t or "") for t in texts)  # AI answer posted


# ── R. THREAD SUMMARY ────────────────────────────────────────────────────────


async def test_flow_r_thread_summary(stack: Stack) -> None:
    summary_id = await stack.bot.send_summary(make_email(), EmailSummary(subject_es="Asunto"))
    await stack.send("resume toda la conversación", reply_to=stack.bot_message(summary_id))
    await asyncio.sleep(0.05)
    assert any("Texto de prueba" in (m.text or "") for m in stack.sender.messages)


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
    await stack.join_background()
    assert len(stack.gmail.sent) == 1
    assert stack.gmail.sent[0].to[0].email == "daniel@forward.ch"
    assert stack.gmail.sent[0].subject.startswith("Fwd:")
    assert stack.gmail.sent[0].attachments[0].filename == "presupuesto.pdf"
    assert stack.draft_row(1)["status"] == DraftStatus.SENT_VERIFIED.value


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
