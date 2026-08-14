"""Integration tests: reply flow (Telegram → thread → draft → confirm → send).

Covers: full happy path (send + Gmail reconciliation → sent_verified), kill
switch (SEND_EMAILS=false blocks sending), cancel discards the draft to
CANCELLED without sending, unknown thread notifies the user.
"""

from __future__ import annotations

import asyncio

from inboxbridge.config import Settings
from inboxbridge.db import DraftStatus, Storage
from inboxbridge.responder import ReplyCoordinator
from inboxbridge.telegram.bot import ReplyRequest
from tests.mocks.coordinator import FakeGmail, FakeReplyBot, make_thread


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"_env_file": None, "SEND_EMAILS": False}
    base.update(overrides)
    return Settings(**base)


def make_storage(tmp_path: object) -> Storage:
    storage = Storage(str(tmp_path) + "/reply.db")
    storage.connect()
    return storage


def _run_request(coordinator: ReplyCoordinator, request: ReplyRequest) -> None:
    # Drive the async handler to completion in a fresh loop.
    asyncio.run(coordinator._handle_request(request))


def _draft_rows(storage: Storage) -> list[dict[str, object]]:
    return storage.drafts_in_statuses(
        [
            DraftStatus.PENDING,
            DraftStatus.CONFIRMED,
            DraftStatus.SENDING,
            DraftStatus.SENT_UNVERIFIED,
            DraftStatus.SENT_VERIFIED,
            DraftStatus.SEND_FAILED,
            DraftStatus.CANCELLED,
        ]
    )


def test_reply_happy_path_with_confirmation(tmp_path: object) -> None:
    settings = make_settings(SEND_EMAILS=True)
    storage = make_storage(tmp_path)
    gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
    llm = MockLLM()
    bot = FakeReplyBot()

    coordinator = ReplyCoordinator(settings, gmail, llm, bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(
            thread_id="t1",
            user_instructions="Sag, dass ich am Montag Bescheid gebe.",
            source_message_id=1,
        ),
    )

    assert len(gmail.sent) == 1
    assert gmail.sent[0].thread_id == "t1"
    assert gmail.sent[0].subject == "Re: Projektbericht"
    assert "Bescheid" in gmail.sent[0].body
    assert gmail.sent[0].to[0].email == "anna@example.com"  # reply to sender, not all

    rows = _draft_rows(storage)
    assert len(rows) == 1
    # Strong success only after Gmail reconciliation.
    assert rows[0]["status"] == DraftStatus.SENT_VERIFIED.value
    assert rows[0]["sent_message_id"]
    assert any("verificado" in n for n in bot.notices)
    assert "Enviado y verificado" in " ".join(bot.notices)


def test_kill_switch_blocks_sending_and_keeps_draft(tmp_path: object) -> None:
    """SEND_EMAILS=false: confirmed draft is stored but never sent."""
    settings = make_settings(SEND_EMAILS=False)
    storage = make_storage(tmp_path)
    gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=False)
    llm = MockLLM()
    bot = FakeReplyBot()

    coordinator = ReplyCoordinator(settings, gmail, llm, bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(thread_id="t1", user_instructions="Danke sagen", source_message_id=2),
    )

    assert gmail.sent == []  # technically impossible to send
    rows = _draft_rows(storage)
    assert len(rows) == 1
    assert rows[0]["status"] == DraftStatus.CONFIRMED.value
    assert any("SEND_EMAILS=false" in n for n in bot.notices)


def test_not_confirmed_draft_is_cancelled_without_send(tmp_path: object) -> None:
    settings = make_settings(SEND_EMAILS=True)
    storage = make_storage(tmp_path)
    gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
    llm = MockLLM()
    bot = FakeReplyBot()
    bot.default_confirmation = False  # no one presses confirm

    coordinator = ReplyCoordinator(settings, gmail, llm, bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(thread_id="t1", user_instructions="Antwort", source_message_id=3),
    )

    assert gmail.sent == []
    rows = _draft_rows(storage)
    assert len(rows) == 1
    assert rows[0]["status"] == DraftStatus.CANCELLED.value
    assert any("cancelado" in n for n in bot.notices)


def test_unknown_thread_notifies_user(tmp_path: object) -> None:
    settings = make_settings()
    storage = make_storage(tmp_path)
    gmail = FakeGmail(threads={})
    bot = FakeReplyBot()

    coordinator = ReplyCoordinator(settings, gmail, MockLLM(), bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(thread_id="", user_instructions="Hola", source_message_id=4),
    )

    assert any("hilo" in n for n in bot.notices)


def test_preview_includes_spanish_translation(tmp_path: object) -> None:
    """The presented draft carries a display-only Spanish translation of the
    exact German body (derived from the body, never sent)."""
    settings = make_settings(SEND_EMAILS=True)
    storage = make_storage(tmp_path)
    gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
    llm = MockLLM()
    bot = FakeReplyBot()

    coordinator = ReplyCoordinator(settings, gmail, llm, bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=5),
    )

    shown = bot.drafts_shown[0]
    assert shown.body.startswith("Sehr geehrte Frau Muster")
    assert shown.body_es == "[ES] " + shown.body  # translation derived from the body
    # The Spanish translation is display-only: it is not persisted on the row.
    row = storage.get_draft(1)
    assert "body_es" not in row
    assert row["body"].startswith("Sehr geehrte")


class MockLLM:
    """Deterministic LLMProvider double for the responder tests."""

    async def summarize_email(self, email: object) -> object:
        from inboxbridge.models import EmailSummary

        return EmailSummary(summary_es="resumen")

    async def draft_reply(self, request: object, thread: object) -> object:
        from inboxbridge.models import DraftReply, ThreadContext

        thread_ctx: ThreadContext = thread  # type: ignore[assignment]
        return DraftReply(
            thread_id=thread_ctx.thread_id,
            subject=thread_ctx.subject,
            to=[thread_ctx.messages[0].from_] if thread_ctx.messages else [],
            cc=[],
            body=(
                "Sehr geehrte Frau Muster,\n\nich gebe Ihnen am Montag Bescheid.\n\n"
                "Mit freundlichen Grüßen"
            ),
            in_reply_to=thread_ctx.messages[-1].message_id if thread_ctx.messages else "",
            references="",
        )

    async def translate_to_spanish(self, body: str) -> str:
        return "[ES] " + body
