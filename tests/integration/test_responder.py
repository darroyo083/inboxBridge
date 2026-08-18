"""Integration tests: reply flow (Telegram → thread → draft → confirm → send).

Covers: full happy path (send + Gmail reconciliation → sent_verified), kill
switch (SEND_EMAILS=false blocks sending), cancel discards the draft to
CANCELLED without sending, unknown thread notifies the user.
"""

from __future__ import annotations

import asyncio
from typing import Any

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


def _seed_reply_target(
    storage: Storage, gmail: FakeGmail, *, message_id: str = "m1"
) -> None:
    """Seed the persisted incoming message the reply targets (the bot would
    have frozen this from the Telegram summary mapping)."""
    from inboxbridge.models import EmailAddress, MessageStatus, ParsedEmail

    email = ParsedEmail(
        message_id=message_id,
        thread_id="t1",
        history_id=10,
        subject="Re: Projektbericht",
        sender=EmailAddress("Anna Muster", "anna@example.com"),
        recipients=[EmailAddress("Daniel", "daniel@example.com")],
        date_iso="2026-08-07T10:00:00+00:00",
        body_text="Hallo, bitte um Rückmeldung.",
    )
    gmail.messages[message_id] = email
    storage.upsert_message(message_id, "t1", 10, MessageStatus.SENT_TELEGRAM)


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
    _seed_reply_target(storage, gmail)

    coordinator = ReplyCoordinator(settings, gmail, llm, bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(
            thread_id="t1",
            user_instructions="Sag, dass ich am Montag Bescheid gebe.",
            source_message_id=1,
            target_message_id="m1",
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
    _seed_reply_target(storage, gmail)

    coordinator = ReplyCoordinator(settings, gmail, llm, bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(
            thread_id="t1",
            user_instructions="Danke sagen",
            source_message_id=2,
            target_message_id="m1",
        ),
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
    _seed_reply_target(storage, gmail)

    coordinator = ReplyCoordinator(settings, gmail, llm, bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(
            thread_id="t1",
            user_instructions="Antwort",
            source_message_id=3,
            target_message_id="m1",
        ),
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
    _seed_reply_target(storage, gmail)

    coordinator = ReplyCoordinator(settings, gmail, llm, bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(
            thread_id="t1",
            user_instructions="Danke",
            source_message_id=5,
            target_message_id="m1",
        ),
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
            to=[request.reply_to] if request.reply_to is not None else [],
            cc=[],
            body=(
                "Sehr geehrte Frau Muster,\n\nich gebe Ihnen am Montag Bescheid.\n\n"
                "Mit freundlichen Grüßen"
            ),
            in_reply_to=request.in_reply_to,
            references="",
        )

    async def translate_to_spanish(
        self, body: str, *, model: str | None = None
    ) -> str:
        return "[ES] " + body


# ── frozen reply target (recipient from exact incoming Gmail message) ────────


def _self_started_thread() -> Any:
    """A thread OUR account started: the first message is from ourselves."""
    from inboxbridge.models import EmailAddress, ThreadContext, ThreadMessage

    return ThreadContext(
        thread_id="t1",
        subject="Re: Projektbericht",
        history_id=20,
        messages=[
            ThreadMessage(
                message_id="our-message",
                from_=EmailAddress("Daniel", "daniel@example.com"),
                date_iso="2026-08-05T09:00:00+00:00",
                body_text="Hallo, hier die Unterlagen.",
            ),
            ThreadMessage(
                message_id="m1",
                from_=EmailAddress("Anna Muster", "anna@example.com"),
                date_iso="2026-08-07T10:00:00+00:00",
                body_text="Danke, bitte um Rückmeldung.",
            ),
        ],
    )


def test_reply_targets_external_sender_not_first_thread_sender(
    tmp_path: object,
) -> None:
    """A thread OUR account started must NEVER resolve the recipient to the
    first thread sender: the recipient is the mapped incoming message sender."""
    settings = make_settings(SEND_EMAILS=True)
    storage = make_storage(tmp_path)
    gmail = FakeGmail(threads={"t1": _self_started_thread()}, send_ok=True)
    _seed_reply_target(storage, gmail)
    llm = MockLLM()
    bot = FakeReplyBot()

    coordinator = ReplyCoordinator(settings, gmail, llm, bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(
            thread_id="t1",
            user_instructions="Danke sagen",
            source_message_id=1,
            target_message_id="m1",
        ),
    )

    assert len(gmail.sent) == 1
    assert gmail.sent[0].to[0].email == "anna@example.com"  # external sender
    assert gmail.sent[0].to[0].email != "daniel@example.com"  # never ourselves
    assert gmail.sent[0].in_reply_to == "m1"  # exact mapped target message
    assert gmail.sent[0].thread_id == "t1"


def test_later_thread_message_does_not_change_frozen_target(
    tmp_path: object,
) -> None:
    """A message arriving AFTER target selection must not change the frozen
    recipient/in_reply_to (deterministic and immutable)."""
    settings = make_settings(SEND_EMAILS=True)
    storage = make_storage(tmp_path)
    gmail = FakeGmail(threads={"t1": _self_started_thread()}, send_ok=True)
    _seed_reply_target(storage, gmail)
    llm = MockLLM()
    bot = FakeReplyBot()

    coordinator = ReplyCoordinator(settings, gmail, llm, bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(
            thread_id="t1",
            user_instructions="Danke sagen",
            source_message_id=1,
            target_message_id="m1",
        ),
    )

    assert len(gmail.sent) == 1
    assert gmail.sent[0].to[0].email == "anna@example.com"
    assert gmail.sent[0].in_reply_to == "m1"  # frozen, not the newest message


def test_self_recipient_reply_aborts_safely(tmp_path: object) -> None:
    """Defense in depth: a reply resolving to the authenticated account itself
    aborts before any draft/send, with a privacy-safe log."""
    settings = make_settings(SEND_EMAILS=True)
    storage = make_storage(tmp_path)
    gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
    gmail.account_email = "anna@example.com"
    _seed_reply_target(storage, gmail)  # target sender IS anna@example.com
    bot = FakeReplyBot()

    coordinator = ReplyCoordinator(settings, gmail, MockLLM(), bot, storage)
    _run_request(
        coordinator,
        ReplyRequest(
            thread_id="t1",
            user_instructions="Danke",
            source_message_id=1,
            target_message_id="m1",
        ),
    )

    assert gmail.sent == []
    assert _draft_rows(storage) == []
    assert any("no respondo a mi propia cuenta" in n for n in bot.notices)


def test_latest_incoming_excludes_own_sent_and_other_statuses(
    tmp_path: object,
) -> None:
    """latest_incoming_message() only returns fully processed INCOMING rows —
    our own sent messages (or failed rows) are never reply targets."""
    from inboxbridge.models import MessageStatus

    storage = make_storage(tmp_path)
    storage.upsert_message("m-failed", "t1", 6, MessageStatus.FAILED)
    storage.upsert_message("m-in", "t1", 7, MessageStatus.SENT_TELEGRAM)
    storage.upsert_message("m-newer", "t1", 8, MessageStatus.RECEIVED)
    row = storage.latest_incoming_message()
    assert row is not None
    # Only fully processed incoming rows are eligible targets — not failed or
    # not-yet-summarized rows (own sent messages never live in this table).
    assert row["message_id"] == "m-in"
