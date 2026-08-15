"""Verified-delivery integration tests (coordinator + FakeGmail + FakeReplyBot).

Covers the goal's hard requirements:

- strong success ONLY after Gmail reconciliation (never from an HTTP return);
- ambiguous send (timeout after transmission) → reconcile → found → verified,
  with NO second send;
- ambiguous send → not found → controlled retry (never blind);
- ambiguous + inconclusive → never resend;
- definitive failure → send_failed + retry button → safe retry;
- resend already-sent → no duplicate;
- double confirmation → single send;
- restart recovery: sent_unverified/sending drafts reconciled, not resent;
- temp attachment cleanup on terminal states;
- orphan temp sweep.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from inboxbridge.config import Settings
from inboxbridge.db import DraftStatus, Storage
from inboxbridge.models import DraftReply, EmailAddress, OutgoingAttachment, ThreadContext
from inboxbridge.responder import ReplyCoordinator
from inboxbridge.telegram.bot import ReplyRequest
from tests.mocks.coordinator import FakeGmail, FakeReplyBot, make_thread


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "SEND_EMAILS": True,
        "send_verification_attempts": 2,
        "send_verification_backoff_seconds": 0.01,
    }
    base.update(overrides)
    return Settings(**base)


def make_storage(tmp_path: Path) -> Storage:
    storage = Storage(str(tmp_path / "state.sqlite"))
    storage.connect()
    return storage


def make_coordinator(
    tmp_path: Path,
    gmail: FakeGmail,
    bot: FakeReplyBot | None = None,
    settings: Settings | None = None,
) -> tuple[ReplyCoordinator, FakeReplyBot, Storage]:
    storage = make_storage(tmp_path)
    fake_bot = bot or FakeReplyBot()
    coordinator = ReplyCoordinator(
        settings or make_settings(tmp_dir=str(tmp_path / "tmp")),
        gmail,
        MockLLM(),
        fake_bot,
        storage,
    )
    return coordinator, fake_bot, storage


def run_request(coordinator: ReplyCoordinator, request: ReplyRequest) -> None:
    asyncio.run(coordinator._handle_request(request))


def status_of(storage: Storage, draft_id: int) -> str:
    row = storage.get_draft(draft_id)
    assert row is not None
    return row["status"]


def write_attachment(tmp_path: Path, name: str = "factura.pdf") -> OutgoingAttachment:
    path = tmp_path / "tmp" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 fake content")
    return OutgoingAttachment(
        filename=name, mime_type="application/pdf", size_bytes=path.stat().st_size, path=str(path)
    )


class TestVerifiedDelivery:
    def test_success_is_only_reported_after_reconciliation(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=1),
        )
        row = storage.get_draft(1)
        assert row["status"] == DraftStatus.SENT_VERIFIED.value
        assert row["sent_message_id"]  # Gmail message id recorded
        assert len(gmail.sent) == 1
        assert any("verificado" in n for n in bot.notices)

    def test_incomplete_reply_never_presents_sendable_draft(self, tmp_path: Path) -> None:
        """A truncated (incomplete) reply body must never reach the draft flow:
        no draft row, no send, and a safe 'incomplete' notice."""
        from inboxbridge.llm.base import LLMIncompleteResponse

        class IncompleteLLM(MockLLM):
            async def draft_reply(self, request: object, thread: object) -> object:
                raise LLMIncompleteResponse("truncated body")

        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        storage = make_storage(tmp_path)
        bot = FakeReplyBot()
        coordinator = ReplyCoordinator(
            make_settings(tmp_dir=str(tmp_path / "tmp")),
            gmail,
            IncompleteLLM(),
            bot,
            storage,
        )
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=99),
        )
        assert any("incompleta" in n for n in bot.notices)
        assert storage.drafts_in_statuses(
            [DraftStatus.PENDING, DraftStatus.CONFIRMED, DraftStatus.SENT_VERIFIED]
        ) == []
        assert gmail.sent == []

    def test_ambiguous_timeout_reconciliation_finds_message_no_second_send(
        self, tmp_path: Path
    ) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.send_error = "ambiguous"  # client receives timeout AFTER Gmail accepted
        gmail.verify_delay_ok = False  # message is immediately visible in Gmail
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=2),
        )
        row = storage.get_draft(1)
        assert row["status"] == DraftStatus.SENT_VERIFIED.value
        assert len(gmail.sent_store) == 1  # exactly ONE accepted message
        assert any("verificado" in n for n in bot.notices)

    def test_ambiguous_timeout_message_visible_after_delay_verified_without_resend(
        self, tmp_path: Path
    ) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.send_error = "ambiguous"
        gmail.verify_delay_ok = True  # first verification: not found; later: found
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=3),
        )
        row = storage.get_draft(1)
        assert row["status"] == DraftStatus.SENT_VERIFIED.value
        assert len(gmail.sent_store) == 1  # still no second send
        assert not bot.resend_offers  # verified before offering retry

    def test_ambiguous_not_found_offers_controlled_retry_only(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.send_error = "ambiguous"
        gmail.ambiguous_accepts = False  # Gmail never received the message
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=4),
        )
        row = storage.get_draft(1)
        assert row["status"] == DraftStatus.SENT_UNVERIFIED.value
        assert len(gmail.sent_store) == 0  # Gmail never accepted; no blind resend
        assert bot.resend_offers == [1]  # controlled retry offered (user-driven)

        # User presses "Reintentar envío" → re-verify → still not found → safe resend.
        gmail.send_error = ""  # transient network condition cleared
        asyncio.run(coordinator.resend_draft(1))
        row = storage.get_draft(1)
        assert row["status"] == DraftStatus.SENT_VERIFIED.value
        assert len(gmail.sent_store) == 1  # exactly one additional, evidence-based send

    def test_ambiguous_inconclusive_never_resends(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.send_error = "ambiguous"
        gmail.verify_error = True  # Gmail unreachable → inconclusive
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=5),
        )
        row = storage.get_draft(1)
        assert row["status"] == DraftStatus.SENT_UNVERIFIED.value
        assert len(gmail.sent_store) == 1
        assert bot.resend_offers == []  # no blind retry on inconclusive evidence
        assert any("duplicados" in n for n in bot.notices)

    def test_definitive_failure_reports_and_retry_is_safe(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.send_error = "definitive"
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=6),
        )
        assert storage.get_draft(1)["status"] == DraftStatus.SEND_FAILED.value
        assert bot.resend_offers == [1]

        gmail.send_error = ""  # transient condition cleared
        asyncio.run(coordinator.resend_draft(1))
        assert storage.get_draft(1)["status"] == DraftStatus.SENT_VERIFIED.value
        assert len(gmail.sent) == 1

    def test_resend_of_already_sent_draft_does_not_duplicate(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=7),
        )
        assert len(gmail.sent) == 1
        # A late retry request arrives; Gmail already has the message.
        asyncio.run(coordinator.resend_draft(1))
        assert len(gmail.sent) == 1  # no duplicate
        assert storage.get_draft(1)["status"] == DraftStatus.SENT_VERIFIED.value

    def test_resend_of_non_retryable_status_is_ignored(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=8),
        )
        asyncio.run(coordinator.resend_draft(1))  # status = sent_verified
        assert len(gmail.sent) == 1
        assert any("reintento" in n.lower() for n in bot.notices)

    def test_double_confirmation_does_not_double_send(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        # Confirm twice: wait_for_confirmation resolves once; the second call
        # must not produce another send.
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=9),
        )
        assert len(gmail.sent) == 1
        assert storage.get_draft(1)["status"] == DraftStatus.SENT_VERIFIED.value

    def test_double_tap_resend_sends_exactly_once(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.send_error = "definitive"
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        run_request(
            coordinator,
            ReplyRequest(thread_id="t1", user_instructions="Danke", source_message_id=13),
        )
        assert storage.get_draft(1)["status"] == DraftStatus.SEND_FAILED.value

        gmail.send_error = ""
        # Two retry taps arrive "concurrently": the second must be refused by
        # the atomic claim (no duplicate email).
        async def double_tap() -> None:
            await asyncio.gather(
                coordinator.resend_draft(1),
                coordinator.resend_draft(1),
            )

        asyncio.run(double_tap())
        assert len(gmail.sent) == 1
        assert storage.get_draft(1)["status"] == DraftStatus.SENT_VERIFIED.value
        # The second tap was refused (either while the first was in flight, or
        # because the draft is no longer retryable).
        assert any(
            "proceso de reenvío" in n or "no está en estado de reintento" in n
            for n in bot.notices
        )

    def test_claim_fails_when_draft_not_in_allowed_state(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        draft = DraftReply(
            thread_id="t1",
            subject="Re: Projektbericht",
            to=[EmailAddress("Anna Muster", "anna@example.com")],
            cc=[],
            body="Danke!",
        )
        draft_id = storage.create_draft("t1", None, draft)
        # Verified drafts are not claimable for another send.
        storage.set_draft_status(draft_id, DraftStatus.SENT_VERIFIED)
        assert not storage.claim_draft_for_send(draft_id, [DraftStatus.SEND_FAILED])
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SENT_VERIFIED.value


class TestRestartRecovery:
    def test_startup_reconciles_sent_unverified_to_verified_without_resend(
        self, tmp_path: Path
    ) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        # Simulate: a previous process sent successfully but crashed before
        # verifying; the draft row is stuck in sent_unverified with no message id.
        storage = make_storage(tmp_path)
        draft = DraftReply(
            thread_id="t1",
            subject="Re: Projektbericht",
            to=[EmailAddress("Anna Muster", "anna@example.com")],
            cc=[],
            body="Danke!",
        )
        draft_id = storage.create_draft("t1", None, draft, telegram_user_id=7)
        storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
        storage.set_draft_send_started(draft_id, int(time.time() * 1000))
        # Gmail accepted the message (the ambiguous send actually went through).
        gmail.sent_store.append(
            gmail.accept(draft)  # type: ignore[attr-defined]
        )

        bot = FakeReplyBot()
        coordinator = ReplyCoordinator(
            make_settings(tmp_dir=str(tmp_path / "tmp")), gmail, MockLLM(), bot, storage
        )
        asyncio.run(coordinator.reconcile_on_startup())
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SENT_VERIFIED.value
        assert gmail.sent == []  # nothing resent
        assert any("confirmado" in n for n in bot.notices)

    def test_startup_reconciles_sending_to_failed_when_never_sent(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        storage = make_storage(tmp_path)
        draft = DraftReply(
            thread_id="t1",
            subject="Re: Projektbericht",
            to=[EmailAddress("Anna Muster", "anna@example.com")],
            cc=[],
            body="Danke!",
        )
        draft_id = storage.create_draft("t1", None, draft)
        storage.set_draft_status(draft_id, DraftStatus.SENDING)
        storage.set_draft_send_started(draft_id, int(time.time() * 1000))
        # Gmail never received anything.

        bot = FakeReplyBot()
        coordinator = ReplyCoordinator(
            make_settings(tmp_dir=str(tmp_path / "tmp")), gmail, MockLLM(), bot, storage
        )
        asyncio.run(coordinator.reconcile_on_startup())
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SEND_FAILED.value
        assert gmail.sent == []
        assert bot.resend_offers == [draft_id]  # user may retry safely

    def test_sweep_verifies_stuck_draft_once_gmail_becomes_visible(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        storage = make_storage(tmp_path)
        draft = DraftReply(
            thread_id="t1",
            subject="Re: Projektbericht",
            to=[EmailAddress("Anna Muster", "anna@example.com")],
            cc=[],
            body="Danke!",
        )
        draft_id = storage.create_draft("t1", None, draft)
        storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
        storage.set_draft_send_started(draft_id, int(time.time() * 1000))

        bot = FakeReplyBot()
        coordinator = ReplyCoordinator(
            make_settings(tmp_dir=str(tmp_path / "tmp")), gmail, MockLLM(), bot, storage
        )
        # Gmail eventually shows the message.
        gmail.sent_store.append(gmail.accept(draft))  # type: ignore[attr-defined]
        asyncio.run(coordinator.sweep_unverified())
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SENT_VERIFIED.value
        assert gmail.sent == []

    def test_sweep_exhausted_draft_watchdog_marks_terminal_without_resend(
        self, tmp_path: Path
    ) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        storage = make_storage(tmp_path)
        draft = DraftReply(
            thread_id="t1",
            subject="Re: Projektbericht",
            to=[EmailAddress("Anna Muster", "anna@example.com")],
            cc=[],
            body="Danke!",
        )
        draft_id = storage.create_draft("t1", None, draft)
        storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
        for _ in range(999):  # exhausted verification budget
            storage.bump_verification_attempts(draft_id)
        bot = FakeReplyBot()
        coordinator = ReplyCoordinator(
            make_settings(tmp_dir=str(tmp_path / "tmp")), gmail, MockLLM(), bot, storage
        )
        asyncio.run(coordinator.sweep_unverified())
        assert storage.get_draft(draft_id)["status"] == DraftStatus.VERIFICATION_FAILED.value
        assert gmail.sent == []  # never resent
        assert any("duplicados" in n for n in bot.notices)  # one definitive notice


class TestWatchdog:
    """sent_unverified watchdog: exhaustion → definitive one-shot 'could not
    verify', never a resend."""

    @staticmethod
    def _draft() -> DraftReply:
        return DraftReply(
            thread_id="t1",
            subject="Re: Projektbericht",
            to=[EmailAddress("Anna Muster", "anna@example.com")],
            cc=[],
            body="Danke!",
        )

    @staticmethod
    def _seed(
        storage: Storage,
        *,
        attempts: int,
        started: bool = True,
        user_id: int = 7,
    ) -> int:
        draft_id = storage.create_draft("t1", None, TestWatchdog._draft(), telegram_user_id=user_id)
        storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
        if started:
            storage.set_draft_send_started(draft_id, int(time.time() * 1000))
        for _ in range(attempts):
            storage.bump_verification_attempts(draft_id)
        return draft_id

    @staticmethod
    def _coordinator(
        tmp_path: Path, gmail: FakeGmail, bot: FakeReplyBot, storage: Storage
    ) -> ReplyCoordinator:
        return ReplyCoordinator(
            make_settings(
                tmp_dir=str(tmp_path / "tmp"), send_verification_max_attempts=3
            ),
            gmail,
            MockLLM(),
            bot,
            storage,
        )

    def test_below_threshold_is_not_terminal(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.verify_error = True  # inconclusive → stays sent_unverified
        storage = make_storage(tmp_path)
        draft_id = self._seed(storage, attempts=0)
        bot = FakeReplyBot()
        coordinator = self._coordinator(tmp_path, gmail, bot, storage)
        asyncio.run(coordinator.sweep_unverified())
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SENT_UNVERIFIED.value
        assert not any("duplicados" in n for n in bot.notices)  # no terminal warning yet

    def test_exhausted_reaches_threshold_one_shot_warning(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.verify_error = True
        storage = make_storage(tmp_path)
        draft_id = self._seed(storage, attempts=3)  # >= max
        bot = FakeReplyBot()
        coordinator = self._coordinator(tmp_path, gmail, bot, storage)

        asyncio.run(coordinator.sweep_unverified())
        assert storage.get_draft(draft_id)["status"] == DraftStatus.VERIFICATION_FAILED.value
        assert gmail.sent == []  # never resent
        assert sum("duplicados" in n for n in bot.notices) == 1

        # A second sweep must not re-notify (draft is no longer sent_unverified).
        asyncio.run(coordinator.sweep_unverified())
        assert sum("duplicados" in n for n in bot.notices) == 1

    def test_verifies_before_threshold_no_warning(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        storage = make_storage(tmp_path)
        draft_id = self._seed(storage, attempts=0)
        gmail.sent_store.append(gmail.accept(self._draft()))  # Gmail has it after all
        bot = FakeReplyBot()
        coordinator = self._coordinator(tmp_path, gmail, bot, storage)
        asyncio.run(coordinator.sweep_unverified())
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SENT_VERIFIED.value
        assert not any("duplicados" in n for n in bot.notices)

    def test_restart_recovery_same_safe_behavior(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.verify_error = True
        storage = make_storage(tmp_path)
        draft_id = self._seed(storage, attempts=3)
        # Simulated restart: a fresh coordinator over the same store.
        bot = FakeReplyBot()
        coordinator = self._coordinator(tmp_path, gmail, bot, storage)
        asyncio.run(coordinator.reconcile_on_startup())  # re-verifies (inconclusive)
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SENT_UNVERIFIED.value
        asyncio.run(coordinator.sweep_unverified())  # watchdog → terminal
        assert storage.get_draft(draft_id)["status"] == DraftStatus.VERIFICATION_FAILED.value
        assert gmail.sent == []
        assert sum("duplicados" in n for n in bot.notices) == 1

    def test_legacy_row_without_send_time_is_not_terminal(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.verify_error = True
        storage = make_storage(tmp_path)
        draft_id = self._seed(storage, attempts=0, started=False)  # no send_started_at
        bot = FakeReplyBot()
        coordinator = self._coordinator(tmp_path, gmail, bot, storage)
        asyncio.run(coordinator.sweep_unverified())
        # Below threshold → reconciled (inconclusive), NOT immediately terminal.
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SENT_UNVERIFIED.value
        assert not any("duplicados" in n for n in bot.notices)

    def test_multiple_exhausted_drafts_handled_independently(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.verify_error = True
        storage = make_storage(tmp_path)
        first = self._seed(storage, attempts=3, user_id=7)
        second = self._seed(storage, attempts=3, user_id=8)
        bot = FakeReplyBot()
        coordinator = self._coordinator(tmp_path, gmail, bot, storage)
        asyncio.run(coordinator.sweep_unverified())
        assert storage.get_draft(first)["status"] == DraftStatus.VERIFICATION_FAILED.value
        assert storage.get_draft(second)["status"] == DraftStatus.VERIFICATION_FAILED.value
        assert storage.get_draft(first)["telegram_user_id"] == 7  # owner preserved
        assert storage.get_draft(second)["telegram_user_id"] == 8
        assert sum("duplicados" in n for n in bot.notices) == 2


class TestAttachmentLifecycle:
    def test_attachments_claimed_then_cleaned_after_verified_send(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        attachment = write_attachment(tmp_path)
        run_request(
            coordinator,
            ReplyRequest(
                thread_id="t1",
                user_instructions="Danke",
                source_message_id=10,
                attachments=(attachment,),
            ),
        )
        # The attachment was included in the sent draft.
        assert len(gmail.sent) == 1
        assert gmail.sent[0].attachments[0].filename == "factura.pdf"
        assert storage.get_draft(1)["status"] == DraftStatus.SENT_VERIFIED.value
        # Claimed into the draft's temp dir, then deleted after verification.
        draft_dir = Path(str(tmp_path / "tmp" / "draft-1"))
        assert not draft_dir.exists()
        assert not Path(attachment.path).exists()

    def test_cancellation_cleans_attachments_and_never_sends(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        bot = FakeReplyBot()
        bot.default_confirmation = False
        coordinator, bot, storage = make_coordinator(tmp_path, gmail, bot)
        attachment = write_attachment(tmp_path)
        run_request(
            coordinator,
            ReplyRequest(
                thread_id="t1",
                user_instructions="Danke",
                source_message_id=11,
                attachments=(attachment,),
            ),
        )
        assert gmail.sent == []
        assert storage.get_draft(1)["status"] == DraftStatus.CANCELLED.value
        assert not Path(str(tmp_path / "tmp" / "draft-1")).exists()

    def test_resend_after_failure_keeps_attachments(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        gmail.send_error = "definitive"
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        attachment = write_attachment(tmp_path, name="scan.pdf")
        run_request(
            coordinator,
            ReplyRequest(
                thread_id="t1",
                user_instructions="Danke",
                source_message_id=12,
                attachments=(attachment,),
            ),
        )
        assert storage.get_draft(1)["status"] == DraftStatus.SEND_FAILED.value
        # The claimed file is still on disk, so the retry can rebuild the draft.
        draft_dir = Path(str(tmp_path / "tmp" / "draft-1"))
        assert (draft_dir / "01_scan.pdf").is_file()

        gmail.send_error = ""
        asyncio.run(coordinator.resend_draft(1))
        assert storage.get_draft(1)["status"] == DraftStatus.SENT_VERIFIED.value
        assert len(gmail.sent) == 1
        assert gmail.sent[0].attachments[0].filename == "scan.pdf"
        assert not draft_dir.exists()  # cleaned after verified delivery

    def test_confirmation_shows_attachments(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        bot = FakeReplyBot()
        coordinator, bot, storage = make_coordinator(tmp_path, gmail, bot)
        attachment = write_attachment(tmp_path, name="scan.pdf")
        run_request(
            coordinator,
            ReplyRequest(
                thread_id="t1",
                user_instructions="Danke",
                source_message_id=12,
                attachments=(attachment,),
            ),
        )
        shown = bot.drafts_shown[0]
        assert shown.attachments[0].filename == "scan.pdf"

    def test_orphan_tmp_swept(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        orphan_dir = Path(str(tmp_path / "tmp" / "draft-999"))
        orphan_dir.mkdir(parents=True)
        (orphan_dir / "leak.pdf").write_bytes(b"junk")
        # Make it look old.
        import os

        old = time.time() - 48 * 3600
        os.utime(orphan_dir / "leak.pdf", (old, old))
        os.utime(orphan_dir, (old, old))
        coordinator.cleanup_orphan_tmp()
        assert not orphan_dir.exists()

    def test_stale_incoming_download_swept(self, tmp_path: Path) -> None:
        gmail = FakeGmail(threads={"t1": make_thread()}, send_ok=True)
        coordinator, bot, storage = make_coordinator(tmp_path, gmail)
        import os

        stale = Path(str(tmp_path / "tmp" / "incoming" / "abc123_scan.pdf"))
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"junk")
        old = time.time() - 48 * 3600
        os.utime(stale, (old, old))
        coordinator.cleanup_orphan_tmp()
        assert not stale.exists()


class TestThreadlessReconciliation:
    """Compose/forward (threadless) ambiguous sends: reconciled via sent-mail
    search, never blind-resent."""

    def _threadless_draft(self) -> DraftReply:
        return DraftReply(
            thread_id="",
            subject="Presupuesto",
            to=[EmailAddress("Roman", "roman@example.com")],
            cc=[],
            body="Sehr geehrte Damen und Herren, vielen Dank.",
        )

    def _seed_unverified(self, tmp_path: Path, draft: DraftReply) -> tuple[Storage, int]:
        storage = make_storage(tmp_path)
        draft_id = storage.create_draft("", None, draft, telegram_user_id=7)
        storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
        storage.set_draft_send_started(draft_id, int(time.time() * 1000))
        return storage, draft_id

    def test_threadless_ambiguous_found_in_sent_mail_verified_without_resend(
        self, tmp_path: Path
    ) -> None:
        gmail = FakeGmail(send_ok=True)
        draft = self._threadless_draft()
        storage, draft_id = self._seed_unverified(tmp_path, draft)
        # Gmail actually accepted the ambiguous send (we just lost the response).
        gmail.sent_store.append(gmail.accept(draft))

        bot = FakeReplyBot()
        coordinator = ReplyCoordinator(
            make_settings(tmp_dir=str(tmp_path / "tmp")), gmail, MockLLM(), bot, storage
        )
        asyncio.run(coordinator.sweep_unverified())
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SENT_VERIFIED.value
        assert gmail.sent == []  # nothing resent
        assert any("verificado" in n or "confirmado" in n for n in bot.notices)

    def test_threadless_ambiguous_not_found_offers_controlled_retry(
        self, tmp_path: Path
    ) -> None:
        gmail = FakeGmail(send_ok=True)
        draft = self._threadless_draft()
        storage, draft_id = self._seed_unverified(tmp_path, draft)
        # Gmail never received the message.
        bot = FakeReplyBot()
        coordinator = ReplyCoordinator(
            make_settings(tmp_dir=str(tmp_path / "tmp")), gmail, MockLLM(), bot, storage
        )
        asyncio.run(coordinator.sweep_unverified())
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SEND_FAILED.value
        assert gmail.sent == []
        assert bot.resend_offers == [draft_id]  # controlled retry offered

        # User presses retry → re-verify (still not found) → one evidence-based send.
        asyncio.run(coordinator.resend_draft(draft_id))
        assert storage.get_draft(draft_id)["status"] == DraftStatus.SENT_VERIFIED.value
        assert len(gmail.sent) == 1

    def test_threadless_send_gets_fresh_thread_id_and_preserves_subject(self) -> None:
        """Fidelity: the fake mirrors real Gmail — a threadless send is assigned a
        fresh thread id and keeps its subject (no Re: prefix)."""
        gmail = FakeGmail(send_ok=True)
        asyncio.run(gmail.send_reply(self._threadless_draft()))
        assert len(gmail.sent_store) == 1
        record = gmail.sent_store[0]
        assert record.thread_id != ""
        assert record.thread_id.startswith("new-thread-")
        assert record.subject == "Presupuesto"


class MockLLM:
    """Deterministic LLMProvider double for the responder tests."""

    async def summarize_email(self, email: object) -> object:
        from inboxbridge.models import EmailSummary

        return EmailSummary(summary_es="resumen")

    async def draft_reply(self, request: object, thread: object) -> object:
        from inboxbridge.models import DraftReply

        thread_ctx: ThreadContext = thread  # type: ignore[assignment]
        return DraftReply(
            thread_id=thread_ctx.thread_id,
            subject=thread_ctx.subject,
            to=[thread_ctx.messages[0].from_] if thread_ctx.messages else [],
            cc=[],
            body="Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen",
            in_reply_to=thread_ctx.messages[-1].message_id if thread_ctx.messages else "",
            references="",
        )

    async def translate_to_spanish(self, body: str) -> str:
        return "[ES] " + body
