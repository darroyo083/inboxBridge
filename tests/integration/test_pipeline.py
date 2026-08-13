"""Integration tests: inbound pipeline end-to-end with mocks.

Covers: happy path, dedup (event processed twice — single Telegram post),
LLM failure → FAILED + retry, non-Primary/no-op events, kill switch on
sending, and the /status report.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from inboxbridge.config import Settings
from inboxbridge.db import Storage
from inboxbridge.gmail.history import HistoryDelta
from inboxbridge.models import MessageStatus, PubSubEvent
from inboxbridge.pipeline import InboundPipeline
from inboxbridge.status import build_status_text
from tests.mocks.coordinator import FakeGmail, make_email
from tests.mocks.llm import FakeLLM
from tests.mocks.telegram import FakeTelegram


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "SEND_EMAILS": False,
        "LLM_MAX_RETRIES": 2,
        "RETRY_BACKOFF_BASE": 0.01,
    }
    base.update(overrides)
    return Settings(**base)


def make_event(history_id: int = 10, message_id: str = "pubsub-1") -> PubSubEvent:
    return PubSubEvent(
        message_id=message_id,
        history_id=history_id,
        email_address="me@example.com",
        raw={"historyId": history_id},
    )


def make_storage(tmp_path: object) -> Storage:
    storage = Storage(str(tmp_path) + "/test.db")
    storage.connect()
    return storage


async def test_happy_path(tmp_path: object) -> None:
    settings = make_settings()
    storage = make_storage(tmp_path)
    gmail = FakeGmail(messages={"m1": make_email()})
    llm = FakeLLM(summary="Reunión el viernes: confirmar asistencia.")
    telegram = FakeTelegram()

    # The watch establishes the baseline on first boot (like WatchManager).
    storage.set_meta("last_history_id", "10")

    pipeline = InboundPipeline(
        settings, gmail, llm, telegram, storage,
        history_provider=_delta_provider(["m1"]),
    )
    result = await pipeline.process_event(make_event(history_id=20))
    assert result.status == MessageStatus.SENT_TELEGRAM
    assert telegram.typing_calls == 1
    assert len(telegram.sent) == 1
    assert telegram.sent[0].kind == "summary"
    assert "viernes" in telegram.sent[0].content
    # Baseline advanced.
    assert storage.get_meta("last_history_id") == "20"
    assert storage.get_status("m1") == MessageStatus.SENT_TELEGRAM


async def test_spanish_subject_flows_to_telegram_and_original_stays_untouched(
    tmp_path: object,
) -> None:
    """The LLM produces subject_es + summary_es in ONE call; the original
    email.subject (source of truth for threading/drafts) is never overwritten."""
    settings = make_settings()
    storage = make_storage(tmp_path)
    storage.set_meta("last_history_id", "10")
    email = make_email(subject="Arbeitsplan nächste Woche")
    gmail = FakeGmail(messages={"m1": email})
    llm = FakeLLM(
        summary="Reunión el viernes: confirmar asistencia.",
        subject_es="Plan de trabajo de la próxima semana",
    )
    telegram = FakeTelegram()

    pipeline = InboundPipeline(
        settings, gmail, llm, telegram, storage,
        history_provider=_delta_provider(["m1"]),
    )
    result = await pipeline.process_event(make_event(history_id=20))
    assert result.status == MessageStatus.SENT_TELEGRAM

    # Spanish subject displayed on Telegram; original subject NOT shown.
    assert telegram.sent[0].summary is not None
    assert telegram.sent[0].summary.subject_es == "Plan de trabajo de la próxima semana"
    # Original subject untouched (source of truth for Gmail threading).
    assert email.subject == "Arbeitsplan nächste Woche"
    # One LLM call total (summary + subject in the same request, no extra call).
    assert len(llm.summarize_calls) == 1


async def test_missing_spanish_subject_falls_back_and_pipeline_succeeds(
    tmp_path: object,
) -> None:
    """Malformed/missing subject_es must not fail the pipeline: Telegram shows
    the original subject and the message is still marked SENT_TELEGRAM."""
    settings = make_settings()
    storage = make_storage(tmp_path)
    storage.set_meta("last_history_id", "10")
    gmail = FakeGmail(messages={"m1": make_email(subject="Arbeitsplan nächste Woche")})
    llm = FakeLLM(summary="Reunión el viernes.", subject_es="")  # no translated subject
    telegram = FakeTelegram()

    pipeline = InboundPipeline(
        settings, gmail, llm, telegram, storage,
        history_provider=_delta_provider(["m1"]),
    )
    result = await pipeline.process_event(make_event(history_id=20))
    assert result.status == MessageStatus.SENT_TELEGRAM
    assert telegram.sent[0].summary is not None
    assert telegram.sent[0].summary.subject_es == ""
    assert telegram.sent[0].summary.summary_es == "Reunión el viernes."


async def test_duplicate_event_is_idempotent(tmp_path: object) -> None:
    settings = make_settings()
    storage = make_storage(tmp_path)
    storage.set_meta("last_history_id", "10")
    gmail = FakeGmail(messages={"m1": make_email()})
    llm = FakeLLM()
    telegram = FakeTelegram()

    pipeline = InboundPipeline(
        settings, gmail, llm, telegram, storage,
        history_provider=_delta_provider(["m1"]),
    )
    await pipeline.process_event(make_event(history_id=20))
    # Pub/Sub redelivers the same event; history provider returns nothing new.
    await pipeline.process_event(make_event(history_id=20))
    assert len(telegram.sent) == 1
    assert len(gmail.fetched) == 1


async def test_no_new_messages_is_noop(tmp_path: object) -> None:
    settings = make_settings()
    storage = make_storage(tmp_path)
    telegram = FakeTelegram()

    pipeline = InboundPipeline(
        settings, FakeGmail(), FakeLLM(), telegram, storage,
        history_provider=_empty_delta,
    )
    result = await pipeline.process_event(make_event(history_id=20))
    assert result.status == MessageStatus.SENT_TELEGRAM  # no-op result
    assert telegram.sent == []


async def test_baseline_advances_past_non_primary_only_delta(tmp_path: object) -> None:
    """A delta with only definitively non-Primary history (no new messages, no
    unknowns) advances the baseline — repeated re-scanning is avoided."""
    settings = make_settings()
    storage = make_storage(tmp_path)
    storage.set_meta("last_history_id", "10")

    pipeline = InboundPipeline(
        settings, FakeGmail(), FakeLLM(), FakeTelegram(), storage,
        history_provider=_delta_full_provider(history_id=30, message_ids=[]),
    )
    result = await pipeline.process_event(make_event(history_id=30))
    assert result.status == MessageStatus.SENT_TELEGRAM
    assert storage.get_meta("last_history_id") == "30"


async def test_baseline_does_not_advance_on_unknown_candidate(tmp_path: object) -> None:
    """A delta with a candidate of UNKNOWN Primary status must not advance the
    baseline — advancing could silently lose a real Primary email."""
    settings = make_settings()
    storage = make_storage(tmp_path)
    storage.set_meta("last_history_id", "10")

    pipeline = InboundPipeline(
        settings, FakeGmail(), FakeLLM(), FakeTelegram(), storage,
        history_provider=_delta_full_provider(history_id=30, message_ids=[], unknown_count=1),
    )
    await pipeline.process_event(make_event(history_id=30))
    assert storage.get_meta("last_history_id") == "10"  # unchanged → retried next push


async def test_llm_failure_marks_failed_and_retry_recovers(tmp_path: object) -> None:
    settings = make_settings()
    storage = make_storage(tmp_path)
    gmail = FakeGmail(messages={"m1": make_email()})
    # First call fails; retry (via retry_failed) succeeds.
    llm = FakeLLM(transient_failures=1, summary="Resumen tras reintento.")
    telegram = FakeTelegram()

    pipeline = InboundPipeline(
        settings, gmail, llm, telegram, storage,
        history_provider=_delta_provider(["m1"]),
    )
    result = await pipeline.process_event(make_event(history_id=20))
    assert result.status == MessageStatus.FAILED
    assert storage.get_status("m1") == MessageStatus.FAILED
    assert telegram.sent == []

    # Retry after next_retry_at.
    storage.bump_retry("m1", (datetime.now(UTC) - timedelta(seconds=1)).timestamp())
    retried = await pipeline.retry_failed()
    assert retried == 1
    assert storage.get_status("m1") == MessageStatus.SENT_TELEGRAM
    assert len(telegram.sent) == 1
    assert telegram.sent[0].content == "Resumen tras reintento."


async def test_failed_message_not_lost_on_restart(tmp_path: object) -> None:
    """A FAILED row survives a storage restart (new Storage over same file)."""
    settings = make_settings()
    storage = make_storage(tmp_path)
    gmail = FakeGmail(messages={"m1": make_email()})
    llm = FailingLLM()
    telegram = FakeTelegram()

    pipeline = InboundPipeline(
        settings, gmail, llm, telegram, storage,
        history_provider=_delta_provider(["m1"]),
    )
    result = await pipeline.process_event(make_event(history_id=20))
    assert result.status == MessageStatus.FAILED

    storage.close()
    storage2 = Storage(storage.path)
    storage2.connect()
    pending = storage2.pending_failures(datetime.now(UTC).timestamp() + 3600)
    assert len(pending) == 1
    assert pending[0]["message_id"] == "m1"
    storage2.close()


async def test_attachment_password_failure_notifies_telegram(tmp_path: object) -> None:
    """Wrong PDF password → FAILED + notice (typed error path)."""
    from inboxbridge.gmail.attachments import AttachmentPasswordError

    settings = make_settings()
    storage = make_storage(tmp_path)

    class PdfGmail(FakeGmail):
        async def fetch_message(self, message_id: str) -> object:
            raise AttachmentPasswordError("wrong PDF password for x.pdf")

    telegram = FakeTelegram()
    pipeline = InboundPipeline(
        settings, PdfGmail(), FakeLLM(), telegram, storage,
        history_provider=_delta_provider(["m1"]),
    )
    result = await pipeline.process_event(make_event(history_id=20))
    assert result.status == MessageStatus.FAILED
    assert any("PDF_PASSWORD" in n for n in telegram.notices)


def test_status_report_has_no_secrets(tmp_path: object) -> None:
    settings = make_settings(
        TELEGRAM_BOT_TOKEN="super-secret-token",
        LLM_API_KEY="super-secret-key",
        LLM_BASE_URL="https://llm.example",
    )
    storage = make_storage(tmp_path)
    text = build_status_text(settings, storage)
    assert "super-secret-token" not in text
    assert "super-secret-key" not in text
    assert "configured" in text
    assert "SEND_EMAILS=false" in text


def _delta(event: PubSubEvent, message_ids: list[str]) -> HistoryDelta:
    return HistoryDelta(history_id=event.history_id, message_ids=message_ids)


def _delta_provider(message_ids: list[str]):
    async def provider(event: PubSubEvent) -> HistoryDelta:
        return _delta(event, message_ids)

    return provider


async def _empty_delta(event: PubSubEvent) -> HistoryDelta:
    return HistoryDelta(history_id=event.history_id, message_ids=[])


def _delta_full_provider(
    *, history_id: int, message_ids: list[str], unknown_count: int = 0
):
    async def provider(event: PubSubEvent) -> HistoryDelta:
        return HistoryDelta(
            history_id=history_id, message_ids=message_ids, unknown_count=unknown_count
        )

    return provider


class FailingLLM(FakeLLM):
    async def summarize_email(self, email: object) -> object:
        from inboxbridge.llm.base import LLMUnavailable

        raise LLMUnavailable("simulated outage")
