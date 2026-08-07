"""Unit tests for gmail/history.py — deltas, dedup, Primary approximation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from inboxbridge.config import Settings
from inboxbridge.db import Storage
from inboxbridge.gmail.history import (
    HistoryDelta,
    HistoryProcessor,
)
from inboxbridge.gmail.watcher import META_HISTORY_ID
from tests.mocks.gmail import FakeGmailService

Route = tuple[str, ...]


def make_settings() -> Settings:
    return Settings(
        _env_file=None, gmail_user_id="me", google_cloud_project="proj", gmail_pubsub_topic="t"
    )


@pytest.fixture
def storage(tmp_path: object) -> Iterator[Storage]:
    db = Storage(str(tmp_path) + "/test.db")
    db.connect()
    yield db
    db.close()


def history_page(*records: dict[str, object], page_token: str | None = None) -> dict[str, object]:
    page: dict[str, object] = {"history": list(records)}
    if page_token:
        page["nextPageToken"] = page_token
    return page


def history_record(record_id: int, *message_ids: str) -> dict[str, object]:
    return {
        "id": str(record_id),
        "messagesAdded": [{"message": {"id": mid, "threadId": "t1"}} for mid in message_ids],
    }


def labels_response(message_id: str, labels: list[str]) -> dict[str, object]:
    return {"id": message_id, "labelIds": labels}


class TestNewMessageIds:
    def test_first_boot_returns_nothing(self, storage: Storage) -> None:
        service = FakeGmailService({})
        processor = HistoryProcessor(make_settings(), service, storage)
        delta = processor.new_message_ids(event_history_id=777)
        assert delta == HistoryDelta(history_id=777, message_ids=[])
        assert service.calls == []

    def test_returns_new_primary_messages(self, storage: Storage) -> None:
        routes: dict[Route, object] = {
            ("users", "history", "list"): history_page(
                history_record(50, "m1", "m2", "m3")
            ),
            ("users", "messages", "get"): lambda kwargs: labels_response(
                kwargs["id"],
                ["CATEGORY_PERSONAL"] if kwargs["id"] in ("m1", "m3") else ["CATEGORY_SOCIAL"],
            ),
        }
        storage.set_meta(META_HISTORY_ID, "49")
        processor = HistoryProcessor(make_settings(), FakeGmailService(routes), storage)
        delta = processor.new_message_ids(event_history_id=50)
        assert delta.message_ids == ["m1", "m3"]
        assert delta.history_id == 50

    def test_unlabeled_messages_are_primary(self, storage: Storage) -> None:
        routes: dict[Route, object] = {
            ("users", "history", "list"): history_page(history_record(60, "m1", "m2")),
            ("users", "messages", "get"): lambda kwargs: labels_response(kwargs["id"], []),
        }
        storage.set_meta(META_HISTORY_ID, "59")
        processor = HistoryProcessor(make_settings(), FakeGmailService(routes), storage)
        delta = processor.new_message_ids(event_history_id=60)
        assert delta.message_ids == ["m1", "m2"]

    def test_all_category_labels_filtered_out(self, storage: Storage) -> None:
        routes: dict[Route, object] = {
            ("users", "history", "list"): history_page(history_record(70, "m1")),
            ("users", "messages", "get"): labels_response(
                "m1", ["CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL"]
            ),
        }
        storage.set_meta(META_HISTORY_ID, "69")
        processor = HistoryProcessor(make_settings(), FakeGmailService(routes), storage)
        assert processor.new_message_ids(event_history_id=70).message_ids == []

    def test_dedup_against_db_and_within_batch(self, storage: Storage) -> None:
        routes: dict[Route, object] = {
            ("users", "history", "list"): history_page(
                history_record(80, "m1", "m2", "m2", "m3")
            ),
            ("users", "messages", "get"): labels_response("m1", ["CATEGORY_PERSONAL"]),
        }
        storage.set_meta(META_HISTORY_ID, "79")
        from inboxbridge.models import MessageStatus

        storage.upsert_message("m1", "t1", 79, MessageStatus.SENT_TELEGRAM)
        processor = HistoryProcessor(make_settings(), FakeGmailService(routes), storage)
        delta = processor.new_message_ids(event_history_id=80)
        assert delta.message_ids == ["m2", "m3"]

    def test_pagination_follows_page_token(self, storage: Storage) -> None:
        routes: dict[Route, object] = {
            ("users", "history", "list"): lambda kwargs: history_page(
                history_record(90, "m1"), page_token="p2"
            )
            if kwargs.get("pageToken") is None
            else history_page(history_record(91, "m2")),
            ("users", "messages", "get"): labels_response("m1", ["CATEGORY_PERSONAL"]),
        }
        storage.set_meta(META_HISTORY_ID, "89")
        service = FakeGmailService(routes)
        processor = HistoryProcessor(make_settings(), service, storage)
        delta = processor.new_message_ids(event_history_id=91)
        assert delta.message_ids == ["m1", "m2"]
        assert delta.history_id == 91
        list_calls = [c for c in service.calls if c[0] == ("users", "history", "list")]
        assert len(list_calls) == 2
        first, second = list_calls
        assert "pageToken" not in first[1]
        assert first[1]["startHistoryId"] == 90
        assert second[1]["pageToken"] == "p2"

    def test_inbox_label_filter_used(self, storage: Storage) -> None:
        routes: dict[Route, object] = {
            ("users", "history", "list"): history_page(history_record(1)),
        }
        storage.set_meta(META_HISTORY_ID, "0")
        service = FakeGmailService(routes)
        processor = HistoryProcessor(make_settings(), service, storage)
        processor.new_message_ids(event_history_id=1)
        assert service.calls[0][1]["labelId"] == "INBOX"

    def test_fetch_failure_skips_message(self, storage: Storage) -> None:
        def _explode(**kwargs: object) -> None:
            raise AssertionError("message gone")

        routes: dict[Route, object] = {
            ("users", "history", "list"): history_page(history_record(1, "m1")),
            ("users", "messages", "get"): _explode,
        }
        storage.set_meta(META_HISTORY_ID, "0")
        processor = HistoryProcessor(make_settings(), FakeGmailService(routes), storage)
        assert processor.new_message_ids(event_history_id=1).message_ids == []


class TestPersist:
    def test_persist_history_id_advances_baseline(self, storage: Storage) -> None:
        processor = HistoryProcessor(make_settings(), FakeGmailService({}), storage)
        storage.set_meta(META_HISTORY_ID, "10")
        processor.persist_history_id(25)
        assert storage.get_meta(META_HISTORY_ID) == "25"

    def test_corrupt_baseline_treated_as_first_boot(self, storage: Storage) -> None:
        storage.set_meta(META_HISTORY_ID, "abc")
        processor = HistoryProcessor(make_settings(), FakeGmailService({}), storage)
        delta = processor.new_message_ids(event_history_id=5)
        assert delta.message_ids == []
