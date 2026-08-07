"""Unit tests for gmail/watcher.py — watch lifecycle and history baseline."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from inboxbridge.config import Settings
from inboxbridge.db import Storage
from inboxbridge.gmail.watcher import (
    META_HISTORY_ID,
    META_WATCH_EXPIRES,
    WatchError,
    WatchManager,
)
from tests.mocks.gmail import FakeGmailService

Route = tuple[str, ...]

WATCH_RESPONSE = {"expiration": 1_757_000_000_000, "historyId": "12345"}


def make_settings(*, project: str = "proj-1", topic: str = "gmail-events") -> Settings:
    return Settings(
        _env_file=None,
        google_cloud_project=project,
        gmail_pubsub_topic=topic,
        gmail_user_id="me",
    )


@pytest.fixture
def storage(tmp_path: object) -> Iterator[Storage]:
    db = Storage(str(tmp_path) + "/test.db")
    db.connect()
    yield db
    db.close()


class TestRegisterWatch:
    def test_persists_expiration_and_baseline(self, storage: Storage) -> None:
        service = FakeGmailService({("users", "watch"): WATCH_RESPONSE})
        manager = WatchManager(make_settings(), service, storage)
        assert manager.register_watch() is True
        expected = 1_757_000_000_000 / 1000
        assert float(storage.get_meta(META_WATCH_EXPIRES) or "0") == pytest.approx(expected)
        assert storage.get_meta(META_HISTORY_ID) == "12345"
        assert manager.history_start() == 12345

    def test_watch_body_uses_inbox_and_topic(self, storage: Storage) -> None:
        service = FakeGmailService({("users", "watch"): WATCH_RESPONSE})
        manager = WatchManager(make_settings(), service, storage)
        manager.register_watch()
        path, kwargs = service.calls[0]
        assert path == ("users", "watch")
        assert kwargs["body"] == {
            "topicName": "projects/proj-1/topics/gmail-events",
            "labelIds": ["INBOX"],
        }

    def test_first_boot_baseline_not_overwritten(self, storage: Storage) -> None:
        service = FakeGmailService({("users", "watch"): WATCH_RESPONSE})
        manager = WatchManager(make_settings(), service, storage)
        manager.register_watch()
        manager.register_watch()
        assert storage.get_meta(META_HISTORY_ID) == "12345"

    def test_missing_topic_config_raises(self, storage: Storage) -> None:
        service = FakeGmailService({})
        manager = WatchManager(make_settings(project="", topic=""), service, storage)
        with pytest.raises(WatchError):
            manager.register_watch()

    def test_missing_history_id_raises(self, storage: Storage) -> None:
        service = FakeGmailService({("users", "watch"): {"expiration": 1_757_000_000_000}})
        manager = WatchManager(make_settings(), service, storage)
        with pytest.raises(WatchError):
            manager.register_watch()


class TestEnsureWatch:
    def _expiry(self, storage: Storage, delta: timedelta) -> None:
        storage.set_meta(
            META_WATCH_EXPIRES,
            str((datetime.now(UTC) + delta).timestamp()),
        )

    def test_registers_when_missing(self, storage: Storage) -> None:
        service = FakeGmailService({("users", "watch"): WATCH_RESPONSE})
        manager = WatchManager(make_settings(), service, storage)
        assert manager.ensure_watch() is True
        assert service.calls

    def test_skips_when_far_from_expiry(self, storage: Storage) -> None:
        service = FakeGmailService({("users", "watch"): WATCH_RESPONSE})
        manager = WatchManager(make_settings(), service, storage)
        self._expiry(storage, timedelta(days=6))
        assert manager.ensure_watch() is False
        assert service.calls == []

    def test_renews_within_24h(self, storage: Storage) -> None:
        service = FakeGmailService({("users", "watch"): WATCH_RESPONSE})
        manager = WatchManager(make_settings(), service, storage)
        self._expiry(storage, timedelta(hours=6))
        assert manager.ensure_watch() is True
        assert service.calls

    def test_renews_within_24h_exactly(self, storage: Storage) -> None:
        service = FakeGmailService({("users", "watch"): WATCH_RESPONSE})
        manager = WatchManager(make_settings(), service, storage)
        self._expiry(storage, timedelta(hours=24))
        assert manager.ensure_watch() is True

    def test_expires_at_parses_and_falls_back(self, storage: Storage) -> None:
        service = FakeGmailService({})
        manager = WatchManager(make_settings(), service, storage)
        assert manager.expires_at() is None
        storage.set_meta(META_WATCH_EXPIRES, str(time.time() + 1000))
        assert manager.expires_at() is not None
        storage.set_meta(META_WATCH_EXPIRES, "not-a-number")
        assert manager.expires_at() is None
