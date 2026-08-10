"""Deterministic schema migration tests: a legacy drafts table (without the
send-state columns) is upgraded idempotently on connect."""

from __future__ import annotations

import sqlite3

from inboxbridge.db import DraftStatus, Storage
from inboxbridge.models import DraftReply, EmailAddress


def _legacy_db(path: str) -> None:
    """Create the pre-verified-delivery schema (no send columns)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            message_id TEXT,
            body TEXT NOT NULL,
            to_json TEXT NOT NULL,
            subject TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            history_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            telegram_message_id INTEGER,
            retry_count INTEGER NOT NULL DEFAULT 0,
            next_retry_at REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO drafts(thread_id, message_id, body, to_json, subject,
                           status, created_at, updated_at)
        VALUES('t1', NULL, 'cuerpo', '["ana@example.com"]', 'Re: X',
               'confirmed', '2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()


def test_legacy_draft_table_is_migrated_without_data_loss(tmp_path: object) -> None:
    path = str(tmp_path) + "/legacy.db"
    _legacy_db(path)
    storage = Storage(path)
    storage.connect()

    row = storage.get_draft(1)
    assert row is not None
    assert row["status"] == "confirmed"  # pre-existing data intact
    assert row["telegram_user_id"] == 0  # new column with default
    assert row["sent_message_id"] == ""
    assert row["verification_attempts"] == 0
    assert row["attachments_json"] == "[]"

    # The migrated row is fully usable by the new flow.
    storage.set_draft_status(1, DraftStatus.SENT_UNVERIFIED)
    storage.set_draft_sent_message(1, "sent-9")
    storage.bump_verification_attempts(1)
    assert storage.get_draft(1)["status"] == DraftStatus.SENT_UNVERIFIED.value
    assert storage.get_draft(1)["sent_message_id"] == "sent-9"
    assert storage.get_draft(1)["verification_attempts"] == 1


def test_reconnecting_is_idempotent(tmp_path: object) -> None:
    path = str(tmp_path) + "/twice.db"
    _legacy_db(path)
    storage = Storage(path)
    storage.connect()
    storage.close()
    storage.connect()  # second connect must not fail or duplicate columns
    assert storage.get_draft(1)["telegram_user_id"] == 0
    storage.close()


def test_fresh_database_has_all_columns(tmp_path: object) -> None:
    storage = Storage(str(tmp_path) + "/fresh.db")
    storage.connect()
    draft = DraftReply(
        thread_id="t1",
        subject="Re: X",
        to=[EmailAddress("Ana", "ana@example.com")],
        cc=[],
        body="danke",
    )
    draft_id = storage.create_draft("t1", None, draft, telegram_user_id=7)
    assert storage.get_draft(draft_id)["telegram_user_id"] == 7
    assert storage.get_draft(draft_id)["status"] == DraftStatus.PENDING.value


def test_draft_roundtrip_preserves_recipient_names(tmp_path: object) -> None:
    storage = Storage(str(tmp_path) + "/names.db")
    storage.connect()
    draft = DraftReply(
        thread_id="t1",
        subject="Re: X",
        to=[EmailAddress("Ana Muster", "ana@example.com")],
        cc=[],
        body="danke",
    )
    draft_id = storage.create_draft("t1", None, draft)
    row = storage.get_draft(draft_id)
    assert "Ana Muster" in row["to_json"]  # full display identity preserved
    assert "ana@example.com" in row["to_json"]
