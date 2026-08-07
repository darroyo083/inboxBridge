"""SQLite persistence — minimal state only.

Stores dedup/status identifiers, never email bodies or attachment text.
Schema is portable: `inboxbridge.db` in a small volume is all that must
survive a VPS migration.

Ownership note: this is a shared contract (used by both Gmail and Telegram
workers). The schema lives here; nobody else writes SQL against SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC
from pathlib import Path
from typing import Any

from .models import DraftReply, DraftStatus, MessageStatus


class Storage:
    """Thin, synchronous SQLite wrapper (single writer task in app)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=True)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _migrate(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
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
            CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
            CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status);

            CREATE TABLE IF NOT EXISTS drafts (
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

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # ── meta ────────────────────────────────────────────────────────────────
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        assert self._conn is not None
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    # ── messages ────────────────────────────────────────────────────────────
    def message_exists(self, message_id: str) -> bool:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT 1 FROM messages WHERE message_id = ?", (message_id,)
        ).fetchone()
        return row is not None

    def upsert_message(
        self,
        message_id: str,
        thread_id: str,
        history_id: int,
        status: MessageStatus,
        telegram_message_id: int | None = None,
    ) -> None:
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO messages(message_id, thread_id, history_id, status,
                                 telegram_message_id, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                thread_id = excluded.thread_id,
                history_id = excluded.history_id,
                status = excluded.status,
                telegram_message_id = excluded.telegram_message_id,
                updated_at = excluded.updated_at
            """,
            (message_id, thread_id, history_id, status.value,
             telegram_message_id, now, now),
        )
        self._conn.commit()

    def mark_status(self, message_id: str, status: MessageStatus,
                    telegram_message_id: int | None = None) -> None:
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE messages SET status = ?, telegram_message_id = ?, updated_at = ? "
            "WHERE message_id = ?",
            (status.value, telegram_message_id, now, message_id),
        )
        self._conn.commit()

    def bump_retry(self, message_id: str, next_retry_at: float) -> int:
        """Increment retry_count; returns the new count."""
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE messages SET retry_count = retry_count + 1, "
            "next_retry_at = ?, updated_at = ? WHERE message_id = ?",
            (next_retry_at, now, message_id),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT retry_count FROM messages WHERE message_id = ?", (message_id,)
        ).fetchone()
        return int(row["retry_count"]) if row else 0

    def get_status(self, message_id: str) -> MessageStatus | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT status FROM messages WHERE message_id = ?", (message_id,)
        ).fetchone()
        return MessageStatus(row["status"]) if row else None

    def pending_failures(self, before_ts: float, limit: int = 50) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT message_id, thread_id, history_id, retry_count "
            "FROM messages WHERE status = 'failed' AND next_retry_at <= ? "
            "ORDER BY next_retry_at LIMIT ?",
            (before_ts, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── drafts ──────────────────────────────────────────────────────────────
    def create_draft(self, thread_id: str, message_id: str | None,
                     reply: DraftReply) -> int:
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        to_json = json.dumps([f"{a.email}" for a in reply.to])
        cur = self._conn.execute(
            """
            INSERT INTO drafts(thread_id, message_id, body, to_json, subject,
                               status, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (thread_id, message_id, reply.body, to_json, reply.subject,
             DraftStatus.PENDING.value, now, now),
        )
        self._conn.commit()
        lastrowid = cur.lastrowid
        if lastrowid is None:
            raise RuntimeError("SQLite did not return a row id for the draft insert")
        return int(lastrowid)

    def set_draft_status(self, draft_id: int, status: DraftStatus) -> None:
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE drafts SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, now, draft_id),
        )
        self._conn.commit()

    def get_draft(self, draft_id: int) -> dict[str, Any] | None:
        assert self._conn is not None
        row = self._conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        return dict(row) if row else None
