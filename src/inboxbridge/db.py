"""SQLite persistence — minimal state only.

Stores dedup/status identifiers, never email bodies or attachment text.
Also stores explicit user memories (facts the team saves via /remember);
nothing is ever learned from emails automatically. Schema is portable:
`inboxbridge.db` in a small volume is all that must survive a VPS migration.

Ownership note: this is a shared contract (used by both Gmail and Telegram
workers). The schema lives here; nobody else writes SQL against SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC
from pathlib import Path
from typing import Any

from .models import DraftReply, DraftStatus, MessageStatus, OutgoingAttachment


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
                telegram_user_id INTEGER NOT NULL DEFAULT 0,
                sent_message_id TEXT NOT NULL DEFAULT '',
                send_started_at REAL,
                verification_attempts INTEGER NOT NULL DEFAULT 0,
                attachments_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(telegram_user_id, key)
            );
            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(telegram_user_id);

            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name TEXT NOT NULL,
                email TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email);

            CREATE TABLE IF NOT EXISTS contact_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contact_id INTEGER NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
                alias TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(alias)
            );
            CREATE INDEX IF NOT EXISTS idx_contact_aliases_alias ON contact_aliases(alias);

            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL DEFAULT '',
                thread_id TEXT NOT NULL DEFAULT '',
                telegram_user_id INTEGER NOT NULL DEFAULT 0,
                due_at REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                note TEXT NOT NULL DEFAULT '',
                telegram_message_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, due_at);
            """
        )
        self._conn.commit()
        self._ensure_column("drafts", "telegram_user_id", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("drafts", "sent_message_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("drafts", "send_started_at", "REAL")
        self._ensure_column("drafts", "verification_attempts", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("drafts", "attachments_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("drafts", "forward_of", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("drafts", "telegram_token", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("drafts", "telegram_message_id", "INTEGER NOT NULL DEFAULT 0")

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        """Idempotent ALTER TABLE ADD COLUMN for pre-existing databases.

        Deterministic migration: a missing column is added exactly once;
        existing data keeps its defaults.
        """
        assert self._conn is not None
        cols = {
            str(row["name"]) for row in self._conn.execute(f"PRAGMA table_info({table})")
        }
        if column not in cols:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
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

    def delete_meta(self, key: str) -> None:
        """Remove a meta key entirely (e.g. temporary original-view state)."""
        assert self._conn is not None
        self._conn.execute("DELETE FROM meta WHERE key = ?", (key,))
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

    def latest_incoming_message(self) -> dict[str, Any] | None:
        """The most recent INCOMING message fully processed by InboxBridge.

        Only messages that were successfully summarized and posted to Telegram
        (status ``sent_telegram``) are eligible — this excludes our own sent
        messages (those live in ``drafts``, never ``messages``), failed rows
        and never-processed history. Used to resolve "al último correo" to a
        concrete, immutable thread target.
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM messages WHERE status = ? "
            "ORDER BY created_at DESC, history_id DESC LIMIT 1",
            (MessageStatus.SENT_TELEGRAM.value,),
        ).fetchone()
        return dict(row) if row else None

    # ── drafts ──────────────────────────────────────────────────────────────
    def create_draft(
        self,
        thread_id: str,
        message_id: str | None,
        reply: DraftReply,
        *,
        telegram_user_id: int = 0,
    ) -> int:
        """Persist a draft row (status PENDING) at presentation time.

        The draft body is the user's own generated reply (needed for safe
        retry/recovery); attachment binaries are never stored — only their
        metadata.
        """
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        to_json = json.dumps([str(a) for a in reply.to])
        attachments_json = json.dumps(
            [
                {
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                    "size_bytes": a.size_bytes,
                }
                for a in reply.attachments
            ]
        )
        cur = self._conn.execute(
            """
            INSERT INTO drafts(thread_id, message_id, body, to_json, subject,
                               status, telegram_user_id, attachments_json,
                               forward_of, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (thread_id, message_id, reply.body, to_json, reply.subject,
             DraftStatus.PENDING.value, telegram_user_id, attachments_json,
             reply.forward_of, now, now),
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

    def set_draft_telegram(self, draft_id: int, token: str, message_id: int) -> None:
        """Persist the Telegram preview identity so a draft callback survives
        restart: the callback token can be resolved back to the draft row."""
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE drafts SET telegram_token = ?, telegram_message_id = ?, updated_at = ? "
            "WHERE id = ?",
            (token, message_id, now, draft_id),
        )
        self._conn.commit()

    def get_draft_by_token(self, token: str) -> dict[str, Any] | None:
        """Resolve a Telegram callback token to its persisted draft row."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM drafts WHERE telegram_token = ?", (token,)
        ).fetchone()
        return dict(row) if row else None

    def set_draft_body(self, draft_id: int, body: str) -> None:
        """Update the persisted draft body after an edit (retry coherence)."""
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE drafts SET body = ?, updated_at = ? WHERE id = ?",
            (body, now, draft_id),
        )
        self._conn.commit()

    def set_draft_attachments(
        self, draft_id: int, attachments: tuple[OutgoingAttachment, ...]
    ) -> None:
        """Persist attachment metadata for an existing draft (binaries never
        stored; temp paths are derived from the draft id at load time)."""
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        attachments_json = json.dumps(
            [
                {
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                    "size_bytes": a.size_bytes,
                }
                for a in attachments
            ]
        )
        self._conn.execute(
            "UPDATE drafts SET attachments_json = ?, updated_at = ? WHERE id = ?",
            (attachments_json, now, draft_id),
        )
        self._conn.commit()

    def set_draft_send_started(self, draft_id: int, started_at_ms: int) -> None:
        """Record when the send attempt began (epoch ms, Gmail internalDate scale)."""
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE drafts SET send_started_at = ?, updated_at = ? WHERE id = ?",
            (started_at_ms, now, draft_id),
        )
        self._conn.commit()

    def set_draft_sent_message(self, draft_id: int, sent_message_id: str) -> None:
        """Record the Gmail message id returned by the send call (when known)."""
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE drafts SET sent_message_id = ?, updated_at = ? WHERE id = ?",
            (sent_message_id, now, draft_id),
        )
        self._conn.commit()

    def bump_verification_attempts(self, draft_id: int) -> int:
        """Increment the reconciliation attempt counter; returns the new count."""
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE drafts SET verification_attempts = verification_attempts + 1, "
            "updated_at = ? WHERE id = ?",
            (now, draft_id),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT verification_attempts FROM drafts WHERE id = ?", (draft_id,)
        ).fetchone()
        return int(row["verification_attempts"]) if row else 0

    def get_draft(self, draft_id: int) -> dict[str, Any] | None:
        assert self._conn is not None
        row = self._conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        return dict(row) if row else None

    def drafts_in_statuses(self, statuses: list[DraftStatus]) -> list[dict[str, Any]]:
        """All draft rows currently in the given states (for recovery sweeps)."""
        assert self._conn is not None
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        rows = self._conn.execute(
            f"SELECT * FROM drafts WHERE status IN ({placeholders}) ORDER BY id",
            [s.value for s in statuses],
        ).fetchall()
        return [dict(r) for r in rows]

    def claim_draft_for_send(self, draft_id: int, allowed: list[DraftStatus]) -> bool:
        """Atomically claim a draft for a send attempt (status → sending).

        Returns False when the draft is not in an allowed state — callers
        treat that as "another flow already owns this draft" (prevents
        double-send races between resend taps and sweeps).
        """
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        placeholders = ", ".join("?" for _ in allowed)
        cur = self._conn.execute(
            f"UPDATE drafts SET status = ?, updated_at = ? "
            f"WHERE id = ? AND status IN ({placeholders})",
            (DraftStatus.SENDING.value, now, draft_id, *[s.value for s in allowed]),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── memories (explicit user facts, per Telegram user) ──────────────────
    def set_memory(self, telegram_user_id: int, key: str, value: str) -> None:
        """Create or update a memory. ``key`` is the normalized fact key."""
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO memories(telegram_user_id, key, value, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(telegram_user_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (telegram_user_id, key, value, now, now),
        )
        self._conn.commit()

    def list_memories(
        self, telegram_user_id: int, query: str | None = None
    ) -> list[dict[str, Any]]:
        """All memories for a user ordered by key; optional substring filter."""
        assert self._conn is not None
        if query:
            pattern = f"%{self._escape_like(query)}%"
            rows = self._conn.execute(
                "SELECT key, value, updated_at FROM memories "
                "WHERE telegram_user_id = ? AND key LIKE ? ESCAPE '\\' "
                "ORDER BY key",
                (telegram_user_id, pattern),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key, value, updated_at FROM memories "
                "WHERE telegram_user_id = ? ORDER BY key",
                (telegram_user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_memories(self, telegram_user_id: int, query: str) -> int:
        """Delete memories whose key contains ``query``; returns count deleted."""
        assert self._conn is not None
        if not query:
            return 0
        pattern = f"%{self._escape_like(query)}%"
        cur = self._conn.execute(
            "DELETE FROM memories WHERE telegram_user_id = ? AND key LIKE ? ESCAPE '\\'",
            (telegram_user_id, pattern),
        )
        self._conn.commit()
        return cur.rowcount

    @staticmethod
    def _escape_like(text: str) -> str:
        """Escape LIKE wildcards so user queries match literally."""
        return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    # ── contacts + aliases (explicit user-configured application data) ──────
    def create_contact(self, display_name: str, email: str) -> int:
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "INSERT INTO contacts(display_name, email, created_at, updated_at) "
            "VALUES(?, ?, ?, ?)",
            (display_name, email, now, now),
        )
        self._conn.commit()
        lastrowid = cur.lastrowid
        if lastrowid is None:
            raise RuntimeError("SQLite did not return a row id for the contact insert")
        return int(lastrowid)

    def get_contact(self, contact_id: int) -> dict[str, Any] | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM contacts WHERE id = ?", (contact_id,)
        ).fetchone()
        return dict(row) if row else None

    def find_contact_by_email(self, email: str) -> dict[str, Any] | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM contacts WHERE email = ? COLLATE NOCASE LIMIT 1", (email,)
        ).fetchone()
        return dict(row) if row else None

    def find_contacts_by_email(self, email: str) -> list[dict[str, Any]]:
        """All contacts using this address (shared mailboxes are supported)."""
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM contacts WHERE email = ? COLLATE NOCASE ORDER BY id",
            (email,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_contact_email(self, contact_id: int, email: str) -> None:
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE contacts SET email = ?, updated_at = ? WHERE id = ?",
            (email, now, contact_id),
        )
        self._conn.commit()

    def rename_contact(self, contact_id: int, display_name: str) -> None:
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE contacts SET display_name = ?, updated_at = ? WHERE id = ?",
            (display_name, now, contact_id),
        )
        self._conn.commit()

    def delete_contact(self, contact_id: int) -> None:
        """Delete a contact; aliases cascade (foreign_keys=ON)."""
        assert self._conn is not None
        self._conn.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        self._conn.commit()

    def list_contacts(self) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM contacts ORDER BY display_name COLLATE NOCASE"
        ).fetchall()
        return [dict(r) for r in rows]

    def add_alias(self, contact_id: int, alias: str) -> bool:
        """Create an alias; returns False on normalized-alias collision."""
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        try:
            self._conn.execute(
                "INSERT INTO contact_aliases(contact_id, alias, created_at) "
                "VALUES(?, ?, ?)",
                (contact_id, alias, now),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def remove_alias(self, alias: str) -> int:
        assert self._conn is not None
        cur = self._conn.execute("DELETE FROM contact_aliases WHERE alias = ?", (alias,))
        self._conn.commit()
        return cur.rowcount

    def remove_alias_by_id(self, alias_id: int) -> int:
        assert self._conn is not None
        cur = self._conn.execute(
            "DELETE FROM contact_aliases WHERE id = ?", (alias_id,)
        )
        self._conn.commit()
        return cur.rowcount

    def find_alias(self, alias: str) -> dict[str, Any] | None:
        """Alias row with its contact joined (for resolution)."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT a.id AS alias_id, a.alias, c.id AS contact_id, "
            "c.display_name, c.email "
            "FROM contact_aliases a JOIN contacts c ON c.id = a.contact_id "
            "WHERE a.alias = ?",
            (alias,),
        ).fetchone()
        return dict(row) if row else None

    def list_aliases(self, contact_id: int) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT id, alias FROM contact_aliases WHERE contact_id = ? ORDER BY alias",
            (contact_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── reminders (minimal workflow metadata, never email bodies) ───────────
    def create_reminder(
        self,
        *,
        message_id: str,
        thread_id: str,
        telegram_user_id: int,
        due_at: float,
        note: str = "",
    ) -> int:
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "INSERT INTO reminders(message_id, thread_id, telegram_user_id, due_at, "
            "status, note, created_at, updated_at) VALUES(?, ?, ?, ?, 'pending', ?, ?, ?)",
            (message_id, thread_id, telegram_user_id, due_at, note, now, now),
        )
        self._conn.commit()
        lastrowid = cur.lastrowid
        if lastrowid is None:
            raise RuntimeError("SQLite did not return a row id for the reminder insert")
        return int(lastrowid)

    def get_reminder(self, reminder_id: int) -> dict[str, Any] | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
        ).fetchone()
        return dict(row) if row else None

    def due_reminders(self, before_ts: float, limit: int = 50) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM reminders WHERE status = 'pending' AND due_at <= ? "
            "ORDER BY due_at LIMIT ?",
            (before_ts, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_reminders(
        self, telegram_user_id: int | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        assert self._conn is not None
        if telegram_user_id is None:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE status = 'pending' "
                "ORDER BY due_at LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE status = 'pending' "
                "AND telegram_user_id = ? ORDER BY due_at LIMIT ?",
                (telegram_user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def claim_reminder(self, reminder_id: int) -> bool:
        """Atomically mark a reminder fired (prevents duplicate ticks)."""
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "UPDATE reminders SET status = 'fired', updated_at = ? "
            "WHERE id = ? AND status = 'pending'",
            (now, reminder_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def cancel_reminder(self, reminder_id: int, telegram_user_id: int) -> bool:
        assert self._conn is not None
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "UPDATE reminders SET status = 'cancelled', updated_at = ? "
            "WHERE id = ? AND telegram_user_id = ? AND status = 'pending'",
            (now, reminder_id, telegram_user_id),
        )
        self._conn.commit()
        return cur.rowcount > 0
