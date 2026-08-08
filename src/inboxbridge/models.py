"""Shared data contracts for InboxBridge.

These models are the ONLY cross-module interfaces (Gmail ↔ LLM ↔ Telegram).
Worker A (gmail) produces ParsedEmail; Worker B (llm/telegram) consumes it.
Never store full bodies or attachment text in the DB — they travel in memory
only, and attachments are never persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MessageStatus(StrEnum):
    """Lifecycle of a Gmail message through the pipeline."""

    RECEIVED = "received"
    SUMMARIZING = "summarizing"
    SENT_TELEGRAM = "sent_telegram"
    FAILED = "failed"


class DraftStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SENT = "sent"
    REJECTED = "rejected"


@dataclass(frozen=True)
class EmailAddress:
    name: str
    email: str

    def __str__(self) -> str:
        return f"{self.name} <{self.email}>" if self.name else self.email


@dataclass(frozen=True)
class AttachmentMeta:
    """Metadata only — content is never stored."""

    filename: str
    mime_type: str
    size_bytes: int
    extracted_text: str = ""


@dataclass(frozen=True)
class ParsedEmail:
    """Cleaned, LLM-ready representation of an incoming email."""

    message_id: str
    thread_id: str
    history_id: int
    subject: str
    sender: EmailAddress
    recipients: list[EmailAddress]
    date_iso: str
    body_text: str
    attachments: list[AttachmentMeta] = field(default_factory=list)

    @property
    def attachment_texts(self) -> list[tuple[str, str]]:
        """(filename, extracted text) for attachments with extractable text."""
        return [
            (a.filename, a.extracted_text)
            for a in self.attachments
            if a.extracted_text
        ]


@dataclass(frozen=True)
class ThreadMessage:
    """A message within a Gmail thread, used as reply context."""

    message_id: str
    from_: EmailAddress
    date_iso: str
    body_text: str
    snippet: str = ""


@dataclass(frozen=True)
class ThreadContext:
    thread_id: str
    subject: str
    messages: list[ThreadMessage]
    history_id: int


@dataclass(frozen=True)
class DraftRequest:
    """User asks (from Telegram) for a reply to a thread."""

    thread_id: str
    user_instructions: str
    language: str = "de"


@dataclass(frozen=True)
class DraftReply:
    """LLM-produced draft for a Gmail reply."""

    thread_id: str
    subject: str
    to: list[EmailAddress]
    cc: list[EmailAddress]
    body: str
    in_reply_to: str = ""
    references: str = ""


@dataclass(frozen=True)
class TelegramTarget:
    chat_id: int


@dataclass(frozen=True)
class PubSubEvent:
    """Parsed notification from Gmail Push (Pub/Sub message)."""

    message_id: str
    history_id: int
    email_address: str
    raw: dict[str, object]


@dataclass(frozen=True)
class EmailSummary:
    """LLM result for an incoming email: Spanish subject + Spanish summary.

    ``subject_es`` may be empty when the LLM did not produce one — the caller
    falls back to the original ``ParsedEmail.subject`` (source of truth for
    threading/drafts; never overwritten).
    """

    subject_es: str = ""
    summary_es: str = ""


@dataclass(frozen=True)
class PipelineResult:
    message_id: str
    status: MessageStatus
    telegram_message_id: int | None = None
    error: str | None = None
