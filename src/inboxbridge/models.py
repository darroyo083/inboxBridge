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
    """Lifecycle of an outgoing reply draft (send/verification states).

    Sending states follow the goal's verified-delivery contract:

    - ``SENDING``: the send request is in flight.
    - ``SENT_UNVERIFIED``: Gmail may or may not have accepted the message
      (ambiguous result) or verification is still pending — never report
      success in this state.
    - ``SENT_VERIFIED``: Gmail evidence confirms the message exists in the
      expected thread with the expected recipients/attachments.
    - ``SEND_FAILED``: a definitive failure; safe to report and offer a
      controlled retry.
    - ``VERIFICATION_FAILED``: reconciliation was exhausted while still
      inconclusive (Gmail could not be queried within budget) — the outcome is
      unknown; never resend automatically.
    """

    PENDING = "pending"
    CONFIRMED = "confirmed"
    SENDING = "sending"
    SENT_UNVERIFIED = "sent_unverified"
    SENT_VERIFIED = "sent_verified"
    SEND_FAILED = "send_failed"
    VERIFICATION_FAILED = "verification_failed"
    CANCELLED = "cancelled"
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
class OutgoingAttachment:
    """A Telegram-supplied file to include in an outgoing Gmail reply.

    Metadata only travels into the DB/confirmation view; the binary lives in a
    temporary directory for the bounded send workflow and is deleted once the
    draft reaches a terminal state (verified, cancelled, failed, expired).
    """

    filename: str
    mime_type: str
    size_bytes: int
    #: Absolute path of the temp file (display-only in the DB; never persisted).
    path: str = ""


@dataclass(frozen=True)
class SendVerification:
    """Deterministic Gmail-side evidence for one outgoing reply.

    ``checked_ok`` distinguishes "Gmail was queried and the message is NOT
    there" (safe to retry) from "Gmail could not be queried" (inconclusive —
    a retry would risk duplicates).
    """

    found: bool
    message_id: str
    thread_match: bool
    recipients_match: bool
    attachments_match: bool
    subject_match: bool
    checked_ok: bool = True

    @property
    def verified(self) -> bool:
        """Strong success: the expected message exists and matches."""
        return (
            self.found
            and self.checked_ok
            and self.thread_match
            and self.recipients_match
            and self.attachments_match
            and self.subject_match
        )

    @property
    def category(self) -> str:
        """Operational outcome category (for logs/observability, never control flow).

        One of: ``verified``, ``inconclusive`` (Gmail unreachable), ``partial_match``
        (found but not fully matching), ``not_found`` (Gmail queried, message absent).
        """
        if self.verified:
            return "verified"
        if not self.checked_ok:
            return "inconclusive"
        if self.found:
            return "partial_match"
        return "not_found"


@dataclass(frozen=True)
class DraftRequest:
    """User asks (from Telegram) for a reply to a thread."""

    thread_id: str
    user_instructions: str
    language: str = "de"
    #: Explicit facts the requesting member asked the bot to remember
    #: (untrusted context for the LLM; capped inside the draft prompt).
    memory: tuple[str, ...] = ()


@dataclass(frozen=True)
class DraftReply:
    """LLM-produced draft for a Gmail reply.

    ``attachments`` are Telegram-supplied files attached AFTER generation
    (the LLM never decides what to attach); they travel in memory and in a
    temp directory, never in SQLite.
    """

    thread_id: str
    subject: str
    to: list[EmailAddress]
    cc: list[EmailAddress]
    body: str
    in_reply_to: str = ""
    references: str = ""
    attachments: tuple[OutgoingAttachment, ...] = ()


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
