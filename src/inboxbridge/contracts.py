"""Shared service contracts (Protocols) between workers.

Worker A (gmail) implements GmailClient.
Worker B (llm/telegram) implements LLMProvider and TelegramNotifier.
The coordinator's pipeline/responder depend only on these protocols —
never on concrete implementations — so both sides can be developed and
tested against mocks in parallel.
"""

from __future__ import annotations

from typing import Protocol

from .models import (
    DraftReply,
    DraftRequest,
    EmailSummary,
    ParsedEmail,
    SendVerification,
    ThreadContext,
)


class GmailClient(Protocol):
    """Gmail API operations needed by the pipeline and responder."""

    async def fetch_message(self, message_id: str) -> ParsedEmail:
        """Fetch and parse one message (body cleaned, attachments extracted)."""
        ...

    async def fetch_thread_context(
        self, thread_id: str, *, with_attachments: bool = False
    ) -> ThreadContext:
        """Fetch full thread (recent messages) for reply/Q&A context.

        ``with_attachments`` also extracts bounded attachment text.
        """
        ...

    async def send_reply(self, draft: DraftReply) -> str:
        """Send a reply in the existing thread. Returns the new message_id.

        May raise ``AmbiguousSendError`` when the server-side outcome is
        unknown — the caller must reconcile before retrying.
        """
        ...

    async def verify_delivery(
        self,
        draft: DraftReply,
        *,
        expected_message_id: str = "",
        since_ms: int = 0,
    ) -> SendVerification:
        """Reconcile a send attempt against Gmail (source of truth).

        ``expected_message_id`` is the id returned by send when known;
        ``since_ms`` bounds the thread search when it is not. Returns
        evidence, never raises.
        """
        ...


class LLMProvider(Protocol):
    """LLM abstraction. Must never raise on transient errors — pipeline
    retries; failures surface via dedicated exception types."""

    async def summarize_email(self, email: ParsedEmail) -> EmailSummary:
        """Produce a natural Spanish summary (and Spanish subject) of an
        incoming email. One LLM call: subject translation + summary together."""
        ...

    async def draft_reply(
        self, request: DraftRequest, thread: ThreadContext
    ) -> DraftReply:
        """Draft a professional German reply using thread context."""
        ...

    async def translate_to_spanish(
        self, body: str, *, model: str | None = None
    ) -> str:
        """Translate a German draft body to Spanish for the Telegram preview.

        Display-only: the Spanish text is never sent to Gmail. ``model``
        overrides the model for this call (``None`` = primary)."""
        ...


class TelegramNotifier(Protocol):
    """Telegram side. Only the configured group chat is ever touched."""

    async def send_summary(self, email: ParsedEmail, summary: EmailSummary) -> int:
        """Post summary (with Spanish subject, falling back to the original);
        returns telegram message_id."""
        ...

    async def send_notice(self, text: str) -> int:
        """Post a short notice (errors, pdf password failure, status)."""
        ...

    async def send_typing(self) -> None:
        """Show typing indicator (no-op if unsupported)."""
        ...

    async def send_draft_for_confirmation(self, draft: DraftReply) -> int:
        """Show draft + recipients; returns message_id for confirmation."""
        ...
