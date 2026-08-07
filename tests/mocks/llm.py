"""Async LLMProvider double for coordinator integration tests."""

from __future__ import annotations

from inboxbridge.llm.base import LLMUnavailable
from inboxbridge.models import DraftReply, DraftRequest, EmailAddress, ParsedEmail, ThreadContext


class FakeLLM:
    """Deterministic LLMProvider double: canned outputs, recorded calls.

    ``transient_failures`` makes the first N summarize/draft calls raise
    :class:`LLMUnavailable` (to exercise the retry wrapper).
    """

    def __init__(
        self,
        *,
        summary: str = "Resumen de prueba.",
        draft_body: str = (
            "Sehr geehrte Frau Muster,\n\nvielen Dank für Ihre Nachricht. "
            "Ich melde mich Anfang nächster Woche.\n\nMit freundlichen Grüßen\nInboxBridge"
        ),
        draft_to: list[EmailAddress] | None = None,
        transient_failures: int = 0,
    ) -> None:
        self.summary = summary
        self.draft_body = draft_body
        self.draft_to = draft_to
        self.transient_failures = transient_failures
        self.summarize_calls: list[ParsedEmail] = []
        self.draft_calls: list[tuple[DraftRequest, ThreadContext]] = []

    async def summarize_email(self, email: ParsedEmail) -> str:
        self.summarize_calls.append(email)
        if self.transient_failures > 0:
            self.transient_failures -= 1
            raise LLMUnavailable("simulated transient outage")
        return self.summary

    async def draft_reply(self, request: DraftRequest, thread: ThreadContext) -> DraftReply:
        self.draft_calls.append((request, thread))
        recipients = self.draft_to or ([thread.messages[0].from_] if thread.messages else [])
        return DraftReply(
            thread_id=request.thread_id,
            subject=thread.subject,
            to=recipients,
            cc=[],
            body=self.draft_body,
            in_reply_to=thread.messages[-1].message_id if thread.messages else "",
            references="",
        )


class FailingLLM:
    """Always-failing LLMProvider double (retry/error-path tests)."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or LLMUnavailable("simulated outage")

    async def summarize_email(self, email: ParsedEmail) -> str:
        raise self.error

    async def draft_reply(self, request: DraftRequest, thread: ThreadContext) -> DraftReply:
        raise self.error
