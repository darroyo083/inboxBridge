"""Coordinator test doubles: FakeGmail (GmailClient protocol) and a minimal
reply bot (TelegramBot surface) for end-to-end integration tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from inboxbridge.gmail.client import SendingDisabledError
from inboxbridge.models import (
    DraftReply,
    EmailAddress,
    ParsedEmail,
    ThreadContext,
    ThreadMessage,
)
from inboxbridge.telegram.bot import ReplyRequest


@dataclass
class FakeGmail:
    """GmailClient protocol double with canned messages/threads.

    ``send_ok`` toggles the kill switch: False simulates SEND_EMAILS=false.
    Records every call for assertions.
    """

    messages: dict[str, ParsedEmail] = field(default_factory=dict)
    threads: dict[str, ThreadContext] = field(default_factory=dict)
    send_ok: bool = True
    fetched: list[str] = field(default_factory=list)
    sent: list[DraftReply] = field(default_factory=list)

    async def fetch_message(self, message_id: str) -> ParsedEmail:
        self.fetched.append(message_id)
        try:
            return self.messages[message_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown message {message_id}") from exc

    async def fetch_thread_context(self, thread_id: str) -> ThreadContext:
        try:
            return self.threads[thread_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown thread {thread_id}") from exc

    async def send_reply(self, draft: DraftReply) -> str:
        if not self.send_ok:
            raise SendingDisabledError("SEND_EMAILS=false")
        self.sent.append(draft)
        return f"new-msg-{len(self.sent)}"


@dataclass
class FakeReplyBot:
    """Minimal TelegramBot surface for the responder: queue + confirmation."""

    allowed: bool = True
    default_confirmation: bool = True
    confirmations: dict[int, bool] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)
    drafts_shown: list[DraftReply] = field(default_factory=list)
    requests: list[ReplyRequest] = field(default_factory=list)
    _next_message_id: int = 1000
    _next_draft_id: int = 1

    async def reply_requests(self):
        for request in self.requests:
            yield request
        while True:
            import asyncio

            await asyncio.sleep(3600)

    async def send_typing(self) -> None:
        return None

    async def send_draft_for_confirmation(self, draft: DraftReply, *, user_id: int = 0) -> int:
        self.drafts_shown.append(draft)
        message_id = self._next_message_id
        self._next_message_id += 1
        self.confirmations[message_id] = self.default_confirmation
        return message_id

    async def wait_for_confirmation(self, message_id: int) -> bool:
        return self.confirmations.get(message_id, False)

    async def send_notice(self, text: str) -> int:
        self.notices.append(text)
        return self._next_draft_id


def make_email(
    message_id: str = "m1",
    thread_id: str = "t1",
    subject: str = "Re: Projektbericht",
    body: str = "Hallo, bitte um Rückmeldung bis Freitag.",
) -> ParsedEmail:
    return ParsedEmail(
        message_id=message_id,
        thread_id=thread_id,
        history_id=10,
        subject=subject,
        sender=EmailAddress("Anna Muster", "anna@example.com"),
        recipients=[EmailAddress("Daniel", "daniel@example.com")],
        date_iso="2026-08-07T10:00:00+00:00",
        body_text=body,
    )


def make_thread(thread_id: str = "t1") -> ThreadContext:
    return ThreadContext(
        thread_id=thread_id,
        subject="Re: Projektbericht",
        history_id=11,
        messages=[
            ThreadMessage(
                message_id="m1",
                from_=EmailAddress("Anna Muster", "anna@example.com"),
                date_iso="2026-08-07T10:00:00+00:00",
                body_text="Hallo, bitte um Rückmeldung bis Freitag.",
            )
        ],
    )
