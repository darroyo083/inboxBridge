"""Coordinator test doubles: FakeGmail (GmailClient protocol) and a minimal
reply bot (TelegramBot surface) for end-to-end integration tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from inboxbridge.gmail.client import AmbiguousSendError, SendingDisabledError
from inboxbridge.models import (
    DraftReply,
    EmailAddress,
    ParsedEmail,
    SendVerification,
    ThreadContext,
    ThreadMessage,
)
from inboxbridge.telegram.bot import ReplyRequest


@dataclass
class SentRecord:
    """A message Gmail 'accepted', used by FakeGmail.verify_delivery."""

    message_id: str
    thread_id: str
    recipients: tuple[str, ...]
    subject: str
    attachment_filenames: tuple[str, ...] = ()


@dataclass
class FakeGmail:
    """GmailClient protocol double with canned messages/threads.

    ``send_ok`` toggles the kill switch: False simulates SEND_EMAILS=false.
    ``send_error`` raises during send ('' = success, 'definitive' = hard
    failure, 'ambiguous' = AmbiguousSendError).
    Records every call for assertions; accepted sends land in ``sent_store``
    so ``verify_delivery`` can reconcile against them.
    """

    messages: dict[str, ParsedEmail] = field(default_factory=dict)
    threads: dict[str, ThreadContext] = field(default_factory=dict)
    send_ok: bool = True
    send_error: str = ""
    fetched: list[str] = field(default_factory=list)
    sent: list[DraftReply] = field(default_factory=list)
    sent_store: list[SentRecord] = field(default_factory=list)
    verify_error: bool = False
    verify_delay_ok: bool = False  # first verification returns not-found, then found
    ambiguous_accepts: bool = True  # False: ambiguous send never reached Gmail
    _next_id: int = 100

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
        if self.send_error == "definitive":
            raise RuntimeError("simulated definitive send failure")
        record = SentRecord(
            message_id=f"sent-{self._next_id}",
            thread_id=draft.thread_id,
            recipients=tuple(a.email for a in draft.to),
            subject=draft.subject,
            attachment_filenames=tuple(a.filename for a in draft.attachments),
        )
        self._next_id += 1
        if self.send_error == "ambiguous":
            if self.ambiguous_accepts:
                # Gmail ACCEPTED the message but the client lost the response.
                self.sent_store.append(record)
            raise AmbiguousSendError("simulated transport timeout after transmission")
        self.sent.append(draft)
        self.sent_store.append(record)
        return record.message_id

    def accept(self, draft: DraftReply) -> SentRecord:
        """Simulate Gmail accepting a message outside a send call (crash recovery)."""
        record = SentRecord(
            message_id=f"sent-{self._next_id}",
            thread_id=draft.thread_id,
            recipients=tuple(a.email for a in draft.to),
            subject=draft.subject,
            attachment_filenames=tuple(a.filename for a in draft.attachments),
        )
        self._next_id += 1
        self.sent_store.append(record)
        return record

    async def verify_delivery(
        self,
        draft: DraftReply,
        *,
        expected_message_id: str = "",
        since_ms: int = 0,
    ) -> SendVerification:
        if self.verify_error:
            return SendVerification(
                found=False, message_id="", thread_match=False,
                recipients_match=False, attachments_match=False,
                subject_match=False, checked_ok=False,
            )
        records = [
            r
            for r in self.sent_store
            if r.thread_id == draft.thread_id
            and (not expected_message_id or r.message_id == expected_message_id)
        ]
        if self.verify_delay_ok and not getattr(self, "_verify_second_pass", False):
            records = []  # first call: not visible yet
            self._verify_second_pass = True
        if not records:
            return SendVerification(
                found=False, message_id="", thread_match=False,
                recipients_match=False, attachments_match=False,
                subject_match=False, checked_ok=True,
            )
        record = records[-1]
        expected_recipients = {a.email.casefold() for a in draft.to}
        recipients_match = expected_recipients <= {r.casefold() for r in record.recipients}
        expected_files = {a.filename.casefold() for a in draft.attachments}
        attachments_match = expected_files <= {f.casefold() for f in record.attachment_filenames}
        return SendVerification(
            found=True,
            message_id=record.message_id,
            thread_match=record.thread_id == draft.thread_id,
            recipients_match=recipients_match,
            attachments_match=attachments_match,
            subject_match=record.subject.casefold() == draft.subject.casefold(),
            checked_ok=True,
        )


@dataclass
class FakeReplyBot:
    """Minimal TelegramBot surface for the responder: queue + confirmation."""

    allowed: bool = True
    default_confirmation: bool = True
    confirmations: dict[int, bool] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)
    drafts_shown: list[DraftReply] = field(default_factory=list)
    resend_offers: list[int] = field(default_factory=list)
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

    async def send_draft_for_confirmation(
        self, draft: DraftReply, *, user_id: int = 0, draft_id: int = 0
    ) -> int:
        self.drafts_shown.append(draft)
        self.confirmations[draft_id] = self.default_confirmation
        return draft_id

    async def wait_for_confirmation(
        self, draft_id: int, timeout_seconds: float = 900.0
    ) -> bool:
        return self.confirmations.get(draft_id, False)

    async def send_notice(self, text: str) -> int:
        self.notices.append(text)
        return self._next_draft_id

    async def offer_resend(self, draft_id: int, user_id: int = 0) -> int:
        self.resend_offers.append(draft_id)
        return self._next_draft_id

    def register_resend_callback(self, callback) -> None:
        return None


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
