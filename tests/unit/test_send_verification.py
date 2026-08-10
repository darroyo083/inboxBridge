"""Unit tests for GmailClient.verify_delivery — deterministic reconciliation.

Covers: message-id path, thread-search path (account From filter), recipient
mismatch, attachment mismatch, subject equivalence, 404 fallback, and
checked_ok=False when Gmail is unreachable.
"""

from __future__ import annotations

import asyncio
import email
from email.message import EmailMessage
from typing import Any

from inboxbridge.config import Settings
from inboxbridge.gmail.client import GmailClient
from inboxbridge.models import DraftReply, EmailAddress, OutgoingAttachment
from tests.mocks.gmail import FakeGmailService

Route = tuple[str, ...]


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        SEND_EMAILS=True,
        gmail_user_id="me",
        PDF_PASSWORD="",
    )


def make_draft(*, attachments: tuple[OutgoingAttachment, ...] = ()) -> DraftReply:
    return DraftReply(
        thread_id="t1",
        subject="Re: Projektbericht",
        to=[EmailAddress("Anna Muster", "anna@example.com")],
        cc=[],
        body="Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen",
        attachments=attachments,
    )


def raw_message(
    *,
    sender: str = "daniel@example.com",
    to: str = "Anna Muster <anna@example.com>",
    subject: str = "Re: Projektbericht",
    attachments: list[tuple[str, str, str, bytes]] | None = None,
) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.set_content("Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen")
    for filename, maintype, subtype, data in attachments or []:
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    return msg.as_bytes()


def full_payload(raw: bytes, message_id: str) -> dict[str, object]:
    """Build a users.messages.get format=full style payload."""
    msg = email.message_from_bytes(raw)
    headers = [{"name": k, "value": v} for k, v in msg.items()]
    parts: list[dict[str, object]] = []
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        ctype = part.get_content_type()
        data = part.get_payload(decode=True) or b""
        if part.get_filename():
            parts.append(
                {
                    "partId": str(len(parts)),
                    "mimeType": ctype,
                    "filename": part.get_filename(),
                    "body": {"size": len(data)},
                }
            )
    return {
        "id": message_id,
        "threadId": "t1",
        "labelIds": ["SENT"],
        "internalDate": "1754400000000",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": headers,
            "parts": parts,
        },
    }


def message_route(*payloads: dict[str, object]) -> Any:
    """users.messages.get responder dispatching on the requested message id."""
    by_id = {str(p["id"]): p for p in payloads}

    def route(kwargs: dict[str, Any]) -> Any:
        return by_id.get(str(kwargs.get("id")), next(iter(by_id.values())))

    return route


def thread_route(*payloads: dict[str, object]) -> Any:
    return {"id": "t1", "historyId": "300", "messages": list(payloads)}


def run_verify(
    client: GmailClient,
    expected_message_id: str,
    *,
    since_ms: int = 0,
    draft: DraftReply | None = None,
):
    return asyncio.run(
        client.verify_delivery(
            draft or make_draft(),
            expected_message_id=expected_message_id,
            since_ms=since_ms,
        )
    )


class TestVerifyByMessageId:
    def test_verified_when_message_matches(self) -> None:
        payload = full_payload(raw_message(), "sent-1")
        service = FakeGmailService({("users", "messages", "get"): message_route(payload)})
        result = run_verify(GmailClient(make_settings(), service=service), "sent-1")
        assert result.verified
        assert result.message_id == "sent-1"
        assert result.thread_match and result.recipients_match and result.subject_match

    def test_wrong_thread_is_not_verified(self) -> None:
        payload = full_payload(raw_message(), "sent-1")
        payload["threadId"] = "t2"
        service = FakeGmailService({("users", "messages", "get"): message_route(payload)})
        result = run_verify(GmailClient(make_settings(), service=service), "sent-1")
        assert not result.verified
        assert not result.thread_match

    def test_missing_recipient_is_not_verified(self) -> None:
        payload = full_payload(raw_message(to="Other <other@example.com>"), "sent-1")
        service = FakeGmailService({("users", "messages", "get"): message_route(payload)})
        result = run_verify(GmailClient(make_settings(), service=service), "sent-1")
        assert not result.verified
        assert not result.recipients_match

    def test_attachment_present_verified_absent_not(self) -> None:
        att = OutgoingAttachment(
            filename="factura.pdf",
            mime_type="application/pdf",
            size_bytes=10,
            path="C:/tmp/factura.pdf",
        )
        with_att = FakeGmailService(
            {
                ("users", "messages", "get"): message_route(
                    full_payload(
                        raw_message(
                            attachments=[("factura.pdf", "application", "pdf", b"%PDF-1.4")]
                        ),
                        "sent-1",
                    )
                )
            }
        )
        assert run_verify(
            GmailClient(make_settings(), service=with_att),
            "sent-1",
            draft=make_draft(attachments=(att,)),
        ).verified

        without_att = FakeGmailService(
            {("users", "messages", "get"): message_route(full_payload(raw_message(), "sent-1"))}
        )
        result = run_verify(
            GmailClient(make_settings(), service=without_att),
            "sent-1",
            draft=make_draft(attachments=(att,)),
        )
        assert not result.verified
        assert not result.attachments_match

    def test_subject_equivalence_ignores_re_prefix(self) -> None:
        payload = full_payload(raw_message(subject="Re[2]: Projektbericht"), "sent-1")
        service = FakeGmailService({("users", "messages", "get"): message_route(payload)})
        result = run_verify(GmailClient(make_settings(), service=service), "sent-1")
        assert result.subject_match
        assert result.verified


class TestVerifyByThreadSearch:
    def test_finds_own_message_and_skips_incoming(self) -> None:
        incoming = full_payload(
            raw_message(
                sender="anna@example.com",
                to="daniel@example.com",
                subject="Projektbericht",
            ),
            "in-1",
        )
        incoming["internalDate"] = "1754300000000"
        outgoing = full_payload(raw_message(), "sent-9")
        service = FakeGmailService(
            {
                ("users", "threads", "get"): thread_route(incoming, outgoing),
                ("users", "messages", "get"): message_route(incoming, outgoing),
                ("users", "getProfile"): {"emailAddress": "daniel@example.com"},
            }
        )
        result = run_verify(
            GmailClient(make_settings(), service=service), "", since_ms=1754399000000
        )
        assert result.verified
        assert result.message_id == "sent-9"

    def test_own_latest_message_matches_when_both_outgoing(self) -> None:
        older = full_payload(raw_message(subject="Projektbericht"), "sent-1")
        older["internalDate"] = "1754300000000"
        newer = full_payload(raw_message(), "sent-9")
        service = FakeGmailService(
            {
                ("users", "threads", "get"): thread_route(older, newer),
                ("users", "messages", "get"): message_route(older, newer),
                ("users", "getProfile"): {"emailAddress": "daniel@example.com"},
            }
        )
        result = run_verify(
            GmailClient(make_settings(), service=service), "", since_ms=1754399000000
        )
        assert result.verified
        assert result.message_id == "sent-9"

    def test_not_found_returns_checked_ok_true(self) -> None:
        incoming = full_payload(
            raw_message(sender="anna@example.com", to="daniel@example.com"), "in-1"
        )
        service = FakeGmailService(
            {
                ("users", "threads", "get"): thread_route(incoming),
                ("users", "messages", "get"): message_route(incoming),
                ("users", "getProfile"): {"emailAddress": "daniel@example.com"},
            }
        )
        result = run_verify(GmailClient(make_settings(), service=service), "", since_ms=0)
        assert not result.found
        assert result.checked_ok
        assert not result.verified

    def test_gmail_unreachable_is_inconclusive(self) -> None:
        class Boom:
            def execute(self):
                raise OSError("network down")

        service = FakeGmailService({("users", "threads", "get"): Boom()})
        result = run_verify(GmailClient(make_settings(), service=service), "", since_ms=0)
        assert not result.checked_ok
        assert not result.verified

    def test_404_on_known_id_falls_back_to_thread_search(self) -> None:
        from googleapiclient.errors import HttpError

        def not_found_for_unknown_route(kwargs: dict[str, Any]) -> Any:
            if str(kwargs.get("id")) == "sent-1":
                class _Resp:
                    status = 404
                    reason = "Not Found"

                return HttpError(_Resp(), b"not found", uri="https://gmail.googleapis.com")
            return outgoing

        outgoing = full_payload(raw_message(), "sent-9")
        service = FakeGmailService(
            {
                ("users", "messages", "get"): not_found_for_unknown_route,
                ("users", "threads", "get"): thread_route(outgoing),
                ("users", "getProfile"): {"emailAddress": "daniel@example.com"},
            }
        )
        result = run_verify(GmailClient(make_settings(), service=service), "sent-1", since_ms=0)
        assert result.verified
        assert result.message_id == "sent-9"
