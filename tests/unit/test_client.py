"""Unit tests for gmail/client.py — Gmail API faked, no credentials needed."""

from __future__ import annotations

import base64
import email
from email.message import Message

import pytest

from inboxbridge.config import Settings
from inboxbridge.gmail.client import (
    GmailClient,
    SendingDisabledError,
    ensure_re_prefix,
)
from inboxbridge.models import DraftReply, EmailAddress, ParsedEmail, ThreadContext
from tests.mocks.gmail import FakeGmailService, build_raw_email

Route = tuple[str, ...]


def make_settings(*, send_emails: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        SEND_EMAILS=send_emails,
        gmail_user_id="me",
        PDF_PASSWORD="",
        attachment_max_bytes=10 * 1024 * 1024,
        attachment_max_text_chars=20_000,
        attachment_max_count=5,
    )


def full_response(
    raw: bytes,
    *,
    message_id: str,
    thread_id: str,
    history_id: str = "100",
    internal_date: str = "1754400000000",
) -> dict[str, object]:
    return {
        "id": message_id,
        "threadId": thread_id,
        "historyId": history_id,
        "internalDate": internal_date,
        "raw": base64.urlsafe_b64encode(raw).decode("ascii"),
    }


def thread_response(
    *messages: dict[str, object], thread_id: str, history_id: str = "200"
) -> dict[str, object]:
    return {"id": thread_id, "historyId": history_id, "messages": list(messages)}


def send_call_mime(client: GmailClient) -> Message:
    """Decode the MIME payload of the messages.send request body."""
    call = next(
        c for c in client._service.calls if c[0] == ("users", "messages", "send")
    )
    raw = call[1]["body"]["raw"]
    return email.message_from_bytes(base64.urlsafe_b64decode(raw))


def thread_message(
    message_id: str,
    internal_date: str,
    *,
    subject: str = "",
    message_id_header: str = "",
    references_header: str = "",
    in_reply_to_header: str = "",
) -> dict[str, object]:
    headers: list[dict[str, str]] = []
    if subject:
        headers.append({"name": "Subject", "value": subject})
    if message_id_header:
        headers.append({"name": "Message-ID", "value": message_id_header})
    if references_header:
        headers.append({"name": "References", "value": references_header})
    if in_reply_to_header:
        headers.append({"name": "In-Reply-To", "value": in_reply_to_header})
    return {
        "id": message_id,
        "threadId": "t1",
        "internalDate": internal_date,
        "payload": {"headers": headers},
    }


class TestFetchMessage:
    async def test_returns_parsed_email(self) -> None:
        raw = build_raw_email(
            subject="Termin am 07.08.2025",
            sender="Alice <alice@example.com>",
            body_html="<p>Hola, el <b>termino</b> es el 07.08.2025.</p>",
            body_text="plain",
            attachments=[("nota.txt", "text", "plain", b"contenido de la nota")],
        )
        service = FakeGmailService(
            {
                ("users", "messages", "get"): full_response(
                    raw, message_id="m1", thread_id="t1"
                )
            }
        )
        client = GmailClient(make_settings(), service=service)
        email_ = await client.fetch_message("m1")

        assert isinstance(email_, ParsedEmail)
        assert email_.message_id == "m1"
        assert email_.thread_id == "t1"
        assert email_.history_id == 100
        assert email_.subject == "Termin am 07.08.2025"
        assert email_.sender == EmailAddress("Alice", "alice@example.com")
        assert "termino" in email_.body_text
        assert email_.attachments[0].filename == "nota.txt"
        assert email_.attachments[0].extracted_text == "contenido de la nota"

    async def test_requests_raw_format(self) -> None:
        """fetch_message must request format='raw' so the API returns raw RFC822."""
        raw = build_raw_email(subject="Tema", body_text="Cuerpo.", body_html=None)
        service = FakeGmailService(
            {
                ("users", "messages", "get"): full_response(
                    raw, message_id="m1", thread_id="t1"
                )
            }
        )
        client = GmailClient(make_settings(), service=service)
        await client.fetch_message("m1")

        calls = [c for c in service.calls if c[0] == ("users", "messages", "get")]
        assert len(calls) == 1
        _path, kwargs = calls[0]
        assert kwargs["format"] == "raw"

    async def test_raises_when_no_raw(self) -> None:
        service = FakeGmailService({("users", "messages", "get"): {"id": "m1"}})
        client = GmailClient(make_settings(), service=service)
        with pytest.raises(RuntimeError):
            await client.fetch_message("m1")


class TestFetchThreadContext:
    async def test_returns_recent_messages_with_bodies(self) -> None:
        raw1 = build_raw_email(subject="Tema", body_text="Mensaje uno.", body_html=None)
        raw2 = build_raw_email(subject="Tema", body_text="Mensaje dos.", body_html=None)
        routes: dict[Route, object] = {
            ("users", "threads", "get"): thread_response(
                thread_message("m1", "100", subject="Tema"),
                thread_message("m2", "200"),
                thread_id="t1",
            ),
            ("users", "messages", "get"): lambda kwargs: full_response(
                raw1 if kwargs["id"] == "m1" else raw2,
                message_id=kwargs["id"],
                thread_id="t1",
            ),
        }
        client = GmailClient(make_settings(), service=FakeGmailService(routes))
        ctx = await client.fetch_thread_context("t1")

        assert isinstance(ctx, ThreadContext)
        assert ctx.thread_id == "t1"
        assert ctx.subject == "Tema"
        assert ctx.history_id == 200
        assert [m.message_id for m in ctx.messages] == ["m1", "m2"]
        assert ctx.messages[1].body_text == "Mensaje dos."


class TestSendReply:
    async def test_reply_continues_thread(self) -> None:
        routes: dict[Route, object] = {
            ("users", "threads", "get"): thread_response(
                thread_message(
                    "m2",
                    "200",
                    subject="Re: Tema",
                    message_id_header="<m2@example.com>",
                    references_header="<m1@example.com>",
                    in_reply_to_header="<m1@example.com>",
                ),
                thread_id="t1",
            ),
            ("users", "messages", "send"): {"id": "m3"},
        }
        service = FakeGmailService(routes)
        client = GmailClient(make_settings(send_emails=True), service=service)
        draft = DraftReply(
            thread_id="t1",
            subject="Re: Tema",
            to=[EmailAddress("Bob", "bob@example.com")],
            cc=[],
            body="Hallo Bob, gerne.",
        )
        new_id = await client.send_reply(draft)
        assert new_id == "m3"

        send_call = next(c for c in service.calls if c[0] == ("users", "messages", "send"))
        body = send_call[1]["body"]
        assert body["threadId"] == "t1"
        mime = email.message_from_bytes(base64.urlsafe_b64decode(body["raw"]))
        assert isinstance(mime, Message)
        assert mime["To"] == "Bob <bob@example.com>"
        assert mime["In-Reply-To"] == "<m2@example.com>"
        assert mime["References"] == "<m1@example.com> <m2@example.com>"
        assert mime["Subject"] == "Re: Tema"
        assert mime.get("Cc") is None

    async def test_default_is_not_reply_all(self) -> None:
        routes: dict[Route, object] = {
            ("users", "threads", "get"): thread_response(
                thread_message("m2", "200", message_id_header="<m2@x>"), thread_id="t1"
            ),
            ("users", "messages", "send"): {"id": "m3"},
        }
        client = GmailClient(make_settings(), service=FakeGmailService(routes))
        draft = DraftReply(
            thread_id="t1",
            subject="Tema",
            to=[EmailAddress("Alice", "alice@example.com")],
            cc=[],
            body="ok",
        )
        await client.send_reply(draft)
        mime = send_call_mime(client)
        assert mime["To"] == "Alice <alice@example.com>"
        assert mime.get("Cc") is None

    async def test_new_email_keeps_subject_without_re_prefix(self) -> None:
        """A threadless draft (compose/forward) must NOT get a 'Re:' prefix."""
        routes: dict[Route, object] = {
            ("users", "messages", "send"): {"id": "m3"},
        }
        client = GmailClient(make_settings(), service=FakeGmailService(routes))
        draft = DraftReply(
            thread_id="", subject="Presupuesto", to=[EmailAddress("R", "r@b.c")], cc=[], body="ok"
        )
        await client.send_reply(draft)
        mime = send_call_mime(client)
        assert mime["Subject"] == "Presupuesto"
        assert mime.get("In-Reply-To") is None
        send_call = next(c for c in client._service.calls if c[0] == ("users", "messages", "send"))
        assert "threadId" not in send_call[1]["body"]

    async def test_forward_keeps_fwd_prefix(self) -> None:
        routes: dict[Route, object] = {
            ("users", "messages", "send"): {"id": "m3"},
        }
        client = GmailClient(make_settings(), service=FakeGmailService(routes))
        draft = DraftReply(
            thread_id="",
            subject="Fwd: Presupuesto",
            to=[EmailAddress("R", "r@b.c")],
            cc=[],
            body="ok",
        )
        await client.send_reply(draft)
        mime = send_call_mime(client)
        assert mime["Subject"] == "Fwd: Presupuesto"

    async def test_spanish_translation_never_sent(self) -> None:
        """The display-only Spanish translation must never enter the sent MIME."""
        routes: dict[Route, object] = {
            ("users", "threads", "get"): thread_response(
                thread_message("m2", "200", message_id_header="<m2@x>"), thread_id="t1"
            ),
            ("users", "messages", "send"): {"id": "m3"},
        }
        client = GmailClient(make_settings(), service=FakeGmailService(routes))
        draft = DraftReply(
            thread_id="t1",
            subject="Tema",
            to=[EmailAddress("A", "a@b.c")],
            cc=[],
            body="Sehr geehrte Frau Muster, vielen Dank für Ihre Nachricht.",
            body_es="Estimada señora Muster, muchas gracias por su mensaje.",
        )
        await client.send_reply(draft)
        mime = send_call_mime(client)
        assert "vielen Dank" in mime.get_payload()
        assert "muchas gracias" not in mime.get_payload()

    async def test_subject_gets_re_prefix(self) -> None:
        routes: dict[Route, object] = {
            ("users", "threads", "get"): thread_response(
                thread_message("m2", "200"), thread_id="t1"
            ),
            ("users", "messages", "send"): {"id": "m3"},
        }
        client = GmailClient(make_settings(), service=FakeGmailService(routes))
        draft = DraftReply(
            thread_id="t1", subject="Tema", to=[EmailAddress("A", "a@b.c")], cc=[], body="ok"
        )
        await client.send_reply(draft)
        mime = send_call_mime(client)
        assert mime["Subject"] == "Re: Tema"

    async def test_falls_back_to_draft_headers_when_thread_unavailable(self) -> None:
        routes: dict[Route, object] = {
            ("users", "messages", "send"): {"id": "m3"},
        }
        client = GmailClient(make_settings(), service=FakeGmailService(routes))
        draft = DraftReply(
            thread_id="t1",
            subject="Tema",
            to=[EmailAddress("A", "a@b.c")],
            cc=[],
            body="ok",
            in_reply_to="<parent@x>",
            references="<g1@x> <g2@x>",
        )
        await client.send_reply(draft)
        mime = send_call_mime(client)
        assert mime["In-Reply-To"] == "<parent@x>"
        assert mime["References"] == "<g1@x> <g2@x>"

    async def test_kill_switch_blocks_sending(self) -> None:
        client = GmailClient(
            make_settings(send_emails=False), service=FakeGmailService({})
        )
        draft = DraftReply(thread_id="t1", subject="S", to=[], cc=[], body="x")
        with pytest.raises(SendingDisabledError):
            await client.send_reply(draft)

    async def test_returns_new_message_id(self) -> None:
        routes: dict[Route, object] = {
            ("users", "threads", "get"): thread_response(
                thread_message("m2", "200"), thread_id="t1"
            ),
            ("users", "messages", "send"): {"id": "new-123"},
        }
        client = GmailClient(make_settings(), service=FakeGmailService(routes))
        draft = DraftReply(thread_id="t1", subject="S", to=[], cc=[], body="x")
        assert await client.send_reply(draft) == "new-123"


class TestEnsureRePrefix:
    def test_prepends_when_missing(self) -> None:
        assert ensure_re_prefix("Hello") == "Re: Hello"

    def test_preserves_existing(self) -> None:
        assert ensure_re_prefix("Re: Hello") == "Re: Hello"
        assert ensure_re_prefix("RE: Hello") == "RE: Hello"
        assert ensure_re_prefix("Re[2]: Hello") == "Re[2]: Hello"

    def test_handles_empty_and_junk(self) -> None:
        assert ensure_re_prefix("") == "Re:"
        assert ensure_re_prefix("Re:\nX") == "Re: X"
        assert ensure_re_prefix("   ") == "Re:"
