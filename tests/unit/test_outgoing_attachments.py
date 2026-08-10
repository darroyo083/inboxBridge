"""Outgoing attachment tests: filename sanitation, MIME construction,
size/count limits, and verification helper coverage."""

from __future__ import annotations

import asyncio
import base64
import email
from email.message import Message
from pathlib import Path

from inboxbridge.config import Settings
from inboxbridge.gmail.client import GmailClient
from inboxbridge.models import DraftReply, EmailAddress, OutgoingAttachment
from inboxbridge.telegram.bot import _guess_mime, sanitize_filename
from tests.mocks.gmail import FakeGmailService


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        SEND_EMAILS=True,
        gmail_user_id="me",
        PDF_PASSWORD="",
        outgoing_attachment_max_count=5,
        outgoing_attachment_max_bytes=10 * 1024 * 1024,
    )


def make_draft(attachments: tuple[OutgoingAttachment, ...] = ()) -> DraftReply:
    return DraftReply(
        thread_id="t1",
        subject="Re: Projektbericht",
        to=[EmailAddress("Anna Muster", "anna@example.com")],
        cc=[],
        body="Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen",
        attachments=attachments,
    )


class TestSanitizeFilename:
    def test_strips_directory_traversal(self) -> None:
        assert sanitize_filename("../../etc/passwd") == "passwd"
        assert sanitize_filename("C:\\fakepath\\scan.pdf") == "scan.pdf"
        assert sanitize_filename("..\\..\\evil.txt") == "evil.txt"

    def test_strips_control_characters(self) -> None:
        assert sanitize_filename("bad\x00\x1fname.txt") == "badname.txt"

    def test_dotdot_becomes_attachment(self) -> None:
        assert sanitize_filename("..") == "attachment"
        assert sanitize_filename(".") == "attachment"
        assert sanitize_filename("") == "attachment"

    def test_long_names_are_capped_with_suffix_kept(self) -> None:
        name = sanitize_filename("x" * 300 + ".pdf")
        assert len(name) <= 180
        assert name.endswith(".pdf")

    def test_normal_names_pass_through(self) -> None:
        assert sanitize_filename("factura_2026.pdf") == "factura_2026.pdf"


class TestGuessMime:
    def test_known_extension(self) -> None:
        assert _guess_mime("scan.pdf") == "application/pdf"

    def test_unknown_extension_defaults_octet(self) -> None:
        # No dot → no extension → no mapping on any platform.
        assert _guess_mime("attachment") == "application/octet-stream"


class TestOutgoingMime:
    def test_attachment_is_included_in_send_payload(self, tmp_path: Path) -> None:
        pdf = tmp_path / "factura.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        txt = tmp_path / "nota.txt"
        txt.write_bytes(b"contenido de la nota")
        attachments = (
            OutgoingAttachment(
                filename="factura.pdf",
                mime_type="application/pdf",
                size_bytes=pdf.stat().st_size,
                path=str(pdf),
            ),
            OutgoingAttachment(
                filename="nota.txt",
                mime_type="text/plain",
                size_bytes=txt.stat().st_size,
                path=str(txt),
            ),
        )
        service = FakeGmailService(
            {
                ("users", "threads", "get"): {"id": "t1", "messages": []},
                ("users", "messages", "send"): {"id": "sent-1"},
            }
        )
        client = GmailClient(make_settings(), service=service)
        asyncio.run(client.send_reply(make_draft(attachments)))

        call = next(c for c in service.calls if c[0] == ("users", "messages", "send"))
        raw = call[1]["body"]["raw"]
        assert call[1]["body"]["threadId"] == "t1"  # same-thread semantics preserved
        mime: Message = email.message_from_bytes(base64.urlsafe_b64decode(raw))
        payloads = [part for part in mime.walk() if part is not mime]
        filenames = {p.get_filename() for p in payloads if p.get_filename()}
        assert filenames == {"factura.pdf", "nota.txt"}
        contents = {p.get_payload(decode=True) for p in payloads}
        assert b"%PDF-1.4 fake" in contents
        assert b"contenido de la nota" in contents

    def test_missing_temp_file_fails_definitively(self, tmp_path: Path) -> None:
        attachments = (
            OutgoingAttachment(
                filename="gone.pdf",
                mime_type="application/pdf",
                size_bytes=5,
                path=str(tmp_path / "does-not-exist.pdf"),
            ),
        )
        service = FakeGmailService(
            {
                ("users", "threads", "get"): {"id": "t1", "messages": []},
                ("users", "messages", "send"): {"id": "sent-1"},
            }
        )
        client = GmailClient(make_settings(), service=service)
        try:
            asyncio.run(client.send_reply(make_draft(attachments)))
        except Exception as exc:
            assert "gone" in str(exc)
        else:
            raise AssertionError("expected a send failure for a missing temp file")

    def test_weird_mime_type_falls_back_to_octet_stream(self, tmp_path: Path) -> None:
        blob = tmp_path / "blob.bin"
        blob.write_bytes(b"data")
        attachments = (
            OutgoingAttachment(
                filename="blob.bin",
                mime_type="not-a-mime",
                size_bytes=4,
                path=str(blob),
            ),
        )
        service = FakeGmailService(
            {
                ("users", "threads", "get"): {"id": "t1", "messages": []},
                ("users", "messages", "send"): {"id": "sent-1"},
            }
        )
        client = GmailClient(make_settings(), service=service)
        asyncio.run(client.send_reply(make_draft(attachments)))
        call = next(c for c in service.calls if c[0] == ("users", "messages", "send"))
        mime: Message = email.message_from_bytes(
            base64.urlsafe_b64decode(call[1]["body"]["raw"])
        )
        part = next(
            p for p in mime.walk() if p is not mime and p.get_filename() == "blob.bin"
        )
        assert part.get_content_type() == "application/octet-stream"


class TestMimeInjectionSafety:
    def test_sanitize_never_allows_header_injection(self) -> None:
        # Newlines in filenames would let a crafted name inject headers.
        for candidate in ("evil\r\nBcc: victim@example.com", "a\nb.pdf"):
            cleaned = sanitize_filename(candidate)
            assert "\r" not in cleaned and "\n" not in cleaned
