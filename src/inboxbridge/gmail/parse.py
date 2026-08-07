"""RFC822/MIME parsing into a cleaned, LLM-ready text representation.

Emails are UNTRUSTED DATA. This module only ever interprets them as text:
no scripts, no links, no attachments are executed or rendered, and body
content is never treated as instructions (prompt-injection defense lives in
the LLM layer; here we only sanitize the markup).

Strategy for body selection: the HTML part is the primary source (richer
structure for summaries); a plain-text part is the fallback when no usable
HTML exists. HTML is stripped with BeautifulSoup+lxml, trackers (images,
scripts, hidden elements) are removed, and repeated quotes/signatures are
collapsed with basic heuristics. Names, dates and numbers are preserved
intentionally — cleanup never runs regexes over them.
"""

from __future__ import annotations

import email
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime

from bs4 import BeautifulSoup, Comment  # type: ignore[import-untyped]
from bs4.element import Tag  # type: ignore[import-untyped]

from ..models import EmailAddress

# Bound for LLM input; arbitrarily large bodies are truncated (documented).
MAX_BODY_CHARS = 50_000

# Elements whose content is never wanted (trackers, chrome, scripts).
_JUNK_TAGS = {
    "script", "style", "iframe", "object", "embed", "noscript", "svg", "canvas",
    "form", "input", "select", "textarea", "button", "link", "meta", "head",
    "template", "img", "picture", "video", "audio", "source", "track", "area",
    "map", "nav", "dialog",
}
_BLOCK_TAGS = {
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "table",
    "blockquote", "section", "article", "ul", "ol", "pre", "hr", "address",
}
_QUOTED_LINE = re.compile(r"^[ \t]*>")
_WROTE_LINE = re.compile(r"^On .+ wrote:$")
_MOBILE_SIG = re.compile(
    r"^(Sent|Gesendet|Enviado|Inviato|Envoyé) (from|von|de|da|dal|depuis) "
    r"(my )?(iPhone|iPad|Android|Galaxy|Outlook|Samsung|phone)",
    re.IGNORECASE,
)
_SIG_DELIM = re.compile(r"^--[ ]*$")


@dataclass(frozen=True)
class RawAttachment:
    """Attachment bytes in memory only — never persisted by this project."""

    filename: str
    mime_type: str
    size_bytes: int
    content: bytes


@dataclass(frozen=True)
class ParsedMessage:
    """Parsed envelope + cleaned text of one RFC822 message."""

    subject: str
    sender: EmailAddress
    recipients: list[EmailAddress]
    date_iso: str
    body_text: str
    attachments: list[RawAttachment] = field(default_factory=list)


def parse_rfc822(raw: bytes | str, *, internal_date_ms: int | None = None) -> ParsedMessage:
    """Parse raw RFC822 bytes (or str) into a ParsedMessage.

    ``internal_date_ms`` (Gmail's ``internalDate``, epoch ms) wins over the
    ``Date`` header when provided — it is the reliable delivery timestamp.
    """
    msg = (
        email.message_from_bytes(raw)
        if isinstance(raw, bytes)
        else email.message_from_string(raw)
    )
    subject = _clean_header(_decode_header(msg.get("Subject")))
    sender = _first_address(_decode_header(msg.get("From")))
    address_fields = [
        str(v) for v in (*msg.get_all("To", []), *msg.get_all("Cc", []))
    ]
    recipients = [
        _make_address(name, addr) for name, addr in getaddresses(address_fields) if addr
    ]
    date_iso = _header_date_iso(msg, internal_date_ms)
    body_text, attachments = _extract_parts(msg)
    return ParsedMessage(
        subject=subject,
        sender=sender,
        recipients=recipients,
        date_iso=date_iso,
        body_text=body_text[:MAX_BODY_CHARS],
        attachments=attachments,
    )


def clean_html(html: str) -> str:
    """Strip HTML to plain text: trackers removed, block structure kept."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")  # stdlib fallback, no deps
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for tag in soup.find_all(_JUNK_TAGS):
        tag.decompose()
    for tag in soup.find_all(_is_hidden):
        tag.decompose()
    for tag in soup.find_all("br"):
        tag.replace_with("\n")
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append("\n")
    for tag in soup.find_all("a"):
        tag.unwrap()
    text = soup.get_text()
    return _normalize_text(text)


def collapse_quotes(text: str) -> str:
    """Drop quoted lines, "On ... wrote:" leads, and signature blocks."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _QUOTED_LINE.match(line):
            continue
        if _WROTE_LINE.match(stripped):
            continue
        if _MOBILE_SIG.match(stripped):
            break
        if _SIG_DELIM.match(stripped):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _extract_parts(msg: Message) -> tuple[str, list[RawAttachment]]:
    html_parts: list[str] = []
    text_parts: list[str] = []
    attachments: list[RawAttachment] = []
    for part in msg.walk():
        ctype = part.get_content_type()
        disposition = part.get_content_disposition() or ""
        filename = _decode_header(part.get_filename())
        is_body = ctype in ("text/plain", "text/html")
        is_image = ctype.startswith("image/")
        is_attachment = disposition == "attachment" or (
            bool(filename)
            and disposition != "inline"
            and not is_body
            and not is_image
        )
        if is_attachment:
            attachments.append(_raw_attachment(part, filename))
            continue
        if ctype == "text/html":
            html_parts.append(_decode_text(part))
        elif ctype == "text/plain":
            text_parts.append(_decode_text(part))
    for html in html_parts:
        cleaned = clean_html(html)
        if cleaned:
            return collapse_quotes(cleaned), attachments
    for text in text_parts:
        if text.strip():
            return collapse_quotes(text), attachments
    return "", attachments


def _raw_attachment(part: Message, filename: str) -> RawAttachment:
    content = _decode_bytes(part)
    return RawAttachment(
        filename=filename or "attachment",
        mime_type=part.get_content_type(),
        size_bytes=len(content),
        content=content,
    )


def _decode_text(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    return str(payload)


def _decode_bytes(part: Message) -> bytes:
    payload = part.get_payload(decode=True)
    if payload is None:
        return b""
    if isinstance(payload, bytes):
        return payload
    return str(payload).encode("utf-8", errors="replace")


def _decode_header(value: str | None) -> str:
    if not value:
        return ""
    return str(email.header.make_header(email.header.decode_header(value)))


def _clean_header(value: str) -> str:
    return re.sub(r"[\r\n\t]+", " ", value).strip()


def _first_address(header: str) -> EmailAddress:
    for name, addr in getaddresses([header]):
        if addr:
            return _make_address(name, addr)
    return EmailAddress(name="", email="")


def _make_address(name: str, addr: str) -> EmailAddress:
    return EmailAddress(name=_clean_header(_decode_header(name)), email=addr.strip())


def _header_date_iso(msg: Message, internal_date_ms: int | None) -> str:
    if internal_date_ms:
        try:
            return datetime.fromtimestamp(internal_date_ms / 1000, tz=UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            pass
    try:
        parsed = parsedate_to_datetime(msg.get("Date") or "")
    except (TypeError, ValueError, OverflowError):
        return ""
    if parsed is None:
        return ""
    return parsed.astimezone(UTC).isoformat()


def _normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_hidden(tag: Tag) -> bool:
    style = (tag.get("style") or "").lower().replace(" ", "")
    if "display:none" in style or "visibility:hidden" in style:
        return True
    classes = set(tag.get("class") or [])
    return bool(classes & {"hidden", "ad", "ads", "advertisement", "sponsor", "tracking"})
