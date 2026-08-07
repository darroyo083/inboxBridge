"""Gmail API client implementing the ``contracts.GmailClient`` protocol.

Transport decision: ``google-api-python-client`` (already pinned in
``pyproject.toml``) with google-auth credentials — no new dependency. Its
HTTP layer auto-refreshes OAuth tokens on 401. Blocking API calls are
wrapped in ``asyncio.to_thread`` so the event loop never stalls.

Reply semantics: ``send_reply`` always continues the SAME thread — the
request body carries ``threadId`` and the MIME headers set ``In-Reply-To`` /
``References`` derived from the thread's latest message, with the thread's
subject preserved (exactly one ``Re:`` prefix). Replies are plain replies,
never reply-all, unless the draft explicitly lists CC recipients.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from email.message import EmailMessage
from typing import Any

from googleapiclient.discovery import build  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

from ..config import Settings
from ..models import DraftReply, ParsedEmail, ThreadContext, ThreadMessage
from .attachments import extract_attachments
from .auth import get_credentials
from .parse import parse_rfc822

logger = logging.getLogger(__name__)

# Reply context is bounded: only the most recent N thread messages are parsed.
MAX_THREAD_CONTEXT_MESSAGES = 10

_RE_PREFIX = re.compile(r"^Re(\[\d+\])?:", re.IGNORECASE)


class GmailAPIError(RuntimeError):
    """Gmail API call failed (transport or API error)."""


class SendingDisabledError(GmailAPIError):
    """Kill switch: SEND_EMAILS=false makes sending technically impossible."""


class GmailClient:
    """Typed Gmail client; satisfies contracts.GmailClient."""

    def __init__(
        self,
        settings: Settings,
        *,
        service: Any | None = None,
        credentials: Any | None = None,
    ) -> None:
        self._settings = settings
        self._user_id = settings.gmail_user_id
        self._service = (
            service
            if service is not None
            else build("gmail", "v1", credentials=credentials or get_credentials(settings))
        )

    async def fetch_message(self, message_id: str) -> ParsedEmail:
        resp: dict[str, Any] = await self._run(
            self._service.users().messages().get(
                userId=self._user_id, id=message_id, format="full"
            )
        )
        raw = resp.get("raw")
        if not raw:
            raise GmailAPIError(f"message {message_id}: API response has no raw payload")
        parsed = parse_rfc822(
            _b64url_decode(str(raw)), internal_date_ms=_to_int(resp.get("internalDate"))
        )
        attachments = extract_attachments(parsed.attachments, self._settings)
        return ParsedEmail(
            message_id=message_id,
            thread_id=str(resp.get("threadId") or ""),
            history_id=_to_int(resp.get("historyId")) or 0,
            subject=parsed.subject,
            sender=parsed.sender,
            recipients=parsed.recipients,
            date_iso=parsed.date_iso,
            body_text=parsed.body_text,
            attachments=attachments,
        )

    async def fetch_thread_context(self, thread_id: str) -> ThreadContext:
        thread_resp: dict[str, Any] = await self._run(
            self._service.users().threads().get(
                userId=self._user_id, id=thread_id, format="metadata"
            )
        )
        messages = thread_resp.get("messages") or []
        subject = _first_subject(messages)
        history_id = _to_int(thread_resp.get("historyId")) or 0
        ordered = sorted(messages, key=lambda m: _to_int(m.get("internalDate")) or 0)
        recent = ordered[-MAX_THREAD_CONTEXT_MESSAGES:]
        thread_messages: list[ThreadMessage] = []
        for m in recent:
            mid = str(m.get("id") or "")
            if not mid:
                continue
            msg_resp: dict[str, Any] = await self._run(
                self._service.users().messages().get(
                    userId=self._user_id, id=mid, format="raw"
                )
            )
            parsed = parse_rfc822(
                _b64url_decode(str(msg_resp.get("raw") or "")),
                internal_date_ms=_to_int(msg_resp.get("internalDate")),
            )
            thread_messages.append(
                ThreadMessage(
                    message_id=mid,
                    from_=parsed.sender,
                    date_iso=parsed.date_iso,
                    body_text=parsed.body_text,
                    snippet=str(m.get("snippet") or ""),
                )
            )
        return ThreadContext(
            thread_id=thread_id, subject=subject, messages=thread_messages, history_id=history_id
        )

    async def send_reply(self, draft: DraftReply) -> str:
        if not self._settings.send_emails:
            raise SendingDisabledError(
                "SEND_EMAILS=false: sending is disabled (kill switch); "
                "set SEND_EMAILS=true to enable"
            )
        subject = ensure_re_prefix(draft.subject)
        in_reply_to, references = await self._threading_headers(draft.thread_id, draft)

        mime = EmailMessage()
        mime["To"] = ", ".join(str(a) for a in draft.to)
        if draft.cc:
            mime["Cc"] = ", ".join(str(a) for a in draft.cc)
        mime["Subject"] = subject
        if in_reply_to:
            mime["In-Reply-To"] = in_reply_to
        if references:
            mime["References"] = references
        mime.set_content(draft.body)

        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")
        resp: dict[str, Any] = await self._run(
            self._service.users().messages().send(
                userId=self._user_id,
                body={"raw": raw, "threadId": draft.thread_id},
            )
        )
        return str(resp.get("id") or "")

    async def _threading_headers(self, thread_id: str, draft: DraftReply) -> tuple[str, str]:
        """Best-effort In-Reply-To/References from the thread's latest message.

        Falls back to the draft's own values when the thread lookup fails, so
        a threading-header glitch never blocks sending.
        """
        in_reply_to, references = draft.in_reply_to, draft.references
        try:
            thread_resp: dict[str, Any] = await self._run(
                self._service.users().threads().get(
                    userId=self._user_id, id=thread_id, format="metadata"
                )
            )
            messages = thread_resp.get("messages") or []
            if not messages:
                return in_reply_to, references
            latest = max(messages, key=lambda m: _to_int(m.get("internalDate")) or 0)
            headers = {
                h.get("name", "").lower(): str(h.get("value") or "")
                for h in (latest.get("payload") or {}).get("headers") or []
            }
            parent_id = headers.get("message-id", "")
            if not in_reply_to and parent_id:
                in_reply_to = parent_id
            refs: list[str] = []
            for chunk in (headers.get("references", ""), parent_id, draft.references):
                for rid in chunk.split():
                    if rid and rid not in refs:
                        refs.append(rid)
            references = " ".join(refs)
        except Exception:
            logger.exception("could not resolve threading headers for thread %s", thread_id)
        return in_reply_to, references

    async def _run(self, request: Any) -> Any:
        try:
            return await asyncio.to_thread(request.execute)
        except HttpError as exc:
            raise GmailAPIError(f"Gmail API error {exc.status_code}") from exc


def ensure_re_prefix(subject: str) -> str:
    """Preserve the thread subject: exactly one leading 'Re:' prefix.

    ``Re:``, ``RE:`` and ``Re[2]:`` forms are left untouched; otherwise a
    single ``Re: `` is prepended. Control characters are stripped first
    (the subject is a header we write — newlines would be injection).
    """
    cleaned = re.sub(r"[\r\n\t]+", " ", subject).strip()
    if not cleaned:
        return "Re:"
    if _RE_PREFIX.match(cleaned):
        return cleaned
    return f"Re: {cleaned}"


def _b64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_subject(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        for h in (m.get("payload") or {}).get("headers") or []:
            if str(h.get("name") or "").lower() == "subject" and h.get("value"):
                return str(h.get("value"))
    return ""
