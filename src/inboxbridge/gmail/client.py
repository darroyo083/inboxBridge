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
import os
import re
from email.message import EmailMessage
from email.utils import getaddresses
from typing import Any

import httplib2  # type: ignore[import-untyped]
from googleapiclient.discovery import build  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

from ..config import Settings
from ..models import (
    DraftReply,
    EmailAddress,
    OutgoingAttachment,
    ParsedEmail,
    SendVerification,
    ThreadContext,
    ThreadMessage,
)
from .attachments import extract_attachments
from .auth import get_credentials
from .parse import parse_rfc822

logger = logging.getLogger(__name__)

# Reply context is bounded: only the most recent N thread messages are parsed.
MAX_THREAD_CONTEXT_MESSAGES = 10
#: Bounded search window for reconciling a send without a message id.
MAX_RECONCILE_CANDIDATES = 20
#: Clock-skew margin for internalDate comparisons (send started vs. Gmail dates).
_SKEW_MS = 60_000

_RE_PREFIX = re.compile(r"^Re(\[\d+\])?:", re.IGNORECASE)

#: Transport-level failures after/before transmission. The send outcome is
#: UNKNOWN in these cases (the request may have reached Gmail): never retry
#: blindly — reconcile first.
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    TimeoutError,
    httplib2.HttpLib2Error,
)


class GmailAPIError(RuntimeError):
    """Gmail API call failed (transport or API error)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SendingDisabledError(GmailAPIError):
    """Kill switch: SEND_EMAILS=false makes sending technically impossible."""


class AmbiguousSendError(GmailAPIError):
    """The send request's outcome is unknown (transport error / uncertain 5xx).

    Gmail may or may not have accepted the message. The caller MUST reconcile
    against Gmail before offering any retry — a blind resend risks a
    duplicate email.
    """


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
        self._my_email: str | None = None
        self._service = (
            service
            if service is not None
            else build("gmail", "v1", credentials=credentials or get_credentials(settings))
        )

    async def fetch_message(self, message_id: str) -> ParsedEmail:
        resp: dict[str, Any] = await self._run(
            self._service.users().messages().get(
                userId=self._user_id, id=message_id, format="raw"
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
        for attachment in draft.attachments:
            _attach_outgoing(mime, attachment)

        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")
        resp: dict[str, Any] = await self._run(
            self._service.users().messages().send(
                userId=self._user_id,
                body={"raw": raw, "threadId": draft.thread_id},
            ),
            ambiguous=True,
        )
        return str(resp.get("id") or "")

    async def verify_delivery(
        self,
        draft: DraftReply,
        *,
        expected_message_id: str = "",
        since_ms: int = 0,
    ) -> SendVerification:
        """Reconcile a send attempt against Gmail (source of truth).

        - With a known message id: fetch it and check thread/recipients/
          subject/attachments.
        - Without one (response lost): search the thread for our own sent
          message with internalDate >= ``since_ms``.

        Never raises: any Gmail query failure yields ``checked_ok=False``
        (inconclusive — a retry is NOT offered on that evidence).
        """
        try:
            if expected_message_id:
                try:
                    resp: dict[str, Any] = await self._run(
                        self._service.users().messages().get(
                            userId=self._user_id,
                            id=expected_message_id,
                            format="full",
                        )
                    )
                except GmailAPIError as exc:
                    if exc.status_code == 404:
                        resp = {}  # id unknown → fall back to thread search
                    else:
                        raise
                if resp:
                    return _verify_message_payload(draft, resp)
            return await self._search_thread_for_sent(draft, since_ms)
        except Exception:
            logger.exception(
                "delivery verification failed for thread %s", draft.thread_id
            )
            return SendVerification(
                found=False,
                message_id="",
                thread_match=False,
                recipients_match=False,
                attachments_match=False,
                subject_match=False,
                checked_ok=False,
            )

    async def _search_thread_for_sent(
        self, draft: DraftReply, since_ms: int
    ) -> SendVerification:
        """Find OUR sent message in the thread (matches account From header)."""
        thread_resp: dict[str, Any] = await self._run(
            self._service.users().threads().get(
                userId=self._user_id, id=draft.thread_id, format="full"
            )
        )
        messages = thread_resp.get("messages") or []
        ordered = sorted(messages, key=lambda m: _to_int(m.get("internalDate")) or 0)
        floor = since_ms - _SKEW_MS
        candidates = [
            m
            for m in ordered[-MAX_RECONCILE_CANDIDATES:]
            if not since_ms or (_to_int(m.get("internalDate")) or 0) >= floor
        ]
        my_email = await self._my_email_address()
        for candidate in reversed(candidates):
            mid = str(candidate.get("id") or "")
            if not mid:
                continue
            full: dict[str, Any] = await self._run(
                self._service.users().messages().get(
                    userId=self._user_id, id=mid, format="full"
                )
            )
            payload = full.get("payload") or {}
            headers = {
                str(h.get("name") or "").lower(): str(h.get("value") or "")
                for h in payload.get("headers") or []
            }
            if my_email and headers.get("from", "").casefold() != my_email.casefold():
                continue  # incoming message in the same thread — not ours
            verification = _verify_message_payload(draft, full)
            if verification.found and verification.thread_match:
                return verification
        return SendVerification(
            found=False,
            message_id="",
            thread_match=False,
            recipients_match=False,
            attachments_match=False,
            subject_match=False,
            checked_ok=True,
        )

    async def _my_email_address(self) -> str:
        """Cache the account's own address (one profile call per process)."""
        if self._my_email is None:
            profile: dict[str, Any] = await self._run(
                self._service.users().getProfile(userId=self._user_id)
            )
            self._my_email = str(profile.get("emailAddress") or "")
        return self._my_email

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

    async def _run(self, request: Any, *, ambiguous: bool = False) -> Any:
        """Execute a Gmail request off the event loop.

        ``ambiguous=True`` (send operations): transport failures and 5xx
        responses are mapped to :class:`AmbiguousSendError` because the
        server-side outcome is unknown.
        """
        try:
            return await asyncio.to_thread(request.execute)
        except HttpError as exc:
            status = getattr(exc, "resp", None)
            code = int(getattr(status, "status", 0) or 0) if status is not None else 0
            if ambiguous and code >= 500:
                raise AmbiguousSendError(
                    f"Gmail send returned {code}: server-side outcome unknown",
                    status_code=code,
                ) from exc
            raise GmailAPIError(
                f"Gmail API error {code}" if code else f"Gmail API error: {exc}",
                status_code=code or None,
            ) from exc
        except _TRANSPORT_ERRORS as exc:
            if ambiguous:
                raise AmbiguousSendError(
                    f"Gmail send transport error ({type(exc).__name__}): "
                    "outcome unknown — reconcile before retrying"
                ) from exc
            raise GmailAPIError(
                f"Gmail transport error ({type(exc).__name__})"
            ) from exc


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


# ── delivery verification helpers ────────────────────────────────────────────


def _verify_message_payload(draft: DraftReply, message: dict[str, Any]) -> SendVerification:
    """Score one Gmail message against the draft's expected identity."""
    message_id = str(message.get("id") or "")
    thread_id = str(message.get("threadId") or "")
    payload = message.get("payload") or {}
    headers = {
        str(h.get("name") or "").lower(): str(h.get("value") or "")
        for h in payload.get("headers") or []
    }
    return SendVerification(
        found=bool(message_id),
        message_id=message_id,
        thread_match=bool(thread_id) and thread_id == draft.thread_id,
        recipients_match=_recipients_covered(draft.to, headers.get("to", "")),
        attachments_match=_attachments_covered(
            draft.attachments, _attachment_filenames(payload)
        ),
        subject_match=_subjects_equivalent(headers.get("subject", ""), draft.subject),
    )


def _recipients_covered(expected: list[EmailAddress], to_header: str) -> bool:
    """Every expected recipient must appear in the sent To header."""
    expected_emails = {a.email.casefold() for a in expected}
    expected_emails.discard("")
    if not expected_emails:
        return True
    actual = {addr.casefold() for _, addr in getaddresses([to_header]) if addr}
    return expected_emails <= actual


def _attachment_filenames(payload: dict[str, Any]) -> set[str]:
    """Display filenames of the parts of a sent message (casefolded)."""
    names: set[str] = set()
    for part in payload.get("parts") or []:
        filename = str(part.get("filename") or "").strip()
        if filename:
            names.add(filename.casefold())
    return names


def _attachments_covered(
    expected: tuple[OutgoingAttachment, ...], actual: set[str]
) -> bool:
    if not expected:
        return True
    expected_names = {a.filename.casefold() for a in expected if a.filename}
    if not expected_names:
        return True
    return expected_names <= actual


def _subjects_equivalent(sent_subject: str, draft_subject: str) -> bool:
    """Compare thread subjects ignoring leading Re:/Fwd: prefixes (case-insensitive)."""
    draft_norm = _strip_subject_prefix(draft_subject).casefold()
    if not draft_norm:
        return True
    sent_norm = _strip_subject_prefix(sent_subject).casefold()
    return (
        sent_norm == draft_norm
        or draft_norm in sent_norm
        or sent_norm in draft_norm
    )


def _strip_subject_prefix(subject: str) -> str:
    return re.sub(r"^(re|fwd|fw|aw|antw)(\[\d+\])?:", "", subject, flags=re.IGNORECASE).strip()


# ── outgoing MIME construction ───────────────────────────────────────────────


def _attach_outgoing(mime: EmailMessage, attachment: OutgoingAttachment) -> None:
    """Attach one Telegram-supplied temp file to the outgoing message."""
    if not attachment.path or not os.path.isfile(attachment.path):
        raise GmailAPIError(f"attachment file is gone for {attachment.filename}")
    maintype, sep, subtype = attachment.mime_type.partition("/")
    if not sep or not maintype or not subtype:
        maintype, subtype = "application", "octet-stream"
    with open(attachment.path, "rb") as fh:
        mime.add_attachment(
            fh.read(),
            maintype=maintype,
            subtype=subtype,
            filename=attachment.filename,
        )
