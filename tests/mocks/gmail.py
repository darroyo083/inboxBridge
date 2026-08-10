"""Test doubles for the Gmail API and Pub/Sub — no real credentials needed.

``FakeGmailService`` mimics the googleapiclient chain
``service.users().messages().get(...).execute()`` and records every call so
tests can assert on request bodies. Route values are either static responses
or callables receiving the kwargs.
"""

from __future__ import annotations

from collections.abc import Callable
from email.message import EmailMessage
from typing import Any

Route = tuple[str, ...]
Responder = Callable[[dict[str, Any]], Any]


class FakeRequest:
    def __init__(self, response: Any) -> None:
        self._response = response

    def execute(self) -> Any:
        # Route responders may return an exception instance: raise it at
        # execute-time like the real googleapiclient does.
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


class _Node:
    """Chainable resource node: attribute access and call return request stubs.

    Calls with no kwargs (``users()``, ``messages()``) are pure navigation;
    only calls with kwargs (``get(...)``, ``send(...)``, ...) are requests.
    """

    def __init__(self, service: FakeGmailService, path: Route) -> None:
        self._service = service
        self._path = path

    def __getattr__(self, name: str) -> _Node:
        return _Node(self._service, self._path + (name,))

    def __call__(self, **kwargs: Any) -> _Node | FakeRequest:
        if not kwargs:
            return self
        self._service._record(self._path, kwargs)
        return FakeRequest(self._service._respond(self._path, kwargs))


class FakeGmailService:
    def __init__(self, routes: dict[Route, Any]) -> None:
        self._routes = routes
        self.calls: list[tuple[Route, dict[str, Any]]] = []

    def users(self) -> _Node:
        return _Node(self, ("users",))

    def _record(self, path: Route, kwargs: dict[str, Any]) -> None:
        self.calls.append((path, kwargs))

    def _respond(self, path: Route, kwargs: dict[str, Any]) -> Any:
        if path not in self._routes:
            raise AssertionError(f"unexpected Gmail API call: {'.'.join(path)} {kwargs}")
        value = self._routes[path]
        return value(kwargs) if callable(value) else value


class FakePubSubMessage:
    def __init__(self, data: bytes, *, message_id: str = "pubsub-1") -> None:
        self.data = data
        self.message_id = message_id
        self.acked = False
        self.nacked = False

    def ack(self) -> None:
        self.acked = True

    def nack(self) -> None:
        self.nacked = True


class FakeSubscriberClient:
    """Minimal StreamingPull stand-in: subscribe, then deliver() messages."""

    def __init__(self, messages: list[FakePubSubMessage] | None = None) -> None:
        self.messages = messages or []
        self.subscription: str | None = None
        self.closed = False
        self.cancelled = False
        self._callback: Callable[[Any], None] | None = None

    def subscribe(self, subscription: str, callback: Callable[[Any], None]) -> FakeSubscriberClient:
        self.subscription = subscription
        self._callback = callback
        return self

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True

    def deliver(self) -> None:
        assert self._callback is not None
        for msg in self.messages:
            self._callback(msg)


def build_raw_email(
    *,
    subject: str = "Test subject",
    sender: str = "Alice Example <alice@example.com>",
    to: str = "Bob <bob@example.com>",
    cc: str = "",
    date: str = "Tue, 05 Aug 2025 10:30:00 +0200",
    body_text: str | None = "Plain body.",
    body_html: str | None = "<p>HTML <b>body</b>.</p>",
    attachments: list[tuple[str, str, str, bytes]] | None = None,
) -> bytes:
    """Build a small multipart/alternative RFC822 message in memory."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    if cc:
        msg["Cc"] = cc
    msg["Date"] = date
    msg.set_content(body_text or "")
    if body_html:
        msg.add_alternative(body_html, subtype="html")
    for filename, maintype, subtype, data in attachments or []:
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    return msg.as_bytes()
