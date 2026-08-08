"""Async TelegramNotifier double for coordinator integration tests."""

from __future__ import annotations

from dataclasses import dataclass

from inboxbridge.models import DraftReply, EmailSummary, ParsedEmail


@dataclass
class SentMessage:
    kind: str
    content: str
    draft: DraftReply | None = None
    email: ParsedEmail | None = None
    summary: EmailSummary | None = None


class FakeTelegram:
    """In-memory TelegramNotifier double; records everything it sends."""

    def __init__(self) -> None:
        self.sent: list[SentMessage] = []
        self.notices: list[str] = []
        self.typing_calls: int = 0
        self._next_id: int = 1

    async def send_summary(self, email: ParsedEmail, summary: EmailSummary) -> int:
        return self._record(
            SentMessage(kind="summary", content=summary.summary_es, email=email, summary=summary)
        )

    async def send_notice(self, text: str) -> int:
        self.notices.append(text)
        return self._record(SentMessage(kind="notice", content=text))

    async def send_typing(self) -> None:
        self.typing_calls += 1

    async def send_draft_for_confirmation(self, draft: DraftReply) -> int:
        return self._record(SentMessage(kind="draft", content=draft.body, draft=draft))

    def _record(self, message: SentMessage) -> int:
        message_id = self._next_id
        self._next_id += 1
        self.sent.append(message)
        return message_id
