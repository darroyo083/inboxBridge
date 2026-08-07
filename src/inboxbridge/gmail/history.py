"""History delta processing: new INBOX messages since the stored historyId.

This is where dedup + Primary-tab approximation + "no historical mail on
first boot" all meet:

- no stored baseline  -> return nothing (first boot; watcher set the
  baseline, we never process mail that predates the watch);
- ``users.history.list(startHistoryId=baseline+1, labelId="INBOX")``
  (paged) -> candidate message ids;
- skip ids already in the DB (``message_exists``) — Pub/Sub is at-least-once,
  so redelivery is safe;
- Primary approximation: Gmail has no ``Primary`` label. Gmail auto-assigns
  ``CATEGORY_*`` labels when tabs are enabled; we keep a message when its
  labels contain ``CATEGORY_PERSONAL`` or NO ``CATEGORY_*`` label at all
  (tabs disabled => everything is Primary). Label lookups are one cheap
  ``messages.get(format="metadata", fields="id,labelIds")`` per candidate.

The returned ``HistoryDelta.history_id`` is the newest historyId actually
seen; the caller persists it via :meth:`HistoryProcessor.persist_history_id`
ONLY after the returned ids were processed successfully (never before, or a
crash would skip mail).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..db import Storage
from .watcher import META_HISTORY_ID

logger = logging.getLogger(__name__)

PRIMARY_LABEL = "CATEGORY_PERSONAL"
CATEGORY_PREFIX = "CATEGORY_"
INBOX_LABEL = "INBOX"
# Safety cap: history.list pages (~1000 records each) must never loop forever.
MAX_HISTORY_PAGES = 100


class HistoryError(RuntimeError):
    """History processing failed (API error, runaway pagination, ...)."""


@dataclass(frozen=True)
class HistoryDelta:
    history_id: int
    message_ids: list[str]


class HistoryProcessor:
    def __init__(self, settings: Settings, service: Any, storage: Storage) -> None:
        self._settings = settings
        self._service = service
        self._storage = storage
        self._user_id = settings.gmail_user_id

    def new_message_ids(self, event_history_id: int) -> HistoryDelta:
        """Return new Primary-tab INBOX message ids since the stored baseline.

        ``event_history_id`` is only used as a floor for the returned
        history_id when nothing was fetched (e.g. first boot).
        """
        start = self._stored_history_start()
        if start is None:
            logger.info("no history baseline yet — processing nothing (first boot)")
            return HistoryDelta(history_id=event_history_id, message_ids=[])
        cursor = start + 1  # startHistoryId is exclusive
        max_seen = start
        candidates: set[str] = set()
        page_token: str | None = None
        for _ in range(MAX_HISTORY_PAGES):
            kwargs: dict[str, Any] = {"userId": self._user_id, "labelId": INBOX_LABEL}
            if page_token:
                kwargs["pageToken"] = page_token
            else:
                kwargs["startHistoryId"] = cursor
            resp: dict[str, Any] = (
                self._service.users().history().list(**kwargs).execute()
            )
            for record in resp.get("history", []):
                record_id = _to_int(record.get("id"))
                if record_id is not None:
                    max_seen = max(max_seen, record_id)
                for added in record.get("messagesAdded", []):
                    mid = str((added.get("message") or {}).get("id") or "")
                    if mid and not self._storage.message_exists(mid):
                        candidates.add(mid)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        else:
            raise HistoryError(f"history.list exceeded {MAX_HISTORY_PAGES} pages")
        primary = [mid for mid in sorted(candidates) if self._is_primary_tab(mid)]
        if primary:
            logger.info("history delta: %d new Primary message(s) since %d", len(primary), start)
        return HistoryDelta(history_id=max_seen, message_ids=primary)

    def persist_history_id(self, history_id: int) -> None:
        """Advance the baseline. Call AFTER successful processing of the delta."""
        self._storage.set_meta(META_HISTORY_ID, str(history_id))

    def _stored_history_start(self) -> int | None:
        raw = self._storage.get_meta(META_HISTORY_ID)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            logger.warning("stored last_history_id %r is not an int; treating as first boot", raw)
            return None

    def _is_primary_tab(self, message_id: str) -> bool:
        try:
            resp: dict[str, Any] = (
                self._service.users()
                .messages()
                .get(
                    userId=self._user_id,
                    id=message_id,
                    format="metadata",
                    fields="id,labelIds",
                )
                .execute()
            )
        except Exception:
            logger.exception("could not fetch labels for %s; skipping", message_id)
            return False
        labels = {str(label) for label in resp.get("labelIds") or []}
        return PRIMARY_LABEL in labels or not any(
            label.startswith(CATEGORY_PREFIX) for label in labels
        )


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
