"""``users.watch`` lifecycle: register, persist historyId, auto-renew.

Gmail Push setup: ``users.watch`` is registered against the Pub/Sub topic
from config (``GOOGLE_CLOUD_PROJECT`` + ``GMAIL_PUBSUB_TOPIC``). The watch
expires after 7 days, so we persist the expiration (meta key
``watch_expires_at``, epoch seconds UTC) and re-register whenever less than
24h remain.

History baseline: on first boot there is NO stored ``last_history_id`` —
we store the current ``historyId`` from the watch response and process
NOTHING (never process historical mail). From then on, ``history_start()``
returns the baseline the delta processor should continue from.

Primary-tab note: Gmail has no ``Primary`` label. ``CATEGORY_PERSONAL`` is
the closest (Gmail auto-tabs messages when tabs are enabled) but relying on
it at watch time would miss unlabeled Primary messages. Decision: watch
``INBOX`` (all inbox mail), and approximate Primary in the history processor
(keep messages labeled ``CATEGORY_PERSONAL`` or with no ``CATEGORY_*`` at
all). See ``gmail/history.py``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from ..config import Settings
from ..db import Storage

logger = logging.getLogger(__name__)

META_HISTORY_ID = "last_history_id"
META_WATCH_EXPIRES = "watch_expires_at"  # epoch seconds (UTC) as str

# Gmail watches expire after 7 days; renew when under 24h remain.
RENEW_BEFORE = timedelta(hours=24)
WATCH_LABELS = ("INBOX",)


class WatchError(RuntimeError):
    """Watch registration failed (config missing or API error)."""


class WatchManager:
    """Owns watch registration and the historyId baseline in meta storage."""

    def __init__(self, settings: Settings, service: Any, storage: Storage) -> None:
        self._settings = settings
        self._service = service
        self._storage = storage
        self._user_id = settings.gmail_user_id

    def topic_name(self) -> str:
        project, topic = self._settings.google_cloud_project, self._settings.gmail_pubsub_topic
        if not project or not topic:
            raise WatchError(
                "GOOGLE_CLOUD_PROJECT and GMAIL_PUBSUB_TOPIC must be set for Gmail watch"
            )
        return f"projects/{project}/topics/{topic}"

    def register_watch(self) -> bool:
        """Register/renew the watch; persists baseline historyId on first boot."""
        resp: dict[str, Any] = (
            self._service.users()
            .watch(
                userId=self._user_id,
                body={"topicName": self.topic_name(), "labelIds": list(WATCH_LABELS)},
            )
            .execute()
        )
        expiration_ms = _to_int(resp.get("expiration"))
        if expiration_ms is not None:
            self._storage.set_meta(META_WATCH_EXPIRES, str(expiration_ms / 1000.0))
        current_history_id = _to_int(resp.get("historyId"))
        if current_history_id is None:
            raise WatchError("users.watch response is missing historyId")
        if self._storage.get_meta(META_HISTORY_ID) is None:
            logger.info(
                "first watch: storing history baseline %d, processing nothing",
                current_history_id,
            )
            self._storage.set_meta(META_HISTORY_ID, str(current_history_id))
        logger.info(
            "gmail watch registered on %s (expires %s)",
            self.topic_name(),
            self.expires_at(),
        )
        return True

    def ensure_watch(self) -> bool:
        """Register only if the watch is missing or expiring within 24h.

        Returns True when a (re)registration happened.
        """
        expires_at = self.expires_at()
        if expires_at is not None and expires_at > datetime.now(UTC) + RENEW_BEFORE:
            return False
        return self.register_watch()

    def expires_at(self) -> datetime | None:
        """UTC datetime of watch expiration, or None when unknown."""
        raw = self._storage.get_meta(META_WATCH_EXPIRES)
        if raw is None:
            return None
        try:
            return datetime.fromtimestamp(float(raw), tz=UTC)
        except (TypeError, ValueError, OSError, OverflowError):
            return None

    def history_start(self) -> int | None:
        """Stored historyId baseline, or None on first boot (process nothing)."""
        raw = self._storage.get_meta(META_HISTORY_ID)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
