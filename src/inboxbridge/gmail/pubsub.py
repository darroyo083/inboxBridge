"""StreamingPull consumer for Gmail push notifications.

Transport note: the pinned ``google-cloud-pubsub`` (2.23.1) ships no stable
asyncio ``SubscriberAsyncClient``, so we use the sync ``SubscriberClient``
whose StreamingPull callbacks run on background threads, and bridge each
message into an asyncio queue (the officially supported pattern). Ack/nack
are thread-safe on the pubsub ``Message`` object.

Semantics: a message is ACKed only after the handler completed successfully
(at-least-once). On handler failure we NACK, so Pub/Sub redelivers; on
shutdown, messages still queued are simply dropped and redelivered after the
ack deadline. Parsing is strict: anything that is not a Gmail push JSON
(``emailAddress`` + ``historyId``) raises :class:`PubSubError` and NACKs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from google.cloud import pubsub_v1  # type: ignore[import-untyped]
from google.cloud.pubsub_v1.subscriber.message import Message  # type: ignore[import-untyped]

from ..config import Settings
from ..models import PubSubEvent

logger = logging.getLogger(__name__)


class PubSubError(RuntimeError):
    """Pub/Sub setup or message parsing failed."""


def parse_pubsub_data(data: bytes, *, message_id: str = "") -> PubSubEvent:
    """Parse a Gmail push payload (JSON: ``emailAddress``, ``historyId``)."""
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PubSubError("pub/sub payload is not valid UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise PubSubError("pub/sub payload is not a JSON object")
    history_id = _to_int(raw.get("historyId"))
    if history_id is None:
        raise PubSubError("pub/sub payload is missing historyId")
    return PubSubEvent(
        message_id=message_id or str(raw.get("messageId") or ""),
        history_id=history_id,
        email_address=str(raw.get("emailAddress") or ""),
        raw=raw,
    )


class PubSubConsumer:
    """StreamingPull consumer; ``consume`` runs until cancelled."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client if client is not None else pubsub_v1.SubscriberClient()
        self._owns_client = client is None

    def subscription_path(self) -> str:
        project, sub = (
            self._settings.google_cloud_project,
            self._settings.gmail_pubsub_subscription,
        )
        if not project or not sub:
            raise PubSubError(
                "GOOGLE_CLOUD_PROJECT and GMAIL_PUBSUB_SUBSCRIPTION must be set"
            )
        return f"projects/{project}/subscriptions/{sub}"

    async def consume(self, handler: Callable[[PubSubEvent], Awaitable[None]]) -> None:
        """Stream messages, acking only after the handler succeeded.

        Blocks until cancelled; on cancellation the subscription is closed.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Message] = asyncio.Queue()

        def on_message(msg: Message) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, msg)

        future = self._client.subscribe(self.subscription_path(), on_message)
        try:
            while True:
                msg = await queue.get()
                try:
                    event = parse_pubsub_data(msg.data, message_id=msg.message_id)
                    await handler(event)
                    msg.ack()
                except Exception:
                    logger.exception(
                        "pub/sub handler failed for message %s — nacking for redelivery",
                        msg.message_id,
                    )
                    msg.nack()
        except asyncio.CancelledError:
            logger.info("pub/sub consumer cancelled; closing subscription")
            future.cancel()
            if self._owns_client:
                self._client.close()
            raise

    def close(self) -> None:
        """Close the client if this consumer created it."""
        if self._owns_client:
            self._client.close()


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
