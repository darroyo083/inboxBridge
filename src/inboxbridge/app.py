"""Application entrypoint: wires Gmail + Pub/Sub + LLM + Telegram + SQLite.

Graceful shutdown: on SIGINT/SIGTERM, consumers and workers are stopped in
order (pub/sub first — acks only after handler success — then retry scheduler,
then telegram, then DB). Nothing is lost: unacked Pub/Sub messages are
redelivered, FAILED messages stay queued for retry.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings
from .db import Storage
from .gmail.auth import get_credentials
from .gmail.client import GmailClient
from .gmail.history import HistoryDelta, HistoryProcessor
from .gmail.pubsub import PubSubConsumer
from .gmail.watcher import WatchManager
from .llm.openai_compat import OpenAICompatLLM
from .models import PubSubEvent
from .pipeline import InboundPipeline, RetryScheduler
from .responder import ReplyCoordinator, ReplyWorker
from .status import build_status_text
from .telegram.bot import TelegramBot

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@dataclass
class Services:
    """Fully-wired runtime services (no Optionals)."""

    settings: Settings
    storage: Storage
    gmail: GmailClient
    watcher: WatchManager
    history: HistoryProcessor
    llm: OpenAICompatLLM
    bot: TelegramBot

    @classmethod
    def build(cls) -> Services:
        settings = get_settings()
        storage = Storage(Path("data") / "inboxbridge.db")
        storage.connect()
        credentials = get_credentials(settings)  # may start interactive OAuth flow
        gmail = GmailClient(settings, credentials=credentials)
        service = gmail._service  # private by design (no public accessor yet)
        watcher = WatchManager(settings, service, storage)
        history = HistoryProcessor(settings, service, storage)
        llm = OpenAICompatLLM(settings)
        bot = TelegramBot(settings, storage)
        bot._status_provider = _status_provider_factory(settings, storage)  # private by design
        return cls(
            settings=settings,
            storage=storage,
            gmail=gmail,
            watcher=watcher,
            history=history,
            llm=llm,
            bot=bot,
        )

    def close(self) -> None:
        self.storage.close()


async def _status_provider(settings: Settings, storage: Storage) -> str:
    return build_status_text(settings, storage)


def _status_provider_factory(
    settings: Settings, storage: Storage
) -> Callable[[], Awaitable[str]]:
    async def provider() -> str:
        return build_status_text(settings, storage)

    return provider


def _history_provider(services: Services) -> Callable[[PubSubEvent], Awaitable[HistoryDelta]]:
    async def provider(event: PubSubEvent) -> HistoryDelta:
        return services.history.new_message_ids(event.history_id)

    return provider


class App:
    """Composes everything; run() is the asyncio entrypoint."""

    def __init__(self) -> None:
        self._services: Services | None = None
        self._consumer: PubSubConsumer | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._retry_scheduler: RetryScheduler | None = None
        self._reply_worker: ReplyWorker | None = None

    async def run(self) -> None:
        services = Services.build()
        self._services = services

        # Watch must exist before any event can arrive (no historical mail).
        services.watcher.ensure_watch()

        pipeline = InboundPipeline(
            services.settings,
            services.gmail,
            services.llm,
            services.bot,
            services.storage,
            history_provider=_history_provider(services),
        )

        consumer = PubSubConsumer(services.settings)
        self._consumer = consumer

        async def handle_event(event: PubSubEvent) -> None:
            await pipeline.process_event(event)

        self._consumer_task = asyncio.create_task(
            consumer.consume(handle_event), name="pubsub-consumer"
        )

        retry_scheduler = RetryScheduler(pipeline)
        retry_scheduler.start()
        self._retry_scheduler = retry_scheduler

        reply_coordinator = ReplyCoordinator(
            services.settings,
            services.gmail,
            services.llm,
            services.bot,
            services.storage,
        )
        reply_worker = ReplyWorker(reply_coordinator)
        reply_worker.start()
        self._reply_worker = reply_worker

        await services.bot.start()

        logger.info("InboxBridge started (SEND_EMAILS=%s)", services.settings.send_emails)
        try:
            assert self._consumer_task is not None
            await self._consumer_task
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        logger.info("shutting down InboxBridge…")
        if self._consumer is not None:
            self._consumer.close()
        if self._reply_worker is not None:
            await self._reply_worker.stop()
        if self._retry_scheduler is not None:
            await self._retry_scheduler.stop()
        services = self._services
        if services is not None:
            await services.bot.stop()
            await services.llm.close()
            services.close()
        logger.info("shutdown complete")


def main() -> None:
    _setup_logging(get_settings().log_level)

    async def _run() -> None:
        app = App()
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        def _request_stop() -> None:
            stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, _request_stop)

        run_task = asyncio.create_task(app.run())
        await stop.wait()
        await app.shutdown()
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run())


if __name__ == "__main__":
    main()
