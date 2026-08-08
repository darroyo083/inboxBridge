"""Unit tests for gmail/pubsub.py — parse + ack/nack consumer semantics."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from inboxbridge.config import Settings
from inboxbridge.gmail.pubsub import PubSubConsumer, PubSubError, parse_pubsub_data
from inboxbridge.models import PubSubEvent
from tests.mocks.gmail import FakePubSubMessage, FakeSubscriberClient

PUSH_DATA = json.dumps({"emailAddress": "me@gmail.com", "historyId": 424242}).encode()


def make_settings(*, project: str = "proj-1", subscription: str = "gmail-sub",
                  sa_key: str = "") -> Settings:
    return Settings(
        _env_file=None,
        google_cloud_project=project,
        gmail_pubsub_subscription=subscription,
        GOOGLE_APPLICATION_CREDENTIALS=sa_key,
    )


class TestParsePubSubData:
    def test_parses_push_payload(self) -> None:
        event = parse_pubsub_data(PUSH_DATA, message_id="ps-9")
        assert event == PubSubEvent(
            message_id="ps-9",
            history_id=424242,
            email_address="me@gmail.com",
            raw={"emailAddress": "me@gmail.com", "historyId": 424242},
        )

    def test_defaults_message_id_from_payload(self) -> None:
        data = json.dumps({"emailAddress": "a@b.c", "historyId": 1, "messageId": "gm-1"}).encode()
        assert parse_pubsub_data(data).message_id == "gm-1"

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(PubSubError):
            parse_pubsub_data(b"not json")

    def test_rejects_missing_history_id(self) -> None:
        with pytest.raises(PubSubError):
            parse_pubsub_data(b'{"emailAddress": "a@b.c"}')

    def test_rejects_non_object(self) -> None:
        with pytest.raises(PubSubError):
            parse_pubsub_data(b"[1, 2]")


class TestConsumer:
    @pytest.mark.asyncio
    async def test_acks_after_handler_success(self) -> None:
        msg = FakePubSubMessage(PUSH_DATA, message_id="ps-1")
        client = FakeSubscriberClient([msg])
        consumer = PubSubConsumer(make_settings(), client=client)
        received: list[PubSubEvent] = []

        async def handler(event: PubSubEvent) -> None:
            received.append(event)

        task = asyncio.create_task(consumer.consume(handler))
        await asyncio.sleep(0.02)
        assert client.subscription == "projects/proj-1/subscriptions/gmail-sub"
        client.deliver()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert received[0].history_id == 424242
        assert received[0].email_address == "me@gmail.com"
        assert received[0].message_id == "ps-1"
        assert msg.acked and not msg.nacked

    @pytest.mark.asyncio
    async def test_nacks_when_handler_fails(self) -> None:
        msg = FakePubSubMessage(PUSH_DATA, message_id="ps-2")
        client = FakeSubscriberClient([msg])
        consumer = PubSubConsumer(make_settings(), client=client)

        async def handler(event: PubSubEvent) -> None:
            raise RuntimeError("processing failed")

        task = asyncio.create_task(consumer.consume(handler))
        await asyncio.sleep(0.02)
        client.deliver()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert msg.nacked and not msg.acked

    @pytest.mark.asyncio
    async def test_injected_client_not_closed(self) -> None:
        client = FakeSubscriberClient()
        consumer = PubSubConsumer(make_settings(), client=client)

        async def handler(event: PubSubEvent) -> None:
            return None

        task = asyncio.create_task(consumer.consume(handler))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client.closed
        consumer.close()
        assert not client.closed

    def test_subscription_path_error(self) -> None:
        consumer = PubSubConsumer(
            make_settings(project="", subscription=""), client=FakeSubscriberClient()
        )
        with pytest.raises(PubSubError):
            consumer.subscription_path()


class TestCredentialsSelection:
    def test_service_account_key_used_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        key_file = tmp_path / "sa.json"
        key_file.write_text("{}", encoding="utf-8")

        captured: dict[str, object] = {}

        def fake_from_file(path: str):
            captured["path"] = path
            return object()

        def fake_subscriber(*, credentials: object = None):
            captured["credentials"] = credentials
            return FakeSubscriberClient()

        monkeypatch.setattr(
            "inboxbridge.gmail.pubsub.Credentials.from_service_account_file", fake_from_file
        )
        monkeypatch.setattr("inboxbridge.gmail.pubsub.pubsub_v1.SubscriberClient", fake_subscriber)

        consumer = PubSubConsumer(make_settings(sa_key=str(key_file)))
        assert consumer._client is not None  # built from the SA path
        assert captured["path"] == str(key_file)
        assert captured["credentials"] is not None
        consumer.close()

    def test_empty_key_falls_back_to_adc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_subscriber(*, credentials: object = None):
            captured["credentials"] = credentials
            return FakeSubscriberClient()

        monkeypatch.setattr("inboxbridge.gmail.pubsub.pubsub_v1.SubscriberClient", fake_subscriber)

        consumer = PubSubConsumer(make_settings(sa_key=""))
        assert consumer._client is not None
        assert captured["credentials"] is None  # ADC: no explicit credentials
        consumer.close()

    def test_injected_client_wins_over_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        injected = FakeSubscriberClient()
        monkeypatch.setattr(
            "inboxbridge.gmail.pubsub.Credentials.from_service_account_file",
            lambda path: pytest.fail("should not load SA when client is injected"),
        )
        consumer = PubSubConsumer(
            make_settings(sa_key="credentials/service_account_pubsub.json"), client=injected
        )
        assert consumer._client is injected
        consumer.close()
