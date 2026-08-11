"""Reminders: deterministic Spanish time parsing + storage + fire-once."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from inboxbridge.db import Storage
from inboxbridge.reminders import (
    ReminderParseError,
    ReminderService,
    parse_due_at,
)

FIXED_NOW = datetime(2026, 8, 11, 10, 0, tzinfo=UTC).timestamp()  # Tuesday


@pytest.fixture
def service(tmp_path: object) -> ReminderService:
    storage = Storage(str(tmp_path) + "/reminders.db")
    storage.connect()
    return ReminderService(storage, clock=lambda: FIXED_NOW)


class TestParseDueAt:
    def test_in_hours(self) -> None:
        due = parse_due_at("recuérdamelo en dos horas", FIXED_NOW)
        assert due == pytest.approx(FIXED_NOW + 2 * 3600)

    def test_in_minutes(self) -> None:
        due = parse_due_at("en 45 minutos", FIXED_NOW)
        assert due == pytest.approx(FIXED_NOW + 45 * 60)

    def test_in_days(self) -> None:
        due = parse_due_at("en 3 días", FIXED_NOW)
        assert due == pytest.approx(FIXED_NOW + 3 * 86400)

    def test_mañana(self) -> None:
        due = parse_due_at("recuérdamelo mañana", FIXED_NOW)
        assert datetime.fromtimestamp(due, tz=UTC).hour == 9
        assert datetime.fromtimestamp(due, tz=UTC).day == 12  # tomorrow

    def test_esta_tarde(self) -> None:
        due = parse_due_at("esta tarde", FIXED_NOW)
        assert datetime.fromtimestamp(due, tz=UTC).hour == 17

    def test_esta_noche(self) -> None:
        due = parse_due_at("esta noche", FIXED_NOW)
        assert datetime.fromtimestamp(due, tz=UTC).hour == 21

    def test_weekday_with_time(self) -> None:
        due = parse_due_at("recuérdamelo el viernes a las 18:00", FIXED_NOW)
        instant = datetime.fromtimestamp(due, tz=UTC)
        assert instant.weekday() == 4  # Friday
        assert instant.hour == 18
        assert instant.minute == 0

    def test_weekday_defaults_morning(self) -> None:
        due = parse_due_at("el jueves", FIXED_NOW)
        assert datetime.fromtimestamp(due, tz=UTC).hour == 9
        assert datetime.fromtimestamp(due, tz=UTC).weekday() == 3

    def test_today_time_rolls_to_tomorrow_when_past(self) -> None:
        due = parse_due_at("a las 08:00", FIXED_NOW)  # 10:00 now → tomorrow 08:00
        instant = datetime.fromtimestamp(due, tz=UTC)
        assert instant.day == 12 and instant.hour == 8

    def test_future_today_time_stays_today(self) -> None:
        due = parse_due_at("a las 22:30", FIXED_NOW)
        assert datetime.fromtimestamp(due, tz=UTC).day == 11
        assert datetime.fromtimestamp(due, tz=UTC).hour == 22

    def test_unparseable_raises(self) -> None:
        with pytest.raises(ReminderParseError):
            parse_due_at("recuérdame esto", FIXED_NOW)


class TestReminderService:
    def test_create_list_cancel(self, service: ReminderService) -> None:
        created = service.create(
            message_id="m1", thread_id="t1", telegram_user_id=7,
            instruction="recuérdamelo en una hora",
        )
        rows = service.list(7)
        assert [r["id"] for r in rows] == [created.reminder_id]
        assert rows[0]["thread_id"] == "t1"
        assert service.cancel(created.reminder_id, 7)
        assert service.list(7) == []

    def test_cancel_scoped_to_owner(self, service: ReminderService) -> None:
        created = service.create(
            message_id="m1", thread_id="t1", telegram_user_id=7,
            instruction="en dos horas",
        )
        assert not service.cancel(created.reminder_id, 8)  # other user
        assert service.list(7)  # still pending
        assert service.cancel(created.reminder_id, 7)

    def test_due_and_atomic_claim(self, service: ReminderService) -> None:
        created = service.create(
            message_id="m1", thread_id="t1", telegram_user_id=7,
            instruction="en 5 minutos",  # due at 10:05
        )
        assert service.due() == []  # not yet due at 10:00
        service2 = ReminderService(
            service._storage, clock=lambda: FIXED_NOW + 6 * 60
        )
        due = service2.due()
        assert [r["id"] for r in due] == [created.reminder_id]
        # Atomic claim: two ticks only fire once.
        assert service2.claim(created.reminder_id)
        assert not service2.claim(created.reminder_id)
        assert service2.due() == []

    def test_restart_safe(self, service: ReminderService, tmp_path: object) -> None:
        created = service.create(
            message_id="m1", thread_id="t1", telegram_user_id=7,
            instruction="mañana",
        )
        # Simulated restart: a new service instance over the same DB.
        storage = Storage(str(tmp_path) + "/reminders.db")
        storage.connect()
        restarted = ReminderService(storage, clock=lambda: FIXED_NOW)
        assert [r["id"] for r in restarted.list(7)] == [created.reminder_id]

    def test_never_stores_body(self, service: ReminderService) -> None:
        created = service.create(
            message_id="m1", thread_id="t1", telegram_user_id=7,
            instruction="recuérdame esto mañana",
        )
        row = service.get(created.reminder_id)
        assert row is not None
        # Only workflow metadata + the user's own note — never email content.
        assert set(row) == {
            "id", "message_id", "thread_id", "telegram_user_id", "due_at",
            "status", "note", "telegram_message_id", "created_at", "updated_at",
        }
        assert row["message_id"] == "m1"
        assert row["thread_id"] == "t1"
        assert row["note"] == "recuérdame esto mañana"  # user's own words only
