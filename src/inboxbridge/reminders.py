"""Lightweight reminders — SQLite metadata only, restart-safe.

Deterministic Spanish time parsing (no LLM dependency for the core):
"en dos horas", "mañana", "esta tarde", "el viernes a las 18:00", "a las 15:30"...

Persistence is minimal: Gmail message/thread IDs, Telegram user context, due
timestamp, status. Email bodies are NEVER stored. Firing is atomic
(claim → fire once) and late-fires safely after downtime.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

from .db import Storage

logger = logging.getLogger(__name__)

#: Weekday names (Spanish) for "el viernes a las 18:00" parsing.
_WEEKDAYS = {
    "lunes": 0,
    "martes": 1,
    "miércoles": 2,
    "miercoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sábado": 5,
    "sabado": 5,
    "domingo": 6,
}

_NUM_WORDS = {
    "una": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "diez": 10, "doce": 12, "veinte": 20, "media": 0.5,
}
_HOURS_RE = re.compile(
    r"en (una?|dos|tres|cuatro|cinco|seis|siete|ocho|diez|doce|\d+) hora(?:s)?",
    re.IGNORECASE,
)
_MINUTES_RE = re.compile(r"en (\d+|quince|veinte|treinta|cuarenta|cincuenta) minuto(?:s)?", re.IGNORECASE)
_MEDIA_HORA_RE = re.compile(r"media hora", re.IGNORECASE)
_DAYS_RE = re.compile(
    r"en (una?|dos|tres|cuatro|cinco|\d+) d[ií]a(?:s)?", re.IGNORECASE
)
_WEEKDAY_AT_RE = re.compile(
    r"el (lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)"
    r"(?: a las (\d{1,2})[:.](\d{2}))?",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"a las (\d{1,2})[:.](\d{2})", re.IGNORECASE)
_THIS_AFTERNOON_RE = re.compile(r"esta tarde", re.IGNORECASE)
_THIS_NIGHT_RE = re.compile(r"esta noche", re.IGNORECASE)
_TOMORROW_RE = re.compile(r"ma[nñ]ana", re.IGNORECASE)

_DEFAULT_MORNING = 9
_DEFAULT_AFTERNOON = 17
_DEFAULT_NIGHT = 21


class ReminderParseError(ValueError):
    """The instruction has no parseable time (caller asks for clarification)."""


def parse_due_at(instruction: str, now_ts: float | None = None) -> float:
    """Deterministic due timestamp (epoch seconds) from a Spanish instruction.

    Raises :class:`ReminderParseError` when nothing parseable is found.
    """
    now = datetime.fromtimestamp(now_ts if now_ts is not None else _now(), tz=UTC)
    text = instruction.strip()

    match = _HOURS_RE.search(text)
    if match:
        return (now + timedelta(hours=_num(match.group(1)))).timestamp()
    if _MEDIA_HORA_RE.search(text):
        return (now + timedelta(minutes=30)).timestamp()
    match = _MINUTES_RE.search(text)
    if match:
        return (now + timedelta(minutes=_num(match.group(1)))).timestamp()
    match = _DAYS_RE.search(text)
    if match:
        return (now + timedelta(days=_num(match.group(1)))).timestamp()

    if _THIS_AFTERNOON_RE.search(text):
        target = now.replace(hour=_DEFAULT_AFTERNOON, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target.timestamp()
    if _THIS_NIGHT_RE.search(text):
        target = now.replace(hour=_DEFAULT_NIGHT, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target.timestamp()
    if _TOMORROW_RE.search(text):
        target = (now + timedelta(days=1)).replace(
            hour=_DEFAULT_MORNING, minute=0, second=0, microsecond=0
        )
        return target.timestamp()

    match = _WEEKDAY_AT_RE.search(text)
    if match:
        weekday = _WEEKDAYS[match.group(1)]
        days_ahead = (weekday - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # "el viernes" means NEXT Friday when today is Friday
        target = (now + timedelta(days=days_ahead)).replace(
            hour=int(match.group(2)) if match.group(2) else _DEFAULT_MORNING,
            minute=int(match.group(3)) if match.group(3) else 0,
            second=0,
            microsecond=0,
        )
        return target.timestamp()

    match = _TIME_RE.search(text)
    if match:
        target = now.replace(
            hour=int(match.group(1)), minute=int(match.group(2)), second=0, microsecond=0
        )
        if target <= now:
            target += timedelta(days=1)
        return target.timestamp()

    raise ReminderParseError("no se pudo determinar la hora del recordatorio")


def _num(value: str) -> float:
    """Digits or Spanish number words → number."""
    if value.isdigit():
        return float(value)
    return float(_NUM_WORDS.get(value.casefold(), 1))


def format_due(due_at: float) -> str:
    """Human-readable due time (UTC → local display is Telegram's job; we show
    the stored UTC instant with a clear marker)."""
    instant = datetime.fromtimestamp(due_at, tz=UTC)
    return f"{instant:%d/%m/%Y %H:%M} UTC"


@dataclass(frozen=True)
class ReminderCreate:
    reminder_id: int
    due_at: float


def _now() -> float:
    return datetime.now(UTC).timestamp()


class ReminderService:
    """Reminder CRUD over SQLite with an injectable clock (deterministic tests)."""

    def __init__(
        self,
        storage: Storage,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._storage = storage
        self._clock = clock or _now

    def create(
        self,
        *,
        message_id: str,
        thread_id: str,
        telegram_user_id: int,
        instruction: str,
    ) -> ReminderCreate:
        due_at = parse_due_at(instruction, self._clock())
        reminder_id = self._storage.create_reminder(
            message_id=message_id,
            thread_id=thread_id,
            telegram_user_id=telegram_user_id,
            due_at=due_at,
            note=instruction,
        )
        return ReminderCreate(reminder_id=reminder_id, due_at=due_at)

    def list(self, telegram_user_id: int) -> list[dict[str, Any]]:
        return self._storage.list_reminders(telegram_user_id)

    def cancel(self, reminder_id: int, telegram_user_id: int) -> bool:
        return self._storage.cancel_reminder(reminder_id, telegram_user_id)

    def due(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._storage.due_reminders(self._clock(), limit=limit)

    def claim(self, reminder_id: int) -> bool:
        """Atomically claim a due reminder for firing (duplicate-tick safe)."""
        return self._storage.claim_reminder(reminder_id)

    def get(self, reminder_id: int) -> dict[str, Any] | None:
        return self._storage.get_reminder(reminder_id)
