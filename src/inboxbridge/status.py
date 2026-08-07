"""/status: connectivity checks for Gmail, Pub/Sub, Telegram, LLM.

Never reveals secrets: output is a short human-readable report with
boolean/pass-fail info only. Values like tokens, keys or full config
are never included.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from .config import Settings
from .db import Storage
from .gmail.watcher import META_HISTORY_ID, META_WATCH_EXPIRES

logger = logging.getLogger(__name__)


def _redacted(value: str) -> str:
    """Return a presence hint ('configured' / 'missing') for a secret value."""
    return "configured" if value else "missing"


def build_status_text(
    settings: Settings,
    storage: Storage | None = None,
    *,
    llm_probe: Callable[[], bool] | None = None,
    gmail_probe: Callable[[], bool] | None = None,
    telegram_probe: Callable[[], bool] | None = None,
) -> str:
    """Compose the /status report. Probe objects are optional callables
    returning True/False; when absent, config-presence is reported instead.
    """
    lines: list[str] = ["InboxBridge — estado"]

    # ── Telegram ────────────────────────────────────────────────────────────
    tg_token = settings.telegram_bot_token.get_secret_value()
    tg_ok = telegram_probe() if telegram_probe else bool(tg_token)
    lines.append(f"Telegram: {'OK' if tg_ok else 'NO configurado'}")

    # ── Gmail / OAuth / watch ───────────────────────────────────────────────
    secret_file_ok = _file_exists(settings.google_client_secret_file)
    token_file_ok = _file_exists(settings.google_token_file)
    gmail_ok = gmail_probe() if gmail_probe else (secret_file_ok and token_file_ok)
    lines.append(f"Gmail: {'OK' if gmail_ok else 'NO configurado'}")

    if storage is not None:
        history = storage.get_meta(META_HISTORY_ID)
        watch_expires = storage.get_meta(META_WATCH_EXPIRES)
        if history:
            lines.append(f"History baseline: sí ({history})")
        else:
            lines.append("History baseline: no (primer arranque)")
        if watch_expires:
            try:
                expires_dt = datetime.fromtimestamp(float(watch_expires), tz=UTC)
                remaining = expires_dt - datetime.now(UTC)
                lines.append(
                    f"Watch caduca: {expires_dt:%Y-%m-%d %H:%M} UTC "
                    f"({int(remaining.total_seconds() // 3600)}h restantes)"
                )
            except (TypeError, ValueError, OverflowError):
                lines.append("Watch caduca: desconocido")
        else:
            lines.append("Watch: sin registrar")

    # ── Pub/Sub ─────────────────────────────────────────────────────────────
    pubsub_ok = bool(settings.google_cloud_project and settings.gmail_pubsub_subscription)
    lines.append(f"Pub/Sub: {'OK' if pubsub_ok else 'NO configurado'}")

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_key = settings.llm_api_key.get_secret_value()
    llm_ok = llm_probe() if llm_probe else bool(llm_key and settings.llm_base_url)
    lines.append(
        f"LLM: {'OK' if llm_ok else 'NO configurado'} "
        f"({_redacted(llm_key)} · {settings.llm_model})"
    )

    # ── Kill switch (not a secret) ──────────────────────────────────────────
    if settings.send_emails:
        lines.append("Envío de emails: ACTIVADO")
    else:
        lines.append("Envío de emails: desactivado (SEND_EMAILS=false)")

    return "\n".join(lines)


def _file_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).is_file()
