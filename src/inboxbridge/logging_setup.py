"""Secure logging: secrets must never appear in log output.

Two layers:

- Third-party HTTP clients (httpx/httpcore) log full request URLs at INFO,
  which would print ``https://api.telegram.org/bot<TOKEN>/getUpdates`` — they
  are silenced to WARNING so their INFO chatter never reaches stdout.
- A defensive :class:`SecretRedactionFilter` on the root handler rewrites any
  log record that contains a known secret (Telegram bot token, LLM API key,
  PDF password, OAuth refresh token) or a ``bot<TOKEN>``-shaped URL, replacing
  it with a placeholder regardless of level. InboxBridge's own INFO logs stay
  untouched (and useful).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Matches Telegram API URLs / references carrying an inline token:
#: ``bot123456:AA...``. Defensive even if the configured token differs.
_TG_BOT_TOKEN_INLINE = re.compile(r"bot\d{5,}:[A-Za-z0-9_-]+")

#: HTTP clients that log full request URLs (with tokens) at INFO.
_NOISY_HTTP_LOGGERS = ("httpx", "httpcore")

REDACTED = "***REDACTED***"


class SecretRedactionFilter(logging.Filter):
    """Rewrite records that contain any configured secret or bot-token URLs."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._secret_values = _collect_secret_values(settings)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = message
        for secret in self._secret_values:
            if secret:
                redacted = redacted.replace(secret, REDACTED)
        redacted = _TG_BOT_TOKEN_INLINE.sub(REDACTED, redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root logging with secret redaction + silenced HTTP clients.

    Only touches the root when it has no handlers yet (fresh process), so
    pre-existing handlers (e.g. pytest's caplog) are preserved and still get
    the redaction filter.
    """
    settings = settings or get_settings()
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            force=True,
        )
    else:
        root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    for handler in root.handlers:
        handler.addFilter(SecretRedactionFilter(settings))
    for name in _NOISY_HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _collect_secret_values(settings: Settings) -> list[str]:
    values = [
        settings.telegram_bot_token.get_secret_value(),
        settings.llm_api_key.get_secret_value(),
        settings.pdf_password.get_secret_value(),
    ]
    refresh_token = _read_refresh_token(Path(settings.google_token_file))
    if refresh_token:
        values.append(refresh_token)
    return values


def _read_refresh_token(token_file: Path) -> str | None:
    """Best-effort read of the OAuth refresh token (never logs it)."""
    try:
        if not token_file.is_file():
            return None
        data = json.loads(token_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    token = data.get("refresh_token") if isinstance(data, dict) else None
    return str(token) if token else None
