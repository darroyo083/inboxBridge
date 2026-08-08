"""Security tests: secrets never reach log output.

Covers: Telegram token in URLs/messages, LLM API key, PDF password, OAuth
refresh token, and the bot-token URL shape — redacted by the defensive
filter; plus httpx INFO suppression and InboxBridge INFO still emitted.
"""

from __future__ import annotations

import json
import logging

from inboxbridge.config import Settings
from inboxbridge.logging_setup import (
    REDACTED,
    SecretRedactionFilter,
    configure_logging,
)

TG_TOKEN = "123456789:AAHhExampleToken000111222333444"


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "SEND_EMAILS": False,
        "TELEGRAM_BOT_TOKEN": TG_TOKEN,
        "LLM_API_KEY": "llm-secret-key-abc",
        "PDF_PASSWORD": "pdf-secret-pw",
    }
    base.update(overrides)
    return Settings(**base)


def _format_with_filter(settings: Settings, message: str) -> str:
    """Format a log record as a handler would, with the redaction filter."""
    record = logging.LogRecord(
        name="inboxbridge.test",
        level=logging.INFO,
        pathname="test_logging.py",
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    assert SecretRedactionFilter(settings).filter(record)
    return logging.Formatter("%(message)s").format(record)


class TestTelegramTokenRedaction:
    def test_url_with_inline_token_is_redacted(self) -> None:
        settings = make_settings()
        url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates?timeout=30"
        out = _format_with_filter(settings, f"GET {url} -> 200 OK")
        assert TG_TOKEN not in out
        assert REDACTED in out

    def test_plain_message_containing_token_is_redacted(self) -> None:
        settings = make_settings()
        out = _format_with_filter(settings, f"sending via token {TG_TOKEN} now")
        assert TG_TOKEN not in out
        assert REDACTED in out

    def test_unknown_bot_token_shape_is_redacted(self) -> None:
        """Defensive: a bot<digits>:<alnum> URL is scrubbed even when the
        configured token differs from the leaked one."""
        settings = make_settings()
        other = "987654321:ZZyyOtherToken000"
        out = _format_with_filter(settings, f"https://api.telegram.org/bot{other}/getUpdates")
        assert other not in out
        assert REDACTED in out


class TestOtherSecretsRedaction:
    def test_llm_api_key_redacted(self) -> None:
        out = _format_with_filter(make_settings(), "LLM call failed, key=llm-secret-key-abc")
        assert "llm-secret-key-abc" not in out
        assert REDACTED in out

    def test_pdf_password_redacted(self) -> None:
        out = _format_with_filter(make_settings(), "unlocking PDF with pdf-secret-pw failed")
        assert "pdf-secret-pw" not in out
        assert REDACTED in out

    def test_oauth_refresh_token_redacted(self, tmp_path: object) -> None:
        token_file = str(tmp_path) + "/token.json"
        with open(token_file, "w", encoding="utf-8") as fh:
            json.dump({"refresh_token": "1//refresh-secret-abc", "token_uri": "x"}, fh)
        settings = make_settings(google_token_file=token_file)
        out = _format_with_filter(settings, "refreshing token 1//refresh-secret-abc")
        assert "1//refresh-secret-abc" not in out
        assert REDACTED in out


class TestConfigureLogging:
    def test_httpx_info_is_suppressed(self) -> None:
        configure_logging(make_settings(log_level="INFO"))
        assert logging.getLogger("httpx").isEnabledFor(logging.INFO) is False
        assert logging.getLogger("httpcore").isEnabledFor(logging.INFO) is False

    def test_inboxbridge_info_still_emitted(self, caplog: object) -> None:
        configure_logging(make_settings(log_level="INFO"))
        with caplog.at_level(logging.INFO):
            logging.getLogger("inboxbridge.test").info("InboxBridge started normally")
        assert "InboxBridge started normally" in caplog.text

    def test_token_redacted_end_to_end(self, caplog: object) -> None:
        configure_logging(make_settings(log_level="INFO"))
        with caplog.at_level(logging.INFO):
            logging.getLogger("inboxbridge.test").error(
                "polling failed for https://api.telegram.org/bot%s/getUpdates", TG_TOKEN
            )
        assert TG_TOKEN not in caplog.text
        assert REDACTED in caplog.text
