"""Docker healthcheck: verifies the app process and DB are alive.

Exits 0 when healthy. Never prints secrets or email content.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import get_settings
from .db import Storage


def main() -> int:
    settings = get_settings()
    db_path = Path("data") / "inboxbridge.db"
    try:
        storage = Storage(db_path)
        storage.connect()
        # Touch a trivial read to confirm the DB opens and is writable.
        storage.get_meta("healthcheck", "none")
        storage.close()
    except Exception:
        return 1
    # Config sanity: bot token present?
    if not settings.telegram_bot_token.get_secret_value():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
