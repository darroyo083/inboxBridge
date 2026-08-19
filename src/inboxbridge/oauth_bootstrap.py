"""CLI: run the one-time local OAuth flow and store the refresh token.

Usage (on a machine with a browser):

    python -m inboxbridge.oauth_bootstrap

Opens the Google authorization page, saves credentials to
GOOGLE_TOKEN_FILE, then exits. Copy the token file to the VPS.
To re-authorize a DIFFERENT account, run with --reauth (deletes the
stored token first) — no code changes needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import get_settings
from .gmail.auth import CredentialsStore, get_credentials, reauthenticate

SECRET_EXAMPLE = (
    "credentials/client_secret.json  (see .env.example / docs/INSTALL.md)"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reauth",
        action="store_true",
        help="delete the stored token and authorize a different account",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    store = CredentialsStore(settings.google_token_file)
    secret = Path(settings.google_client_secret_file)
    if not secret.is_file():
        print(f"client secret file not found: {secret}\n  Provide: {SECRET_EXAMPLE}")
        return 1

    creds = reauthenticate(settings, store=store) if args.reauth else get_credentials(
        settings, store=store
    )
    if creds is None:
        print("OAuth flow did not return credentials")
        return 1
    store.save(creds)
    print(f"credentials saved to {store.path}")
    print(
        "On the VPS the token lives in the writable data volume; "
        "no copy is needed when running this on the target host."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
