"""OAuth 2.0 (installed-app, local) for the Gmail API.

Flow: run interactively ONCE on a desktop machine — a browser opens, the user
authorizes, and the resulting refresh token is stored in the file configured
via ``GOOGLE_TOKEN_FILE`` (default ``data/token.json`` — a writable, persistent
path; in Docker this lives in the ``inboxbridge-data`` volume, NOT in the
read-only ``credentials`` bind mount). Refresh tokens from installed-app flows
do not expire, but the API client silently refreshes the short-lived access
token on every expiry and SAVES the updated credentials back to that file.

Re-auth on a different account needs no code changes: delete the token file
and run the auth flow again (see :func:`reauthenticate`).

Scope decision: we use ``gmail.modify`` (not ``gmail.readonly`` + ``gmail.send``)
because (a) ``users.watch`` and history processing work under it, (b) the
architecture assigns "mark read" to the Gmail client, which needs ``modify``,
and (c) a single scope keeps the token small. ``readonly`` + ``send`` would
suffice for the current protocol only.

Never log secrets: this module only ever logs file paths, never token or
secret contents.
"""

from __future__ import annotations

import contextlib
import logging
import stat
from collections.abc import Callable
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]

from ..config import Settings

logger = logging.getLogger(__name__)

SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/gmail.modify",)


class AuthError(RuntimeError):
    """OAuth setup failed (missing secret file, unreadable token, ...)."""


class CredentialsStore:
    """File-backed credentials storage (token JSON)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Credentials | None:
        if not self.path.is_file():
            return None
        try:
            return Credentials.from_authorized_user_file(  # type: ignore[no-untyped-call, no-any-return]
                str(self.path), SCOPES
            )
        except (ValueError, KeyError, OSError) as exc:
            logger.warning("token file %s is unusable, ignoring: %s", self.path, exc)
            return None

    def save(self, creds: Credentials) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(creds.to_json(), encoding="utf-8")  # type: ignore[no-untyped-call]
        with contextlib.suppress(OSError):  # best effort (Windows ignores chmod semantics)
            self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)


FlowBuilder = Callable[[str, list[str]], InstalledAppFlow]
FlowRunner = Callable[[InstalledAppFlow], Credentials]


def get_credentials(
    settings: Settings,
    *,
    store: CredentialsStore | None = None,
    flow_builder: FlowBuilder | None = None,
    flow_runner: FlowRunner | None = None,
) -> Credentials:
    """Return valid credentials, refreshing or running the local flow as needed."""
    store = store or CredentialsStore(settings.google_token_file)
    creds = store.load()
    if creds is not None:
        if creds.valid:
            return creds
        if creds.expired and creds.refresh_token:
            logger.info("refreshing expired OAuth token from %s", store.path)
            creds.refresh(Request())  # type: ignore[no-untyped-call]
            store.save(creds)
            return creds
    logger.info("no valid token at %s — starting interactive OAuth flow", store.path)
    creds = _run_flow(settings, flow_builder, flow_runner)
    store.save(creds)
    return creds


def reauthenticate(
    settings: Settings,
    *,
    store: CredentialsStore | None = None,
    flow_builder: FlowBuilder | None = None,
    flow_runner: FlowRunner | None = None,
) -> Credentials:
    """Drop the stored token and run the flow again (new account, revoke, ...)."""
    store = store or CredentialsStore(settings.google_token_file)
    store.delete()
    logger.info("deleted token %s; re-running OAuth flow", store.path)
    return get_credentials(
        settings, store=store, flow_builder=flow_builder, flow_runner=flow_runner
    )


def _run_flow(
    settings: Settings,
    flow_builder: FlowBuilder | None,
    flow_runner: FlowRunner | None,
) -> Credentials:
    secret = Path(settings.google_client_secret_file)
    if not secret.is_file():
        raise AuthError(
            f"client secret file not found: {secret} "
            "(set GOOGLE_CLIENT_SECRET_FILE, see .env.example)"
        )
    builder = flow_builder or InstalledAppFlow.from_client_secrets_file
    flow = builder(str(secret), list(SCOPES))
    runner = flow_runner or (lambda f: f.run_local_server(port=0))
    return runner(flow)
