"""Deployment-contract regression tests.

The Docker deployment must keep the static Gmail secrets read-only while
letting the OAuth refresh token be written back on every access-token refresh.
This guards the exact contract that previously broke: ``credentials/`` was
mounted ``:ro`` but ``GOOGLE_TOKEN_FILE`` pointed inside it, so a token refresh
crashed with ``OSError: [Errno 30] Read-only file system``.
"""

from __future__ import annotations

from pathlib import Path

from inboxbridge.config import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_FILE = _REPO_ROOT / "docker-compose.yml"


def _compose_text() -> str:
    return _COMPOSE_FILE.read_text(encoding="utf-8")


def test_credentials_mount_is_read_only() -> None:
    """The static secrets directory must stay read-only in Docker."""
    text = _compose_text()
    assert "./credentials:/app/credentials:ro" in text


def test_token_file_default_lives_on_writable_volume() -> None:
    """The OAuth refresh token must be writable — never under the read-only
    ``credentials`` mount, always on the writable ``data`` volume."""
    token = Settings(_env_file=None).google_token_file
    assert not token.startswith("credentials/")
    assert token.startswith("data/")


def test_static_secrets_default_under_credentials() -> None:
    """client_secret.json and the Pub/Sub service account stay under
    ``credentials/`` (the read-only mount)."""
    settings = Settings(_env_file=None)
    assert settings.google_client_secret_file.startswith("credentials/")
    assert settings.google_application_credentials == ""
    assert "credentials/" in _compose_text()
