"""Unit tests for gmail/auth.py — OAuth flow mocked, no network, no secrets."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from google.oauth2.credentials import Credentials

from inboxbridge.config import Settings
from inboxbridge.gmail import auth
from inboxbridge.gmail.auth import AuthError, CredentialsStore, get_credentials, reauthenticate

SCOPES = list(auth.SCOPES)


def make_settings(tmp_path: object) -> Settings:
    return Settings(
        _env_file=None,
        google_token_file=str(tmp_path) + "/token.json",
        google_client_secret_file=str(tmp_path) + "/client_secret.json",
    )


def make_credentials(*, expired: bool = False) -> Credentials:
    kwargs: dict[str, object] = {
        "token": "access-token",
        "refresh_token": "refresh-token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "client-id",
        "client_secret": "client-secret",
    }
    # google-auth auto-expires tokens whose JSON lacks "expiry" — real flows
    # always persist one, so fixtures do too.
    if expired:
        expiry = datetime.now(UTC) - timedelta(minutes=5)
    else:
        expiry = datetime.now(UTC) + timedelta(hours=2)
    kwargs["expiry"] = expiry
    return Credentials(scopes=SCOPES, **kwargs)  # type: ignore[no-untyped-call]


def write_secret_file(tmp_path: object) -> None:
    import json
    import os

    path = os.path.join(str(tmp_path), "client_secret.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "installed": {
                    "client_id": "test.apps.googleusercontent.com",
                    "project_id": "test",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "client_secret": "test-secret",
                    "redirect_uris": ["http://localhost"],
                }
            },
            fh,
        )
    return None


class FakeFlow:
    def __init__(self) -> None:
        self.run_called = False
        self.credentials = make_credentials()

    def run_local_server(self, **kwargs: object) -> Credentials:
        self.run_called = True
        self.run_kwargs = kwargs
        return self.credentials


class TestCredentialsStore:
    def test_load_missing_returns_none(self, tmp_path: object) -> None:
        store = CredentialsStore(str(tmp_path) + "/absent.json")
        assert store.load() is None

    def test_save_load_roundtrip(self, tmp_path: object) -> None:
        store = CredentialsStore(str(tmp_path) + "/token.json")
        creds = make_credentials()
        store.save(creds)
        loaded = store.load()
        assert loaded is not None
        assert loaded.token == "access-token"
        assert loaded.refresh_token == "refresh-token"
        assert set(loaded.scopes) == set(SCOPES)

    def test_delete(self, tmp_path: object) -> None:
        store = CredentialsStore(str(tmp_path) + "/token.json")
        store.save(make_credentials())
        store.delete()
        assert store.load() is None

    def test_corrupt_file_returns_none(self, tmp_path: object) -> None:
        import os

        path = os.path.join(str(tmp_path), "token.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("not json")
        assert CredentialsStore(path).load() is None


class TestGetCredentials:
    def test_loads_existing_valid_token(self, tmp_path: object) -> None:
        settings = make_settings(tmp_path)
        CredentialsStore(settings.google_token_file).save(make_credentials())
        creds = get_credentials(settings)
        assert creds.token == "access-token"

    def test_refreshes_expired_token(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = make_settings(tmp_path)
        store = CredentialsStore(settings.google_token_file)
        store.save(make_credentials(expired=True))

        class FakeResponse:
            ok = True
            status = 200
            data = b'{"access_token": "fresh-token", "expires_in": 3600, "token_type": "Bearer"}'

            @staticmethod
            def json() -> dict[str, object]:
                return {
                    "access_token": "fresh-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                }

        class FakeRequest:
            def __call__(self, **kwargs: object) -> FakeResponse:
                return FakeResponse()

        monkeypatch.setattr(auth, "Request", lambda: FakeRequest())
        creds = get_credentials(settings, store=store)
        assert creds.token == "fresh-token"
        assert not creds.expired

    def test_runs_flow_when_no_token(self, tmp_path: object) -> None:
        settings = make_settings(tmp_path)
        write_secret_file(tmp_path)
        fake = FakeFlow()
        captured: dict[str, Any] = {}

        def builder(path: str, scopes: list[str]) -> Any:
            captured["path"] = path
            captured["scopes"] = scopes
            return fake

        creds = get_credentials(settings, flow_builder=builder)
        assert fake.run_called
        assert captured["path"].endswith("client_secret.json")
        assert captured["scopes"] == SCOPES
        assert creds.token == "access-token"
        assert CredentialsStore(settings.google_token_file).load() is not None

    def test_raises_when_secret_missing(self, tmp_path: object) -> None:
        settings = make_settings(tmp_path)
        with pytest.raises(AuthError):
            get_credentials(settings)

    def test_reauthenticate_deletes_and_runs_flow(self, tmp_path: object) -> None:
        settings = make_settings(tmp_path)
        write_secret_file(tmp_path)
        store = CredentialsStore(settings.google_token_file)
        store.save(make_credentials())
        fake = FakeFlow()
        creds = reauthenticate(settings, store=store, flow_builder=lambda *_: fake)
        assert creds.token == "access-token"
        assert fake.run_called
        assert store.load() is not None

    def test_scopes_are_gmail_modify(self) -> None:
        assert auth.SCOPES == ("https://www.googleapis.com/auth/gmail.modify",)
