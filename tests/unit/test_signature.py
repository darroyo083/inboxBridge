"""Trusted signature lifecycle for outgoing drafts: orphan sign-offs rejected,
signature appended deterministically, never duplicated by edits, never invented
from recipient/thread data."""

from __future__ import annotations

import pytest

from inboxbridge.llm.base import LLMIncompleteResponse
from inboxbridge.llm.signature import (
    ensure_signature,
    finalize_draft_body,
    is_sign_off,
    looks_orphan_signoff,
)


def test_is_sign_off_recognizes_german_and_spanish_closings() -> None:
    assert is_sign_off("Mit freundlichen Grüßen")
    assert is_sign_off("Viele Grüße")
    assert is_sign_off("  Hochachtungsvoll  ")
    assert is_sign_off("Atentamente")
    assert not is_sign_off("Danke, bis morgen.")


def test_looks_orphan_signoff() -> None:
    assert looks_orphan_signoff(
        "Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen"
    )
    assert not looks_orphan_signoff("Danke, bis morgen.")
    assert not looks_orphan_signoff(
        "Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen\n\nDaniel"
    )


def test_ensure_signature_appends_after_closing() -> None:
    body = "Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen"
    assert ensure_signature(body, "Daniel") == body + "\n\nDaniel"


def test_ensure_signature_dedups_existing_signature() -> None:
    body = "Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen\n\nDaniel"
    assert ensure_signature(body, "Daniel") == body


def test_ensure_signature_replaces_invented_name() -> None:
    body = (
        "Sehr geehrte Frau Muster,\n\nvielen Dank.\n\n"
        "Mit freundlichen Grüßen\nInboxBridge"
    )
    out = ensure_signature(body, "Daniel")
    assert out.endswith("Mit freundlichen Grüßen\n\nDaniel")
    assert "InboxBridge" not in out


def test_ensure_signature_short_reply_unchanged() -> None:
    assert ensure_signature("Danke, bis morgen.", "Daniel") == "Danke, bis morgen."


def test_ensure_signature_no_config_returns_unchanged() -> None:
    body = "Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen"
    assert ensure_signature(body, "") == body


def test_finalize_draft_body_rejects_orphan_without_config() -> None:
    body = "Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen"
    with pytest.raises(LLMIncompleteResponse):
        finalize_draft_body(body, "")


def test_finalize_draft_body_accepts_with_config() -> None:
    body = "Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen"
    assert finalize_draft_body(body, "Daniel").endswith("Grüßen\n\nDaniel")


def test_repeated_edits_never_duplicate_signature() -> None:
    body = "Sehr geehrte Frau Muster,\n\nvielen Dank.\n\nMit freundlichen Grüßen"
    once = ensure_signature(body, "Daniel")
    twice = ensure_signature(once, "Daniel")
    thrice = ensure_signature(twice, "Daniel")
    assert once == twice == thrice
    assert once.count("Daniel") == 1
