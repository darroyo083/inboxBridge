"""Regression: the new-mail SUBJECT must come from the compose LLM JSON (or a
safe fallback) — never the raw bot command — and must never leak the recipient
address. Guarded by assistant._parse_compose + _strip_recipient_from_subject.
"""

from __future__ import annotations

from inboxbridge.assistant import (
    _FALLBACK_SUBJECT,
    _parse_compose,
    _strip_recipient_from_subject,
)


def test_parse_compose_valid_json() -> None:
    subject, body = _parse_compose(
        '{"subject_de": "Vielen Dank", "body_de": "Sehr geehrte Frau Muster,\\n\\n'
        "vielen Dank für Ihre Nachricht.\\n\\nMit freundlichen Grüßen\"}"
    )
    assert subject == "Vielen Dank"
    assert "Sehr geehrte Frau Muster" in body


def test_parse_compose_raw_text_becomes_body_with_no_subject() -> None:
    subject, body = _parse_compose("Sehr geehrte Frau Muster, vielen Dank.")
    assert subject == ""
    assert body == "Sehr geehrte Frau Muster, vielen Dank."


def test_parse_compose_garbage_never_yields_command_subject() -> None:
    subject, body = _parse_compose("envía un correo a user@example.com diciendo que hola")
    # The raw command text is NEVER a subject — it becomes the body only.
    assert subject == ""
    assert "user@example.com" in body


def test_parse_compose_malformed_json_is_tolerated() -> None:
    subject, body = _parse_compose('{"subject_de": "Solo asunto"}')
    assert subject == ""
    assert body == '{"subject_de": "Solo asunto"}'


def test_strip_recipient_email_from_subject() -> None:
    assert (
        _strip_recipient_from_subject("Reunión con user@example.com", "user@example.com")
        == "Reunión con"
    )
    assert (
        _strip_recipient_from_subject("Reunión con USER@EXAMPLE.COM", "user@example.com")
        == "Reunión con"
    )


def test_strip_recipient_email_alone_falls_back() -> None:
    assert (
        _strip_recipient_from_subject("user@example.com", "user@example.com")
        == _FALLBACK_SUBJECT
    )


def test_strip_recipient_email_keeps_unrelated_subject() -> None:
    assert (
        _strip_recipient_from_subject("Vielen Dank", "user@example.com") == "Vielen Dank"
    )
