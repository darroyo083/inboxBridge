"""Logical contacts + aliases: validation, CRUD, resolution, hijack defenses."""

from __future__ import annotations

import pytest

from inboxbridge.contacts import ContactError, ContactService, normalize_alias, validate_email
from inboxbridge.db import Storage


@pytest.fixture
def service(tmp_path: object) -> ContactService:
    storage = Storage(str(tmp_path) + "/contacts.db")
    storage.connect()
    return ContactService(storage)


class TestValidation:
    def test_valid_emails(self) -> None:
        assert validate_email("femo@femo.ch")
        assert validate_email("roman.schmidt@example.com")
        assert validate_email("a+b@sub.example.co.uk")

    def test_invalid_emails(self) -> None:
        assert not validate_email("")
        assert not validate_email("roman")
        assert not validate_email("a@b")
        assert not validate_email("a b@example.com")
        assert not validate_email("a\nb@example.com")
        assert not validate_email("a@b@c.com")
        assert not validate_email("a@example.com ")
        assert not validate_email("mailto:a@example.com")

    def test_alias_normalization(self) -> None:
        assert normalize_alias("  Mi   Jefe ") == "mi jefe"
        assert normalize_alias("ROMÁN") == "román"
        assert normalize_alias("ａｂｃ") == "abc"  # fullwidth NFKC

    def test_alias_rejects_junk(self) -> None:
        with pytest.raises(ContactError):
            ContactService.check_alias("")
        with pytest.raises(ContactError):
            ContactService.check_alias("roman@femo.ch")  # alias cannot be an address
        with pytest.raises(ContactError):
            ContactService.check_alias("a\x00b")
        with pytest.raises(ContactError):
            ContactService.check_alias("x" * 100)


class TestCrud:
    def test_create_and_resolve_via_alias(self, service: ContactService) -> None:
        contact = service.create_contact("Roman", "femo@femo.ch")
        service.add_alias(contact["id"], "mi jefe")
        result = service.resolve("Mi Jefe")
        assert result.resolved
        assert result.contact["email"] == "femo@femo.ch"
        assert result.matched_alias == "mi jefe"

    def test_shared_mailbox_multiple_logical_names(self, service: ContactService) -> None:
        # Roman and FEMO legitimately map to the same shared inbox.
        roman = service.create_contact("Roman", "femo@femo.ch")
        femo = service.create_contact("FEMO", "femo@femo.ch")
        assert roman["email"] == femo["email"] == "femo@femo.ch"
        # Resolving by the shared address surfaces BOTH (ambiguity, never silent).
        result = service.resolve("femo@femo.ch")
        assert result.ambiguous
        assert len(result.candidates) == 2

    def test_change_email_visible_and_deterministic(self, service: ContactService) -> None:
        contact = service.create_contact("Roman", "femo@femo.ch")
        updated = service.change_email(contact["id"], "roman@femo.ch")
        assert updated["email"] == "roman@femo.ch"
        assert service.resolve("Roman").contact["email"] == "roman@femo.ch"

    def test_delete_removes_aliases(self, service: ContactService) -> None:
        contact = service.create_contact("Roman", "femo@femo.ch")
        service.add_alias(contact["id"], "mi jefe")
        service.delete(contact["id"])
        assert not service.resolve("Roman").resolved
        assert not service.resolve("mi jefe").resolved
        assert service.list_contacts() == []

    def test_duplicate_alias_rejected(self, service: ContactService) -> None:
        contact = service.create_contact("Roman", "femo@femo.ch")
        service.add_alias(contact["id"], "mi jefe")
        with pytest.raises(ContactError):
            service.add_alias(contact["id"], "Mi Jefe")  # normalized duplicate

    def test_alias_colliding_with_contact_email_rejected(self, service: ContactService) -> None:
        service.create_contact("Roman", "femo@femo.ch")
        other = service.create_contact("Manuela", "manuela@example.ch")
        with pytest.raises(ContactError):
            service.add_alias(other["id"], "femo@femo.ch")

    def test_remove_alias(self, service: ContactService) -> None:
        contact = service.create_contact("Roman", "femo@femo.ch")
        service.add_alias(contact["id"], "mi jefe")
        assert service.remove_alias("MI   JEFE") == 1
        assert not service.resolve("mi jefe").resolved


class TestResolution:
    def test_display_name_match(self, service: ContactService) -> None:
        service.create_contact("Roman", "femo@femo.ch")
        assert service.resolve("roman").resolved
        assert service.resolve("Roman").contact["email"] == "femo@femo.ch"

    def test_email_match(self, service: ContactService) -> None:
        service.create_contact("Roman", "femo@femo.ch")
        assert service.resolve("femo@femo.ch").contact["email"] == "femo@femo.ch"

    def test_unknown_never_invents(self, service: ContactService) -> None:
        result = service.resolve("Pepe")
        assert not result.resolved
        assert result.candidates == ()

    def test_ambiguous_candidates_surface(self, service: ContactService) -> None:
        service.create_contact("Roman", "femo@femo.ch")
        service.create_contact("ROMAN", "roman@example.ch")  # same normalized name
        result = service.resolve("Roman")
        assert result.ambiguous  # never silently choose
        assert {c["email"] for c in result.candidates} == {
            "femo@femo.ch",
            "roman@example.ch",
        }

    def test_case_and_accent_insensitive_alias(self, service: ContactService) -> None:
        contact = service.create_contact("Roman", "femo@femo.ch")
        service.add_alias(contact["id"], "mi jefe")
        assert service.resolve("MI JEFE").resolved

    def test_empty_phrase(self, service: ContactService) -> None:
        assert not service.resolve("   ").resolved


class TestHijackDefense:
    def test_control_chars_never_persist(self, service: ContactService) -> None:
        contact = service.create_contact("Roman", "femo@femo.ch")
        with pytest.raises(ContactError):
            service.add_alias(contact["id"], "jefe\x00redirige")
        assert service.aliases_of(contact["id"]) == []

    def test_fullwidth_homoglyph_alias_is_distinct_not_silent(
        self, service: ContactService
    ) -> None:
        # A true homoglyph alias (e.g. Cyrillic 'м') stays distinct from the
        # real alias: "Roman" resolves to the real contact, and the attacker's
        # alias resolves ONLY to the attacker — never a silent redirect.
        service.create_contact("Roman", "femo@femo.ch")
        attacker = service.create_contact("Attacker", "attacker@example.com")
        service.add_alias(attacker["id"], "roмan")  # Cyrillic м
        assert service.resolve("Roman").contact["email"] == "femo@femo.ch"
        assert service.resolve("roмan").contact["email"] == "attacker@example.com"
        assert service.resolve("roмan").matched_alias == "roмan"
