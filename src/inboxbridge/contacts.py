"""Logical contacts + aliases — deterministic, user-configured.

Model: a LogicalContact has a display name, ONE preferred email, and zero or
more aliases. Several logical names may legitimately resolve to the same
shared mailbox (e.g. Roman → femo@femo.ch and FEMO → femo@femo.ch).

Hard rules:

- aliases are normalized (NFKC, casefold, whitespace-collapsed) and unique;
- email addresses are syntactically validated (no control chars, no spaces,
  single @, sane TLD/domain shape);
- resolution is deterministic: exact normalized alias match, then display
  name match, then exact email match;
- ambiguous matches NEVER silently choose — the caller asks;
- nothing is ever learned or overwritten silently (contact mutation is
  explicit and visible);
- the LLM never invents an address: unknown names resolve to nothing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .db import Storage

#: Disallow control characters, whitespace and '@' anywhere in an address.
_EMAIL_RE = re.compile(
    r"^[^\s\x00-\x1f\x7f<>()\[\]\\,;:\"@]+@[^\s\x00-\x1f\x7f<>()\[\]\\,;:\"@]+\.[^\s\x00-\x1f\x7f]{2,}$"
)
_ALIAS_BLOCKED_RE = re.compile(r"[\x00-\x1f\x7f@<>]")
_ALIAS_MAX_LEN = 60
_EMAIL_MAX_LEN = 254


class ContactError(ValueError):
    """Invalid contact data or operation (message is user-safe)."""


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving a recipient phrase."""

    contact: dict[str, Any] | None
    candidates: tuple[dict[str, Any], ...]
    matched_alias: str = ""

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    @property
    def resolved(self) -> bool:
        return self.contact is not None and not self.ambiguous


def normalize_alias(text: str) -> str:
    """Canonical alias form: NFKC, casefold, whitespace collapsed, trimmed."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(folded.split())


def validate_email(address: str) -> bool:
    """Syntactic email validation (defense in depth; Gmail does the rest)."""
    if not address or len(address) > _EMAIL_MAX_LEN:
        return False
    if address != address.strip():
        return False
    if re.search(r"[\x00-\x1f\x7f]", address):
        return False
    return _EMAIL_RE.match(address) is not None


class ContactService:
    """Deterministic contact/alias operations over the SQLite store."""

    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    # ── validation ──────────────────────────────────────────────────────────

    @staticmethod
    def check_alias(alias: str) -> str:
        """Validate + normalize an alias; raises ContactError when invalid."""
        normalized = normalize_alias(alias)
        if not normalized:
            raise ContactError("El alias no puede estar vacío.")
        if len(normalized) > _ALIAS_MAX_LEN:
            raise ContactError("El alias es demasiado largo.")
        if _ALIAS_BLOCKED_RE.search(alias):
            raise ContactError("El alias no puede contener @ ni caracteres raros.")
        return normalized

    @staticmethod
    def check_email(email: str) -> str:
        """Validate + normalize an email; raises ContactError when invalid."""
        cleaned = email.strip()
        if not validate_email(cleaned):
            raise ContactError("Esa dirección de correo no parece válida.")
        return cleaned.casefold()

    # ── CRUD ────────────────────────────────────────────────────────────────

    def create_contact(self, display_name: str, email: str) -> dict[str, Any]:
        """Create a contact; returns the row.

        Several logical names may share one mailbox (e.g. Roman and FEMO →
        femo@femo.ch), so the same email may exist on multiple contacts.
        """
        clean_email = self.check_email(email)
        name = display_name.strip() or clean_email
        contact_id = self._storage.create_contact(name, clean_email)
        row = self._storage.get_contact(contact_id)
        assert row is not None
        return row

    def change_email(self, contact_id: int, email: str) -> dict[str, Any]:
        """Change the preferred address of a contact (explicit, visible)."""
        clean_email = self.check_email(email)
        self._storage.update_contact_email(contact_id, clean_email)
        row = self._storage.get_contact(contact_id)
        assert row is not None
        return row

    def rename(self, contact_id: int, display_name: str) -> dict[str, Any]:
        name = display_name.strip()
        if not name:
            raise ContactError("El nombre no puede estar vacío.")
        self._storage.rename_contact(contact_id, name)
        row = self._storage.get_contact(contact_id)
        assert row is not None
        return row

    def delete(self, contact_id: int) -> None:
        self._storage.delete_contact(contact_id)

    def add_alias(self, contact_id: int, alias: str) -> str:
        normalized = self.check_alias(alias)
        # An alias must never equal another contact's email (confusable).
        if self._storage.find_contact_by_email(normalized) is not None:
            raise ContactError("Ese alias coincide con un correo guardado.")
        if not self._storage.add_alias(contact_id, normalized):
            raise ContactError("Ese alias ya existe.")
        return normalized

    def remove_alias(self, alias: str) -> int:
        return self._storage.remove_alias(normalize_alias(alias))

    # ── queries / resolution ────────────────────────────────────────────────

    def list_contacts(self) -> list[dict[str, Any]]:
        return self._storage.list_contacts()

    def get(self, contact_id: int) -> dict[str, Any] | None:
        return self._storage.get_contact(contact_id)

    def aliases_of(self, contact_id: int) -> list[str]:
        return [str(r["alias"]) for r in self._storage.list_aliases(contact_id)]

    def resolve(self, phrase: str) -> Resolution:
        """Resolve a recipient phrase to a contact (never invents anything).

        Priority: normalized alias → display name → exact email. If several
        contacts match, all candidates are returned (ambiguity must surface).
        """
        normalized = normalize_alias(phrase)
        if not normalized:
            return Resolution(contact=None, candidates=())

        candidates: list[dict[str, Any]] = []

        # 1. exact alias
        alias_row = self._storage.find_alias(normalized)
        if alias_row is not None:
            contact = self.get(int(alias_row["contact_id"]))
            if contact is not None:
                return Resolution(
                    contact=contact,
                    candidates=(contact,),
                    matched_alias=str(alias_row["alias"]),
                )

        # 2. display name (normalized compare)
        for contact in self.list_contacts():
            if normalize_alias(str(contact["display_name"])) == normalized:
                candidates.append(contact)

        # 3. exact email (shared mailboxes may match several contacts)
        if not candidates:
            for contact in self._storage.find_contacts_by_email(normalized):
                candidates.append(contact)

        return Resolution(
            contact=candidates[0] if len(candidates) == 1 else None,
            candidates=tuple(candidates),
        )
