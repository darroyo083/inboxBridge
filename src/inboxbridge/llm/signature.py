"""Trusted sender signature lifecycle for outgoing drafts.

The sender identity is explicit application configuration
(``EMAIL_SIGNATURE_NAME``) — never inferred from email, LLM or Telegram
content, and never invented by the model. The LLM writes the semantic body; this
module deterministically attaches the canonical closing + signature so a formal
email never ends on an orphan sign-off, and repeated edits never duplicate the
signature.
"""

from __future__ import annotations

from .base import LLMIncompleteResponse

#: Recognized formal closings (casefolded — note Python casefold maps ß→ss).
#: A draft ending on one of these (with nothing after it) is structurally
#: incomplete unless the trusted signature follows.
_SIGN_OFFS = frozenset(
    {
        # German
        "mit freundlichen grüssen",
        "mit freundlichem gruss",
        "viele grüsse",
        "freundliche grüsse",
        "beste grüsse",
        "liebe grüsse",
        "herzliche grüsse",
        "hochachtungsvoll",
        # Spanish
        "atentamente",
        "un saludo",
        "saludos cordiales",
        "cordialmente",
        # English
        "best regards",
        "kind regards",
        "regards",
        "sincerely",
    }
)


def is_sign_off(line: str) -> bool:
    """Whether a trimmed line is a recognized formal closing."""
    return line.strip().casefold() in _SIGN_OFFS


def looks_orphan_signoff(body: str) -> bool:
    """True when the body ends on a formal closing with nothing after it."""
    lines = [ln for ln in body.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    return is_sign_off(lines[-1])


def _is_name_like(line: str) -> bool:
    """A single-word, capitalized, punctuation-free line (an invented name)."""
    word = line.strip()
    if not word or " " in word or "\t" in word:
        return False
    if word[-1] in ".!?,":
        return False
    if any(ch.isdigit() for ch in word):
        return False
    return word[0].isupper()


def ensure_signature(body: str, signature_name: str) -> str:
    """Normalize the final sendable body with the trusted signature.

    - No configured signature → body returned unchanged (an orphan sign-off is
      then rejected by the completeness validator).
    - Drops trailing lines equal to the trusted signature (dedup after edits).
    - Replaces a single invented name line directly after a recognized closing.
    - If the body ends on a recognized closing, appends the trusted signature.
    """
    text = body.strip()
    if not text or not signature_name:
        return text
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and lines[-1].strip().casefold() == signature_name.casefold():
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    if (
        len(lines) >= 2
        and is_sign_off(lines[-2].strip())
        and _is_name_like(lines[-1].strip())
    ):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    if lines and is_sign_off(lines[-1].strip()):
        lines.append("")
        lines.append(signature_name)
    return "\n".join(lines).strip()


def finalize_draft_body(body: str, signature_name: str) -> str:
    """Apply the trusted signature and validate structural completeness.

    The returned body is the EXACT sendable German text (the Spanish translation
    is derived from it). Raises :class:`LLMIncompleteResponse` when the final
    body ends on a recognized formal closing with no signature after it (i.e.
    no trusted signature is configured) — such a draft must never be sendable.
    """
    body = ensure_signature(body, signature_name)
    if looks_orphan_signoff(body):
        raise LLMIncompleteResponse(
            "draft ends on an orphan sign-off without a trusted signature"
        )
    return body
