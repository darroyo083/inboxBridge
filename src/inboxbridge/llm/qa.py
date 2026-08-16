"""Structured informational-response contract for Q&A.

The Q&A model returns a small JSON document instead of free-form prose so the
application can render deterministic, safe Telegram formatting:

    {"answer": "...", "sections": [{"emoji": "💰", "title": "...", "items": [...]}]}

All model output is treated as untrusted data: parsing is defensive and
rendering escapes everything (see telegram.bot.render_qa_answer). When the
model does not produce a valid document the raw text is preserved and rendered
with the plain safe formatter — information is never lost.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

#: Emojis the model may use as section anchors. This is the SINGLE source of
#: truth: the prompt lists them for the model and the renderer only honors
#: these (any other emoji falls back to a neutral one). Unknown emojis are
#: never rendered as markup.
CONTEXTUAL_EMOJIS = frozenset("💰📍📅⏰🕐✈️🏠🏢📄📎🖇️✅⚠️ℹ️📞✉️👤📬📦🔔❓👋")


@dataclass(frozen=True)
class QaSection:
    """One fact block: emoji anchor, bold title, one or more items."""

    emoji: str
    title: str
    items: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QaAnswer:
    """Parsed structured Q&A response (answer line + optional sections)."""

    answer: str
    sections: list[QaSection] = field(default_factory=list)


@dataclass(frozen=True)
class ThreadSummary:
    """Parsed structured thread-summary response (headline + sections).

    Uses the same safe ``QaSection`` building block; a one-section summary
    with ``title == headline`` renders as a compact bullet list.
    """

    headline: str
    sections: list[QaSection] = field(default_factory=list)


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)


def _clean_candidate(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = _FENCE_RE.sub("", candidate)
        if candidate.endswith("```"):
            candidate = candidate[:-3]
    return candidate.strip()


def _extract_object(text: str) -> dict[str, object] | None:
    candidate = _clean_candidate(text)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(candidate[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _parse_sections(raw_sections: object) -> list[QaSection]:
    """Tolerant section parsing shared by Q&A and thread-summary contracts:
    invalid entries are dropped, never fatal."""
    sections: list[QaSection] = []
    if not isinstance(raw_sections, list):
        return sections
    for raw in raw_sections:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        emoji = str(raw.get("emoji") or "").strip()
        items_raw = raw.get("items")
        items: list[str] = []
        if isinstance(items_raw, list):
            items = [str(item).strip() for item in items_raw if str(item).strip()]
        if not title or not items:
            continue
        sections.append(QaSection(emoji=emoji, title=title, items=items))
    return sections


def parse_qa_answer(text: str) -> QaAnswer | None:
    """Parse the structured Q&A contract; None when it is not valid JSON.

    Defensive by design: strips optional markdown fences, locates the JSON
    object between the first ``{`` and last ``}``, tolerates wrong field
    types by dropping invalid sections, and requires at least an answer line
    or one valid section.
    """
    data = _extract_object(text)
    if data is None:
        return None
    answer = str(data.get("answer") or "").strip()
    sections = _parse_sections(data.get("sections"))
    if not answer and not sections:
        return None
    return QaAnswer(answer=answer, sections=sections)


def parse_thread_summary(text: str) -> ThreadSummary | None:
    """Parse the structured thread-summary contract; None when unusable.

    Same tolerance as ``parse_qa_answer``; the headline defaults to
    ``Resumen`` when missing or empty.
    """
    data = _extract_object(text)
    if data is None:
        return None
    headline = str(data.get("headline") or "Resumen").strip() or "Resumen"
    sections = _parse_sections(data.get("sections"))
    if not sections:
        return None
    return ThreadSummary(headline=headline, sections=sections)
