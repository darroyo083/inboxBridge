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


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)


def _clean_candidate(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = _FENCE_RE.sub("", candidate)
        if candidate.endswith("```"):
            candidate = candidate[:-3]
    return candidate.strip()


def parse_qa_answer(text: str) -> QaAnswer | None:
    """Parse the structured Q&A contract; None when it is not valid JSON.

    Defensive by design: strips optional markdown fences, locates the JSON
    object between the first ``{`` and last ``}``, tolerates wrong field
    types by dropping invalid sections, and requires at least an answer line
    or one valid section.
    """
    candidate = _clean_candidate(text)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(candidate[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    answer = str(data.get("answer") or "").strip()
    sections: list[QaSection] = []
    raw_sections = data.get("sections")
    if isinstance(raw_sections, list):
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
    if not answer and not sections:
        return None
    return QaAnswer(answer=answer, sections=sections)
