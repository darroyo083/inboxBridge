"""Structured Q&A contract: parser and safe deterministic renderer tests."""

import json

from inboxbridge.llm.qa import QaSection, parse_qa_answer
from inboxbridge.telegram.bot import (
    render_qa_answer,
    render_qa_plain,
    render_summary,
    render_summary_plain,
)


def _answer(
    answer: str = "Hay que pagar 125 CHF. La cita es en Bahnhofstrasse 10, Zürich.",
    sections: list[QaSection] | None = None,
):
    return render_qa_answer(answer, sections or [])


# ── parser ──────────────────────────────────────────────────────────────────


def test_parse_valid_contract() -> None:
    parsed = parse_qa_answer(
        '{"answer": "ok", "sections": [{"emoji": "💰", "title": "Importe", '
        '"items": ["125 CHF"]}]}'
    )
    assert parsed is not None
    assert parsed.answer == "ok"
    assert len(parsed.sections) == 1
    assert parsed.sections[0].emoji == "💰"
    assert parsed.sections[0].title == "Importe"
    assert parsed.sections[0].items == ["125 CHF"]


def test_parse_strips_markdown_fences() -> None:
    parsed = parse_qa_answer(
        '```json\n{"answer": "sí", "sections": []}\n```'
    )
    assert parsed is not None
    assert parsed.answer == "sí"


def test_parse_tolerates_surrounding_prose() -> None:
    parsed = parse_qa_answer(
        'Claro:\n{"answer": "sí", "sections": [{"emoji": "✅", "title": "A", '
        '"items": ["x"]}]}\nEspero que sirva.'
    )
    assert parsed is not None
    assert parsed.answer == "sí"


def test_parse_drops_invalid_sections_keeps_valid() -> None:
    parsed = parse_qa_answer(
        '{"answer": "sí", "sections": [{"emoji": "💰", "title": "Bien", '
        '"items": ["1"]}, {"title": "sin items"}, "texto", {"emoji": "📅", '
        '"title": "Sin items tampoco", "items": []}]}'
    )
    assert parsed is not None
    assert len(parsed.sections) == 1
    assert parsed.sections[0].title == "Bien"


def test_parse_malformed_returns_none() -> None:
    assert parse_qa_answer("no es json en absoluto") is None
    assert parse_qa_answer("{json roto") is None
    assert parse_qa_answer('{"answer": ""}') is None  # nothing usable
    assert parse_qa_answer("") is None


def test_parse_plain_prose_returns_none() -> None:
    assert parse_qa_answer("💰 Importe\n• 500 EUR\n📍 Cita\n• Zürich") is None


# ── renderer: multi-fact sections ───────────────────────────────────────────


def test_multi_fact_sections_deterministic() -> None:
    rendered = _answer(
        "Hay que pagar 125 CHF. Cita en Bahnhofstrasse 10, Zürich.",
        [
            QaSection("💰", "Importe", ["125 CHF"]),
            QaSection(
                "📍",
                "Cita",
                ["Bahnhofstrasse 10, Zürich", "18 de agosto de 2026, 14:30"],
            ),
            QaSection("📄", "Documentos", ["Contrato firmado", "DNI o pasaporte"]),
            QaSection("⏰", "Fecha límite", ["17 de agosto de 2026, 18:00"]),
        ],
    )
    assert rendered == (
        "Hay que pagar 125 CHF. Cita en Bahnhofstrasse 10, Zürich.\n"
        "💰 <b>Importe</b>\n125 CHF\n"
        "📍 <b>Cita</b>\n• Bahnhofstrasse 10, Zürich\n• 18 de agosto de 2026, 14:30\n"
        "📄 <b>Documentos</b>\n• Contrato firmado\n• DNI o pasaporte\n"
        "⏰ <b>Fecha límite</b>\n17 de agosto de 2026, 18:00"
    )


def test_single_item_section_renders_bare() -> None:
    rendered = _answer("", [QaSection("💰", "Importe", ["125 CHF"])])
    assert rendered == "💰 <b>Importe</b>\n125 CHF"


def test_multi_item_section_renders_bullets() -> None:
    rendered = _answer("", [QaSection("📄", "Documentos", ["A", "B", "C"])])
    assert rendered == "📄 <b>Documentos</b>\n• A\n• B\n• C"


def test_no_sections_renders_answer_plain() -> None:
    rendered = _answer("La cita es el 18 de agosto.")
    assert rendered == "La cita es el 18 de agosto."
    assert "<b>" not in rendered


# ── renderer: compact single-fact ───────────────────────────────────────────


def test_compact_single_fact_bolds_fact_inline() -> None:
    rendered = _answer(
        "El contacto es Markus Schneider.",
        [QaSection("👤", "Contacto", ["Markus Schneider"])],
    )
    assert rendered == "👤 El contacto es <b>Markus Schneider</b>."


def test_compact_falls_back_to_block_when_item_not_in_answer() -> None:
    rendered = _answer(
        "La cita es el 18 de agosto.",
        [QaSection("📍", "Cita", ["Bahnhofstrasse 10, Zürich"])],
    )
    assert rendered == "La cita es el 18 de agosto.\n📍 <b>Cita</b>\nBahnhofstrasse 10, Zürich"


# ── renderer: security ──────────────────────────────────────────────────────


def test_dynamic_values_escaped() -> None:
    rendered = _answer(
        "",
        [QaSection("💰", "Precio < 100", ["100 < 200 & 50% > 90"])],
    )
    assert "<b>Precio" in rendered
    assert "Precio &lt; 100" in rendered
    assert "100 &lt; 200 &amp; 50% &gt; 90" in rendered
    assert "< 100" not in rendered.replace("<b>", "").replace("</b>", "")


def test_model_html_cannot_inject_markup() -> None:
    rendered = _answer(
        "",
        [
            QaSection(
                "💰", "<b>fake</b>", ["<script>alert(1)</script> <i>x</i>"]
            )
        ],
    )
    assert "<script>" not in rendered
    assert "<i>" not in rendered
    assert "&lt;b&gt;fake&lt;/b&gt;" in rendered
    # The only real tags are the two this function emitted.
    assert rendered.count("<b>") == 1
    assert rendered.count("</b>") == 1


def test_unknown_emoji_falls_back_to_neutral() -> None:
    rendered = _answer("", [QaSection("🚀", "Despegue", ["08:00"])])
    assert rendered.startswith("ℹ️ <b>Despegue</b>")


def test_output_always_well_formed() -> None:
    rendered = _answer(
        "a<b> & \" ' <i> [x] <hr>",
        [
            QaSection("💰", "a<b> & \" '", ["x <y> z"] * 3),
            QaSection("📍", "t2", ["u"]),
        ],
    )
    assert rendered.count("<b>") == rendered.count("</b>") == 2
    assert "&lt;" in rendered


# ── plain variant ───────────────────────────────────────────────────────────


def test_plain_variant_has_no_tags() -> None:
    rendered = render_qa_plain(
        "Hay que pagar 125 CHF.",
        [
            QaSection("💰", "Importe", ["125 CHF"]),
            QaSection("📍", "Cita", ["Zürich", "18.08.2026"]),
        ],
    )
    assert "<b>" not in rendered
    assert "💰 Importe" in rendered
    assert "• Zürich" in rendered
    assert "125 CHF" in rendered


def test_json_roundtrip_render() -> None:
    payload = {
        "answer": "Hay que pagar 125 CHF y la cita es en Bahnhofstrasse 10, Zürich.",
        "sections": [
            {"emoji": "💰", "title": "Importe", "items": ["125 CHF"]},
            {
                "emoji": "📍",
                "title": "Cita",
                "items": ["Bahnhofstrasse 10, Zürich", "18 de agosto de 2026, 14:30"],
            },
        ],
    }
    parsed = parse_qa_answer(json.dumps(payload))
    assert parsed is not None
    rendered = render_qa_answer(parsed.answer, parsed.sections)
    assert "💰 <b>Importe</b>\n125 CHF" in rendered
    assert "📍 <b>Cita</b>" in rendered
    assert "• Bahnhofstrasse 10, Zürich" in rendered


# ── thread-summary contract ─────────────────────────────────────────────────


def test_parse_summary_valid() -> None:
    from inboxbridge.llm.qa import parse_thread_summary

    parsed = parse_thread_summary(
        '{"headline": "Resumen", "sections": [{"emoji": "📅", "title": "Cita", '
        '"items": ["18 de agosto de 2026, 14:30", "Zürich"]}, {"emoji": "💰", '
        '"title": "Importe", "items": ["125 CHF"]}]}'
    )
    assert parsed is not None
    assert parsed.headline == "Resumen"
    assert len(parsed.sections) == 2
    assert parsed.sections[0].emoji == "📅"
    assert parsed.sections[0].items == ["18 de agosto de 2026, 14:30", "Zürich"]


def test_parse_summary_default_headline() -> None:
    from inboxbridge.llm.qa import parse_thread_summary

    parsed = parse_thread_summary(
        '{"sections": [{"emoji": "📬", "title": "Resumen", "items": ["a"]}]}'
    )
    assert parsed is not None
    assert parsed.headline == "Resumen"


def test_parse_summary_malformed_returns_none() -> None:
    from inboxbridge.llm.qa import parse_thread_summary

    assert parse_thread_summary("texto plano") is None
    assert parse_thread_summary('{"headline": "Resumen", "sections": []}') is None
    assert parse_thread_summary("{roto") is None


def test_parse_summary_strips_fences() -> None:
    from inboxbridge.llm.qa import parse_thread_summary

    parsed = parse_thread_summary(
        '```json\n{"headline": "Resumen", "sections": [{"emoji": "📬", '
        '"title": "Resumen", "items": ["x"]}]}\n```'
    )
    assert parsed is not None
    assert parsed.headline == "Resumen"


def test_render_summary_header_and_sections() -> None:
    rendered = render_summary(
        "Resumen",
        [
            QaSection("📅", "Cita", ["18 de agosto de 2026, 14:30", "Zürich"]),
            QaSection("👤", "Contacto", ["Markus Schneider"]),
        ],
    )
    assert rendered == (
        "📬 <b>Resumen</b>\n"
        "📅 <b>Cita</b>\n• 18 de agosto de 2026, 14:30\n• Zürich\n"
        "👤 <b>Contacto</b>\nMarkus Schneider"
    )


def test_render_summary_simple_form_dedupes_header() -> None:
    rendered = render_summary(
        "Resumen",
        [QaSection("📬", "Resumen", ["a", "b", "c"])],
    )
    assert rendered == "📬 <b>Resumen</b>\n• a\n• b\n• c"
    assert rendered.count("Resumen") == 1  # header never repeated


def test_render_summary_single_item_section_bare() -> None:
    rendered = render_summary("Resumen", [QaSection("💰", "Importe", ["125 CHF"])])
    assert "💰 <b>Importe</b>\n125 CHF" in rendered


def test_render_summary_escapes_values_and_blocks_injection() -> None:
    rendered = render_summary(
        "Resumen <b>x</b>",
        [QaSection("📄", "Docs", ["<script>alert(1)</script>", "a < b"])],
    )
    assert "<script>" not in rendered
    assert "&lt;b&gt;x&lt;/b&gt;" in rendered
    assert "a &lt; b" in rendered
    # Only the app's own tags: header + one section.
    assert rendered.count("<b>") == rendered.count("</b>") == 2


def test_render_summary_unknown_emoji_neutral() -> None:
    rendered = render_summary("Resumen", [QaSection("🚀", "Tema", ["x"])])
    assert "ℹ️ <b>Tema</b>" in rendered


def test_render_summary_plain_variant_no_tags() -> None:
    plain = render_summary_plain(
        "Resumen",
        [QaSection("📅", "Cita", ["18 de agosto", "Zürich"])],
    )
    assert "<b>" not in plain
    assert plain == "📬 Resumen\n📅 Cita\n• 18 de agosto\n• Zürich"


def test_render_summary_plain_simple_form() -> None:
    plain = render_summary_plain("Resumen", [QaSection("📬", "Resumen", ["a", "b"])])
    assert plain == "📬 Resumen\n• a\n• b"
