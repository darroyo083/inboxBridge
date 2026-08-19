"""Unit tests for safe rich-text rendering of AI answers (Telegram HTML)."""

from inboxbridge.telegram.bot import render_rich_text


def test_emoji_heading_line_is_bolded() -> None:
    assert render_rich_text("💰 Importe: 125 CHF") == "💰 <b>Importe: 125 CHF</b>"


def test_multiline_sections_and_bullets() -> None:
    text = "💰 Importe\n• Total: 125 CHF\n📍 Cita\n• Bahnhofstrasse 10, Zürich"
    assert render_rich_text(text) == (
        "💰 <b>Importe</b>\n• Total: 125 CHF\n📍 <b>Cita</b>\n• Bahnhofstrasse 10, Zürich"
    )


def test_model_html_is_escaped_not_executed() -> None:
    attack = "💰 <b>fake bold</b> & <script>alert(1)</script>"
    rendered = render_rich_text(attack)
    assert "<b>fake bold</b>" not in rendered
    assert "&lt;b&gt;fake bold&lt;/b&gt;" in rendered
    assert "<script>" not in rendered
    # The only real tags are the ones this function emits.
    assert rendered.count("<b>") == 1
    assert rendered.count("</b>") == 1


def test_plain_lines_never_get_tags() -> None:
    rendered = render_rich_text("Sin emojis\nsegunda línea")
    assert "<b>" not in rendered
    assert rendered == "Sin emojis\nsegunda línea"


def test_emoji_only_line_is_kept_without_empty_bold() -> None:
    assert render_rich_text("✅\nLuego texto") == "✅\nLuego texto"


def test_unknown_emoji_not_treated_as_heading() -> None:
    # 🚀 is NOT whitelisted → stays plain escaped text.
    assert render_rich_text("🚀 despegue") == "🚀 despegue"


def test_output_is_always_well_formed_html() -> None:
    nasty = "💰 a<b> & \" ' <i> [x] <hr>" * 20
    rendered = render_rich_text(nasty)
    assert rendered.count("<b>") == rendered.count("</b>")
    assert "&lt;" in rendered
