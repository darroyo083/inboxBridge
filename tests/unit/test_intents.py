"""Intent routing: deterministic rules, explicit semantics, LLM fallback safety."""

from __future__ import annotations

import asyncio
from typing import Any

from inboxbridge.intents import Intent, IntentAction, IntentClassifier


def classify_rule(text: str) -> Intent:
    return IntentClassifier().classify_rule_only(text)


class TestRuleBasics:
    def test_explicit_send_verbs(self) -> None:
        for phrase in ("envíalo", "manda el correo", "sí, mándaselo", "envíalo ya"):
            intent = classify_rule(phrase)
            assert intent.action == IntentAction.SEND_DRAFT
            assert intent.explicit  # explicit verb acts need no button confirm

    def test_ambiguous_acks_never_send(self) -> None:
        for phrase in ("ok", "vale", "sí", "perfecto", "bien", "de acuerdo", "si"):
            intent = classify_rule(phrase)
            assert intent.action == IntentAction.CLARIFY
            assert not intent.explicit

    def test_cancel_draft_explicit(self) -> None:
        intent = classify_rule("cancela el borrador")
        assert intent.action == IntentAction.CANCEL_DRAFT and intent.explicit

    def test_archive_mark_read(self) -> None:
        assert classify_rule("archívalo").action == IntentAction.ARCHIVE
        assert classify_rule("márcalo como leído").action == IntentAction.MARK_READ
        assert classify_rule("archívalo").explicit

    def test_edit_and_regenerate(self) -> None:
        intent = classify_rule("hazlo más corto")
        assert intent.action == IntentAction.MODIFY_DRAFT
        assert not intent.explicit  # edit never implicit-send
        assert classify_rule("hazlo más formal").action == IntentAction.MODIFY_DRAFT
        assert classify_rule("cambia las 18:00 por las 19:00").action == IntentAction.MODIFY_DRAFT
        assert classify_rule("reescríbelo").action == IntentAction.REGENERATE_DRAFT

    def test_questions_and_summaries(self) -> None:
        assert classify_rule("¿qué me está pidiendo?").action == IntentAction.ASK_ABOUT_EMAIL
        assert classify_rule("¿tengo que responder?").action == IntentAction.ASK_ABOUT_EMAIL
        assert classify_rule("resume toda la conversación").action == IntentAction.SUMMARIZE_THREAD
        assert classify_rule("¿qué ha pasado en este hilo?").action == IntentAction.SUMMARIZE_THREAD

    def test_attachment_request(self) -> None:
        assert classify_rule("mándame el pdf").action == IntentAction.GET_ATTACHMENT
        assert classify_rule("enséñame los adjuntos").action == IntentAction.GET_ATTACHMENT

    def test_reminders(self) -> None:
        assert classify_rule("recuérdamelo mañana").action == IntentAction.CREATE_REMINDER
        assert classify_rule("¿qué recordatorios tengo?").action == IntentAction.LIST_REMINDERS
        assert classify_rule("cancela el recordatorio").action == IntentAction.CANCEL_REMINDER

    def test_contacts(self) -> None:
        assert classify_rule("¿qué contactos tengo?").action == IntentAction.LIST_CONTACTS
        assert classify_rule("muéstrame mis contactos").action == IntentAction.LIST_CONTACTS
        assert (
            classify_rule("cuando diga roman usa femo@femo.ch").action
            == IntentAction.CREATE_CONTACT
        )
        assert (
            classify_rule("guarda a Manuela como manuela@example.ch").action
            == IntentAction.CREATE_CONTACT
        )
        assert (
            classify_rule("cambia el correo de roman a x@y.ch").action
            == IntentAction.UPDATE_CONTACT
        )
        assert (
            classify_rule("borra el contacto roman").action == IntentAction.DELETE_CONTACT
        )
        assert (
            classify_rule("añade 'mi jefe' como alias de roman").action
            == IntentAction.ADD_ALIAS
        )
        assert (
            classify_rule("quita el alias 'mi jefe' de roman").action
            == IntentAction.REMOVE_ALIAS
        )

    def test_compose_and_forward(self) -> None:
        intent = classify_rule("escribe a Roman y dile que mañana llego a las seis")
        assert intent.action == IntentAction.COMPOSE_NEW_EMAIL
        assert intent.payload.get("recipient", "").lower().startswith("roman")
        assert "mañana llego a las seis" in intent.payload.get("instruction", "")

        intent = classify_rule("reenvíaselo a Daniel")
        assert intent.action == IntentAction.FORWARD_EMAIL
        assert intent.payload.get("recipient", "").lower() == "daniel"

    def test_help_and_unknown(self) -> None:
        assert classify_rule("ayuda").action == IntentAction.HELP
        assert classify_rule("hola que tal").action == IntentAction.UNKNOWN
        assert classify_rule("gracias").action == IntentAction.UNKNOWN

    def test_high_impact_requires_explicit(self) -> None:
        # A generic instruction that happens to mention send is NOT explicit.
        intent = classify_rule("dile que me envíe el contrato")
        assert intent.action != IntentAction.SEND_DRAFT


class FakeAi:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    async def text(self, messages: list[Any], *, max_tokens: int, task: str) -> str:
        self.calls += 1
        return self._content


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestLlmFallback:
    def test_llm_classification_is_validated(self) -> None:
        ai = FakeAi(
            '{"action": "summarize_thread", "recipient": "", "instruction": '
            '"resume el hilo", "needs_clarification": false}'
        )
        intent = run(
            IntentClassifier(ai).classify("resume el hilo", context="resumen de correo")
        )
        assert intent.action == IntentAction.SUMMARIZE_THREAD
        assert not intent.explicit  # LLM can never be explicit

    def test_llm_send_is_never_explicit(self) -> None:
        # "procede con el envío" does not match the deterministic send rules,
        # so the LLM classifies it. The LLM vocabulary EXCLUDES send/cancel
        # entirely: even if the model returns "send_draft", the result is
        # rejected (UNKNOWN) — the model can never authorize a send.
        ai = FakeAi(
            '{"action": "send_draft", "recipient": "", "instruction": '
            '"procede con el envío", "needs_clarification": false}'
        )
        intent = run(
            IntentClassifier(ai).classify(
                "procede con el envío", context="borrador activo"
            )
        )
        assert intent.action == IntentAction.UNKNOWN
        assert not intent.explicit

    def test_llm_unknown_action_rejected(self) -> None:
        ai = FakeAi(
            '{"action": "send_all_emails_now", "instruction": "x", '
            '"needs_clarification": false}'
        )
        intent = run(IntentClassifier(ai).classify("haz algo", context=""))
        assert intent.action == IntentAction.UNKNOWN

    def test_llm_garbage_rejected(self) -> None:
        ai = FakeAi("no entiendo nada de esto")
        intent = run(IntentClassifier(ai).classify("haz algo", context=""))
        assert intent.action == IntentAction.UNKNOWN

    def test_llm_failure_falls_back_to_unknown(self) -> None:
        from inboxbridge.llm.base import LLMError

        class BrokenAi:
            async def text(self, messages: list[Any], *, max_tokens: int, task: str) -> str:
                raise LLMError("boom")

        intent = run(IntentClassifier(BrokenAi()).classify("haz algo", context=""))
        assert intent.action == IntentAction.UNKNOWN

    def test_ambiguous_llm_result_becomes_clarify(self) -> None:
        ai = FakeAi(
            '{"action": "reply_to_email", "recipient": "", "instruction": '
            '"dile que sí", "needs_clarification": true}'
        )
        intent = run(IntentClassifier(ai).classify("dile que sí", context=""))
        assert intent.action == IntentAction.CLARIFY

    def test_rules_win_over_llm(self) -> None:
        ai = FakeAi('{"action": "archive", "instruction": "x", "needs_clarification": false}')
        intent = run(IntentClassifier(ai).classify("márcalo como leído", context=""))
        assert intent.action == IntentAction.MARK_READ  # rule, not LLM
        assert ai.calls == 0
