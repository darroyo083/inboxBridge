"""LLM tests: retry wrapper behavior and OpenAI-compatible error mapping.

No network: the OpenAI client is monkeypatched; ``asyncio.sleep`` is stubbed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import httpx
import openai
import pytest

from inboxbridge.config import Settings
from inboxbridge.llm import OpenAICompatLLM, base, prompts
from inboxbridge.llm.openai_compat import CompletionMeta
from inboxbridge.models import (
    DraftRequest,
    EmailAddress,
    EmailSummary,
    ParsedEmail,
    ThreadContext,
    ThreadMessage,
)


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Settings:
    values: dict[str, str] = {
        "LLM_API_KEY": "test-key",
        "LLM_BASE_URL": "https://api.test/v1",
        "LLM_MODEL": "test-model",
        "AI_TEXT_MODEL": "test-model",
        "LLM_MAX_RETRIES": "1",
        "EMAIL_SIGNATURE_NAME": "Daniel",
        "TELEGRAM_BOT_TOKEN": "test-token",
    }
    for key, value in overrides.items():
        values[key] = str(value)
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return Settings()


def _email() -> ParsedEmail:
    return ParsedEmail(
        message_id="m1",
        thread_id="t1",
        history_id=1,
        subject="Asunto",
        sender=EmailAddress("Ana", "ana@example.com"),
        recipients=[EmailAddress("Bob", "bob@example.com")],
        date_iso="2026-08-07T10:00:00+00:00",
        body_text="cuerpo",
    )


def _thread() -> ThreadContext:
    return ThreadContext(
        thread_id="t1",
        subject="Re: Proyecto",
        history_id=2,
        messages=[
            ThreadMessage(
                message_id="m1",
                from_=EmailAddress("Ana", "ana@example.com"),
                date_iso="2026-08-06T09:00:00+00:00",
                body_text="Hola",
            ),
            ThreadMessage(
                message_id="m2",
                from_=EmailAddress("Bob", "bob@example.com"),
                date_iso="2026-08-06T10:00:00+00:00",
                body_text="Danke",
            ),
        ],
    )


def _error(exc_cls: type[openai.APIError]) -> Exception:
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    response = httpx.Response(429, request=request)
    if exc_cls is openai.RateLimitError:
        return openai.RateLimitError("boom", response=response, body=None)
    if exc_cls is openai.InternalServerError:
        return openai.InternalServerError("boom", response=response, body=None)
    if exc_cls is openai.AuthenticationError:
        return openai.AuthenticationError("boom", response=response, body=None)
    if exc_cls is openai.BadRequestError:
        return openai.BadRequestError("boom", response=response, body=None)
    if exc_cls is openai.APIResponseValidationError:
        return openai.APIResponseValidationError(response=response, body=None)
    if exc_cls is openai.APIConnectionError:
        return openai.APIConnectionError(message="connection failed", request=request)
    if exc_cls is openai.APITimeoutError:
        return openai.APITimeoutError(request=request)
    raise AssertionError(f"unhandled error class: {exc_cls}")


def _provider(monkeypatch: pytest.MonkeyPatch) -> OpenAICompatLLM:
    return OpenAICompatLLM(_settings(monkeypatch))


def _fake_create(
    provider: OpenAICompatLLM,
    monkeypatch: pytest.MonkeyPatch,
    content: str | None = "  Resumen  ",
) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        captured.append(kwargs)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    return captured


def _fake_create_with_finish(
    provider: OpenAICompatLLM,
    monkeypatch: pytest.MonkeyPatch,
    results: list[tuple[str, str]],
) -> list[dict[str, object]]:
    """Results are (content, finish_reason) returned in order per call."""
    captured: list[dict[str, object]] = []
    it = iter(results)

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        captured.append(kwargs)
        content, finish_reason = next(it)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    return captured


# ── retry wrapper ─────────────────────────────────────────────────────────


async def test_call_with_retry_succeeds_on_first_attempt() -> None:
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await base.call_with_retry(fn, max_attempts=3)
    assert result == "ok"
    assert calls == 1


async def test_call_with_retry_retries_transient_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise base.LLMUnavailable("down")
        return "ok"

    result = await base.call_with_retry(fn, max_attempts=5, base_backoff=1.0)
    assert result == "ok"
    assert calls == 3
    assert len(sleeps) == 2
    assert all(0 <= seconds <= 2.0 for seconds in sleeps)


async def test_call_with_retry_raises_after_exhausting_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise base.LLMRateLimited("quota")

    with pytest.raises(base.LLMRateLimited):
        await base.call_with_retry(fn, max_attempts=3)
    assert calls == 3


async def test_call_with_retry_does_not_retry_permanent_errors() -> None:
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise base.LLMError("permanent")

    with pytest.raises(base.LLMError):
        await base.call_with_retry(fn, max_attempts=3)
    assert calls == 1


async def test_call_with_retry_does_not_retry_invalid_response() -> None:
    """Genuinely invalid output (refusal/garbage) stays permanent: the
    plain LLMInvalidResponse is NOT in the retryable set."""
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise base.LLMInvalidResponse("refusal")

    with pytest.raises(base.LLMInvalidResponse):
        await base.call_with_retry(fn, max_attempts=3)
    assert calls == 1


async def test_call_with_retry_retries_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise base.LLMEmptyResponse("empty")
        return "ok"

    result = await base.call_with_retry(fn, max_attempts=5, base_backoff=1.0)
    assert result == "ok"
    assert calls == 3
    assert len(sleeps) == 2
    assert all(0 <= seconds <= 2.0 for seconds in sleeps)


async def test_call_with_retry_invalid_attempts() -> None:
    async def fn() -> str:
        return "never"

    with pytest.raises(ValueError):
        await base.call_with_retry(fn, max_attempts=0)


# ── error mapping ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("exc_cls", "expected"),
    [
        (openai.RateLimitError, base.LLMRateLimited),
        (openai.APIConnectionError, base.LLMUnavailable),
        (openai.APITimeoutError, base.LLMUnavailable),
        (openai.InternalServerError, base.LLMUnavailable),
        (openai.AuthenticationError, base.LLMError),
        (openai.BadRequestError, base.LLMError),
        (openai.APIResponseValidationError, base.LLMError),
    ],
)
async def test_openai_errors_map_to_llm_exceptions(
    monkeypatch: pytest.MonkeyPatch, exc_cls: type[openai.APIError], expected: type[base.LLMError]
) -> None:
    provider = _provider(monkeypatch)

    async def boom(**kwargs: object) -> SimpleNamespace:
        raise _error(exc_cls)

    monkeypatch.setattr(provider._client.chat.completions, "create", boom)
    with pytest.raises(expected):
        await provider.summarize_email(_email())


async def test_rate_limit_error_is_retried_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    settings = _settings(monkeypatch, LLM_MAX_RETRIES=3)
    provider = OpenAICompatLLM(settings)
    calls = 0

    async def boom(**kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        raise _error(openai.RateLimitError)

    monkeypatch.setattr(provider._client.chat.completions, "create", boom)
    with pytest.raises(base.LLMRateLimited):
        await provider.summarize_email(_email())
    assert calls == 3


# ── provider behavior ─────────────────────────────────────────────────────


async def test_summarize_email_sends_prompt_and_parses_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    content = (
        '{"subject_es": "Plan de trabajo de la próxima semana", '
        '"summary_es": "Resumen breve."}'
    )
    captured = _fake_create(provider, monkeypatch, content=content)
    result = await provider.summarize_email(_email())
    assert result == EmailSummary(
        subject_es="Plan de trabajo de la próxima semana", summary_es="Resumen breve."
    )
    request = cast(dict[str, Any], captured[0])
    assert request["model"] == "test-model"
    assert request["max_tokens"] == provider._settings.llm_max_tokens_summary
    assert request["temperature"] == 0.4
    assert request["messages"][0]["role"] == "system"
    assert prompts.UNTRUSTED_DATA_START in request["messages"][1]["content"]


async def test_summarize_email_plain_text_falls_back_to_raw_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-JSON LLM output must not break the pipeline: raw text becomes the
    summary and the Spanish subject stays empty (original subject shown)."""
    provider = _provider(monkeypatch)
    _fake_create(provider, monkeypatch, content="  Resumen breve en texto plano.  ")
    result = await provider.summarize_email(_email())
    assert result == EmailSummary(subject_es="", summary_es="Resumen breve en texto plano.")


async def test_summarize_email_json_without_subject_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    _fake_create(provider, monkeypatch, content='{"summary_es": "Solo resumen."}')
    result = await provider.summarize_email(_email())
    assert result == EmailSummary(subject_es="", summary_es="Solo resumen.")


async def test_summarize_email_fenced_json_block_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    content = '```json\n{"subject_es": "Reunión", "summary_es": "El viernes."}\n```'
    _fake_create(provider, monkeypatch, content=content)
    result = await provider.summarize_email(_email())
    assert result.subject_es == "Reunión"
    assert result.summary_es == "El viernes."


async def test_summarize_email_empty_content_raises_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    _fake_create(provider, monkeypatch, content=None)
    with pytest.raises(base.LLMInvalidResponse):
        await provider.summarize_email(_email())


async def test_summarize_email_empty_content_is_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty responses are TRANSIENT: retried within the same call_with_retry
    budget; a valid second response succeeds."""
    async def fake_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    settings = _settings(monkeypatch, LLM_MAX_RETRIES=3)
    provider = OpenAICompatLLM(settings)
    calls = 0

    async def empty_then_ok(**kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        content = "Resumen tras reintento." if calls > 1 else None
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(provider._client.chat.completions, "create", empty_then_ok)
    result = await provider.summarize_email(_email())
    assert result.summary_es == "Resumen tras reintento."
    assert calls == 2


async def test_summarize_email_repeated_empty_raises_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All attempts empty → LLMEmptyResponse (an LLMInvalidResponse subclass)
    propagates so the pipeline FAILED + delayed retry path is preserved."""
    async def fake_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    settings = _settings(monkeypatch, LLM_MAX_RETRIES=3)
    provider = OpenAICompatLLM(settings)
    calls = 0

    async def always_empty(**kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        message = SimpleNamespace(content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(provider._client.chat.completions, "create", always_empty)
    with pytest.raises(base.LLMEmptyResponse) as exc_info:
        await provider.summarize_email(_email())
    assert isinstance(exc_info.value, base.LLMInvalidResponse)
    assert calls == 3


async def test_draft_reply_builds_draft_from_trusted_thread_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    content = "  Sehr geehrte Frau Ana,\n\nDanke.\n\nMit freundlichen Grüßen  "
    _fake_create(provider, monkeypatch, content=content)
    request = DraftRequest(
        thread_id="t1",
        user_instructions="Danke sagen",
        language="de",
        reply_to=EmailAddress("Ana", "ana@example.com"),
        in_reply_to="m1",
    )
    reply = await provider.draft_reply(request, _thread())
    assert reply.body == (
        "Sehr geehrte Frau Ana,\n\nDanke.\n\nMit freundlichen Grüßen\n\nDaniel"
    )
    assert reply.thread_id == "t1"
    assert reply.subject == "Re: Proyecto"
    # Recipient/in_reply_to come from the TRUSTED request, never thread
    # heuristics (thread.messages[0] may be our own sent message).
    assert reply.to == [EmailAddress("Ana", "ana@example.com")]
    assert reply.cc == []
    assert reply.in_reply_to == "m1"


async def test_draft_reply_empty_thread_has_no_recipients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    _fake_create(provider, monkeypatch, content="Danke")
    empty_thread = ThreadContext(thread_id="t1", subject="Re: X", history_id=3, messages=[])
    request = DraftRequest(thread_id="t1", user_instructions="", language="de")
    reply = await provider.draft_reply(request, empty_thread)
    assert reply.to == []
    assert reply.body == "Danke"


def test_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        OpenAICompatLLM(Settings())


# ── incomplete/truncated completion rejection (require_complete) ─────────────


async def test_complete_require_complete_rejects_finish_reason_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    _fake_create_with_finish(
        provider, monkeypatch, [("Sehr geehrte Frau Muster,\n\nvielen Dank für die", "length")]
    )
    with pytest.raises(base.LLMIncompleteResponse):
        await provider.complete(
            [{"role": "user", "content": "x"}], max_tokens=100, require_complete=True
        )


async def test_complete_require_complete_rejects_dangling_ending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    # finish_reason says "stop" but the text is obviously cut mid-clause.
    _fake_create_with_finish(
        provider, monkeypatch, [("Sehr geehrter Herr Arroyo,\n\nvielen Dank für die", "stop")]
    )
    with pytest.raises(base.LLMIncompleteResponse):
        await provider.complete(
            [{"role": "user", "content": "x"}], max_tokens=100, require_complete=True
        )


async def test_complete_require_complete_accepts_short_complete_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    _fake_create_with_finish(provider, monkeypatch, [("Danke, bis morgen.", "stop")])
    result = await provider.complete(
        [{"role": "user", "content": "x"}], max_tokens=100, require_complete=True
    )
    assert result == "Danke, bis morgen."


async def test_complete_without_require_complete_keeps_tolerant_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Incoming summaries must NOT fail on truncation (require_complete=False).
    provider = _provider(monkeypatch)
    _fake_create_with_finish(provider, monkeypatch, [("cuerpo incompleto y", "length")])
    result = await provider.complete([{"role": "user", "content": "x"}], max_tokens=100)
    assert result == "cuerpo incompleto y"


# ── structured contract completeness (thread summary / Q&A) ──────────────────


_TRUNCATED_SUMMARY_JSON = (
    '{"headline": "Resumen", "sections": [{"emoji": "📬", "title": "Resumen", "items": ['
)


async def test_complete_require_complete_rejects_truncated_structured_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """finish_reason=length with a truncated structured contract is rejected
    by require_complete (the thread-summary/Q&A paths must request it)."""
    provider = _provider(monkeypatch)
    _fake_create_with_finish(provider, monkeypatch, [(_TRUNCATED_SUMMARY_JSON, "length")])
    with pytest.raises(base.LLMIncompleteResponse):
        await provider.complete(
            [{"role": "user", "content": "x"}], max_tokens=100, require_complete=True
        )


async def test_complete_require_complete_does_not_catch_stop_with_broken_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider metadata alone is NOT enough: finish_reason=stop with
    syntactically broken JSON passes require_complete — the structured parse
    layer must catch it (call_structured → StructuredOutputError)."""
    provider = _provider(monkeypatch)
    _fake_create_with_finish(
        provider, monkeypatch, [(_TRUNCATED_SUMMARY_JSON, "stop")] * 2
    )
    result = await provider.complete(
        [{"role": "user", "content": "x"}], max_tokens=100, require_complete=True
    )
    assert result == _TRUNCATED_SUMMARY_JSON
    # And the structured layer refuses it.
    from inboxbridge.llm.qa import call_structured, parse_thread_summary

    with pytest.raises(base.StructuredOutputError):
        await call_structured(
            lambda: provider.complete(
                [{"role": "user", "content": "x"}], max_tokens=100, require_complete=True
            ),
            parse_thread_summary,
            max_attempts=1,
            base_backoff=0,
        )


async def test_complete_require_complete_accepts_valid_structured_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    valid = (
        '{"headline": "Resumen", "sections": [{"emoji": "📬", "title": '
        '"Resumen", "items": ["a"]}]}'
    )
    _fake_create_with_finish(provider, monkeypatch, [(valid, "stop")])
    result = await provider.complete(
        [{"role": "user", "content": "x"}], max_tokens=100, require_complete=True
    )
    assert result == valid


async def test_draft_reply_retries_incomplete_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatLLM(_settings(monkeypatch, LLM_MAX_RETRIES=2))
    captured = _fake_create_with_finish(
        provider,
        monkeypatch,
        [
            ("Sehr geehrte Frau Muster,\n\nvielen Dank für die", "length"),
            (
                "Sehr geehrte Frau Muster,\n\nvielen Dank für Ihre Nachricht.\n\n"
                "Mit freundlichen Grüßen",
                "stop",
            ),
        ],
    )
    request = DraftRequest(
        thread_id="t1",
        user_instructions="Danke",
        language="de",
        reply_to=EmailAddress("Ana", "ana@example.com"),
    )
    reply = await provider.draft_reply(request, _thread())
    assert "vielen Dank für Ihre Nachricht" in reply.body
    assert reply.to == [EmailAddress("Ana", "ana@example.com")]
    assert len(captured) == 2  # truncated attempt retried once


async def test_draft_reply_all_incomplete_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatLLM(_settings(monkeypatch, LLM_MAX_RETRIES=2))
    captured = _fake_create_with_finish(
        provider, monkeypatch, [("...die", "length"), ("...die", "length")]
    )
    request = DraftRequest(
        thread_id="t1",
        user_instructions="Danke",
        language="de",
        reply_to=EmailAddress("Ana", "ana@example.com"),
    )
    with pytest.raises(base.LLMIncompleteResponse):
        await provider.draft_reply(request, _thread())
    assert len(captured) == 2  # bounded retries exhausted


async def test_draft_reply_signature_from_config_not_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The signature is trusted config, never invented from thread/recipient."""
    provider = _provider(monkeypatch)
    _fake_create(
        provider,
        monkeypatch,
        content="Sehr geehrte Frau Ana,\n\nDanke.\n\nMit freundlichen Grüßen",
    )
    request = DraftRequest(thread_id="t1", user_instructions="Danke", language="de")
    reply = await provider.draft_reply(request, _thread())
    lines = [ln for ln in reply.body.splitlines() if ln.strip()]
    assert lines[-1] == "Daniel"  # config signature, not "Ana"
    assert reply.body.endswith("Mit freundlichen Grüßen\n\nDaniel")


async def test_draft_reply_signature_change_affects_new_drafts_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing EMAIL_SIGNATURE_NAME affects NEW drafts; existing drafts keep
    their frozen body."""
    provider_a = _provider(monkeypatch)
    _fake_create(
        provider_a,
        monkeypatch,
        content="Sehr geehrte Frau Muster,\n\nDanke.\n\nMit freundlichen Grüßen",
    )
    request = DraftRequest(thread_id="t1", user_instructions="Danke", language="de")
    first = await provider_a.draft_reply(request, _thread())
    assert first.body.endswith("Grüßen\n\nDaniel")

    provider_b = OpenAICompatLLM(_settings(monkeypatch, EMAIL_SIGNATURE_NAME="Otro Nombre"))
    _fake_create(
        provider_b,
        monkeypatch,
        content="Sehr geehrte Frau Muster,\n\nDanke.\n\nMit freundlichen Grüßen",
    )
    second = await provider_b.draft_reply(request, _thread())
    assert second.body.endswith("Grüßen\n\nOtro Nombre")
    assert first.body.endswith("Grüßen\n\nDaniel")  # frozen draft untouched


# ── text model alternation (AI_TEXT_FALLBACK_MODEL) ─────────────────────────


def test_alternate_models_disabled_when_unset() -> None:
    models = base.alternate_text_models("primary", None, task="t")
    assert [models() for _ in range(3)] == [None, None, None]


def test_alternate_models_disabled_when_same_as_primary() -> None:
    models = base.alternate_text_models("primary", "primary", task="t")
    assert [models() for _ in range(3)] == [None, None, None]


def test_alternate_models_primary_then_fallback_then_primary() -> None:
    models = base.alternate_text_models("primary", "fb-model", task="t")
    assert [models() for _ in range(3)] == [None, "fb-model", None]


async def test_summarize_email_uses_fallback_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatLLM(
        _settings(monkeypatch, LLM_MAX_RETRIES="2", AI_TEXT_FALLBACK_MODEL="fb-model")
    )
    captured: list[dict[str, object]] = []
    responses = iter(
        [
            _error(openai.APIConnectionError),  # → LLMUnavailable (retryable)
            '{"subject_es": "Plan", "summary_es": "Resumen breve."}',
        ]
    )

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        captured.append(kwargs)
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        message = SimpleNamespace(content=item)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    result = await provider.summarize_email(_email())
    assert result.summary_es == "Resumen breve."
    # Attempt 1 on the primary model, attempt 2 on the configured fallback.
    assert [c["model"] for c in captured] == ["test-model", "fb-model"]


async def test_summarize_email_no_fallback_on_permanent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatLLM(
        _settings(monkeypatch, LLM_MAX_RETRIES="2", AI_TEXT_FALLBACK_MODEL="fb-model")
    )
    captured: list[dict[str, object]] = []

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        captured.append(kwargs)
        raise _error(openai.AuthenticationError)  # → permanent LLMError

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    with pytest.raises(base.LLMError):
        await provider.summarize_email(_email())
    assert len(captured) == 1  # fail closed: no fallback burn on auth/policy


async def test_draft_reply_primary_incomplete_fallback_valid_one_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatLLM(
        _settings(monkeypatch, LLM_MAX_RETRIES="2", AI_TEXT_FALLBACK_MODEL="fb-model")
    )
    captured: list[dict[str, object]] = []
    responses = iter(
        [
            # Primary: truncated (finish_reason=length) → LLMIncompleteResponse.
            ("Sehr geehrte Frau Muster,\n\nvielen Dank für die", "length"),
            # Fallback: valid complete body.
            (
                "Sehr geehrte Frau Muster,\n\nvielen Dank für Ihre Nachricht.\n\n"
                "Mit freundlichen Grüßen\nDaniel",
                "stop",
            ),
        ]
    )

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        captured.append(kwargs)
        content, finish_reason = next(responses)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    draft = await provider.draft_reply(
        DraftRequest(
            thread_id="t1",
            user_instructions="sag einfach danke",
            memory=(),
            reply_to=EmailAddress("Ana", "ana@example.com"),
        ),
        _thread(),
    )
    assert [c["model"] for c in captured] == ["test-model", "fb-model"]
    assert draft.body.endswith("Daniel")  # trusted signature normalized
    # Recipient comes from the trusted request, never from thread heuristics.
    assert draft.to == [EmailAddress("Ana", "ana@example.com")]


# ── completion telemetry + task-aware budgets ───────────────────────────────


def _fake_create_with_usage(
    provider: OpenAICompatLLM,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    *,
    finish_reason: str = "stop",
    completion_tokens: int | None = 118,
    reasoning_tokens: int | None = 114,
    reasoning_text: str = "",
) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        captured.append(kwargs)
        message = SimpleNamespace(content=content, reasoning=reasoning_text or None)
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        details = None
        if reasoning_tokens is not None:
            details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
        usage = SimpleNamespace(
            completion_tokens=completion_tokens,
            completion_tokens_details=details,
        )
        return SimpleNamespace(choices=[choice], usage=usage)

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    return captured


async def test_completion_meta_captures_safe_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    _fake_create_with_usage(provider, monkeypatch, "Resumen breve.", finish_reason="stop")
    meta = CompletionMeta()
    result = await provider.complete(
        [{"role": "user", "content": "x"}], max_tokens=2000, meta=meta
    )
    assert result == "Resumen breve."
    assert meta.finish_reason == "stop"
    assert meta.completion_tokens == 118
    assert meta.reasoning_tokens == 114
    assert meta.reasoning_available is True
    assert meta.max_tokens == 2000


async def test_completion_meta_missing_reasoning_handled_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    # DeepSeek-like: completion_tokens_details is absent/empty.
    _fake_create_with_usage(
        provider, monkeypatch, "ok", completion_tokens=27, reasoning_tokens=None
    )
    meta = CompletionMeta()
    result = await provider.complete([{"role": "user", "content": "x"}], max_tokens=2000, meta=meta)
    assert result == "ok"
    assert meta.reasoning_available is False
    assert meta.reasoning_tokens == 0
    assert meta.completion_tokens == 27


async def test_completion_meta_missing_usage_tolerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    # Provider omitted usage entirely.
    _fake_create_with_usage(
        provider, monkeypatch, "ok", completion_tokens=None, reasoning_tokens=None
    )
    meta = CompletionMeta()
    result = await provider.complete([{"role": "user", "content": "x"}], max_tokens=100, meta=meta)
    assert result == "ok"
    assert meta.completion_tokens == 0
    assert meta.reasoning_available is False


async def test_reasoning_content_never_read_or_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="inboxbridge.llm.openai_compat")
    provider = _provider(monkeypatch)
    _fake_create_with_usage(
        provider, monkeypatch, "visible ok", reasoning_text="REASONING-SECRET-42"
    )
    result = await provider.complete(
        [{"role": "user", "content": "x"}], max_tokens=2000
    )
    assert result == "visible ok"
    assert not any("REASONING-SECRET-42" in (r.message or "") for r in caplog.records)


async def test_direct_completion_logs_safe_telemetry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="inboxbridge.llm.openai_compat")
    provider = _provider(monkeypatch)
    _fake_create_with_usage(provider, monkeypatch, "ok", finish_reason="stop")
    await provider.complete([{"role": "user", "content": "x"}], max_tokens=2000)
    assert any(
        "llm provider=opencode_go model=test-model finish_reason=stop "
        "completion_tokens=118 reasoning_tokens=114 max_tokens=2000" in r.message
        for r in caplog.records
    )


async def test_larger_budget_does_not_bypass_completeness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    _fake_create_with_usage(
        provider, monkeypatch, "truncado", finish_reason="length"
    )
    with pytest.raises(base.LLMIncompleteResponse):
        await provider.complete(
            [{"role": "user", "content": "x"}], max_tokens=2000, require_complete=True
        )


async def test_summarize_email_uses_summary_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatLLM(
        _settings(monkeypatch, LLM_MAX_TOKENS_SUMMARY="1000")
    )
    captured = _fake_create(
        provider, monkeypatch, '{"subject_es": "Plan", "summary_es": "Resumen breve."}'
    )
    await provider.summarize_email(_email())
    assert captured[0]["max_tokens"] == 1000
