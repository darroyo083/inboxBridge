"""LLM tests: retry wrapper behavior and OpenAI-compatible error mapping.

No network: the OpenAI client is monkeypatched; ``asyncio.sleep`` is stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import httpx
import openai
import pytest

from inboxbridge.config import Settings
from inboxbridge.llm import OpenAICompatLLM, base, prompts
from inboxbridge.models import DraftRequest, EmailAddress, ParsedEmail, ThreadContext, ThreadMessage


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Settings:
    values: dict[str, str] = {
        "LLM_API_KEY": "test-key",
        "LLM_BASE_URL": "https://api.test/v1",
        "LLM_MODEL": "test-model",
        "LLM_MAX_RETRIES": "1",
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


async def test_summarize_email_sends_prompt_and_strips_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    captured = _fake_create(provider, monkeypatch, content="  Resumen breve.  ")
    result = await provider.summarize_email(_email())
    assert result == "Resumen breve."
    request = cast(dict[str, Any], captured[0])
    assert request["model"] == "test-model"
    assert request["max_tokens"] == 700
    assert request["temperature"] == 0.4
    assert request["messages"][0]["role"] == "system"
    assert prompts.UNTRUSTED_DATA_START in request["messages"][1]["content"]


async def test_summarize_email_empty_content_raises_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    _fake_create(provider, monkeypatch, content=None)
    with pytest.raises(base.LLMInvalidResponse):
        await provider.summarize_email(_email())


async def test_draft_reply_builds_draft_from_trusted_thread_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    content = "  Sehr geehrte Frau Ana,\n\nDanke.\n\nMit freundlichen Grüßen  "
    _fake_create(provider, monkeypatch, content=content)
    request = DraftRequest(thread_id="t1", user_instructions="Danke sagen", language="de")
    reply = await provider.draft_reply(request, _thread())
    assert reply.body == "Sehr geehrte Frau Ana,\n\nDanke.\n\nMit freundlichen Grüßen"
    assert reply.thread_id == "t1"
    assert reply.subject == "Re: Proyecto"
    assert reply.to == [EmailAddress("Ana", "ana@example.com")]
    assert reply.cc == []
    assert reply.in_reply_to == "m2"


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
