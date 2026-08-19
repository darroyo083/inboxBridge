"""AI routing facade: model selection, bounded vision fallback, observability.

Provider calls are faked at the OpenAICompatLLM boundary; real-provider
validation lives in the opt-in ``test_llm_real.py`` suite.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from inboxbridge.llm.ai_service import AIService
from inboxbridge.llm.base import LLMEmptyResponse, LLMUnavailable, LLMUnsupportedModality


def make_settings(**overrides: object) -> Any:
    from inboxbridge.config import Settings

    base: dict[str, object] = {
        "_env_file": None,
        "LLM_API_KEY": "test-key",
        "LLM_BASE_URL": "https://api.test/v1",
        "AI_TEXT_MODEL": "deepseek-v4-flash",
        "AI_VISION_MODEL": "mimo-v2.5",
        "AI_VISION_FALLBACK_MODEL": "gpt-5.6-luna",
        "AI_AUDIO_ENABLED": False,
        "llm_max_retries": 1,
    }
    base.update(overrides)
    return Settings(**base)


class FakeClient:
    """Replaces OpenAICompatLLM instances; records model used per call."""

    def __init__(self, results: list[Any] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[tuple[str, Any]] = []  # (method, model)
        self.model: str = ""
        self.max_tokens_used: list[int] = []
        self.reasoning_available = True  # simulate MiMo-style usage details

    async def complete(
        self,
        messages: list[Any],
        *,
        max_tokens: int,
        require_complete: bool = False,
        meta: Any = None,
    ) -> str:
        self.calls.append(("complete", self.model))
        self.max_tokens_used.append(max_tokens)
        if meta is not None:
            # Mimic OpenAICompatLLM.complete filling safe response telemetry.
            meta.finish_reason = "stop"
            meta.completion_tokens = 118
            meta.max_tokens = max_tokens
            if self.reasoning_available:
                meta.reasoning_tokens = 114
                meta.reasoning_available = True
        return self._next()

    async def complete_vision(
        self, prompt: str, images: list[tuple[str, bytes]], *, max_tokens: int
    ) -> str:
        self.calls.append(("complete_vision", self.model))
        return self._next()

    async def transcribe_audio(self, mime: str, data: bytes) -> str:
        self.calls.append(("transcribe_audio", self.model))
        return self._next()

    def _next(self) -> str:
        if not self.results:
            return "ok"
        item = self.results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self) -> None:
        return None


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestTextRouting:
    def test_text_uses_configured_text_model(self) -> None:
        service = AIService(make_settings())
        text_fake = FakeClient(["hola"])
        service._text_llm = text_fake  # type: ignore[assignment]
        result = run(service.text([{"role": "user", "content": "hi"}], max_tokens=10))
        assert result == "hola"
        assert text_fake.calls == [("complete", "")]

    def test_text_model_falls_back_to_llm_model_when_unset(self) -> None:
        settings = make_settings(AI_TEXT_MODEL="")
        assert settings.effective_text_model == "deepseek-v4-flash"


class TestVisionRouting:
    def test_vision_uses_primary_model_and_succeeds(self) -> None:
        service = AIService(make_settings())
        vision = FakeClient(["veo un gato"])
        fallback = FakeClient()
        service._vision_llm = vision  # type: ignore[assignment]
        service._vision_fallback_llm = fallback  # type: ignore[assignment]
        result = run(service.vision("¿qué ves?", [("image/png", b"png-bytes")]))
        assert result == "veo un gato"
        assert len(vision.calls) == 1
        assert fallback.calls == []  # no fallback on success
        call = service.calls[0]
        assert call.task == "vision"
        assert call.model == "mimo-v2.5"
        assert call.success and not call.fallback_used

    def test_vision_technical_failure_falls_back_to_luna_once(self) -> None:
        service = AIService(make_settings())
        vision = FakeClient([LLMUnavailable("down")])
        fallback = FakeClient(["leído desde luna"])
        service._vision_llm = vision  # type: ignore[assignment]
        service._vision_fallback_llm = fallback  # type: ignore[assignment]
        result = run(service.vision("¿qué ves?", [("image/png", b"x")]))
        assert result == "leído desde luna"
        assert len(vision.calls) == 1 and len(fallback.calls) == 1
        call = service.calls[0]
        assert call.success and call.fallback_used
        assert call.fallback_model == "gpt-5.6-luna"

    def test_vision_both_fail_raises(self) -> None:
        service = AIService(make_settings())
        service._vision_llm = FakeClient([LLMUnavailable("down")])  # type: ignore[assignment]
        service._vision_fallback_llm = FakeClient([LLMUnavailable("down too")])  # type: ignore[assignment]
        with pytest.raises(LLMUnavailable):
            run(service.vision("¿qué ves?", [("image/png", b"x")]))
        call = service.calls[0]
        assert not call.success and call.fallback_used

    def test_unsupported_modality_falls_back(self) -> None:
        service = AIService(make_settings())
        service._vision_llm = FakeClient([LLMUnsupportedModality("no images")])  # type: ignore[assignment]
        fallback = FakeClient(["ok"])
        service._vision_fallback_llm = fallback  # type: ignore[assignment]
        result = run(service.vision("¿qué ves?", [("image/png", b"x")]))
        assert result == "ok"
        assert fallback.calls

    def test_content_quality_failure_does_not_fall_back(self) -> None:
        from inboxbridge.llm.base import LLMError

        service = AIService(make_settings())
        vision = FakeClient([LLMError("bad request / policy")])
        fallback = FakeClient()
        service._vision_llm = vision  # type: ignore[assignment]
        service._vision_fallback_llm = fallback  # type: ignore[assignment]
        with pytest.raises(LLMError):
            run(service.vision("¿qué ves?", [("image/png", b"x")]))
        assert fallback.calls == []  # "didn't like the answer" is NOT a fallback

    def test_fallback_disabled_raises_directly(self) -> None:
        service = AIService(make_settings())
        service._vision_llm = FakeClient([LLMUnavailable("down")])  # type: ignore[assignment]
        with pytest.raises(LLMUnavailable):
            run(service.vision("¿qué ves?", [("image/png", b"x")], allow_fallback=False))


class TestAudioGating:
    def test_audio_disabled_raises(self) -> None:
        from inboxbridge.llm.base import LLMError

        service = AIService(make_settings(AI_AUDIO_ENABLED=False))
        with pytest.raises(LLMError):
            run(service.audio("audio/ogg", b"data"))

    def test_audio_enabled_routes_to_text_client(self) -> None:
        service = AIService(make_settings(AI_AUDIO_ENABLED=True))
        fake = FakeClient(["transcripción"])
        service._text_llm = fake  # type: ignore[assignment]
        result = run(service.audio("audio/ogg", b"data"))
        assert result == "transcripción"
        assert fake.calls == [("transcribe_audio", "")]


# ── text fallback model (AI_TEXT_FALLBACK_MODEL) ────────────────────────────


class TestTextFallbackConfig:
    def test_fallback_empty_disabled(self) -> None:
        settings = make_settings()
        assert settings.effective_text_fallback_model == ""

    def test_fallback_configured(self) -> None:
        settings = make_settings(AI_TEXT_FALLBACK_MODEL="fb-model")
        assert settings.effective_text_fallback_model == "fb-model"

    def test_fallback_same_as_primary_disabled(self) -> None:
        settings = make_settings(AI_TEXT_FALLBACK_MODEL="deepseek-v4-flash")
        assert settings.effective_text_fallback_model == ""

    def test_lazy_fallback_client_uses_same_provider_config(self) -> None:
        service = AIService(
            make_settings(AI_TEXT_FALLBACK_MODEL="fb-model")
        )
        client = service._text_client("fb-model")
        assert client._model == "fb-model"
        assert str(client._client.base_url) == "https://api.test/v1/"
        assert client._client.api_key == "test-key"
        # Primary client is untouched by the fallback request.
        assert service._text_llm is None


class TestTextWithModelOverride:
    def test_primary_call_has_no_fallback_markers(self) -> None:
        service = AIService(make_settings(AI_TEXT_FALLBACK_MODEL="fb-model"))
        text_fake = FakeClient(["hola"])
        service._text_llm = text_fake  # type: ignore[assignment]
        result = run(service.text([{"role": "user", "content": "hi"}], max_tokens=10))
        assert result == "hola"
        record = service.calls[-1]
        assert record.model == "deepseek-v4-flash"
        assert record.fallback_used is False
        assert record.fallback_model == ""

    def test_fallback_model_call_marks_fallback_used(self) -> None:
        service = AIService(make_settings(AI_TEXT_FALLBACK_MODEL="fb-model"))
        fallback_fake = FakeClient(["hola fb"])
        service._text_fallback_llm = fallback_fake  # type: ignore[assignment]
        result = run(
            service.text(
                [{"role": "user", "content": "hi"}],
                max_tokens=10,
                model="fb-model",
            )
        )
        assert result == "hola fb"
        record = service.calls[-1]
        assert record.model == "fb-model"
        assert record.fallback_used is True
        assert record.fallback_model == "fb-model"

    def test_fallback_failure_records_and_logs_safely(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="inboxbridge.llm.ai_service")
        service = AIService(make_settings(AI_TEXT_FALLBACK_MODEL="fb-model"))
        fallback_fake = FakeClient([LLMUnavailable("down")])
        service._text_fallback_llm = fallback_fake  # type: ignore[assignment]
        with pytest.raises(LLMUnavailable):
            run(
                service.text(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=10,
                    task="qa",
                    model="fb-model",
                )
            )
        record = service.calls[-1]
        assert record.success is False
        assert record.model == "fb-model"
        assert any(
            "text_fallback task=qa model=fb-model outcome=failed error=LLMUnavailable"
            in r.message
            for r in caplog.records
        )
        # Privacy-safe: no prompt text ever logged.
        assert not any("hi" in (r.message or "") for r in caplog.records)

    def test_fallback_success_logs_outcome(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="inboxbridge.llm.ai_service")
        service = AIService(make_settings(AI_TEXT_FALLBACK_MODEL="fb-model"))
        fallback_fake = FakeClient(["ok"])
        service._text_fallback_llm = fallback_fake  # type: ignore[assignment]
        run(service.text([{"role": "user", "content": "hi"}], task="compose", model="fb-model"))
        assert any(
            "text_fallback task=compose model=fb-model outcome=success fallback_used=true"
            in r.message
            for r in caplog.records
        )

    def test_primary_transient_failure_does_not_auto_fallback_inside_text(self) -> None:
        """AIService.text is a SINGLE transport call: model diversity is
        orchestrated by the bounded alternation at the retry layer, never
        silently here (no retry multiplication)."""
        service = AIService(make_settings(AI_TEXT_FALLBACK_MODEL="fb-model"))
        text_fake = FakeClient([LLMEmptyResponse("empty")])
        service._text_llm = text_fake  # type: ignore[assignment]
        with pytest.raises(LLMEmptyResponse):
            run(service.text([{"role": "user", "content": "hi"}], max_tokens=10))
        assert len(service.calls) == 1  # exactly one provider call


# ── completion telemetry (safe counts only) ─────────────────────────────────


class TestCompletionTelemetry:
    def test_success_record_carries_safe_telemetry(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="inboxbridge.llm.ai_service")
        service = AIService(make_settings())
        text_fake = FakeClient(["hola"])
        service._text_llm = text_fake  # type: ignore[assignment]
        result = run(service.text([{"role": "user", "content": "hi"}], max_tokens=1600))
        assert result == "hola"
        record = service.calls[-1]
        assert record.finish_reason == "stop"
        assert record.completion_tokens == 118
        assert record.reasoning_tokens == 114
        assert record.reasoning_available is True
        assert record.max_tokens == 1600
        # The log line carries the safe counts, never prompt content.
        assert any(
            "finish_reason=stop completion_tokens=118 reasoning_tokens=114 "
            "max_tokens=1600" in r.message
            for r in caplog.records
        )
        assert not any("hi" in (r.message or "") for r in caplog.records)

    def test_missing_reasoning_tokens_handled_safely(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="inboxbridge.llm.ai_service")
        service = AIService(make_settings())
        text_fake = FakeClient(["hola"])
        text_fake.reasoning_available = False  # DeepSeek-like: no details
        service._text_llm = text_fake  # type: ignore[assignment]
        run(service.text([{"role": "user", "content": "hi"}], max_tokens=1600))
        record = service.calls[-1]
        assert record.reasoning_available is False
        assert record.reasoning_tokens == 0
        assert any(
            "reasoning_tokens=unavailable" in r.message for r in caplog.records
        )

    def test_fallback_model_telemetry_still_correct(self) -> None:
        service = AIService(make_settings(AI_TEXT_FALLBACK_MODEL="fb-model"))
        fallback_fake = FakeClient(["ok"])
        service._text_fallback_llm = fallback_fake  # type: ignore[assignment]
        run(service.text([{"role": "user", "content": "hi"}], task="qa", model="fb-model"))
        record = service.calls[-1]
        assert record.fallback_used is True
        assert record.model == "fb-model"
        assert record.finish_reason == "stop"
        assert record.completion_tokens == 118


def test_task_budgets_are_distinct_and_bounded() -> None:
    settings = make_settings()
    budgets = {
        settings.llm_max_tokens_intent,
        settings.llm_max_tokens_summary,
        settings.llm_max_tokens_translation,
        settings.llm_max_tokens_thread_summary_plain,
        settings.llm_max_tokens_qa,
        settings.llm_max_tokens_draft,
        settings.llm_max_tokens_thread_summary,
    }
    # No single giant global value: each task class has its own ceiling
    # (draft and Q&A intentionally share the drafts ceiling).
    assert len(budgets) == 6
    assert settings.llm_max_tokens_intent == 400  # intent stays deliberately small
    assert settings.llm_max_tokens_thread_summary == 2000
    assert settings.llm_max_tokens_thread_summary_plain == 1500
    assert settings.llm_max_tokens_qa == 1600
    assert settings.llm_max_tokens_draft == 1600
    assert settings.llm_max_tokens_translation == 1200
    assert settings.llm_max_tokens_summary == 1000


def test_translate_uses_translation_budget() -> None:
    service = AIService(make_settings())
    text_fake = FakeClient(["traducción"])
    service._text_llm = text_fake  # type: ignore[assignment]
    result = run(service.translate_to_spanish("Deutscher Text"))
    assert result == "traducción"
    assert text_fake.max_tokens_used == [1200]  # LLM_MAX_TOKENS_TRANSLATION


def test_text_default_budget_is_draft_ceiling() -> None:
    """A generic text() call without an explicit task budget still uses a
    bounded ceiling (draft), never an unbounded default."""
    service = AIService(make_settings())
    text_fake = FakeClient(["x"])
    service._text_llm = text_fake  # type: ignore[assignment]
    run(service.text([{"role": "user", "content": "hi"}], task="compose", max_tokens=1600))
    assert text_fake.max_tokens_used == [1600]
