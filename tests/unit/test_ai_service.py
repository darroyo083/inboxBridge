"""AI routing facade: model selection, bounded vision fallback, observability.

Provider calls are faked at the OpenAICompatLLM boundary; real-provider
validation lives in the opt-in ``test_llm_real.py`` suite.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from inboxbridge.llm.ai_service import AIService
from inboxbridge.llm.base import LLMUnavailable, LLMUnsupportedModality


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

    async def complete(
        self, messages: list[Any], *, max_tokens: int, require_complete: bool = False
    ) -> str:
        self.calls.append(("complete", self.model))
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
