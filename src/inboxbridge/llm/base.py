"""LLM provider exceptions and retry machinery.

Transient failures (rate limits, outages) are retried with exponential backoff
and full jitter; permanent failures propagate immediately. After the retry
budget is exhausted the last exception is re-raised — work is never silently
lost.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")
_R = TypeVar("_R", bound=Callable[..., Awaitable[Any]])


class LLMError(Exception):
    """Base class for LLM provider failures (permanent unless subclassed)."""


class LLMRateLimited(LLMError):
    """Provider rejected the request due to rate limiting; retryable."""


class LLMUnavailable(LLMError):
    """Provider unreachable or failed transiently (connection, timeout, 5xx)."""


class LLMInvalidResponse(LLMError):
    """Provider answered but the output is unusable (empty, refusal)."""


class LLMEmptyResponse(LLMInvalidResponse):
    """Provider answered with EMPTY content; transient — retried like outages.

    Distinct from :class:`LLMInvalidResponse` so genuinely malformed output
    (refusals, garbage) stays permanent while empty responses — often a
    transient provider glitch — get immediate retries.
    """


def _default_retryable(exc: BaseException) -> bool:
    return isinstance(exc, LLMRateLimited | LLMUnavailable | LLMEmptyResponse)


def _backoff_delay(attempt: int, base_backoff: float, max_backoff: float) -> float:
    """Full-jitter exponential backoff: uniform(0, min(max, base * 2**attempt))."""
    cap = min(max_backoff, base_backoff * (2.0**attempt))
    return random.uniform(0.0, cap)


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_backoff: float = 2.0,
    max_backoff: float = 120.0,
    retryable: Callable[[BaseException], bool] | None = None,
) -> T:
    """Await ``fn()``, retrying transient failures with backoff and jitter.

    ``LLMRateLimited``, ``LLMUnavailable`` and ``LLMEmptyResponse`` are retried
    by default; other exceptions propagate immediately. The last exception is
    re-raised when ``max_attempts`` is exhausted.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    predicate = retryable or _default_retryable
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if not predicate(exc):
                raise
            if attempt + 1 < max_attempts:
                delay = _backoff_delay(attempt, base_backoff, max_backoff)
                logger.warning(
                    "LLM attempt %d/%d failed with %s; retrying in %.1fs",
                    attempt + 1,
                    max_attempts,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def retry(
    *,
    max_attempts: int,
    base_backoff: float = 2.0,
    max_backoff: float = 120.0,
    retryable: Callable[[BaseException], bool] | None = None,
) -> Callable[[_R], _R]:
    """Decorator form of :func:`call_with_retry` for async callables."""

    def decorator(fn: _R) -> _R:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await call_with_retry(
                lambda: fn(*args, **kwargs),
                max_attempts=max_attempts,
                base_backoff=base_backoff,
                max_backoff=max_backoff,
                retryable=retryable,
            )

        return cast(_R, wrapper)

    return decorator
