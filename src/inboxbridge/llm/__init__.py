"""LLM provider abstraction and prompt crafting."""

from .base import LLMError, LLMInvalidResponse, LLMRateLimited, LLMUnavailable
from .openai_compat import OpenAICompatLLM

__all__ = [
    "LLMError",
    "LLMInvalidResponse",
    "LLMRateLimited",
    "LLMUnavailable",
    "OpenAICompatLLM",
]
