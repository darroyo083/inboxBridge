"""Application configuration — loaded from environment / .env.

Never log these values. This module is the single source of truth for env vars.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "inboxbridge"
    log_level: str = "INFO"
    send_emails: bool = Field(default=False, alias="SEND_EMAILS")

    # Telegram
    telegram_bot_token: SecretStr = Field(default=SecretStr(""), alias="TELEGRAM_BOT_TOKEN")
    telegram_allowed_chat_id: int = Field(default=0, alias="TELEGRAM_ALLOWED_CHAT_ID")

    # LLM (OpenAI-compatible) — text
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_api_key: SecretStr = Field(default=SecretStr(""), alias="LLM_API_KEY")
    llm_model: str = "deepseek-v4-flash"
    llm_max_tokens_summary: int = 700
    llm_max_tokens_draft: int = 1000
    llm_temperature: float = 0.4
    llm_max_retries: int = 3

    # AI routing (V1.1): text vs vision vs audio, configuration-driven.
    # Model IDs come from the environment; business logic never hardcodes them.
    ai_text_model: str = Field(default="", alias="AI_TEXT_MODEL")
    ai_vision_model: str = Field(default="", alias="AI_VISION_MODEL")
    ai_vision_fallback_model: str = Field(default="", alias="AI_VISION_FALLBACK_MODEL")
    ai_audio_enabled: bool = Field(default=False, alias="AI_AUDIO_ENABLED")
    #: Bounded scanned-PDF analysis: max pages rendered and max pixel dimension.
    ai_vision_max_pages: int = 5
    ai_vision_max_dimension: int = 2000
    #: Bounded audio: max seconds and max bytes for experimental voice notes.
    ai_audio_max_seconds: int = 120
    ai_audio_max_bytes: int = 8 * 1024 * 1024

    @property
    def effective_text_model(self) -> str:
        return self.ai_text_model or self.llm_model

    # Gmail / Google
    google_client_secret_file: str = "credentials/client_secret.json"
    google_token_file: str = "credentials/token.json"
    # Service account key used to consume the Pub/Sub subscription (StreamingPull).
    # Empty → falls back to Application Default Credentials.
    google_application_credentials: str = Field(
        default="", alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    gmail_user_id: str = "me"
    google_cloud_project: str = ""
    gmail_pubsub_topic: str = ""
    gmail_pubsub_subscription: str = ""

    # Attachments
    pdf_password: SecretStr = Field(default=SecretStr(""), alias="PDF_PASSWORD")
    attachment_max_bytes: int = 10 * 1024 * 1024
    attachment_max_text_chars: int = 20_000
    attachment_max_count: int = 5

    # Outgoing attachments (Telegram → Gmail reply)
    outgoing_attachment_max_count: int = 5
    outgoing_attachment_max_bytes: int = 10 * 1024 * 1024

    # Temporary working directory (attachment binaries; cleaned on terminal states)
    tmp_dir: str = "data/tmp"

    # Verified delivery / reconciliation
    send_verification_attempts: int = 3
    send_verification_backoff_seconds: float = 2.0
    #: Absolute cap on reconciliation attempts for one draft (across retries/restarts).
    send_verification_max_attempts: int = 12
    #: Periodic sweep interval for drafts stuck in sent_unverified (seconds).
    reconcile_sweep_interval_seconds: float = 300.0
    #: Temp attachment files older than this (seconds) are swept unconditionally.
    tmp_max_age_seconds: int = 24 * 3600
    #: How long a resend offer (and its temp attachments) stays valid after the
    #: draft reaches a terminal state; after this, the files are swept.
    resend_offer_ttl_seconds: int = 3600
    #: Per-request Gmail API timeout (seconds). A hung transport can never
    #: block the reply worker forever; a timed-out SEND is ambiguous by design.
    gmail_request_timeout_seconds: float = 30.0

    # Retries
    retry_backoff_base: float = 2.0
    retry_max_attempts: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
