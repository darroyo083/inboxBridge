# InboxBridge — Architecture

Self-hosted email assistant: connects Gmail (Primary inbox) with a private
Telegram group. Summarizes incoming emails in Spanish via LLM; allows drafting
and sending replies in German from Telegram with explicit confirmation.

## High-level flow

```
Gmail Primary inbox
      │  Gmail Push (users.watch → Pub/Sub)
      ▼
InboxBridge (VPS, Docker Compose)
      │  StreamingPull subscription
      ▼
fetch message via Gmail API (thread-aware)
      │
      ▼
extract text (HTML cleanup) + attachments (PDF / password-PDF / DOCX / TXT)
      │
      ▼
LLM provider (OpenAI-compatible, e.g. OpenCode Go / DeepSeek V4 Flash)
      │
      ▼
natural Spanish summary → private Telegram group (authorized chat_id only)

--- reply flow (from Telegram, replying to bot's message) ---
Telegram reply / mention / command
      ▼
recover Gmail thread context
      ▼
LLM drafts professional German reply
      ▼
show draft + recipients in Telegram → explicit confirmation
      ▼
send as reply in same Gmail thread (threadId, In-Reply-To, References, Subject)
```

## Design principles

- **Simple, modular, typed Python (3.12)**. Avoid unnecessary abstractions.
- **Async throughout** (`asyncio`): streaming pub/sub subscriber, Telegram
  polling (python-telegram-bot v21+), async OpenAI client, aiosqlite.
- **Gmail is the source of truth.** Never process history at first boot; only
  process new messages landing in Primary via historyId delta since watch.
- **Idempotent pipeline.** Deduplication by `message_id` + `history_id`;
  Pub/Sub at-least-once delivery is safe.
- **State minimal.** SQLite only for dedup ids, thread mapping, telegram ids,
  statuses, retry timestamps. No full email bodies, no attachment text, no
  secrets. Volume is tiny and portable to another VPS.
- **Untrusted content.** Emails/attachments are data, never instructions.
  Prompt-injection defenses, HTML/tracker cleanup, no link clicks, no
  attachment execution, no OCR/vision.
- **Kill switch.** `SEND_EMAILS=false` (default) makes sending technically
  impossible in dev/test. Sending only happens with `SEND_EMAILS=true` AND
  explicit Telegram confirmation.
- **No secret leakage.** `.env` ignored by git; `.env.example` has placeholders
  only; structured logs never include secrets or bodies.

## Modules

| Module | Responsibility | Owner |
|---|---|---|
| `config.py` | pydantic-settings, .env, limits, kill switch | core |
| `models.py` | Shared contracts (dataclasses/pydantic) | core |
| `db.py` | SQLite state, dedup, retry bookkeeping | core |
| `gmail/auth.py` | OAuth 2.0 local flow, token store, per-install credentials | worker A |
| `gmail/client.py` | Gmail API: get message/thread, send reply, mark read | worker A |
| `gmail/watcher.py` | `users.watch` lifecycle, auto-renew before expiry, historyId | worker A |
| `gmail/pubsub.py` | StreamingPull subscription (no public webhook) | worker A |
| `gmail/parse.py` | HTML cleanup, tracker/signature removal, plain text | worker A |
| `gmail/attachments.py` | PDF (+password), DOCX, TXT local extraction, limits | worker A |
| `llm/base.py` | Provider abstraction | worker B |
| `llm/openai_compat.py` | OpenAI-compatible client (OpenCode Go / DeepSeek / OpenRouter) | worker B |
| `llm/prompts.py` | ES summary + DE reply prompts, injection defense | worker B |
| `telegram/bot.py` | Bot API, group auth, typing/streaming, clean UX | worker B |
| `pipeline.py` | Orchestrates: pubsub event → fetch → parse → LLM → telegram | core (integration) |
| `responder.py` | Telegram request → thread context → draft → confirm → send | core (integration) |
| `status.py` | `/status`: checks Gmail, Pub/Sub, Telegram, LLM w/o secrets | core |
| `app.py` | Entrypoint, asyncio tasks, graceful shutdown, retries | core |

## Persistence (SQLite)

Schema (owned by core `db.py`):

- `messages`: `message_id`, `thread_id`, `history_id`, `telegram_message_id`,
  `status` (received|summarizing|sent_telegram|draft_created|draft_sent|failed),
  `created_at`, `updated_at`, `retry_count`, `next_retry_at`.
- `drafts`: `id`, `thread_id`, `message_id`, `body`, `to`, `subject`,
  `status` (pending|confirmed|sent), `created_at`.
- `meta`: `key`/`value` — `last_history_id`, `watch_expires_at`, `oauth_*` hints.

No email bodies, no attachment contents, no secrets stored.

## LLM provider abstraction

```python
class LLMProvider(Protocol):
    async def summarize_email(self, ctx: SummarizeContext) -> str: ...
    async def draft_reply(self, ctx: DraftContext) -> str: ...
```

OpenAI-compatible via the `openai` SDK (`base_url` configurable). No automatic
paid fallback between providers. On failure, email stays pending for retry
(with backoff); it is never silently lost.

## Operations

- Docker Compose (single container + tiny SQLite volume).
- Healthcheck endpoint/command; graceful shutdown (SIGTERM → stop subscribers,
  finish in-flight, close db).
- Retries with exponential backoff + jitter; idempotent processing.
- `SEND_EMAILS=false` default (dev/test).
- Structured JSON logs, no PII/secrets/bodies.

## Repository layout

```
src/inboxbridge/
  config.py models.py db.py pipeline.py responder.py status.py app.py
  gmail/    (auth, client, watcher, pubsub, parse, attachments)
  llm/      (base, openai_compat, prompts)
  telegram/ (bot)
tests/
  unit/  integration/  mocks/
docs/ARCHITECTURE.md
.env.example  .gitignore  pyproject.toml  Dockerfile  docker-compose.yml
.github/workflows/ci.yml
```

## MVP acceptance (17 criteria)

See README.md "MVP verification" — each maps to a test or an operational check.
