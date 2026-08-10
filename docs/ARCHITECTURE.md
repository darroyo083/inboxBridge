# InboxBridge — Architecture

Self-hosted email assistant: connects Gmail (Primary inbox) with a private
Telegram group. Summarizes incoming emails in Spanish via LLM; allows drafting
and sending replies in German from Telegram with explicit confirmation,
verified delivery against Gmail, and outgoing attachments from Telegram.

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
Telegram reply / mention / command (+ optional document/photo)
      ▼
recover Gmail thread context
      ▼
LLM drafts professional German reply
      ▼
show draft + recipients + attachments in Telegram → explicit confirmation
      ▼
send as reply in same Gmail thread (threadId, In-Reply-To, References, Subject)
      ▼
RECONCILE against Gmail (verify_delivery)
      ▼
sent_verified → "Enviado y verificado ✓" (only after Gmail evidence)
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
  attachment binaries, no secrets. Volume is tiny and portable to another VPS.
- **Untrusted content.** Emails/attachments are data, never instructions.
  Prompt-injection defenses, HTML/tracker cleanup, no link clicks, no
  attachment execution, no OCR/vision.
- **Kill switch.** `SEND_EMAILS=false` (default) makes sending technically
  impossible in dev/test. Sending only happens with `SEND_EMAILS=true` AND
  explicit Telegram confirmation.
- **No secret leakage.** `.env` ignored by git; `.env.example` has placeholders
  only; structured logs never include secrets or bodies.

## Send state machine (verified delivery)

A confirmed draft moves through:

| State | Meaning |
|---|---|
| `pending` | draft shown for confirmation (persisted at presentation time) |
| `confirmed` | member pressed Enviar; send will be attempted |
| `sending` | Gmail send request in flight |
| `sent_unverified` | outcome unknown (timeout/5xx) OR verification still pending |
| `sent_verified` | Gmail evidence: message in the expected thread, correct recipients/subject/attachments |
| `send_failed` | definitive failure; safe to report and offer controlled retry |
| `cancelled` | rejected/timed-out draft; temp attachments cleaned |

Rules:

- Success is NEVER reported from an HTTP return alone. Every send is
  reconciled via `GmailClient.verify_delivery` (message id lookup, or thread
  search filtered by the account's own From header + `since_ms` time bound)
  and checked for thread/recipients/subject/attachment match.
- Ambiguous outcome (transport error, 5xx after transmission): the draft goes
  to `sent_unverified` and is reconciled with bounded attempts. Found →
  `sent_verified` with NO second send; not found (Gmail queried OK) → a
  CONTROLLED retry is offered via a "Reintentar envío" button; Gmail
  unreachable → inconclusive, never resend.
- Restart recovery: drafts left in `sending`/`sent_unverified` are reconciled
  at startup (and periodically by `ReconciliationSweep`) — never blindly
  resent. Orphan temp files are swept.
- Retry (`resend_draft`) re-verifies Gmail FIRST and only sends again on
  definitive evidence the mail never left.

## Outgoing attachments (Telegram → Gmail)

- Telegram documents and photos attached to a reply intent are downloaded to
  `TMP_DIR/incoming/` (bounded: `outgoing_attachment_max_count`,
  `outgoing_attachment_max_bytes`; batches violating limits are rejected with
  a notice).
- Filenames are sanitized display metadata (basename only, no traversal, no
  control chars) — never used as trusted filesystem paths.
- The confirmation message lists every attachment that would be sent.
- MIME: `EmailMessage.add_attachment` with the Telegram mime type (fallback
  `application/octet-stream`); thread semantics unchanged.
- Binaries live ONLY in the temp dir for the bounded workflow: deleted after
  verified delivery, cancellation, terminal failure, or the orphan sweep
  (`TMP_MAX_AGE_SECONDS`). SQLite stores metadata only.

## Modules

| Module | Responsibility | Owner |
|---|---|---|
| `config.py` | pydantic-settings, .env, limits, kill switch | core |
| `models.py` | Shared contracts (dataclasses/pydantic) | core |
| `db.py` | SQLite state, dedup, retry bookkeeping, deterministic migrations | core |
| `gmail/auth.py` | OAuth 2.0 local flow, token store, per-install credentials | worker A |
| `gmail/client.py` | Gmail API: get message/thread, send reply (MIME+attachments), `verify_delivery` | worker A |
| `gmail/watcher.py` | `users.watch` lifecycle, auto-renew before expiry, historyId | worker A |
| `gmail/pubsub.py` | StreamingPull subscription (no public webhook) | worker A |
| `gmail/parse.py` | HTML cleanup, tracker/signature removal, plain text | worker A |
| `gmail/attachments.py` | PDF (+password), DOCX, TXT local extraction, limits | worker A |
| `llm/base.py` | Provider abstraction | worker B |
| `llm/openai_compat.py` | OpenAI-compatible client (OpenCode Go / DeepSeek / OpenRouter) | worker B |
| `llm/prompts.py` | ES summary + DE reply prompts, injection defense | worker B |
| `telegram/bot.py` | Bot API, group auth, confirm/cancel/resend buttons, outgoing attachment download | worker B |
| `pipeline.py` | Orchestrates: pubsub event → fetch → parse → LLM → telegram | core (integration) |
| `responder.py` | Draft → confirm → send → verify state machine, reconciliation, retry, recovery | core (integration) |
| `status.py` | `/status`: checks Gmail, Pub/Sub, Telegram, LLM w/o secrets | core |
| `app.py` | Entrypoint, asyncio tasks, graceful shutdown, retries | core |

## Persistence (SQLite)

Schema (owned by core `db.py`):

- `messages`: `message_id`, `thread_id`, `history_id`, `telegram_message_id`,
  `status` (received|summarizing|sent_telegram|failed), `created_at`,
  `updated_at`, `retry_count`, `next_retry_at`.
- `drafts`: `id`, `thread_id`, `message_id`, `body` (the user's own generated
  reply, needed for safe retry), `to_json`, `subject`, `status`
  (pending|confirmed|sending|sent_unverified|sent_verified|send_failed|
  cancelled|rejected), `telegram_user_id` (ownership), `sent_message_id`,
  `send_started_at`, `verification_attempts`, `attachments_json` (metadata
  only: filename/mime/size), timestamps.
- `memories`: explicit per-member facts (`/remember`), keyed by Telegram user.
- `meta`: `key`/`value` — `last_history_id`, `watch_expires_at`, tg↔thread
  mappings, temporary "ver original" state.

No email bodies, no extracted attachment text, no attachment binaries, no
secrets stored. Draft bodies persist only until terminal states are reached
(retry needs them); incoming content is never persisted. Schema migrations
(`_ensure_column`) are deterministic and tested.

## Data lifecycle (privacy)

```
Gmail/Telegram  →  temporary processing (memory, TMP_DIR)  →  LLM provider
        →  generated result (summary / draft)  →  Telegram
        →  temp data destroyed (verified / cancelled / failed / sweep)
SQLite keeps only minimal workflow metadata (IDs, statuses, timestamps).
```

What is sent to the configured AI provider: the incoming email's cleaned text
body, bounded attachment text (PDF/DOCX/TXT, truncated), and thread context
for drafts — i.e. email content DOES reach the provider while processing.
Nothing else. The provider is self-configured (OpenCode Go / DeepSeek /
OpenRouter endpoint). Secrets, tokens and bot credentials are never part of
LLM payloads.

## LLM provider abstraction

```python
class LLMProvider(Protocol):
    async def summarize_email(self, email: ParsedEmail) -> EmailSummary: ...
    async def draft_reply(self, request: DraftRequest, thread: ThreadContext) -> DraftReply: ...
```

OpenAI-compatible via the `openai` SDK (`base_url` configurable). No automatic
paid fallback between providers. On failure, email stays pending for retry
(with backoff); it is never silently lost. Real-provider validation lives in
`tests/integration/test_llm_real.py` (opt-in via `RUN_REAL_LLM=1`, synthetic
content only).

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
  integration/test_verified_delivery.py   # state machine, ambiguity, retry
  integration/test_llm_real.py            # opt-in real provider validation
docs/ARCHITECTURE.md
.env.example  .gitignore  pyproject.toml  Dockerfile  docker-compose.yml
.github/workflows/ci.yml
```

## MVP acceptance

See README.md "MVP verification" — each maps to a test or an operational check.
