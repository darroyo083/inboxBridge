# InboxBridge — Architecture

Self-hosted email assistant: connects Gmail (Primary inbox) with a private
Telegram group. Summarizes incoming emails in Spanish via LLM; allows drafting
and sending replies in German from Telegram with explicit confirmation,
verified delivery against Gmail, and outgoing attachments from Telegram.

**Natural language first (V1.1):** ordinary usage needs no slash commands.
Messages are routed through a validated intent boundary:

    Telegram text
    → deterministic rule pre-checks (explicit verbs)
    → LLM classification fallback (structured JSON, validated)
    → validated Intent
    → deterministic application state machine
    → allowed action

The LLM never gains authority: sending, cancelling, archiving and contact
mutation stay deterministic application logic behind explicit intents.

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
AI provider (DeepSeek text; MiMo vision for scans/images; Luna fallback)
      │
      ▼
natural Spanish summary → private Telegram group (authorized chat_id only)

--- reply flow (from Telegram, replying to bot's message) ---
Telegram reply / mention / command (+ optional document/photo/voice)
      ▼
intent routing (validated)
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
  statuses, retry timestamps, contacts and reminders. No full email bodies,
  no attachment text, no attachment binaries, no secrets.
- **Untrusted content.** Emails/attachments/images are data, never
  instructions. Prompt-injection defenses, HTML/tracker cleanup, no link
  clicks, no attachment execution, no local OCR.
- **Kill switch.** `SEND_EMAILS=false` (default) makes sending technically
  impossible in dev/test. Sending only happens with `SEND_EMAILS=true` AND
  explicit Telegram confirmation.
- **No secret leakage.** `.env` ignored by git; `.env.example` has placeholders
  only; structured logs never include secrets or bodies.

## Intent routing

`intents.py` classifies every user message:

- **Deterministic rules first**: explicit verbs ("envíalo", "cancela el
  borrador", "archívalo", "márcalo como leído", "reenvíaselo a…",
  "escribe a…", "recuérdame…", contact management…) produce high-confidence,
  `explicit` intents. Ambiguous acks ("ok", "vale", "sí", "perfecto") NEVER
  authorize send/cancel — they surface as CLARIFY and the bot asks.
- **LLM fallback**: for anything unmatched, the text model classifies into a
  fixed JSON vocabulary. The LLM vocabulary EXCLUDES send/cancel entirely
  (rejected → UNKNOWN), and LLM results are never `explicit`. High-impact
  actions only execute on deterministic explicit verbs.
- **Context resolution**: replying to a summary binds the Gmail thread;
  an active draft binds edit/send/cancel intents; ambiguity asks.

## Draft safety (V1.1)

- Every draft preview shows the REAL recipient addresses, CC, subject,
  attachments and type (Respuesta / Nuevo correo / Reenvío).
- Previews are **versioned**: any edit re-renders the full preview and bumps
  the version; a stale preview can never authorize a send.
- Buttons SEND / EDIT / CANCEL are **two-step**: a tap shows
  "¿Seguro?" → [Sí] [Volver]. Callbacks are one-shot, owner-scoped and
  replay-safe.
- Explicit TEXT acts ("envíalo", "cancela el borrador") skip the second
  button confirmation but keep every other safety invariant; "ok"/"sí" never
  suffice.
- Text and button confirmations resolve the SAME pending future; the
  coordinator's atomic `claim_draft_for_send` prevents any double send.

## Logical contacts + aliases

`contacts.py` + SQLite (`contacts`, `contact_aliases`):

- A LogicalContact has a display name, one preferred email and normalized
  unique aliases. Several logical names may map to one shared mailbox
  (Roman → femo@femo.ch, FEMO → femo@femo.ch).
- Aliases normalize with NFKC/casefold; emails are syntactically validated
  (control chars, spaces, multiple @ rejected).
- Resolution is deterministic: normalized alias → display name → exact email.
  Ambiguity ALWAYS asks; unknown names ALWAYS ask (the LLM never invents an
  address).
- All persistent mutations (create/change/delete/alias) render the
  interpreted change and require a one-shot confirmation. Nothing is learned
  silently.
- Reply semantics ignore aliases: replying to a thread always uses the
  original Gmail recipients.

## Reminders

`reminders.py` + SQLite `reminders` (IDs + due + status + user note only —
never email bodies). Deterministic Spanish time parsing ("en dos horas",
"mañana", "el viernes a las 18:00", "a las 15:30"…). `ReminderScheduler`
fires due reminders exactly once (atomic claim), survives restarts, and is
cancelable/listable from Telegram.

## AI routing (V1.1)

`llm/ai_service.py` — configuration-driven, no hardcoded model IDs:

```
AIService
├── text(...)             → configured text model (DeepSeek)
├── vision(...)           → configured vision model (MiMo), bounded fallback
│                            to the fallback model (Luna) ONLY on technical
│                            failures (unavailable/rate-limit/empty/malformed/
│                            unsupported modality) — never on "didn't like it"
├── document_vision(...)  → scanned PDF: bounded page render (PyMuPDF) →
│                            vision model (zero local OCR, zero local models)
└── audio(...)            → experimental, gated by ai_audio_enabled
```

- Text-only work (summaries, drafts, edits, Q&A, intents) uses the text model.
- Images and image-only PDFs go to the vision model. PDFs with a usable text
  layer are extracted deterministically — vision is NOT wasted on them.
- Bounds: `ai_vision_max_pages`, `ai_vision_max_dimension`, per-page size.
- Observability is metadata-only (task, model, duration, success,
  fallback_used) — never content.
- Voice notes (experimental): bounded download → transcription → intent flow;
  gated by `AI_AUDIO_ENABLED` (default off); any failure asks the user to type.
- REAL provider evidence (opt-in suite): MiMo reads synthetic images and
  scanned PDFs; text PDFs avoid vision; the configured fallback model
  (`gpt-5.6-luna`) currently rejects image input via the OpenCode Go gateway
  (HTTP 400) — documented limitation, graceful typed failure, single bounded
  fallback attempt.

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
  definitive evidence the mail never left. Per-draft locks + atomic
  `claim_draft_for_send` serialize all send paths (double-tap, sweep races).
- New emails and forwards reuse this EXACT pipeline (`present_draft` hook).

## Outgoing attachments (Telegram → Gmail)

- Telegram documents and photos attached to a reply intent are downloaded to
  `TMP_DIR/incoming/` (bounded count/size, re-validated after download) and
  claimed into `tmp/draft-<id>/` with deterministic order-prefixed names.
- Filenames are sanitized display metadata — never trusted paths.
- The confirmation message lists every attachment that would be sent.
- MIME: `EmailMessage.add_attachment` with the Telegram mime type (fallback
  `application/octet-stream`); thread semantics unchanged.
- Binaries live ONLY in the temp dir for the bounded workflow: deleted after
  verified delivery, cancellation, terminal failure, or the orphan sweep.

## Incoming attachments (Gmail → Telegram)

- "mándame el pdf" / the 📎 Adjuntos button delivers supported original
  attachments to the group (temp file → send → remove). Types: PDF, DOCX,
  TXT, CSV, images; bounded count/size; no traversal; authorized chat only.
- Vision handles image-only PDFs/images when the user asks questions about
  them (bounded render → MiMo).

## Modules

| Module | Responsibility | Owner |
|---|---|---|
| `config.py` | pydantic-settings, .env, limits, kill switch, AI routing | core |
| `models.py` | Shared contracts (dataclasses/pydantic) | core |
| `db.py` | SQLite state, dedup, retry, contacts, reminders, deterministic migrations | core |
| `intents.py` | NL intent routing (rules + validated LLM fallback) | core (V1.1) |
| `assistant.py` | Executes validated intents: edits, Q&A, summaries, attachments, actions, reminders, contacts, compose/forward, voice | core (V1.1) |
| `contacts.py` | Logical contacts + aliases, resolution, hijack-safe validation | core (V1.1) |
| `reminders.py` | Deterministic time parsing, CRUD, fire-once scheduler | core (V1.1) |
| `gmail/auth.py` | OAuth 2.0 local flow, token store, per-install credentials | worker A |
| `gmail/client.py` | Gmail API: get message/thread, send reply (MIME+attachments), `verify_delivery`, labels, attachment bytes | worker A |
| `gmail/watcher.py` | `users.watch` lifecycle, auto-renew before expiry, historyId | worker A |
| `gmail/pubsub.py` | StreamingPull subscription (no public webhook) | worker A |
| `gmail/parse.py` | HTML cleanup, tracker/signature removal, plain text | worker A |
| `gmail/attachments.py` | PDF (+password), DOCX, TXT local extraction, limits | worker A |
| `llm/base.py` | Provider abstraction + retry machinery | worker B |
| `llm/openai_compat.py` | OpenAI-compatible client (text/vision/audio) | worker B |
| `llm/ai_service.py` | Routing facade: text/vision/document_vision/audio, bounded fallback | worker B (V1.1) |
| `llm/pdf_render.py` | Bounded PDF page rendering (PyMuPDF; no OCR) | worker B (V1.1) |
| `llm/prompts.py` | ES/DE prompts, edits, Q&A, thread summaries, compose/forward, injection defense | worker B |
| `telegram/bot.py` | Bot API, group auth, intent dispatch, two-step buttons, panels, attachment download, voice | worker B |
| `pipeline.py` | Orchestrates: pubsub event → fetch → parse → LLM → telegram | core (integration) |
| `responder.py` | Draft → confirm → send → verify state machine, reconciliation, retry, recovery | core (integration) |
| `status.py` | `/status`: checks Gmail, Pub/Sub, Telegram, LLM w/o secrets | core |
| `app.py` | Entrypoint, asyncio tasks, graceful shutdown, wiring | core |

## Persistence (SQLite)

- `messages`: dedup/status identifiers only.
- `drafts`: the user's own generated reply (needed for retry), recipients,
  subject, status (pending|confirmed|sending|sent_unverified|sent_verified|
  send_failed|cancelled|rejected), owner, sent message id, timestamps.
- `contacts` / `contact_aliases`: explicit user-configured application data
  (display name, preferred email, normalized aliases).
- `reminders`: message/thread IDs, due time, status, user, note (user's words).
- `memories`: explicit per-member facts (`/remember`).
- `meta`: history baseline, watch expiry, tg↔thread mappings, temp view state.

No email bodies, no extracted attachment text, no attachment binaries, no
secrets. Incoming content is never persisted; temp files are swept by age.

## Data lifecycle (privacy)

```
Gmail/Telegram  →  temporary processing (memory, TMP_DIR)  →  AI provider
        →  generated result (summary / draft / answer)  →  Telegram
        →  temp data destroyed (verified / cancelled / failed / sweep)
SQLite keeps only minimal workflow metadata + explicit user data.
```

What is sent to the configured AI provider: the incoming email's cleaned text
body, bounded attachment text, bounded rendered pages of scanned PDFs, user
instructions, and thread context for drafts — email content DOES reach the
provider while processing. Nothing else. Secrets, tokens and credentials are
never part of AI payloads.

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
  config.py models.py db.py intents.py assistant.py contacts.py reminders.py
  pipeline.py responder.py status.py app.py
  gmail/    (auth, client, watcher, pubsub, parse, attachments)
  llm/      (base, openai_compat, ai_service, pdf_render, prompts)
  telegram/ (bot)
tests/
  unit/  integration/  mocks/
  integration/test_verified_delivery.py    # state machine, ambiguity, retry
  integration/test_v11_flows.py            # simulated V1.1 E2E (flows A-T)
  integration/test_v11_security_restart.py # injection/restart/concurrency
  integration/test_llm_real.py             # opt-in real AI validation
docs/ARCHITECTURE.md
.env.example  .gitignore  pyproject.toml  Dockerfile  docker-compose.yml
.github/workflows/ci.yml
```

## MVP acceptance

See README.md "MVP verification" — each maps to a test or an operational check.
