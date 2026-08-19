# InboxBridge

InboxBridge is a self-hosted, multilingual Gmail assistant controlled through
Telegram. The live product landing is at [inboxbridge.dev](https://inboxbridge.dev).
It keeps Gmail as the source of truth while turning email work into a short,
reviewable conversation.

- New Gmail emails (Primary tab) → LLM → natural **Spanish** summary → your
  private Telegram group.
- Reply by simply answering the summary: *"respóndele que el viernes sí puedo"*
  → professional **German** draft in the same Gmail thread → you confirm →
  it is sent.
- Refine drafts naturally: *"hazlo más corto"*, *"más formal"*,
  *"cambia las 18:00 por las 19:00"* — every edit re-renders the full preview.
- **New emails** and **forwards**: *"escribe a Roman y dile que mañana llego a
  las seis"* — recipients resolve through your saved **contacts/aliases**.
- Buttons protect against mis-taps: SEND / EDIT / CANCEL always ask "¿seguro?".
- *"mándame el pdf"* → the original attachment arrives in Telegram (temp only).
- *"márcalo como leído"*, *"archívalo"*, *"¿qué me está pidiendo?"*,
  *"resume toda la conversación"*, *"recuérdamelo mañana"* — all work as text.
- **AI routing**: configured text and vision providers handle summaries, drafts,
  translations, and scanned documents without local OCR.
- **Verified delivery**: after sending, InboxBridge reconciles against Gmail
  and only reports "sent ✓" once the message is confirmed in the right thread
  with the right recipients/attachments. Ambiguous timeouts never trigger a
  blind resend — Gmail is checked first (no duplicate emails).
- Gmail Push + Google Pub/Sub (StreamingPull, no public webhook). SQLite state
  (IDs/statuses only — no email bodies, no attachment content). Docker
  Compose. Kill switch `SEND_EMAILS=false` by default.

The deployed landing describes the current Gmail -> Telegram workflow, including
multilingual summaries and replies, attachment-aware context, reply/compose/
forward flows, reminders, protected PDFs, explicit confirmation, verified
delivery, and restart/recovery safety.

## Current status

InboxBridge is deployed and maintained as a self-hosted product. Sending remains
guarded by the explicit confirmation boundary and the `SEND_EMAILS` kill switch.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — flow, modules, send state machine,
  verified delivery, attachment lifecycle, natural-language routing,
  AI routing, data/privacy model.
- [Installation](docs/INSTALL.md) — VPS + Docker Compose setup.
- [Migration](docs/MIGRATION.md) — moving to another VPS.

## Quick start (development)

```bash
cp .env.example .env          # then fill in credentials
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
pytest
```

## Validation

```bash
pytest                              # deterministic suite (no credentials needed)
RUN_REAL_LLM=1 pytest -m real_llm   # opt-in REAL AI validation (synthetic content)
```

## Product guarantees

- New Gmail Primary mail is summarized in Spanish for the authorized Telegram group.
- Thread context and supported attachments, including protected PDFs, are handled in memory.
- Natural-language replies, new messages, forwards, edits, and reminders stay conversational.
- German drafts show recipients, attachments, and Spanish review before sending.
- Only explicit confirmation can cross the send boundary; delivery is reconciled against Gmail.
- Restart and timeout recovery checks trusted state before retrying, preventing blind duplicates.
- SQLite stores identifiers and statuses only; message bodies and attachments are not persisted.
- Deployment is portable through Docker Compose and a small persistent volume.

## Data & privacy

Email bodies and attachment content are used in memory for the current
processing step and are sent to the configured AI provider (OpenCode Go /
DeepSeek text; MiMo vision; Luna fallback) to produce summaries, drafts,
answers and document understanding. They are never written to SQLite or logs,
and temp files (attachments, voice notes, rendered PDF pages) are deleted
after use, cancellation, failure, or the cleanup sweep. Gmail remains the
source of truth for email content. Contacts/aliases and reminders persist
explicit user-configured application data (names, addresses, IDs, times) —
never email bodies.

## License

MIT
