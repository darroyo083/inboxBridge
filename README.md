# InboxBridge

Self-hosted email assistant that connects your Gmail **Primary** inbox with a
private Telegram group.

- New Gmail emails (Primary tab) → LLM → natural **Spanish** summary → your
  private Telegram group.
- From Telegram you can ask for a reply → InboxBridge drafts a professional
  **German** email in the same Gmail thread → you confirm → it is sent.
- You can attach Telegram documents/photos to a reply; the confirmation shows
  exactly what will be sent.
- **Verified delivery**: after sending, InboxBridge reconciles against Gmail
  and only reports "sent ✓" once the message is confirmed in the right thread
  with the right recipients/attachments. Ambiguous timeouts never trigger a
  blind resend — Gmail is checked first (no duplicate emails).
- Gmail Push + Google Pub/Sub (StreamingPull, no public webhook). SQLite state
  (IDs/statuses only — no email bodies, no attachment content). Docker
  Compose. Kill switch `SEND_EMAILS=false` by default.

> **Development note:** by default `SEND_EMAILS=false`, so sending is
> technically impossible until you explicitly enable it.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — flow, modules, send state machine,
  verified delivery, attachment lifecycle, data/privacy model.
- [Installation](docs/INSTALL.md) — VPS + Docker Compose setup.
- [Migration](docs/MIGRATION.md) — moving to another VPS.

## Quick start (dev)

```bash
cp .env.example .env          # then fill in credentials
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
pytest
```

## Tests

```bash
pytest                              # deterministic suite (no credentials needed)
RUN_REAL_LLM=1 pytest -m real_llm   # opt-in REAL LLM validation (synthetic content)
```

## MVP verification

1. New Gmail email in Primary triggers InboxBridge.
2. No historical emails processed on first boot.
3. No duplicates (idempotent pipeline).
4. Email summarized in natural Spanish.
5. PDF, password-PDF, DOCX, TXT attachments work.
6. Summary only reaches the authorized group.
7. Replying to the bot's message lets you request a reply.
8. InboxBridge recovers thread context.
9. Draft is professional/natural German.
10. Recipients, attachments and content shown before sending.
11. Explicit confirmation required.
12. With `SEND_EMAILS=true` + confirmation, reply lands in the same Gmail thread.
13. Delivery is reconciled against Gmail; success is shown only after verification.
14. Ambiguous send failures are resolved by reconciliation, never by blind resend.
15. Telegram documents/photos can be attached; temp binaries are cleaned up.
16. Container restart breaks nothing; in-flight drafts are reconciled, not resent.
17. LLM/Pub/Sub failure does not silently lose the email (retry queue).
18. No bodies/attachments persisted (SQLite holds metadata only).
19. No secrets committed or logged.
20. Project moves to another VPS via Docker Compose + config + small volume.

## Data & privacy

Email bodies and attachment content are used in memory for the current
processing step and are sent to the configured AI provider (OpenCode Go /
DeepSeek / OpenAI-compatible endpoint) to produce the summary/draft. They are
never written to SQLite or logs, and temp attachment files are deleted after
verified delivery, cancellation, failure, or the cleanup sweep. Gmail remains
the source of truth for email content.

## License

MIT
