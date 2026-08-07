# InboxBridge

Self-hosted email assistant that connects your Gmail **Primary** inbox with a
private Telegram group.

- New Gmail emails (Primary tab) → LLM → natural **Spanish** summary → your
  private Telegram group.
- From Telegram you can ask for a reply → InboxBridge drafts a professional
  **German** email in the same Gmail thread → you confirm → it is sent.
- Gmail Push + Google Pub/Sub (StreamingPull, no public webhook). SQLite state.
  Docker Compose. No email bodies stored. Kill switch `SEND_EMAILS=false` by
  default.

> **Development note:** by default `SEND_EMAILS=false`, so sending is
> technically impossible until you explicitly enable it.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — flow, modules, contracts.
- [Installation](docs/INSTALL.md) — VPS + Docker Compose setup.
- [Migration](docs/MIGRATION.md) — moving to another VPS.

## Quick start (dev)

```bash
cp .env.example .env          # then fill in credentials
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
pytest
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
10. Recipients and content shown before sending.
11. Explicit confirmation required.
12. With `SEND_EMAILS=true` + confirmation, reply lands in the same Gmail thread.
13. Container restart breaks nothing, no duplicates.
14. LLM/Pub/Sub failure does not silently lose the email (retry queue).
15. No bodies/attachments persisted.
16. No secrets committed or logged.
17. Project moves to another VPS via Docker Compose + config + small volume.

## License

MIT
