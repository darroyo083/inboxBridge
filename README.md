# InboxBridge

Self-hosted email assistant that connects your Gmail **Primary** inbox with a
private Telegram group. **Natural language first**: no slash commands needed —
talk to it like a person.

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
- **AI routing**: DeepSeek for text, MiMo for vision (scanned PDFs/images),
  Luna configured as the technical-fallback vision model — zero local OCR.
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
  verified delivery, attachment lifecycle, natural-language routing,
  AI routing, data/privacy model.
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
RUN_REAL_LLM=1 pytest -m real_llm   # opt-in REAL AI validation (synthetic content)
```

## MVP verification

1. New Gmail email in Primary triggers InboxBridge.
2. No historical emails processed on first boot.
3. No duplicates (idempotent pipeline).
4. Email summarized in natural Spanish.
5. PDF, password-PDF, DOCX, TXT attachments work.
6. Summary only reaches the authorized group.
7. Natural language drives everything; buttons and slash commands are extras.
8. Reply drafts are professional/natural German with thread context.
9. Recipients, attachments and content shown before sending — always the real
   addresses, never just friendly names.
10. Explicit confirmation required; SEND/EDIT/CANCEL buttons ask "¿seguro?".
11. "ok"/"sí"/"vale" can never authorize a send — only explicit verbs.
12. Draft edits re-render the full preview; a stale preview cannot be sent.
13. Delivery is reconciled against Gmail; success is shown only after verification.
14. Ambiguous send failures are resolved by reconciliation, never by blind resend.
15. New emails and forwards use the SAME verified-delivery pipeline.
16. Contacts/aliases managed from Telegram (e.g. "cuando diga Roman usa femo@femo.ch");
    the LLM never invents an address; ambiguous recipients always ask.
17. Telegram documents/photos attach to replies; Gmail attachments come to Telegram.
18. Reminders survive restarts, fire once, and are cancelable.
19. Scanned PDFs and images are read by the external vision model (no local OCR).
20. Prompt-injected emails/attachments/images can never mutate contacts or send mail.
21. Container restart breaks nothing; in-flight drafts are reconciled, not resent.
22. No bodies/attachments persisted (SQLite holds metadata only).
23. No secrets committed or logged.
24. Project moves to another VPS via Docker Compose + config + small volume.

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
