# InboxBridge Landing

Public static landing experience for the deployed InboxBridge product:
[inboxbridge.dev](https://inboxbridge.dev).

InboxBridge connects Gmail Primary mail to Telegram with multilingual Spanish
summaries, German reply/compose/forward drafts, attachment-aware context,
explicit confirmation, verified delivery, reminders, and restart-safe recovery.

The app uses real browser routes:

- `/`
- `/how-it-works`
- `/capabilities`
- `/safety`
- `/architecture`

## Local development

```bash
npm install
npm run dev
```

## Production build

```bash
npm run build
```

Deploy the generated `dist/` directory as a static site behind nginx or any
static hosting provider. Configure the server to fall back unknown paths to
`index.html` so direct navigation to a route continues to work. The frontend
does not require the InboxBridge runtime.

Inter and JetBrains Mono are bundled under `public/fonts/` to match the Stitch
design without a runtime Google Fonts dependency. The page uses CSS diagrams and
the downloaded Stitch preview only as a visual reference; no backend assets or
credentials are required.

The visual architecture reflects the actual stack: Python, Gmail API, Telegram
Bot API, Google Pub/Sub, SQLite, OAuth 2.0, Docker Compose, and configured LLM
APIs. No credentials, personal addresses, chat IDs, or VPS details belong here.

The GitHub links point to the repository remote:
`https://github.com/darroyo083/inboxbridge`.
