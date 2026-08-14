# Installation & Setup

InboxBridge runs as a single Docker Compose service. This guide covers a fresh
VPS setup. Everything sensitive stays in `.env` + `credentials/` — nothing is
committed to the repo.

## 1. Prerequisites

- A VPS with Docker + Docker Compose (any provider; nothing OVH-specific).
- A Gmail account for the inbox you want to bridge.
- A Google Cloud project with Gmail API + Pub/Sub enabled.
- A Telegram bot token from [@BotFather](https://t.me/BotFather) and the
  numeric id of your private group.

## 2. Google Cloud setup (per installation)

1. [Create a Google Cloud project](https://console.cloud.google.com) (e.g.
   `inboxbridge`). Enable **Gmail API**.
2. Create an **OAuth 2.0 Client ID** of type *Desktop app*; download the JSON
   as `credentials/client_secret.json`. Every installation uses its OWN OAuth
   credentials — never share them.
3. Enable **Pub/Sub API**. Create a topic (e.g. `gmail-events`) and a pull
   subscription (e.g. `gmail-events-pull`). Subscription type: **Pull** —
   InboxBridge uses StreamingPull from the VPS; no public webhook is exposed.
4. Grant the OAuth client's Google Account (the Gmail address itself) the
   Pub/Sub publisher role on the topic, so Gmail Push can publish to it.

## 3. First boot on the VPS

```bash
git clone git@github.com:darroyo083/inboxBridge.git
cd inboxBridge
cp .env.example .env
# edit .env: Telegram, LLM, Google ids, limits (see .env.example comments)

mkdir -p credentials
# copy credentials/client_secret.json into credentials/
# (do this ONCE with a browser available:)
docker compose run --rm inboxbridge python -m inboxbridge.oauth_bootstrap
```

`oauth_bootstrap` opens the local OAuth flow: you authorize in a browser, and
the refresh token is saved to `data/token.json` — i.e. inside the persistent
`inboxbridge-data` Docker volume, NOT the read-only `credentials/` mount. Run
this on the VPS (SSH port-forward the local server URL it prints) so the token
lands in the volume directly. Refresh tokens from installed-app flows do not
expire; the access token still expires and is refreshed + saved back to
`data/token.json` on every expiry, which is why that path must stay writable.

## 4. Run

```bash
docker compose up -d
docker compose ps          # healthy after ~30s
```

## 5. Verify

- `/status` in the Telegram group shows Gmail / Pub/Sub / Telegram / LLM
  states without revealing secrets.
- Send a test email to the bridged Gmail address → a Spanish summary arrives
  in the group within seconds.
- Replying to the bot's summary asks for a reply draft (German). Confirm to
  send — only if `SEND_EMAILS=true`.

## 6. Updating

```bash
git pull
docker compose build --pull
docker compose up -d
```

## .env essentials

| Variable | Meaning |
|---|---|
| `TELEGRAM_BOT_TOKEN` | bot token from @BotFather |
| `TELEGRAM_ALLOWED_CHAT_ID` | numeric id of the private group (only this chat is processed) |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI-compatible endpoint (OpenCode Go, DeepSeek, OpenRouter…) |
| `GOOGLE_CLOUD_PROJECT` | Cloud project hosting topic/subscription |
| `GMAIL_PUBSUB_TOPIC` / `GMAIL_PUBSUB_SUBSCRIPTION` | short names (path built as `projects/<proj>/topics/<topic>`) |
| `SEND_EMAILS` | `false` in dev; `true` only when you want real sending |
| `PDF_PASSWORD` | password for protected PDFs you expect to receive |
