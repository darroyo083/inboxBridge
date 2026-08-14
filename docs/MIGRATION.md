# Migration to another VPS

The whole state is small and portable by design. Moving takes minutes.

## What to copy

1. **`.env`** — all configuration (secrets stay here, never in git).
2. **`credentials/`** — the static secrets `client_secret.json` and
   `service_account_pubsub.json` (read-only at runtime). The OAuth refresh
   token is NOT here — it lives in the data volume (see below).
3. **The Docker volume** `inboxbridge-data` — SQLite state plus
   `token.json` (the OAuth refresh token; from installed-app flows it does not
   expire, so the new VPS keeps working without re-authorization). It also
   holds dedup ids, statuses, the historyId baseline and the watch expiry —
   copying it means no duplicate summaries and no gap in processing.

## Procedure

On the OLD VPS:

```bash
docker compose stop
docker run --rm -v inboxbridge-data:/data -v "$PWD:/backup" alpine \
  tar czf /backup/inboxbridge-data.tar.gz -C /data .
```

Copy `inboxbridge-data.tar.gz`, `.env` and `credentials/` to the new VPS
(scp / rsync — the tarball is a few KB typically).

On the NEW VPS:

```bash
git clone git@github.com:darroyo083/inboxBridge.git && cd inboxBridge
cp <backup>/.env .env
mkdir -p credentials && cp <backup>/credentials/* credentials/
docker compose up -d      # creates the volume
docker compose stop
docker run --rm -v inboxbridge-data:/data -v "$PWD:/backup" alpine \
  sh -c "rm -rf /data/* && tar xzf /backup/inboxbridge-data.tar.gz -C /data"
docker compose start
```

Verify with `/status` and a test email.

## Notes

- No email bodies or attachment contents are ever stored — the only sensitive
  thing in the volume backup is the OAuth refresh token (`token.json`).
- The Pub/Sub subscription is Pull/StreamingPull: on the new VPS, the same
  subscription streams again; because messages are acked only after successful
  processing, events that arrived while the service was down are redelivered
  (at-least-once), and the dedup table in the volume prevents duplicates.
- If the historyId baseline lags far behind (long downtime), the history delta
  processor catches up from the stored baseline; it never processes mail older
  than the first watch.
