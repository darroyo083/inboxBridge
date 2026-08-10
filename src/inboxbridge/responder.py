"""Reply flow: Telegram request → thread context → draft → confirm → send → verify.

Separation of concerns (MVP requirement):
- ``prepare_draft`` ONLY generates a draft and shows it for confirmation.
- ``send_draft`` ONLY sends an already-confirmed draft.
- Sending is technically impossible unless ``SEND_EMAILS=true`` (the Gmail
  client raises ``SendingDisabledError`` otherwise — kill switch).

Verified delivery contract (hard requirement):

- Success is never reported from an HTTP return alone. Every send attempt is
  RECONCILED against Gmail (``verify_delivery``) and the draft only reaches
  ``sent_verified`` on deterministic Gmail evidence (message exists, correct
  thread, recipients, subject and attachments).
- An ambiguous outcome (transport error / 5xx after transmission) transitions
  to ``sent_unverified`` and is reconciled: found → ``sent_verified`` (NO
  second send); not found → a CONTROLLED retry may be offered; inconclusive
  (Gmail could not be queried) → never a blind resend.
- A definitive failure → ``send_failed`` with a user-facing retry button.
- Restart recovery: drafts left in ``sending``/``sent_unverified`` are
  reconciled at startup (never blindly resent); orphan temp files are swept.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import getaddresses
from pathlib import Path
from typing import Any

from .config import Settings
from .contracts import GmailClient, LLMProvider
from .db import Storage
from .gmail.client import AmbiguousSendError, SendingDisabledError
from .models import (
    DraftReply,
    DraftRequest,
    DraftStatus,
    EmailAddress,
    OutgoingAttachment,
    SendVerification,
)
from .telegram.bot import ReplyRequest, TelegramBot

logger = logging.getLogger(__name__)

#: Active send states that must be reconciled after a restart (never resent).
_UNRECONCILED = (DraftStatus.SENDING, DraftStatus.SENT_UNVERIFIED)


class ReplyCoordinator:
    """Consumes Telegram reply requests and drives the draft/confirm/send/verify cycle."""

    def __init__(
        self,
        settings: Settings,
        gmail: GmailClient,
        llm: LLMProvider,
        bot: TelegramBot,
        storage: Storage,
    ) -> None:
        self._settings = settings
        self._gmail = gmail
        self._llm = llm
        self._bot = bot
        self._storage = storage
        self._tmp_dir = Path(settings.tmp_dir)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        bot.register_resend_callback(self.resend_draft)

    async def run_forever(self) -> None:
        """Process reply requests from the bot queue indefinitely."""
        async for request in self._bot.reply_requests():
            await self._handle_request(request)

    async def _handle_request(self, request: ReplyRequest) -> None:
        if not request.thread_id:
            await self._bot.send_notice(
                "No puedo asociar tu petición a ningún hilo. Responde directamente "
                "a un resumen de InboxBridge y escribe lo que quieres responder."
            )
            return
        try:
            await self._bot.send_typing()
            thread = await self._gmail.fetch_thread_context(request.thread_id)
            draft_request = DraftRequest(
                thread_id=request.thread_id,
                user_instructions=request.user_instructions,
                language="de",
                memory=request.memory,
            )
            draft: DraftReply = await self._llm.draft_reply(draft_request, thread)
            if request.attachments:
                draft = replace(draft, attachments=request.attachments)
            await self._present_draft(draft, user_id=request.user_id)
        except Exception:
            logger.exception("reply flow failed for thread %s", request.thread_id)
            await self._bot.send_notice("No pude preparar la respuesta. Inténtalo de nuevo.")

    async def _present_draft(self, draft: DraftReply, *, user_id: int = 0) -> None:
        """Show recipients/attachments/body and wait for explicit confirmation.

        The draft row is persisted at presentation time (status PENDING) so a
        crash or restart keeps the draft and its temp attachments coherent.
        Cancel/timeout → CANCELLED + temp cleanup.
        """
        draft_id = self._storage.create_draft(
            draft.thread_id, None, draft, telegram_user_id=user_id
        )
        message_id = await self._bot.send_draft_for_confirmation(
            draft, draft_id=draft_id, user_id=user_id
        )
        confirmed = await self._bot.wait_for_confirmation(message_id)
        if not confirmed:
            logger.info(
                "draft %d for thread %s not confirmed; discarding",
                draft_id, draft.thread_id,
            )
            self._storage.set_draft_status(draft_id, DraftStatus.CANCELLED)
            self._cleanup_attachments(draft)
            await self._bot.send_notice("Borrador cancelado.")
            return
        self._storage.set_draft_status(draft_id, DraftStatus.CONFIRMED)
        await self._send_confirmed(draft_id, draft)

    async def _send_confirmed(self, draft_id: int, draft: DraftReply) -> None:
        """Drive one send attempt: sending → (verified | unverified | failed)."""
        started_ms = int(time.time() * 1000)
        self._storage.set_draft_send_started(draft_id, started_ms)
        self._storage.set_draft_status(draft_id, DraftStatus.SENDING)
        try:
            new_message_id = await self._gmail.send_reply(draft)
        except SendingDisabledError:
            # Kill switch active: never attempt sending. Notify, keep pending.
            self._storage.set_draft_status(draft_id, DraftStatus.CONFIRMED)
            await self._bot.send_notice(
                "SEND_EMAILS=false: el envío está desactivado (kill switch). "
                "El borrador queda guardado; actívalo en la configuración para enviar."
            )
            return
        except AmbiguousSendError:
            logger.warning(
                "ambiguous send for draft %d (thread %s); reconciling",
                draft_id, draft.thread_id,
            )
            self._storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
            await self._reconcile(
                draft_id, draft, expected_message_id="", started_ms=started_ms
            )
            return
        except Exception:
            logger.exception("definitive send failure for draft %d", draft_id)
            self._storage.set_draft_status(draft_id, DraftStatus.SEND_FAILED)
            await self._bot.send_notice(
                "El envío falló y no se completó. El borrador queda guardado; "
                "puedes reintentarlo."
            )
            await self._bot.offer_resend(draft_id, user_id=0)
            return
        self._storage.set_draft_sent_message(draft_id, new_message_id)
        await self._reconcile(
            draft_id, draft, expected_message_id=new_message_id, started_ms=started_ms
        )

    async def _reconcile(
        self,
        draft_id: int,
        draft: DraftReply,
        *,
        expected_message_id: str,
        started_ms: int,
    ) -> None:
        """Bounded verification loop against Gmail.

        - verified → sent_verified + strong success message + temp cleanup;
        - inconclusive (Gmail unreachable) → sent_unverified, report later;
        - definitive not-found → sent_unverified + controlled retry button.
        """
        attempts = 0
        while attempts < self._settings.send_verification_attempts:
            attempts += 1
            self._storage.bump_verification_attempts(draft_id)
            result: SendVerification = await self._gmail.verify_delivery(
                draft,
                expected_message_id=expected_message_id,
                since_ms=started_ms,
            )
            if result.verified:
                self._storage.set_draft_sent_message(
                    draft_id, result.message_id or expected_message_id
                )
                self._storage.set_draft_status(draft_id, DraftStatus.SENT_VERIFIED)
                await self._bot.send_notice("Enviado y verificado ✓")
                self._cleanup_attachments(draft)
                return
            if not result.checked_ok:
                break  # inconclusive — never resend without Gmail evidence
            if attempts < self._settings.send_verification_attempts:
                await asyncio.sleep(self._settings.send_verification_backoff_seconds)

        self._storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
        if not result.checked_ok:
            await self._bot.send_notice(
                "El envío no se ha podido verificar contra Gmail todavía. "
                "No reenvío para evitar duplicados; lo comprobaré y te aviso."
            )
            return
        await self._bot.send_notice(
            "No encuentro el correo en Gmail. Para evitar duplicados no reenvío "
            "automáticamente; usa el botón si quieres reintentar."
        )
        await self._bot.offer_resend(draft_id, user_id=0)

    # ── controlled retry (user-initiated, duplicate-safe) ───────────────────

    async def resend_draft(self, draft_id: int) -> None:
        """Retry a failed/uncertain draft. NEVER blind: Gmail is checked first.

        Called from the Telegram "Reintentar envío" button (chat + ownership
        already validated by the bot).
        """
        row = self._storage.get_draft(draft_id)
        if row is None:
            await self._bot.send_notice("Ese borrador ya no existe.")
            return
        status = DraftStatus(row["status"])
        if status not in (DraftStatus.SEND_FAILED, DraftStatus.SENT_UNVERIFIED):
            await self._bot.send_notice("Ese borrador no está en estado de reintento.")
            return
        draft = self._draft_from_row(row)
        attachments = self._load_attachments(row)
        if any(a is None for a in attachments):
            await self._bot.send_notice(
                "Los adjuntos de ese borrador ya no están disponibles; "
                "prepara la respuesta otra vez."
            )
            return
        draft = replace(draft, attachments=tuple(a for a in attachments if a is not None))

        verification = await self._gmail.verify_delivery(
            draft,
            expected_message_id=row["sent_message_id"] or "",
            since_ms=int(row["send_started_at"] or 0),
        )
        if verification.verified:
            self._storage.set_draft_status(draft_id, DraftStatus.SENT_VERIFIED)
            await self._bot.send_notice("Ya estaba enviado (verificado ✓). No he reenviado nada.")
            self._cleanup_attachments(draft)
            return
        if not verification.checked_ok:
            await self._bot.send_notice(
                "No puedo confirmar el estado en Gmail ahora; no reenvío para evitar duplicados."
            )
            return
        if verification.found:
            await self._bot.send_notice(
                "Gmail tiene un mensaje en ese hilo que no coincide con el borrador; "
                "no reenvío."
            )
            return
        # Gmail evidence: the message was NOT sent → safe to send now.
        await self._send_confirmed(draft_id, draft)

    # ── restart / periodic recovery ─────────────────────────────────────────

    async def reconcile_on_startup(self) -> None:
        """Sweep drafts left in-flight by a previous process.

        Only reconciles; never resends. Orphan temp files are cleaned after
        terminal states are resolved.
        """
        rows = self._storage.drafts_in_statuses(list(_UNRECONCILED))
        for row in rows:
            draft_id = int(row["id"])
            status = DraftStatus(row["status"])
            if status == DraftStatus.SENDING:
                # The send may have reached Gmail; the row may lack a message id.
                self._storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
            draft = self._draft_from_row(row)
            logger.info("startup reconciliation for draft %d (was %s)", draft_id, status.value)
            await self._reconcile_row(row, draft, startup=True)
        self.cleanup_orphan_tmp()

    async def sweep_unverified(self) -> None:
        """Periodic: re-verify drafts stuck in sent_unverified (bounded)."""
        for row in self._storage.drafts_in_statuses([DraftStatus.SENT_UNVERIFIED]):
            attempts = int(row["verification_attempts"])
            if attempts >= self._settings.send_verification_max_attempts:
                continue
            draft = self._draft_from_row(row)
            await self._reconcile_row(row, draft, startup=False)
        self.cleanup_orphan_tmp()

    async def _reconcile_row(
        self, row: dict[str, Any], draft: DraftReply, *, startup: bool
    ) -> None:
        draft_id = int(row["id"])
        result = await self._gmail.verify_delivery(
            draft,
            expected_message_id=row["sent_message_id"] or "",
            since_ms=int(row["send_started_at"] or 0),
        )
        self._storage.bump_verification_attempts(draft_id)
        if result.verified:
            self._storage.set_draft_sent_message(draft_id, result.message_id)
            self._storage.set_draft_status(draft_id, DraftStatus.SENT_VERIFIED)
            prefix = "Tras el reinicio" if startup else "Verificación"
            await self._bot.send_notice(f"{prefix}: envío confirmado ✓ (no he reenviado nada).")
            self._cleanup_attachments(draft)
        elif not result.checked_ok:
            logger.warning("reconciliation inconclusive for draft %d (Gmail unreachable)", draft_id)
        elif not result.found:
            self._storage.set_draft_status(draft_id, DraftStatus.SEND_FAILED)
            await self._bot.send_notice(
                "Tras revisar Gmail, el envío pendiente no llegó a salir. "
                "Puedes reintentarlo."
            )
            await self._bot.offer_resend(draft_id, user_id=int(row["telegram_user_id"] or 0))

    # ── temp attachment lifecycle ───────────────────────────────────────────

    def _draft_tmp_dir(self, draft_id: int) -> Path:
        return self._tmp_dir / f"draft-{draft_id}"

    def _cleanup_attachments(self, draft: DraftReply) -> None:
        for attachment in draft.attachments:
            if not attachment.path:
                continue
            with contextlib.suppress(OSError):
                Path(attachment.path).unlink(missing_ok=True)
        for attachment in draft.attachments:
            if not attachment.path:
                continue
            parent = Path(attachment.path).parent
            with contextlib.suppress(OSError):
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
                    break

    def cleanup_orphan_tmp(self) -> None:
        """Remove temp dirs whose draft reached a terminal state, or that are stale."""
        if not self._tmp_dir.is_dir():
            return
        max_age = timedelta(seconds=self._settings.tmp_max_age_seconds)
        for child in self._tmp_dir.iterdir():
            if not child.is_dir():
                continue
            if not child.name.startswith("draft-"):
                continue
            try:
                draft_id = int(child.name.removeprefix("draft-"))
            except ValueError:
                continue
            row = self._storage.get_draft(draft_id)
            if row is None:
                pass  # unknown draft → stale, remove if old
            elif row["status"] in (
                DraftStatus.SENT_VERIFIED.value,
                DraftStatus.SEND_FAILED.value,
                DraftStatus.CANCELLED.value,
                DraftStatus.REJECTED.value,
            ):
                with contextlib.suppress(OSError):
                    shutil.rmtree(child, ignore_errors=True)
                continue
            else:
                continue  # active draft: leave the files
            try:
                mtime = datetime.fromtimestamp(child.stat().st_mtime, tz=UTC)
            except OSError:
                continue
            if datetime.now(UTC) - mtime > max_age:
                with contextlib.suppress(OSError):
                    shutil.rmtree(child, ignore_errors=True)

    # ── row → model reconstruction ──────────────────────────────────────────

    def _draft_from_row(self, row: dict[str, Any]) -> DraftReply:
        to: list[EmailAddress] = []
        try:
            stored = json.loads(row["to_json"])
            if isinstance(stored, list):
                to = [
                    EmailAddress(name=name or "", email=addr)
                    for name, addr in getaddresses([", ".join(str(x) for x in stored)])
                    if addr
                ]
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("draft %s has corrupt to_json; no recipients", row.get("id"))
        return DraftReply(
            thread_id=str(row["thread_id"]),
            subject=str(row["subject"]),
            to=to,
            cc=[],
            body=str(row["body"]),
            attachments=(),
        )

    def _load_attachments(self, row: dict[str, Any]) -> list[OutgoingAttachment | None]:
        """Rebuild attachment metadata from the row; None when a temp file is gone.

        Binaries are never in the DB — only metadata; the temp file path is
        derived from the draft id.
        """
        try:
            items = json.loads(row["attachments_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        draft_id = int(row["id"])
        base = self._draft_tmp_dir(draft_id)
        result: list[OutgoingAttachment | None] = []
        for item in items:
            filename = str(item.get("filename") or "")
            path = base / _safe_basename(filename)
            result.append(
                OutgoingAttachment(
                    filename=filename,
                    mime_type=str(item.get("mime_type") or "application/octet-stream"),
                    size_bytes=int(item.get("size_bytes") or 0),
                    path=str(path),
                )
                if path.is_file()
                else None
            )
        return result

    # ── programmatic send (tests / future automation) ───────────────────────

    async def send_confirmed_draft(self, draft: DraftReply) -> str:
        """Send a draft that has already been confirmed elsewhere.

        Returns the new Gmail message id. Raises ``SendingDisabledError`` when
        the kill switch is active.
        """
        return await self._gmail.send_reply(draft)


class ReconciliationSweep:
    """Periodic loop: re-verify drafts stuck in sent_unverified + sweep temp
    files."""

    def __init__(
        self, coordinator: ReplyCoordinator, interval_seconds: float | None = None
    ) -> None:
        self._coordinator = coordinator
        self._interval = (
            interval_seconds
            if interval_seconds is not None
            else coordinator._settings.reconcile_sweep_interval_seconds
        )
        self._task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        while True:
            try:
                await self._coordinator.sweep_unverified()
            except Exception:
                logger.exception("reconciliation sweep failed")
            await asyncio.sleep(self._interval)

    def start(self) -> None:
        self._task = asyncio.create_task(self.run(), name="reconciliation-sweep")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


class ReplyWorker:
    """Task wrapper so the app can start/stop the reply loop cleanly."""

    def __init__(self, coordinator: ReplyCoordinator) -> None:
        self._coordinator = coordinator
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(
            self._coordinator.run_forever(), name="reply-coordinator"
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


def _safe_basename(filename: str) -> str:
    """Derive a safe temp file name from a (untrusted) display filename."""
    from pathlib import PurePath

    name = PurePath(filename.replace("\\", "/")).name
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return cleaned.strip(".") or "attachment"
