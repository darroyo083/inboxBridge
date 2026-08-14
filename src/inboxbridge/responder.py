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
        #: Per-draft async locks: a draft's send/verify/resend lifecycle is
        #: serialized within the process (double-tap and sweep races).
        self._draft_locks: dict[int, asyncio.Lock] = {}
        #: Draft ids currently being reconciled (sweep skips them).
        self._active_reconciles: set[int] = set()
        #: Background presentation tasks (compose/forward) so their blocking
        #: confirmation wait never stalls the Telegram message handler.
        self._presentation_tasks: set[asyncio.Task[None]] = set()
        bot.register_resend_callback(self.resend_draft)

    def _lock_for(self, draft_id: int) -> asyncio.Lock:
        lock = self._draft_locks.get(draft_id)
        if lock is None:
            lock = asyncio.Lock()
            self._draft_locks[draft_id] = lock
        return lock

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
        # Claim freshly-downloaded files into tmp/draft-<id>/ so the whole
        # attachment lifecycle is per-draft (sweepable, resendable).
        draft = self._claim_attachments(draft_id, draft)
        draft = await self._translate_for_preview(draft)
        await self._bot.send_draft_for_confirmation(
            draft, draft_id=draft_id, user_id=user_id
        )
        confirmed = await self._bot.wait_for_confirmation(draft_id)
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

    async def _translate_for_preview(self, draft: DraftReply) -> DraftReply:
        """Attach a best-effort Spanish translation of the German body.

        The translation is a display-only aid for the Telegram preview and is
        derived from the EXACT body that will be sent (never generated
        independently from the instruction). It is never persisted and never
        included in the Gmail message; any failure just omits it.
        """
        if draft.body_es or not draft.body.strip():
            return draft
        try:
            translated = (await self._llm.translate_to_spanish(draft.body)).strip()
            if translated:
                return replace(draft, body_es=translated)
        except Exception:
            logger.warning("draft translation failed; showing German-only preview")
        return draft

    async def _send_confirmed(self, draft_id: int, draft: DraftReply) -> None:
        """Drive one send attempt: sending → (verified | unverified | failed).

        The draft is atomically claimed (status → sending) so concurrent
        confirm/resend paths can never double-send.
        """
        async with self._lock_for(draft_id):
            allowed = [
                DraftStatus.CONFIRMED,
                DraftStatus.SEND_FAILED,
                DraftStatus.SENT_UNVERIFIED,
            ]
            if not self._storage.claim_draft_for_send(draft_id, allowed):
                logger.warning(
                    "draft %d not claimable for send (concurrent flow); skipping", draft_id
                )
                return
            await self._send_claimed(draft_id, draft)

    async def _send_claimed(self, draft_id: int, draft: DraftReply) -> None:
        """Send attempt for a draft already atomically claimed as SENDING."""
        started_ms = int(time.time() * 1000)
        self._storage.set_draft_send_started(draft_id, started_ms)
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
            await self._bot.offer_resend(draft_id, user_id=self._draft_owner(draft_id))
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
        self._active_reconciles.add(draft_id)
        try:
            await self._reconcile_loop(
                draft_id, draft, expected_message_id=expected_message_id, started_ms=started_ms
            )
        finally:
            self._active_reconciles.discard(draft_id)

    async def _reconcile_loop(
        self,
        draft_id: int,
        draft: DraftReply,
        *,
        expected_message_id: str,
        started_ms: int,
    ) -> None:
        attempts = 0
        max_attempts = max(1, self._settings.send_verification_attempts)
        while attempts < max_attempts:
            attempts += 1
            self._storage.bump_verification_attempts(draft_id)
            result: SendVerification = await self._gmail.verify_delivery(
                draft,
                expected_message_id=expected_message_id,
                since_ms=started_ms,
            )
            logger.info(
                "reconcile draft=%d attempt=%d/%d outcome=%s",
                draft_id, attempts, max_attempts, result.category,
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
            if attempts < max_attempts:
                await asyncio.sleep(self._settings.send_verification_backoff_seconds)

        self._storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
        if not result.checked_ok:
            await self._bot.send_notice(
                "El envío no se ha podido verificar contra Gmail todavía. "
                "No reenvío para evitar duplicados; lo comprobaré y te aviso."
            )
            return
        owner_id = self._draft_owner(draft_id)
        await self._bot.send_notice(
            "No encuentro el correo en Gmail. Para evitar duplicados no reenvío "
            "automáticamente; usa el botón si quieres reintentar."
        )
        await self._bot.offer_resend(draft_id, user_id=owner_id)

    # ── controlled retry (user-initiated, duplicate-safe) ───────────────────

    async def resend_draft(self, draft_id: int) -> None:
        """Retry a failed/uncertain draft. NEVER blind: Gmail is checked first.

        Called from the Telegram "Reintentar envío" button (chat + ownership
        already validated by the bot). The draft is atomically claimed
        (status → sending) before any work, so a double-tap or a concurrent
        sweep can never produce two emails.
        """
        async with self._lock_for(draft_id):
            await self._resend_draft_locked(draft_id)

    async def _resend_draft_locked(self, draft_id: int) -> None:
        row = self._storage.get_draft(draft_id)
        if row is None:
            await self._bot.send_notice("Ese borrador ya no existe.")
            return
        previous_status = DraftStatus(row["status"])
        if previous_status not in (DraftStatus.SEND_FAILED, DraftStatus.SENT_UNVERIFIED):
            await self._bot.send_notice("Ese borrador no está en estado de reintento.")
            return
        # Atomic claim: the first caller wins; any concurrent retry/sweep skips.
        if not self._storage.claim_draft_for_send(
            draft_id, [DraftStatus.SEND_FAILED, DraftStatus.SENT_UNVERIFIED]
        ):
            await self._bot.send_notice("Ese borrador ya está en proceso de reenvío.")
            return
        draft = self._draft_from_row(row)
        attachments = self._load_attachments(row)
        if any(a is None for a in attachments):
            self._storage.set_draft_status(draft_id, previous_status)
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
            self._storage.set_draft_sent_message(draft_id, verification.message_id)
            self._storage.set_draft_status(draft_id, DraftStatus.SENT_VERIFIED)
            await self._bot.send_notice("Ya estaba enviado (verificado ✓). No he reenviado nada.")
            self._cleanup_attachments(draft)
            return
        if not verification.checked_ok:
            self._storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
            await self._bot.send_notice(
                "No puedo confirmar el estado en Gmail ahora; no reenvío para evitar duplicados."
            )
            return
        if verification.found:
            self._storage.set_draft_status(draft_id, previous_status)
            await self._bot.send_notice(
                "Gmail tiene un mensaje en ese hilo que no coincide con el borrador; "
                "no reenvío."
            )
            return
        # Gmail evidence: the message was NOT sent → safe to send now (already
        # claimed as SENDING; keep the lock across the whole attempt).
        await self._send_claimed(draft_id, draft)

    # ── restart / periodic recovery ─────────────────────────────────────────

    async def reconcile_on_startup(self) -> None:
        """Sweep drafts left in-flight by a previous process.

        Only reconciles; never resends. Temp cleanup is left to the periodic
        sweep (which runs AFTER this pass), so a freshly-offered retry button
        never races its own attachment files.
        """
        rows = self._storage.drafts_in_statuses(list(_UNRECONCILED))
        for row in rows:
            draft_id = int(row["id"])
            status = DraftStatus(row["status"])
            if status == DraftStatus.SENDING:
                # The send may have reached Gmail; the row may lack a message id.
                self._storage.set_draft_status(draft_id, DraftStatus.SENT_UNVERIFIED)
                row = {**row, "status": DraftStatus.SENT_UNVERIFIED.value}
            draft = self._draft_from_row(row)
            logger.info("startup reconciliation for draft %d (was %s)", draft_id, status.value)
            await self._reconcile_row(row, draft, startup=True)

    async def sweep_unverified(self) -> None:
        """Periodic: re-verify drafts stuck in sent_unverified (bounded).

        Drafts whose reconciliation budget is exhausted are handed to the
        watchdog, which surfaces a definitive one-shot "could not verify"
        outcome (never a resend).
        """
        for row in self._storage.drafts_in_statuses([DraftStatus.SENT_UNVERIFIED]):
            draft_id = int(row["id"])
            if draft_id in self._active_reconciles:
                continue  # a send/verify flow already owns this draft
            attempts = int(row["verification_attempts"])
            if attempts >= self._settings.send_verification_max_attempts:
                await self._notify_exhausted(row)
                continue
            draft = self._draft_from_row(row)
            await self._reconcile_row(row, draft, startup=False)
        self.cleanup_orphan_tmp()

    async def _notify_exhausted(self, row: dict[str, Any]) -> None:
        """Definitive, one-shot 'could not verify' for an exhausted draft.

        The draft has used up its reconciliation budget while still
        inconclusive (Gmail could not be queried within it). The watchdog NEVER
        resends: it transitions ``sent_unverified → verification_failed`` and
        posts a single explanatory notice. The atomic status transition makes
        the notice one-shot across sweeps and restarts (a second pass sees the
        draft is no longer ``sent_unverified``).
        """
        draft_id = int(row["id"])
        attempts = int(row["verification_attempts"] or 0)
        async with self._lock_for(draft_id):
            current = self._storage.get_draft(draft_id)
            if current is None or current["status"] != DraftStatus.SENT_UNVERIFIED.value:
                return  # already resolved by another flow
            attempts = int(current["verification_attempts"] or 0)
            if attempts < self._settings.send_verification_max_attempts:
                return  # resolved below the threshold in the meantime
            self._storage.set_draft_status(draft_id, DraftStatus.VERIFICATION_FAILED)
        subject = str(row.get("subject") or "")
        subject_hint = f" (asunto «{subject[:60]}»)" if subject else ""
        await self._bot.send_notice(
            "No pude confirmar si Gmail aceptó este envío"
            f"{subject_hint} y ya agoté las comprobaciones automáticas. "
            "No lo he reenviado para evitar duplicados. Revisa Gmail directamente: "
            "si el correo no llegó a salir, escríbelo de nuevo."
        )
        logger.info(
            "watchdog exhausted draft=%d status=verification_failed attempts=%d",
            draft_id, attempts,
        )

    async def _reconcile_row(
        self, row: dict[str, Any], draft: DraftReply, *, startup: bool
    ) -> None:
        draft_id = int(row["id"])
        self._active_reconciles.add(draft_id)
        try:
            async with self._lock_for(draft_id):
                # A user-driven resend may have claimed the draft while we were
                # verifying; never clobber an in-flight flow's state.
                current = self._storage.get_draft(draft_id)
                if current is None or current["status"] != row["status"]:
                    return
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
                    await self._bot.send_notice(
                        f"{prefix}: envío confirmado ✓ (no he reenviado nada)."
                    )
                    self._cleanup_attachments(draft)
                elif not result.checked_ok:
                    logger.warning(
                        "reconciliation inconclusive for draft %d (Gmail unreachable)",
                        draft_id,
                    )
                elif not result.found:
                    self._storage.set_draft_status(draft_id, DraftStatus.SEND_FAILED)
                    await self._bot.send_notice(
                        "Tras revisar Gmail, el envío pendiente no llegó a salir. "
                        "Puedes reintentarlo."
                    )
                    await self._bot.offer_resend(
                        draft_id, user_id=int(row["telegram_user_id"] or 0)
                    )
                logger.info(
                    "reconcile sweep draft=%d startup=%s outcome=%s",
                    draft_id, startup, result.category,
                )
        finally:
            self._active_reconciles.discard(draft_id)

    # ── temp attachment lifecycle ───────────────────────────────────────────

    def _draft_tmp_dir(self, draft_id: int) -> Path:
        return self._tmp_dir / f"draft-{draft_id}"

    def _claim_attachments(self, draft_id: int, draft: DraftReply) -> DraftReply:
        """Move freshly downloaded files into ``tmp/draft-<id>/``.

        Claimed names are deterministic from the attachment order
        (``NN_<safe-name>``) so recovery/resend lookups reproduce them and
        same-name attachments never collide. Returns a draft whose attachment
        paths point at the claimed files.
        """
        if not draft.attachments:
            return draft
        target_dir = self._draft_tmp_dir(draft_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        claimed: list[OutgoingAttachment] = []
        for index, attachment in enumerate(draft.attachments):
            if not attachment.path:
                claimed.append(attachment)
                continue
            source = Path(attachment.path)
            target = target_dir / _claimed_name(index, attachment.filename)
            try:
                if source != target and source.is_file():
                    source.replace(target)
                claimed.append(
                    OutgoingAttachment(
                        filename=attachment.filename,
                        mime_type=attachment.mime_type,
                        size_bytes=attachment.size_bytes,
                        path=str(target),
                    )
                )
            except OSError:
                logger.warning("could not claim attachment %s", attachment.filename)
                claimed.append(attachment)
        return replace(draft, attachments=tuple(claimed))

    def _cleanup_attachments(self, draft: DraftReply) -> None:
        for attachment in draft.attachments:
            if not attachment.path:
                continue
            with contextlib.suppress(OSError):
                Path(attachment.path).unlink(missing_ok=True)
            parent = Path(attachment.path).parent
            with contextlib.suppress(OSError):
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
                    break

    def cleanup_orphan_tmp(self) -> None:
        """Remove temp files/dirs that can no longer be used.

        - terminal drafts: files survive a grace period (a freshly-posted
          retry button must not be dead on arrival);
        - PENDING/CONFIRMED drafts are swept once stale (crashed workflows,
          kill-switch leftovers);
        - unknown drafts and never-claimed downloads are swept by age.
        """
        if not self._tmp_dir.is_dir():
            return
        max_age = timedelta(seconds=self._settings.tmp_max_age_seconds)
        offer_ttl = timedelta(seconds=self._settings.resend_offer_ttl_seconds)
        now = datetime.now(UTC)
        for child in self._tmp_dir.iterdir():
            # Fresh downloads that never became drafts (e.g. crashed mid-request).
            if child.is_file():
                self._sweep_stale(child, now, max_age)
                continue
            if not child.is_dir():
                continue
            if child.name == "incoming":
                for file in child.iterdir():
                    self._sweep_stale(file, now, max_age)
                continue
            if child.name == "delivery":
                for file in child.iterdir():
                    self._sweep_stale(file, now, max_age)
                continue
            if not child.name.startswith("draft-"):
                continue
            try:
                draft_id = int(child.name.removeprefix("draft-"))
            except ValueError:
                continue
            row = self._storage.get_draft(draft_id)
            if row is None:
                self._sweep_stale(child, now, max_age)  # unknown draft → stale
                continue
            status = row["status"]
            if status in (
                DraftStatus.SENT_VERIFIED.value,
                DraftStatus.SEND_FAILED.value,
                DraftStatus.VERIFICATION_FAILED.value,
                DraftStatus.CANCELLED.value,
                DraftStatus.REJECTED.value,
            ):
                # Terminal: keep files only while a resend offer could still be
                # acted on; the dir's own age is the grace clock.
                self._sweep_stale(child, now, offer_ttl)
                continue
            # PENDING/CONFIRMED can only legitimately stay that way briefly;
            # stale ones are crash/kill-switch leftovers → sweep by age.
            self._sweep_stale(child, now, offer_ttl)

    @staticmethod
    def _sweep_stale(path: Path, now: datetime, max_age: timedelta) -> None:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            return
        if now - mtime > max_age:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)

    # ── row → model reconstruction ──────────────────────────────────────────

    def _draft_owner(self, draft_id: int) -> int:
        row = self._storage.get_draft(draft_id)
        return int(row["telegram_user_id"] or 0) if row else 0

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
        for index, item in enumerate(items):
            filename = str(item.get("filename") or "")
            path = base / _claimed_name(index, filename)
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

    async def present_draft(self, draft: DraftReply, *, user_id: int = 0) -> None:
        """V1.1 hook: present ANY draft (reply/compose/forward) through the
        shared verified-delivery confirmation path.

        Runs in a background task so the Telegram message handler returns
        immediately. The preview's SEND/EDIT/CANCEL buttons are callback queries
        that PTB processes sequentially; blocking here (in ``wait_for_confirmation``)
        would deadlock them — the compose/forward flow used to hit exactly that.
        """
        async def _run() -> None:
            try:
                await self._present_draft(draft, user_id=user_id)
            except Exception:
                logger.exception("draft presentation failed for thread %s", draft.thread_id)
                await self._bot.send_notice(
                    "No pude preparar el borrador; inténtalo de nuevo."
                )

        task = asyncio.create_task(_run())
        self._presentation_tasks.add(task)
        task.add_done_callback(self._presentation_tasks.discard)


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


def _claimed_name(index: int, filename: str) -> str:
    """Deterministic per-draft claimed file name (order-prefixed, collision-free)."""
    return f"{index + 1:02d}_{_safe_basename(filename)}"
