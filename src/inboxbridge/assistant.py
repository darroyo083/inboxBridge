"""EmailAssistant — V1.1 natural-language flows over the safe action boundary.

Every user text goes through the intent classifier; this module executes the
VALIDATED intents with deterministic state machines. The LLM never gains
authority: it produces drafts/answers/edits; sending, cancelling, archiving
and contact mutation stay deterministic application logic.

Flows implemented here:

- draft edit / regenerate (re-render preview, versioned, never stale-send)
- Q&A about one email/thread (bounded context)
- thread summary
- Gmail attachment delivery to Telegram (temp, bounded, cleaned)
- mark read / archive (exact Gmail item from Telegram context)
- reminders (create/list/cancel; fired by ReminderScheduler)
- contacts CRUD + aliases (explicit confirmations for persistent changes)
- new-email compose and forwarding (same verified-delivery pipeline)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Settings
from .contacts import ContactError, ContactService, Resolution
from .db import Storage
from .gmail.client import GmailClient
from .llm import prompts
from .llm.ai_service import AIService
from .llm.base import LLMError
from .models import DraftReply, EmailAddress, ParsedEmail
from .reminders import ReminderParseError, ReminderService
from .telegram.bot import TelegramBot

logger = logging.getLogger(__name__)

#: Temp delivery of Gmail attachments: bounded count and bytes.
MAX_ATTACHMENT_DELIVERY_BYTES = 10 * 1024 * 1024
MAX_ATTACHMENT_DELIVERY_COUNT = 5
#: Supported types for attachment delivery to Telegram.
_DELIVERABLE_TYPES = ("application/pdf", "application/msword", "text/plain", "text/csv", "image/")


class AssistantError(RuntimeError):
    """User-safe flow failure (message shown as-is, no internals)."""


class EmailAssistant:
    """Executes validated V1.1 intents against the real application stack."""

    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        gmail: GmailClient,
        ai: AIService,
        bot: TelegramBot,
        contacts: ContactService,
        reminders: ReminderService,
        *,
        draft_presenter: Any | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._gmail = gmail
        self._ai = ai
        self._bot = bot
        self._contacts = contacts
        self._reminders = reminders
        #: Coordinator hook that presents a draft for confirmation (the shared
        #: verified-delivery path). Injected by app wiring to avoid cycles.
        self._draft_presenter = draft_presenter

    @property
    def contacts(self) -> ContactService:
        return self._contacts

    @property
    def reminders(self) -> ReminderService:
        return self._reminders

    @property
    def ai(self) -> AIService:
        return self._ai

    def set_draft_presenter(self, presenter: Any) -> None:
        """Wire the coordinator's shared verified-delivery draft presenter."""
        self._draft_presenter = presenter

    async def transcribe_audio(self, mime: str, data: bytes) -> str:
        """Experimental voice transcription (gated by ai_audio_enabled)."""
        return await self._ai.audio(mime, data)

    # ── dispatcher (called by the bot with validated intents) ───────────────

    async def handle(self, action: str, payload: dict[str, Any]) -> None:
        handler = getattr(self, f"_act_{action}", None)
        if handler is None:
            logger.warning("assistant: unknown action %s", action)
            await self._bot.send_notice("No puedo hacer eso todavía.")
            return
        await handler(payload)

    # ── draft editing / regeneration ────────────────────────────────────────

    async def _act_edit_draft(self, payload: dict[str, Any]) -> None:
        draft_id = int(payload.get("draft_id") or 0)
        instruction = str(payload.get("instruction") or "").strip()
        if not draft_id or not instruction:
            await self._bot.send_notice("No tengo un borrador activo para editar.")
            return
        pending = self._bot.pending_draft_for_owner(draft_id, int(payload.get("user_id") or 0))
        if pending is None:
            await self._bot.send_notice("Ese borrador ya no está activo.")
            return
        try:
            thread = await self._gmail.fetch_thread_context(pending.draft.thread_id)
            await self._bot.send_typing()
            messages = prompts.edit_draft_messages(pending.draft.body, instruction, thread)
            new_body = await self._ai.text(
                messages,
                max_tokens=self._settings.llm_max_tokens_draft,
                task="draft_edit",
            )
        except LLMError:
            await self._bot.send_notice("No pude editar el borrador ahora; inténtalo otra vez.")
            return
        updated = DraftReply(
            thread_id=pending.draft.thread_id,
            subject=pending.draft.subject,
            to=pending.draft.to,
            cc=pending.draft.cc,
            body=new_body,
            in_reply_to=pending.draft.in_reply_to,
            references=pending.draft.references,
            attachments=pending.draft.attachments,
        )
        await self._bot.apply_draft_edit(pending.draft_id, updated)
        # Persisted draft row keeps the latest body (retry coherence).
        self._storage.set_draft_body(draft_id, new_body)
        await self._bot.send_notice("Borrador actualizado.")

    async def _act_regenerate_draft(self, payload: dict[str, Any]) -> None:
        payload["instruction"] = "reescribe el borrador por completo, misma intención"
        await self._act_edit_draft(payload)

    # ── Q&A / thread summary ────────────────────────────────────────────────

    async def _act_ask_about_email(self, payload: dict[str, Any]) -> None:
        thread_id = str(payload.get("thread_id") or "")
        question = str(payload.get("instruction") or "").strip()
        if not thread_id:
            await self._bot.send_notice("Sobre qué correo? Responde a un resumen de InboxBridge.")
            return
        try:
            thread = await self._gmail.fetch_thread_context(thread_id)
            await self._bot.send_typing()
            answer = await self._ai.text(
                prompts.ask_about_email_messages(question or "¿Qué me está pidiendo?", thread),
                max_tokens=600,
                task="qa",
            )
        except LLMError:
            await self._bot.send_notice("No pude responder ahora; inténtalo otra vez.")
            return
        await self._bot.send_notice(_cap(answer, 1800))

    async def _act_summarize_thread(self, payload: dict[str, Any]) -> None:
        thread_id = str(payload.get("thread_id") or "")
        if not thread_id:
            await self._bot.send_notice("¿Qué hilo? Responde a un resumen de InboxBridge.")
            return
        try:
            thread = await self._gmail.fetch_thread_context(thread_id)
            await self._bot.send_typing()
            summary = await self._ai.text(
                prompts.summarize_thread_messages(thread), max_tokens=600, task="thread_summary"
            )
        except LLMError:
            await self._bot.send_notice("No pude resumir el hilo ahora; inténtalo otra vez.")
            return
        await self._bot.send_notice(_cap(summary, 1800))

    # ── Gmail attachment delivery to Telegram ───────────────────────────────

    async def _act_get_attachment(self, payload: dict[str, Any]) -> None:
        tg_message_id = int(payload.get("tg_message_id") or 0)
        index = int(payload.get("index") or -1)  # -1 → list panel
        message_id = self._storage.get_meta(f"tgm:{tg_message_id}") or ""
        if not message_id:
            await self._bot.send_notice("No puedo asociar eso a ningún correo.")
            return
        try:
            email = await self._gmail.fetch_message(message_id)
        except Exception:
            logger.exception("attachment delivery: fetch failed for %s", message_id)
            await self._bot.send_notice("No pude recuperar el correo ahora.")
            return
        if not email.attachments:
            await self._bot.send_notice("Ese correo no tiene adjuntos.")
            return
        if index == -1:
            await self._bot.show_attachments_panel(tg_message_id, email)
            return
        if index < 0 or index >= len(email.attachments):
            await self._bot.send_notice("Ese adjunto ya no existe.")
            return
        await self._deliver_attachment(tg_message_id, email, index)

    async def _deliver_attachment(
        self, tg_message_id: int, email: ParsedEmail, index: int
    ) -> None:
        attachment = email.attachments[index]
        if attachment.size_bytes > MAX_ATTACHMENT_DELIVERY_BYTES:
            await self._bot.send_notice(f"{attachment.filename} es demasiado grande para enviarlo.")
            return
        mime = attachment.mime_type
        if not any(mime.startswith(t) for t in _DELIVERABLE_TYPES):
            await self._bot.send_notice(
                f"No puedo enviar adjuntos de tipo {mime or 'desconocido'}."
            )
            return
        data = await self._gmail.fetch_attachment_bytes(email.message_id, index)
        if data is None:
            await self._bot.send_notice("No pude leer ese adjunto.")
            return
        path = self._bot.write_temp_file(attachment.filename, data)
        try:
            await self._bot.send_document_file(path, attachment.filename)
        finally:
            _remove_file(path)

    # ── mark read / archive ─────────────────────────────────────────────────

    async def _act_mark_read(self, payload: dict[str, Any]) -> None:
        await self._gmail_action(payload, "read")

    async def _act_archive(self, payload: dict[str, Any]) -> None:
        await self._gmail_action(payload, "archive")

    async def _gmail_action(self, payload: dict[str, Any], action: str) -> None:
        message_id = str(payload.get("message_id") or "")
        if not message_id:
            await self._bot.send_notice("¿Qué correo? Responde a un resumen de InboxBridge.")
            return
        try:
            if action == "read":
                await self._gmail.modify_labels(message_id, remove_labels=["UNREAD"])
                await self._bot.send_notice("Marcado como leído ✓")
            else:
                await self._gmail.modify_labels(message_id, remove_labels=["INBOX"])
                await self._bot.send_notice("Archivado ✓")
        except Exception:
            logger.exception("gmail action %s failed for %s", action, message_id)
            await self._bot.send_notice("No pude hacer eso con Gmail ahora; inténtalo otra vez.")

    # ── reminders ───────────────────────────────────────────────────────────

    async def _act_create_reminder(self, payload: dict[str, Any]) -> None:
        instruction = str(payload.get("instruction") or "").strip()
        if not instruction:
            await self._bot.send_notice("¿Qué te recuerdo y cuándo?")
            return
        thread_id = str(payload.get("thread_id") or "")
        message_id = str(payload.get("message_id") or "")
        user_id = int(payload.get("user_id") or 0)
        try:
            created = self._reminders.create(
                message_id=message_id,
                thread_id=thread_id,
                telegram_user_id=user_id,
                instruction=instruction,
            )
        except ReminderParseError:
            await self._bot.send_notice(
                "No entendí para cuándo. Dime por ejemplo: 'en dos horas', "
                "'mañana', 'el viernes a las 18:00'."
            )
            return
        from .reminders import format_due

        due_text = format_due(created.due_at)
        await self._bot.send_notice(f"⏰ Recordatorio guardado para el {due_text}.")

    async def _act_list_reminders(self, payload: dict[str, Any]) -> None:
        await self._bot.show_reminders(int(payload.get("user_id") or 0))

    async def _act_cancel_reminder(self, payload: dict[str, Any]) -> None:
        reminder_id = int(payload.get("reminder_id") or 0)
        user_id = int(payload.get("user_id") or 0)
        if not reminder_id:
            await self._bot.show_reminders(user_id)
            return
        if self._reminders.cancel(reminder_id, user_id):
            await self._bot.send_notice("Recordatorio cancelado.")
        else:
            await self._bot.send_notice("Ese recordatorio ya no está pendiente.")

    # ── contacts ────────────────────────────────────────────────────────────

    async def _act_list_contacts(self, payload: dict[str, Any]) -> None:
        await self._bot.show_contacts_panel()

    async def _act_contact_create(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("name") or "").strip()
        email = str(payload.get("email") or "").strip()
        try:
            contact = self._contacts.create_contact(name, email)
        except ContactError as exc:
            await self._bot.send_notice(str(exc))
            return
        saved = f"{contact['display_name']} <{contact['email']}>"
        await self._bot.send_notice(f"Contacto guardado: {saved}")

    async def _act_contact_update_email(self, payload: dict[str, Any]) -> None:
        try:
            contact = self._contacts.change_email(int(payload["contact_id"]), str(payload["email"]))
        except ContactError as exc:
            await self._bot.send_notice(str(exc))
            return
        await self._bot.send_notice(
            f"Correo actualizado: {contact['display_name']} <{contact['email']}>"
        )

    async def _act_contact_rename(self, payload: dict[str, Any]) -> None:
        try:
            contact = self._contacts.rename(int(payload["contact_id"]), str(payload["name"]))
        except ContactError as exc:
            await self._bot.send_notice(str(exc))
            return
        await self._bot.send_notice(f"Nombre actualizado: {contact['display_name']}")

    async def _act_contact_delete(self, payload: dict[str, Any]) -> None:
        self._contacts.delete(int(payload["contact_id"]))
        await self._bot.send_notice("Contacto borrado.")

    async def _act_contact_add_alias(self, payload: dict[str, Any]) -> None:
        try:
            alias = self._contacts.add_alias(int(payload["contact_id"]), str(payload["alias"]))
        except ContactError as exc:
            await self._bot.send_notice(str(exc))
            return
        await self._bot.send_notice(f"Alias guardado: «{alias}»")

    async def _act_contact_remove_alias(self, payload: dict[str, Any]) -> None:
        removed = self._contacts.remove_alias(str(payload["alias"]))
        if removed:
            await self._bot.send_notice("Alias eliminado.")
        else:
            await self._bot.send_notice("Ese alias no existe.")

    # ── compose / forward ───────────────────────────────────────────────────

    async def _act_compose(self, payload: dict[str, Any]) -> None:
        recipient_phrase = str(payload.get("recipient") or "").strip()
        instruction = str(payload.get("instruction") or "").strip()
        user_id = int(payload.get("user_id") or 0)
        contact = await self._resolve_recipient(recipient_phrase, user_id)
        if contact is None:
            return  # ambiguity/unknown already handled (asked)
        await self._bot.send_typing()
        try:
            body = await self._ai.text(
                prompts.compose_messages(
                    contact["display_name"],
                    instruction or "Saluda y presenta el asunto.",
                ),
                max_tokens=self._settings.llm_max_tokens_draft,
                task="compose",
            )
        except LLMError:
            await self._bot.send_notice("No pude redactar el correo ahora; inténtalo otra vez.")
            return
        subject = _default_subject(instruction)
        draft = DraftReply(
            thread_id="",
            subject=subject,
            to=[EmailAddress(contact["display_name"], contact["email"])],
            cc=[],
            body=body,
        )
        await self._present_new_draft(draft, user_id=user_id)

    async def _act_forward(self, payload: dict[str, Any]) -> None:
        recipient_phrase = str(payload.get("recipient") or "").strip()
        tg_message_id = int(payload.get("tg_message_id") or 0)
        user_id = int(payload.get("user_id") or 0)
        contact = await self._resolve_recipient(recipient_phrase, user_id)
        if contact is None:
            return
        gmail_message_id = self._storage.get_meta(f"tgm:{tg_message_id}") or ""
        if not gmail_message_id:
            await self._bot.send_notice("No puedo asociar eso a ningún correo.")
            return
        try:
            original = await self._gmail.fetch_message(gmail_message_id)
            await self._bot.send_typing()
            body = await self._ai.text(
                prompts.forward_body_messages(original),
                max_tokens=self._settings.llm_max_tokens_draft,
                task="forward",
            )
        except LLMError:
            await self._bot.send_notice("No pude preparar el reenvío ahora; inténtalo otra vez.")
            return
        subject = f"Fwd: {original.subject}"
        draft = DraftReply(
            thread_id="",
            subject=subject,
            to=[EmailAddress(contact["display_name"], contact["email"])],
            cc=[],
            body=body,
        )
        await self._present_new_draft(draft, user_id=user_id)

    async def _present_new_draft(self, draft: DraftReply, *, user_id: int) -> None:
        """Present a new-email/forward draft through the SHARED verified path
        (preview with real addresses → explicit send → reconcile)."""
        if self._draft_presenter is None:
            await self._bot.send_notice("No puedo preparar ese correo ahora.")
            return
        await self._draft_presenter(draft, user_id=user_id)

    async def _resolve_recipient(self, phrase: str, user_id: int) -> dict[str, Any] | None:
        """Contact resolution: unique → contact; ambiguous → ask; unknown → ask.

        The LLM NEVER invents an address — resolution is deterministic.
        """
        phrase = phrase.strip().strip(".,;:!?¿¡")
        # Bare valid email → direct destination (still shown in preview).
        from .contacts import validate_email

        if validate_email(phrase):
            return {"display_name": phrase.split("@")[0], "email": phrase.casefold()}
        if not phrase:
            await self._bot.prompt_compose_recipient(user_id)
            return None
        resolution: Resolution = self._contacts.resolve(phrase)
        if resolution.resolved:
            assert resolution.contact is not None
            return resolution.contact
        if resolution.ambiguous:
            await self._bot.choose_candidate(
                user_id,
                "Hay varios contactos que coinciden; dime cuál:",
                list(resolution.candidates),
                flow="compose",
            )
            return None
        await self._bot.ask_unknown_recipient(
            user_id, phrase, "No tengo a nadie guardado como"
        )
        return None

    # ── help ────────────────────────────────────────────────────────────────

    async def _act_help(self, payload: dict[str, Any]) -> None:
        await self._bot.show_help()


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 30] + "\n[…truncado…]"


def _default_subject(instruction: str) -> str:
    """Heuristic subject from the instruction; the user can change it later."""
    for word in instruction.split():
        if "@" in word:
            continue
    subject = " ".join(instruction.split()[:6])
    return subject[:80] or "Sin asunto"


def _remove_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("could not remove temp file %s", path)
