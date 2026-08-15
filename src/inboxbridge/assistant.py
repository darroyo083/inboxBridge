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

import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import Settings
from .contacts import ContactError, ContactService, Resolution
from .db import Storage
from .gmail.client import GmailClient
from .llm import prompts
from .llm.ai_service import AIService
from .llm.base import LLMError, call_with_retry
from .models import DraftReply, EmailAddress, OutgoingAttachment, ParsedEmail, ThreadContext
from .reminders import ReminderParseError, ReminderService
from .telegram.bot import TelegramBot

logger = logging.getLogger(__name__)

#: Temp delivery of Gmail attachments: bounded count and bytes.
MAX_ATTACHMENT_DELIVERY_BYTES = 10 * 1024 * 1024
MAX_ATTACHMENT_DELIVERY_COUNT = 5
#: Supported types for attachment delivery to Telegram (docs claim DOCX works;
#: modern .docx uses the OpenXML mime, older .doc uses application/msword).
_DELIVERABLE_TYPES = (
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/csv",
    "image/",
)


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

    #: Intent action values → handler suffixes (keeps handler names readable).
    _ALIASES = {
        "compose_new_email": "compose",
        "forward_email": "forward",
        "create_contact": "contact_create",
        "update_contact": "contact_update",
        "delete_contact": "contact_delete",
        "add_alias": "contact_add_alias",
        "remove_alias": "contact_remove_alias",
        "get_attachment": "get_attachment",
        "ask_about_email": "ask_about_email",
        "summarize_thread": "summarize_thread",
        "mark_read": "mark_read",
        "archive": "archive",
        "create_reminder": "create_reminder",
        "list_reminders": "list_reminders",
        "cancel_reminder": "cancel_reminder",
        "list_contacts": "list_contacts",
        "help": "help",
    }

    async def handle(self, action: str, payload: dict[str, Any]) -> None:
        handler_name = self._ALIASES.get(action, action)
        handler = getattr(self, f"_act_{handler_name}", None)
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
            if pending.draft.thread_id:
                thread = await self._gmail.fetch_thread_context(pending.draft.thread_id)
            else:
                # Compose/forward drafts have no thread; the edit prompt just
                # omits thread context rather than failing on an empty id.
                thread = ThreadContext(
                    thread_id="", subject=pending.draft.subject, messages=[], history_id=0
                )
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
        # Regenerate the display-only Spanish translation from the NEW German
        # body, so the preview never shows a stale translation. Bounded retry
        # (one extra attempt) covers transient LLMEmptyResponse; the German
        # body is NEVER regenerated here.
        try:
            new_body_es = (
                await call_with_retry(
                    lambda: self._ai.translate_to_spanish(new_body),
                    max_attempts=2,
                    base_backoff=self._settings.retry_backoff_base,
                )
            ).strip()
            translation_failed = False
        except Exception:
            logger.warning("draft translation failed after retry")
            new_body_es = ""
            translation_failed = True
        updated = DraftReply(
            thread_id=pending.draft.thread_id,
            subject=pending.draft.subject,
            to=pending.draft.to,
            cc=pending.draft.cc,
            body=new_body,
            in_reply_to=pending.draft.in_reply_to,
            references=pending.draft.references,
            attachments=pending.draft.attachments,
            body_es=new_body_es,
            translation_failed=translation_failed,
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
        # Distinguish "no index" (→ list panel) from attachment #0, which is a
        # real, deliverable index. ``payload.get("index") or -1`` would wrongly
        # treat 0 as "no index" and re-show the panel forever.
        raw_index = payload.get("index")
        index = -1 if raw_index is None else int(raw_index)
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
            # Natural-language request: deliver a matching attachment directly
            # ("mándame el pdf" → first pdf); otherwise show the panel.
            wanted = _attachment_kind(str(payload.get("instruction") or ""))
            if wanted:
                for att_index, att in enumerate(email.attachments):
                    if wanted(att):
                        await self._deliver_attachment(tg_message_id, email, att_index)
                        return
                await self._bot.send_notice("No encontré un adjunto de ese tipo.")
                return
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
        instruction = str(payload.get("instruction") or "").strip()
        owner = int(payload.get("user_id") or 0)
        if not name or not email:
            parsed = _parse_contact_instruction(instruction)
            if parsed is None:
                await self._bot.send_notice(
                    "Dime el nombre y el correo, por ejemplo: "
                    "«cuando diga Roman usa femo@femo.ch»."
                )
                return
            name, email = parsed
        await self._bot.request_confirmation(
            f"¿Guardo el contacto {name} <{email}>?",
            "contact_create_confirm",
            {"name": name, "email": email},
            user_id=owner,
        )

    async def _act_contact_update(self, payload: dict[str, Any]) -> None:
        """NL 'cambia el correo de X a Y' / 'cambia el nombre de X a Y'."""
        instruction = str(payload.get("instruction") or "").strip()
        parsed = _parse_contact_update(instruction)
        if parsed is None:
            await self._bot.send_notice(
                "Dime qué cambio, por ejemplo: «cambia el correo de Roman a roman@femo.ch»."
            )
            return
        kind, target, value = parsed
        resolution = self._contacts.resolve(target)
        if not resolution.resolved:
            await self._bot.send_notice(f"No tengo a nadie guardado como «{target}».")
            return
        assert resolution.contact is not None
        contact_id = int(resolution.contact["id"])
        if kind == "email":
            await self._bot.request_confirmation(
                f"¿Cambio el correo de {resolution.contact['display_name']} a {value}?",
                "contact_update_email",
                {"contact_id": contact_id, "email": value},
            )
        else:
            await self._bot.request_confirmation(
                f"¿Renombro {resolution.contact['display_name']} a «{value}»?",
                "contact_rename",
                {"contact_id": contact_id, "name": value},
            )

    async def _act_contact_delete(self, payload: dict[str, Any]) -> None:
        instruction = str(payload.get("instruction") or "").strip()
        name = str(payload.get("name") or "").strip() or _parse_contact_delete(instruction)
        if not name:
            await self._bot.send_notice("¿Qué contacto quieres borrar?")
            return
        resolution = self._contacts.resolve(name)
        if not resolution.resolved:
            await self._bot.send_notice(f"No tengo a nadie guardado como «{name}».")
            return
        assert resolution.contact is not None
        await self._bot.request_confirmation(
            f"¿Borro el contacto {resolution.contact['display_name']} y sus alias?",
            "contact_delete_confirm",
            {"contact_id": int(resolution.contact["id"])},
        )

    async def _act_contact_add_alias(self, payload: dict[str, Any]) -> None:
        instruction = str(payload.get("instruction") or "").strip()
        alias = str(payload.get("alias") or "").strip()
        contact_phrase = str(payload.get("contact") or "").strip()
        if not alias or not contact_phrase:
            parsed = _parse_alias_instruction(instruction, add=True)
            if parsed is None:
                await self._bot.send_notice(
                    "Dime qué alias añadir y a quién, por ejemplo: "
                    "«añade mi jefe como alias de Roman»."
                )
                return
            alias, contact_phrase = parsed
        resolution = self._contacts.resolve(contact_phrase)
        if not resolution.resolved:
            await self._bot.send_notice(f"No tengo a nadie guardado como «{contact_phrase}».")
            return
        assert resolution.contact is not None
        await self._bot.request_confirmation(
            f"¿Guardo «{alias}» como alias de {resolution.contact['display_name']}?",
            "contact_add_alias_confirm",
            {"contact_id": int(resolution.contact["id"]), "alias": alias},
        )

    async def _act_contact_remove_alias(self, payload: dict[str, Any]) -> None:
        instruction = str(payload.get("instruction") or "").strip()
        alias = str(payload.get("alias") or "").strip()
        if not alias:
            parsed = _parse_alias_instruction(instruction, add=False)
            if parsed is None:
                await self._bot.send_notice("¿Qué alias quieres quitar?")
                return
            alias, _contact_phrase = parsed
        await self._bot.request_confirmation(
            f"¿Elimino el alias «{alias}»?",
            "contact_remove_alias_confirm",
            {"alias": alias},
        )

    async def _act_contact_create_confirm(self, payload: dict[str, Any]) -> None:
        try:
            contact = self._contacts.create_contact(
                str(payload.get("name") or ""), str(payload.get("email") or "")
            )
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
        updated = f"{contact['display_name']} <{contact['email']}>"
        await self._bot.send_notice(f"Correo actualizado: {updated}")

    async def _act_contact_rename(self, payload: dict[str, Any]) -> None:
        try:
            contact = self._contacts.rename(int(payload["contact_id"]), str(payload["name"]))
        except ContactError as exc:
            await self._bot.send_notice(str(exc))
            return
        await self._bot.send_notice(f"Nombre actualizado: {contact['display_name']}")

    async def _act_contact_delete_confirm(self, payload: dict[str, Any]) -> None:
        self._contacts.delete(int(payload["contact_id"]))
        await self._bot.send_notice("Contacto borrado.")

    async def _act_contact_add_alias_confirm(self, payload: dict[str, Any]) -> None:
        try:
            alias = self._contacts.add_alias(int(payload["contact_id"]), str(payload["alias"]))
        except ContactError as exc:
            await self._bot.send_notice(str(exc))
            return
        await self._bot.send_notice(f"Alias guardado: «{alias}»")

    async def _act_contact_remove_alias_confirm(self, payload: dict[str, Any]) -> None:
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
            content = await self._ai.text(
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
        subject, body = _parse_compose(content)
        if not body:
            await self._bot.send_notice("No pude redactar el correo ahora; inténtalo otra vez.")
            return
        if not subject:
            subject = _FALLBACK_SUBJECT  # safe deterministic fallback, never the command
        # Defensive: never leak the recipient's address into the subject.
        subject = _strip_recipient_from_subject(subject, contact["email"])
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
        # Include the original's attachments (bounded; the preview shows them
        # and the send pipeline verifies them against Gmail).
        attachments = await self._collect_original_attachments(original)
        draft = DraftReply(
            thread_id="",
            subject=subject,
            to=[EmailAddress(contact["display_name"], contact["email"])],
            cc=[],
            body=body,
            attachments=attachments,
        )
        await self._present_new_draft(draft, user_id=user_id)

    async def _collect_original_attachments(
        self, original: ParsedEmail
    ) -> tuple[OutgoingAttachment, ...]:
        """Claim the original's supported attachments as temp files (bounded)."""
        collected: list[OutgoingAttachment] = []
        for index, meta in enumerate(original.attachments):
            if len(collected) >= self._settings.outgoing_attachment_max_count:
                break
            if meta.size_bytes > self._settings.outgoing_attachment_max_bytes:
                continue
            if not any(
                meta.mime_type.startswith(t) for t in _DELIVERABLE_TYPES
            ) and not meta.mime_type.startswith("image/"):
                continue
            data = await self._gmail.fetch_attachment_bytes(original.message_id, index)
            if data is None:
                continue
            path = self._bot.write_temp_file(meta.filename, data)
            collected.append(
                OutgoingAttachment(
                    filename=meta.filename,
                    mime_type=meta.mime_type,
                    size_bytes=len(data),
                    path=path,
                )
            )
        return tuple(collected)

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
        # Accept a display form "Name <email>" (e.g. from candidate selection).
        import re as _re

        display_match = _re.search(r"<([^<>@\s]+@[^<>\s]+)>", phrase)
        if display_match:
            phrase = display_match.group(1)
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


def _attachment_kind(instruction: str) -> Any | None:
    """Return a matcher for a requested attachment kind, or None when the
    request is generic (panel)."""
    import re

    lowered = instruction.casefold()
    if re.search(r"\b(pdf|documento)\b", lowered):
        def is_pdf(att: Any) -> bool:
            return bool(
                att.mime_type == "application/pdf"
                or att.filename.casefold().endswith(".pdf")
            )

        return is_pdf
    if re.search(r"\b(foto|im[aá]gen(es)?)\b", lowered):
        return lambda att: att.mime_type.startswith("image/")
    if re.search(r"\b(adjuntos?|archivos?)\b", lowered):
        return None  # generic → panel
    return None


# ── NL contact-instruction parsing (deterministic; never LLM-invented) ───────


def _parse_contact_instruction(text: str) -> tuple[str, str] | None:
    """'cuando diga X usa a@b.ch' / 'guarda a X como a@b.ch' / 'añade a X con correo a@b.ch'."""
    import re

    patterns = (
        r"cuando diga (.+?)\s+usa\s+(\S+@\S+)",
        r"guarda a (.+?)\s+como\s+(\S+@\S+)",
        r"a[nñ]ade a (.+?)\s+(?:con correo|como)?\s*(\S+@\S+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip().strip("'\",.;:!?¿¡")
            email = match.group(2).strip().strip("'\",.;:!?¿¡")
            if name and email:
                return name, email
    return None


def _parse_contact_update(text: str) -> tuple[str, str, str] | None:
    """'cambia el correo de X a a@b.ch' → ('email', X, addr)
    'cambia el nombre de X a Y' → ('name', X, Y)."""
    import re

    match = re.search(
        r"cambia el (correo|email|direcci[oó]n|nombre) de (.+?)\s+(?:a|por)\s+(.+)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    kind = match.group(1).lower()
    kind = "email" if kind in ("correo", "email", "dirección", "direccion") else "name"
    target = match.group(2).strip().strip("'\",.;:!?¿¡")
    value = match.group(3).strip().strip("'\",.;:!?¿¡")
    if not target or not value:
        return None
    return kind, target, value


def _parse_contact_delete(text: str) -> str:
    import re

    match = re.search(r"(?:borra|elimina|quita) el contacto (.+)", text, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip().strip("'\",.;:!?¿¡")


def _parse_alias_instruction(text: str, *, add: bool) -> tuple[str, str] | None:
    """'añade X como alias de Y' (add) / 'quita el alias X de Y' (remove)."""
    import re

    if add:
        match = re.search(r"a[nñ]ade (.+?)\s+como alias de (.+)", text, re.IGNORECASE)
        if not match:
            match = re.search(r"pon (.+?)\s+como alias de (.+)", text, re.IGNORECASE)
    else:
        match = re.search(r"quita el alias (.+?)\s+de (.+)", text, re.IGNORECASE)
        if not match:
            match = re.search(r"(?:quita|borra|elimina) el alias (.+)", text, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip("'\",.;:!?¿¡"), ""
    if not match:
        return None
    alias = match.group(1).strip().strip("'\",.;:!?¿¡")
    contact = (
        match.group(2).strip().strip("'\",.;:!?¿¡")
        if match.lastindex and match.lastindex >= 2
        else ""
    )
    return alias, contact


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

#: Safe deterministic fallback subject when the LLM does not produce one.
_FALLBACK_SUBJECT = "Kein Betreff"


def _parse_compose(content: str) -> tuple[str, str]:
    """Parse the compose LLM JSON ``{"subject_de": ..., "body_de": ...}``.

    Tolerant like the summary parser: when the output is not valid JSON, the
    whole text becomes the body and the subject is empty (the caller applies
    the safe fallback). The raw bot command NEVER becomes the subject.
    Returns ``(subject, body)``.
    """
    match = _JSON_BLOCK_RE.search(content)
    if match is not None:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            body = payload.get("body_de")
            subject = payload.get("subject_de")
            if isinstance(body, str) and body.strip():
                subject = subject if isinstance(subject, str) else ""
                return subject.strip(), body.strip()
    return "", content.strip()


def _strip_recipient_from_subject(subject: str, recipient_email: str) -> str:
    """Never leak the recipient's email address into the generated subject."""
    if not subject or not recipient_email:
        return subject
    cleaned = re.sub(
        re.escape(recipient_email), "", subject, flags=re.IGNORECASE
    ).strip(" .,;:!?¿¡")
    return cleaned[:80] or _FALLBACK_SUBJECT


def _remove_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("could not remove temp file %s", path)
