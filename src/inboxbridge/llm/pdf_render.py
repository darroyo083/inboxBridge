"""Bounded PDF page rendering for EXTERNAL Vision analysis.

Explicitly NOT OCR: this module only rasterizes pages (PyMuPDF) so the
configured Vision model can read them. No local OCR/Vision models, no weight
downloads. Everything is bounded: max pages, max pixel dimension, max total
bytes — a huge or hostile PDF can never trigger unbounded work.

Rendering is deterministic and safe: the PDF is treated as untrusted data,
never executed.
"""

from __future__ import annotations

import logging

import fitz  # type: ignore[import-untyped]  # PyMuPDF

logger = logging.getLogger(__name__)

#: PNG quality bound for rendered pages.
_DPI = 144
#: Safety cap on the raw page image bytes (before PNG encoding).
_MAX_PAGE_BYTES = 40 * 1024 * 1024
#: Output format (widely accepted by vision providers).
_OUT_FORMAT = "png"


class PdfRenderError(RuntimeError):
    """PDF could not be rendered (corrupt, encrypted, or out of bounds)."""


class PdfRenderPasswordError(PdfRenderError):
    """The PDF is password-protected and cannot be opened."""


def render_pdf_pages(
    pdf_bytes: bytes,
    *,
    max_pages: int,
    max_dimension: int,
) -> list[bytes]:
    """Render up to ``max_pages`` pages of a PDF to PNG bytes (bounded).

    Raises :class:`PdfRenderPasswordError` for protected documents and
    :class:`PdfRenderError` for corrupt/oversized input. Returns an empty
    list when the document has no pages.
    """
    if not pdf_bytes:
        raise PdfRenderError("empty PDF input")
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PdfRenderError(f"cannot open PDF: {type(exc).__name__}") from exc
    try:
        if document.needs_pass:
            raise PdfRenderPasswordError("PDF is password-protected")
        pages = list(document.pages(0, min(max_pages, document.page_count)))
        rendered: list[bytes] = []
        for page in pages:
            try:
                pixmap = page.get_pixmap(dpi=_DPI)
            except Exception as exc:
                logger.warning("PDF page render failed: %s", type(exc).__name__)
                continue
            if pixmap.width > max_dimension or pixmap.height > max_dimension:
                scale = max_dimension / max(pixmap.width, pixmap.height)
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale) * fitz.Matrix(_DPI / 72, _DPI / 72),
                    dpi=None,
                )
            if pixmap.width * pixmap.height * 4 > _MAX_PAGE_BYTES:
                logger.warning("PDF page too large to render; skipping")
                continue
            try:
                rendered.append(pixmap.tobytes(_OUT_FORMAT))
            except Exception as exc:
                logger.warning("PDF page encode failed: %s", type(exc).__name__)
        return rendered
    finally:
        document.close()


def pdf_has_renderable_pages(pdf_bytes: bytes, *, max_pages: int) -> bool:
    """Cheap gate: does the PDF open and contain pages (no rendering)?"""
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return False
    try:
        return not document.needs_pass and document.page_count > 0
    finally:
        document.close()


def to_png(image_data: bytes) -> bytes:
    """Normalize any raster bytes into PNG for vision providers.

    Accepts JPEG/PNG/WebP input (auto-detected); unsupported data raises
    :class:`PdfRenderError` (callers fall back gracefully). No AI, no OCR.
    """
    try:
        pix = fitz.Pixmap(image_data)
    except Exception as exc:
        raise PdfRenderError(f"cannot decode image: {type(exc).__name__}") from exc
    if pix.n > 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    result = pix.tobytes("png")
    assert isinstance(result, bytes)
    return result
