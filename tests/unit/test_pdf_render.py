"""Bounded PDF page rendering (external vision input) — no local OCR."""

from __future__ import annotations

import fitz  # PyMuPDF

from inboxbridge.llm.pdf_render import (
    PdfRenderError,
    PdfRenderPasswordError,
    render_pdf_pages,
    to_png,
)


def _make_pdf(pages: int = 2, *, password: str | None = None) -> bytes:
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), "Synthetic PDF content for tests")
    if password:
        document.save(
            "mem.pdf",
            encryption=fitz.PDF_ENCRYPT_AES_256,
            user_pw=password,
            owner_pw=password,
        )
        with open("mem.pdf", "rb") as fh:
            data = fh.read()
        import os

        os.remove("mem.pdf")
        return data
    data = document.tobytes()
    document.close()
    return data


class TestRenderPdfPages:
    def test_renders_bounded_pages_as_png(self) -> None:
        pages = render_pdf_pages(_make_pdf(3), max_pages=5, max_dimension=2000)
        assert len(pages) == 3
        for page in pages:
            assert page.startswith(b"\x89PNG")  # PNG magic

    def test_max_pages_is_honored(self) -> None:
        pages = render_pdf_pages(_make_pdf(10), max_pages=5, max_dimension=2000)
        assert len(pages) == 5

    def test_max_dimension_scales_down(self) -> None:
        pages = render_pdf_pages(_make_pdf(1), max_pages=2, max_dimension=64)
        assert len(pages) == 1

    def test_password_protected_raises_typed_error(self) -> None:
        try:
            render_pdf_pages(_make_pdf(1, password="secret"), max_pages=2, max_dimension=2000)
        except PdfRenderPasswordError:
            pass
        else:
            raise AssertionError("expected PdfRenderPasswordError")

    def test_corrupt_pdf_raises(self) -> None:
        try:
            render_pdf_pages(b"not a pdf at all", max_pages=2, max_dimension=2000)
        except PdfRenderError:
            pass
        else:
            raise AssertionError("expected PdfRenderError")

    def test_empty_input_raises(self) -> None:
        try:
            render_pdf_pages(b"", max_pages=2, max_dimension=2000)
        except PdfRenderError:
            pass
        else:
            raise AssertionError("expected PdfRenderError")


class TestToPng:
    def test_jpeg_converts_to_png(self) -> None:
        # Build a tiny JPEG via PyMuPDF pixmap.
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16))
        pix.clear_with(255)
        jpeg = pix.tobytes("jpeg")
        png = to_png(jpeg)
        assert png.startswith(b"\x89PNG")

    def test_png_passthrough(self) -> None:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16))
        pix.clear_with(255)
        png = to_png(pix.tobytes("png"))
        assert png.startswith(b"\x89PNG")

    def test_garbage_raises(self) -> None:
        try:
            to_png(b"garbage-not-an-image")
        except PdfRenderError:
            pass
        else:
            raise AssertionError("expected PdfRenderError")
