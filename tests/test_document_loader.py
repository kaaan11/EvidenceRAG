from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.rag.config import InputLimitError
from app.rag.document_loader import DocumentError, EmptyDocumentError, load_document


def write_pdf(path, texts):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=100, height=100)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 10 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


@pytest.mark.parametrize("extension", [".txt", ".md"])
@pytest.mark.parametrize("size", [4, 5, 6])
def test_text_char_boundary_unicode(tmp_path, extension, size):
    path = tmp_path / ("document" + extension)
    path.write_text("ş" * size, encoding="utf-8")
    if size > 5:
        with pytest.raises(InputLimitError, match="characters"):
            load_document(path, max_text_chars=5)
    else:
        assert load_document(path, max_text_chars=5)[0]["text"] == "ş" * size


def test_text_read_requests_only_limit_plus_one(monkeypatch, tmp_path):
    from io import StringIO
    path = tmp_path / "document.txt"
    path.write_text("exists", encoding="utf-8")
    counts = []
    class MeasuredText(StringIO):
        def read(self, count=-1):
            counts.append(count)
            return super().read(count)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: MeasuredText("x" * 100))
    with pytest.raises(InputLimitError):
        load_document(path, max_text_chars=5)
    assert counts == [6]


def test_invalid_utf8_is_document_error(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(DocumentError, match="UTF-8"):
        load_document(path)


@pytest.mark.parametrize("pages", [1, 2, 3])
def test_native_pdf_page_boundary(tmp_path, pages):
    path = tmp_path / "small.pdf"
    write_pdf(path, ["a"] * pages)
    if pages > 2:
        with pytest.raises(InputLimitError, match="pages"):
            load_document(path, max_pdf_pages=2)
    else:
        result = load_document(path, max_pdf_pages=2)
        assert [p["metadata"]["page"] for p in result] == list(range(1, pages + 1))


@pytest.mark.parametrize("size", [4, 5, 6])
def test_native_pdf_text_boundary(tmp_path, size):
    path = tmp_path / "small.pdf"
    write_pdf(path, ["x" * size])
    if size > 5:
        with pytest.raises(InputLimitError, match="characters"):
            load_document(path, max_text_chars=5)
    else:
        assert load_document(path, max_text_chars=5)[0]["text"] == "x" * size


def test_pdf_stops_extracting_after_total_text_overflow(monkeypatch, tmp_path):
    from pypdf._page import PageObject
    path = tmp_path / "small.pdf"
    write_pdf(path, ["xxx", "xxx", "xxx"])
    calls = []
    original = PageObject.extract_text
    def extract(page, *args, **kwargs):
        calls.append(page)
        return original(page, *args, **kwargs)
    monkeypatch.setattr(PageObject, "extract_text", extract)
    with pytest.raises(InputLimitError):
        load_document(path, max_text_chars=5)
    assert len(calls) == 2


def test_native_blank_pdf_is_empty_document_error(tmp_path):
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=10, height=10)
    writer.write(path)
    with pytest.raises(EmptyDocumentError):
        load_document(path)


def test_native_corrupt_pdf_is_document_error(tmp_path):
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not a PDF")
    with pytest.raises(DocumentError, match="Could not read PDF"):
        load_document(path)
