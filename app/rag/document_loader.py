from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from .config import InputLimitError


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentError(Exception):
    """Base exception for document loading errors."""


class UnsupportedFileError(DocumentError):
    """Raised when a file format is not supported."""


class EmptyDocumentError(DocumentError):
    """Raised when a document contains no extractable text."""


def load_document(path, filename=None, source=None, document_id=None, *, max_text_chars=2000000, max_pdf_pages=500):
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileError(
            f"Unsupported file type '{extension}'. Supported types: {supported}"
        )

    document_id = document_id or uuid4().hex
    filename = filename or path.name
    source = source or str(path)

    if extension == ".pdf":
        pages = _load_pdf(path, max_text_chars=max_text_chars, max_pdf_pages=max_pdf_pages)
    else:
        try:
            # Bound decoded characters while reading, rather than loading an unbounded file.
            with path.open(encoding="utf-8") as stream:
                text = stream.read(max_text_chars + 1)
            if len(text) > max_text_chars:
                raise InputLimitError(f"Extracted text exceeds {max_text_chars} characters")
        except UnicodeDecodeError as exc:
            raise DocumentError(f"Document is not valid UTF-8 text: {filename}") from exc
        pages = [{"text": text, "page": None}]

    if not any(page["text"].strip() for page in pages):
        raise EmptyDocumentError(f"Document contains no extractable text: {filename}")

    return [
        {
            "text": page["text"],
            "metadata": {
                "filename": filename,
                "document_id": document_id,
                "page": page["page"],
                "source": source,
            },
        }
        for page in pages
        if page["text"].strip()
    ]


def _load_pdf(path, *, max_text_chars=2000000, max_pdf_pages=500):
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentError(f"Could not read PDF document: {path.name}") from exc
    if len(reader.pages) > max_pdf_pages:
        raise InputLimitError(f"PDF exceeds {max_pdf_pages} pages")
    pages = []
    total = 0
    for page_number, page in enumerate(reader.pages, start=1):
        # This check bounds retained output, not extract_text's peak CPU/RSS.
        text = page.extract_text() or ""
        total += len(text)
        if total > max_text_chars:
            raise InputLimitError(f"Extracted text exceeds {max_text_chars} characters")
        pages.append({"text": text, "page": page_number})
    return pages
