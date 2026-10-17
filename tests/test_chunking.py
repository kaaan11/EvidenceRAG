from app.rag.chunking import chunk_documents
from app.rag.config import InputLimitError
import pytest


def test_chunking_preserves_metadata_and_overlap():
    pages = [
        {
            "text": "one two three four five six seven eight nine ten",
            "metadata": {
                "filename": "notes.txt",
                "document_id": "doc-1",
                "page": None,
                "source": "notes.txt",
            },
        }
    ]

    chunks = chunk_documents(pages, chunk_size=20, chunk_overlap=5)

    assert len(chunks) > 1
    assert chunks[0]["metadata"]["filename"] == "notes.txt"
    assert chunks[0]["metadata"]["chunk_id"] == "doc-1-0"
    assert chunks[0]["text"].split()[-1] in chunks[1]["text"].split()


@pytest.mark.parametrize("count", [1, 2, 3])
def test_chunk_count_boundary(count):
    pages = [{"text": "x", "metadata": {"document_id": str(i)}} for i in range(count)]
    if count > 2:
        with pytest.raises(InputLimitError, match="chunks"):
            chunk_documents(pages, max_chunks=2)
    else:
        assert len(chunk_documents(pages, max_chunks=2)) == count


def test_chunk_generation_stops_without_consuming_unbounded_pages():
    consumed = []
    def pages():
        for i in range(100):
            consumed.append(i)
            yield {"text": "x", "metadata": {"document_id": str(i)}}
    with pytest.raises(InputLimitError):
        chunk_documents(pages(), max_chunks=2)
    assert consumed == [0, 1, 2]


def test_one_large_page_does_not_build_all_chunks_before_rejecting(monkeypatch):
    import app.rag.chunking as module
    original = module._split_text
    emitted = []
    def measured(*args):
        for part in original(*args):
            emitted.append(part)
            yield part
    monkeypatch.setattr(module, "_split_text", measured)
    with pytest.raises(InputLimitError):
        chunk_documents([{"text": "x" * 1000, "metadata": {"document_id": "d"}}],
                        chunk_size=5, chunk_overlap=0, max_chunks=2)
    assert len(emitted) == 3
