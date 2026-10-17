import json
from pathlib import Path

import numpy as np
import pytest

from app.rag.config import Settings
from app.rag.config import InputLimitError
from app.rag.pipeline import INSUFFICIENT_EVIDENCE, RAGPipeline
from app.rag.vector_store import DuplicateDocumentError


def fake_embed(texts, model_name):
    return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


def fake_generate(messages, model_name, max_new_tokens):
    assert "Only answer using the supplied evidence" in messages[0]["content"]
    assert "The event happened in 2024." in messages[1]["content"]
    return "The evidence says the event happened in 2024. [Kanıt 1]"


def test_pipeline_indexes_documents_and_returns_sources(tmp_path: Path):
    document = tmp_path / "report.txt"
    document.write_text("The event happened in 2024.", encoding="utf-8")
    settings = Settings(chunk_size=100, chunk_overlap=10, relevance_threshold=0.3)
    pipeline = RAGPipeline(
        settings=settings,
        embedding_function=fake_embed,
        generation_function=fake_generate,
    )

    result = pipeline.index_file(document)
    answer = pipeline.ask("When did the event happen?")

    assert result["chunks_indexed"] == 1
    assert answer["answer"].startswith("The evidence says")
    assert answer["sources"][0]["filename"] == "report.txt"
    assert answer["sources"][0]["text"] == "The event happened in 2024."
    assert answer["sources"][0]["evidence_number"] == 1
    assert answer["sources"][0]["cited"] is True
    assert answer["verification"] == {
        "citations": [1],
        "invalid_citations": [],
        "unsupported_numbers": [],
        "question_only_numbers": [],
    }


def test_pipeline_reports_insufficient_evidence(tmp_path: Path):
    document = tmp_path / "report.txt"
    document.write_text("The event happened in 2024.", encoding="utf-8")
    settings = Settings(relevance_threshold=1.1)
    pipeline = RAGPipeline(settings=settings, embedding_function=fake_embed)
    pipeline.index_file(document)

    result = pipeline.ask("What is unrelated?")

    assert result["diagnostics"]["reason"] == "retrieval_no_evidence"
    assert {k: result[k] for k in ("answer", "sources", "verification")} == {"answer": INSUFFICIENT_EVIDENCE, "sources": [], "verification": None}


def test_pipeline_applies_embedding_prefixes(tmp_path: Path):
    document = tmp_path / "report.txt"
    document.write_text("The event happened in 2024.", encoding="utf-8")
    embedded = []

    def recording_embed(texts, model_name):
        embedded.extend(texts)
        return fake_embed(texts, model_name)

    settings = Settings(
        embedding_query_prefix="query: ",
        embedding_passage_prefix="passage: ",
        relevance_threshold=0.3,
    )
    pipeline = RAGPipeline(
        settings=settings,
        embedding_function=recording_embed,
        generation_function=fake_generate,
    )

    pipeline.index_file(document)
    pipeline.ask("When did the event happen?")

    assert embedded == ["passage: The event happened in 2024.", "query: When did the event happen?"]
    assert pipeline.vector_store.chunks[0]["text"] == "The event happened in 2024."


def test_pipeline_preserves_explanation_plus_marker(tmp_path: Path):
    document = tmp_path / "report.txt"
    document.write_text("The event happened in 2024.", encoding="utf-8")

    def declining_generate(messages, model_name, max_new_tokens):
        return "The evidence does not mention the mayor.\n\nNO_EVIDENCE"

    pipeline = RAGPipeline(
        settings=Settings(relevance_threshold=0.3),
        embedding_function=fake_embed,
        generation_function=declining_generate,
    )
    pipeline.index_file(document)

    result = pipeline.ask("Who is the mayor?")

    # The former substring-based refusal expectation violated the exact-marker protocol.
    assert result["answer"] == "The evidence does not mention the mayor.\n\nNO_EVIDENCE"
    assert result["sources"]
    assert result["verification"] is not None


CASES = json.loads((Path(__file__).parent / "fixtures/answer_contract/cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_pipeline_answer_contract(case, monkeypatch):
    # Isolate protocol behavior from retrieval quality and model inference.
    evidence = [{"text": case.get("evidence", "Protokol açıklaması."), "score": 1.0}]
    monkeypatch.setattr("app.rag.pipeline.retrieve", lambda *args, **kwargs: evidence)
    pipeline = RAGPipeline(generation_function=lambda *args, **kwargs: case["answer"])
    result = pipeline.ask(case.get("question", "Protokol nedir?"))
    if case["refused"]:
        assert result["diagnostics"]["reason"] == "model_no_evidence"
        assert {k: result[k] for k in ("answer", "sources", "verification")} == {"answer": INSUFFICIENT_EVIDENCE, "sources": [], "verification": None}
    else:
        assert result["answer"] == case["answer"]  # Preserve whitespace and raw model text.
        assert result["sources"][0]["text"] == evidence[0]["text"]
        if "verification" in case:
            assert result["verification"] == case["verification"]


def test_pipeline_registers_documents_and_skips_duplicate_uploads(tmp_path: Path):
    document = tmp_path / "report.txt"
    document.write_text("The event happened in 2024.", encoding="utf-8")
    copy = tmp_path / "copy-of-report.txt"
    copy.write_text("The event happened in 2024.", encoding="utf-8")
    embedded = []

    def recording_embed(texts, model_name):
        embedded.extend(texts)
        return fake_embed(texts, model_name)

    pipeline = RAGPipeline(settings=Settings(), embedding_function=recording_embed)
    result = pipeline.index_file(document)

    with pytest.raises(DuplicateDocumentError):
        pipeline.index_file(copy)

    [registered] = pipeline.list_documents()
    assert registered["document_id"] == result["document_id"]
    assert registered["filename"] == "report.txt"
    assert registered["chunks"] == 1
    assert registered["pages"] is None
    assert len(registered["sha256"]) == 64
    assert len(embedded) == 1


@pytest.mark.parametrize("kind", ["file", "txt_text", "md_text", "pdf_pages", "chunks", "store_chunks", "store_text"])
def test_early_budget_rejection_never_embeds_or_mutates(tmp_path, kind):
    from app.rag.vector_store import VectorStore
    from pypdf import PdfWriter
    path = tmp_path / "store.npz"
    store = VectorStore(persist_path=path)
    store.add_document({"document_id": "kept", "filename": "kept.txt", "sha256": "kept"},
                       [{"text": "kept", "metadata": {"document_id": "kept", "chunk_id": "kept-0"}}],
                       [[1.0, 0.0]])
    before = path.read_bytes()
    settings = {"chunk_size": 5, "chunk_overlap": 0}
    settings.update({
        "file": {"max_upload_bytes": 3}, "txt_text": {"max_text_chars": 3},
        "md_text": {"max_text_chars": 3}, "pdf_pages": {"max_pdf_pages": 1},
        "chunks": {"max_document_chunks": 1}, "store_chunks": {"max_store_chunks": 1},
        "store_text": {"max_store_text_bytes": 4},
    }[kind])
    calls = []
    rag = RAGPipeline(settings=Settings(**settings), vector_store=store,
                      embedding_function=lambda *args: calls.append(args))
    document = tmp_path / ("document.md" if kind == "md_text" else "document.txt")
    if kind == "pdf_pages":
        document = tmp_path / "document.pdf"
        writer = PdfWriter()
        for _ in range(2):
            writer.add_blank_page(width=10, height=10)
        writer.write(document)
    else:
        document.write_text("abcdefghij", encoding="utf-8")
    with pytest.raises(InputLimitError):
        rag.index_file(document)
    assert calls == []
    assert path.read_bytes() == before
    assert [d["document_id"] for d in store.list_documents()] == ["kept"]
    assert [c["text"] for c in store.chunks] == ["kept"]
    assert store.index.ntotal == 1


@pytest.mark.parametrize("length", [4, 5, 6])
def test_direct_question_boundary_precedes_model_calls(tmp_path, length):
    document = tmp_path / "document.txt"
    document.write_text("An event happened.", encoding="utf-8")
    calls = []
    def embed(texts, model_name):
        calls.append(texts)
        return fake_embed(texts, model_name)
    rag = RAGPipeline(settings=Settings(max_question_chars=5, relevance_threshold=0.3),
                      embedding_function=embed, generation_function=lambda *args, **kwargs: "Answer")
    rag.index_file(document)
    calls.clear()
    if length > 5:
        with pytest.raises(InputLimitError, match="Question"):
            rag.ask("ş" * length)
        assert calls == []
    else:
        assert rag.ask("ş" * length)["answer"] == "Answer"
        assert len(calls) == 1
