import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from app.rag.config import Settings
from app.rag.pipeline import RAGPipeline


def fake_embed(texts, model_name):
    return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


def fake_generate(messages, model_name, max_new_tokens):
    return "Etkinlik 2024'te gerçekleşti. [Kanıt 1]"


@pytest.fixture
def client(monkeypatch):
    pipeline = RAGPipeline(
        settings=Settings(relevance_threshold=0.3, max_upload_mb=1),
        embedding_function=fake_embed,
        generation_function=fake_generate,
    )
    monkeypatch.setattr(main, "rag_pipeline", pipeline)
    return TestClient(main.app)


def test_upload_then_ask_returns_answer_with_sources(client):
    upload = client.post(
        "/documents",
        files={"file": ("rapor.txt", "Etkinlik 2024'te gerçekleşti.".encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 200
    assert upload.json()["chunks_indexed"] == 1
    assert client.get("/health").json() == {"status": "ok", "indexed_chunks": 1}

    answer = client.post("/ask", json={"question": "Etkinlik ne zaman oldu?"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["answer"] == "Etkinlik 2024'te gerçekleşti. [Kanıt 1]"
    assert body["sources"][0]["filename"] == "rapor.txt"


def test_upload_rejects_unsupported_file_type(client):
    response = client.post("/documents", files={"file": ("tablo.csv", b"a,b", "text/csv")})
    assert response.status_code == 400


def test_upload_rejects_empty_file(client):
    response = client.post("/documents", files={"file": ("bos.txt", b"", "text/plain")})
    assert response.status_code == 400


def test_upload_rejects_files_over_size_limit(client):
    oversized = b"a" * (1024 * 1024 + 1)
    response = client.post("/documents", files={"file": ("buyuk.txt", oversized, "text/plain")})
    assert response.status_code == 413
    assert client.get("/health").json()["indexed_chunks"] == 0


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def upload(client, filename="rapor.txt", text="Etkinlik 2024'te gerçekleşti."):
    return client.post("/documents", files={"file": (filename, text.encode("utf-8"), "text/plain")})


def test_documents_can_be_listed_and_deleted(client):
    document_id = upload(client).json()["document_id"]

    listed = client.get("/documents").json()["documents"]
    assert [(d["document_id"], d["filename"], d["chunks"]) for d in listed] == [(document_id, "rapor.txt", 1)]

    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 200
    assert client.get("/documents").json() == {"documents": []}
    assert client.get("/health").json()["indexed_chunks"] == 0
    assert client.delete(f"/documents/{document_id}").status_code == 404


def test_uploading_the_same_content_twice_is_rejected(client):
    assert upload(client, "rapor.txt").status_code == 200

    duplicate = upload(client, "rapor-kopya.txt")

    assert duplicate.status_code == 409
    assert "rapor.txt" in duplicate.json()["detail"]
    assert len(client.get("/documents").json()["documents"]) == 1


@pytest.mark.parametrize("label", ["Kanıt", "Kanit", "Evidence"])
def test_ask_round_trips_number_warnings_without_blocking(client, monkeypatch, label):
    upload(client, text="Etkinlik 2024'te gerçekleşti.")
    raw = f"  2025 ve 2026. [{label} 1] [Evidence 4]  "
    monkeypatch.setattr(main.rag_pipeline, "generation_function", lambda *args, **kwargs: raw)
    response = client.post("/ask", json={"question": "2025 yılında ne oldu?"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "sources", "verification"}
    assert body["answer"] == raw
    assert body["verification"] == {
        "citations": [1], "invalid_citations": [4],
        "unsupported_numbers": ["2026"], "question_only_numbers": ["2025"],
    }
    assert set(body["sources"][0]) == {
        "evidence_number", "cited", "text", "filename", "document_id",
        "page", "source", "chunk_id", "score",
    }
    assert body["sources"][0]["evidence_number"] == 1
    assert body["sources"][0]["cited"] is True


@pytest.mark.parametrize("raw,refused", [
    (" \n no_evidence \t", True),
    ("Ret işareti NO_EVIDENCE olarak yazılır. [Kanıt 1]", False),
])
def test_ask_marker_contract(client, monkeypatch, raw, refused):
    upload(client)
    monkeypatch.setattr(main.rag_pipeline, "generation_function", lambda *args, **kwargs: raw)
    response = client.post("/ask", json={"question": "İşaret nedir?"})
    assert response.status_code == 200
    body = response.json()
    if refused:
        assert body["sources"] == []
        assert body["verification"] is None
    else:
        assert body["answer"] == raw
        assert body["verification"]["question_only_numbers"] == []
