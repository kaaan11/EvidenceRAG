import numpy as np
import pytest
import asyncio
import json
from dataclasses import replace
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


<<<<<<< HEAD
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
=======
def multipart_body(content=b"abc", filename="a.txt", extra_headers=b""):
    return (b'--edge\r\nContent-Disposition: form-data; name="file"; filename="'
            + filename.encode("utf-8") + b'"\r\n' + extra_headers + b'\r\n'
            + content + b'\r\n--edge--\r\n')


def asgi_post(chunks, headers=(), app=None, path="/documents", disconnect=False):
    messages = []
    consumed = []
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "",
             "query_string": b"", "headers": list(headers), "server": ("test", 80), "client": ("127.0.0.1", 1)}
    async def receive():
        position = len(consumed)
        if position >= len(chunks):
            return {"type": "http.disconnect"}
        consumed.append(position)
        return {"type": "http.request", "body": chunks[position], "more_body": disconnect or position < len(chunks) - 1}
    async def send(message):
        messages.append(message)
    asyncio.run((app or main.app)(scope, receive, send))
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, body, consumed


CONTENT_TYPE = (b"content-type", b"multipart/form-data; boundary=edge")


@pytest.mark.parametrize("size", [4, 5, 6])
def test_file_byte_boundary_is_413_not_500(client, monkeypatch, size):
    monkeypatch.setattr(main.rag_pipeline, "settings", replace(main.rag_pipeline.settings, max_upload_bytes=5))
    response = client.post("/documents", files={"file": ("a.txt", b"x" * size, "text/plain")})
    assert response.status_code == (413 if size > 5 else 200)
    assert len(main.rag_pipeline.list_documents()) == (0 if size > 5 else 1)


@pytest.mark.parametrize("size", [4, 5, 6])
def test_api_question_character_boundary(client, monkeypatch, size):
    monkeypatch.setattr(main.rag_pipeline, "settings", replace(main.rag_pipeline.settings, max_question_chars=5))
    response = client.post("/ask", json={"question": "ş" * size})
    assert response.status_code == (422 if size > 5 else 200)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_total_http_body_boundary_includes_multipart_overhead(client, monkeypatch, delta):
    body = multipart_body()
    limit = len(body) - delta
    monkeypatch.setattr(main.rag_pipeline, "settings", replace(main.rag_pipeline.settings, max_http_body_bytes=limit))
    response = client.post("/documents", content=body, headers={"Content-Type": "multipart/form-data; boundary=edge"})
    assert response.status_code == (413 if delta > 0 else 200)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_json_body_byte_boundary(client, monkeypatch, delta):
    body = b'{"question":"ok"}'
    monkeypatch.setattr(main.rag_pipeline, "settings", replace(main.rag_pipeline.settings, max_json_body_bytes=len(body)-delta))
    response = client.post("/ask", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == (413 if delta > 0 else 200)


def test_known_oversized_length_never_calls_receive_or_parser(client, monkeypatch):
    import app.request_limits as limits
    monkeypatch.setattr(main.rag_pipeline, "settings", replace(main.rag_pipeline.settings, max_http_body_bytes=10))
    def forbidden(*args, **kwargs):
        raise AssertionError("Parser must not start")
    monkeypatch.setattr(limits, "LimitedMultipartParser", forbidden)
    status, body, consumed = asgi_post([b"unused"], [CONTENT_TYPE, (b"content-length", b"11")])
    assert status == 413 and consumed == []
    assert main.rag_pipeline.list_documents() == []


@pytest.mark.parametrize("declared_length", [None, b"1"])
def test_stream_overflow_stops_before_parser_and_closes_partial_spool(client, monkeypatch, declared_length):
    import starlette.formparsers as parser_module
    from app.request_limits import LimitedMultipartParser
    first = b'--edge\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\nabcde'
    overflow = b'fghij\r\n--edge--\r\n'
    monkeypatch.setattr(main.rag_pipeline, "settings", replace(main.rag_pipeline.settings, max_http_body_bytes=len(first)+1))
    monkeypatch.setattr(LimitedMultipartParser, "spool_max_size", 2)
    spools = []
    original_spool = parser_module.SpooledTemporaryFile
    def spool(*args, **kwargs):
        obj = original_spool(*args, **kwargs)
        spools.append(obj)
        return obj
    monkeypatch.setattr(parser_module, "SpooledTemporaryFile", spool)
    delivered = []
    original_write = parser_module.multipart.MultipartParser.write
    def write(parser, data):
        delivered.append(data)
        return original_write(parser, data)
    monkeypatch.setattr(parser_module.multipart.MultipartParser, "write", write)
    calls = []
    monkeypatch.setattr(main.rag_pipeline, "embedding_function", lambda *args: calls.append(args))
    headers = [CONTENT_TYPE] + ([(b"content-length", declared_length)] if declared_length else [])
    status, body, consumed = asgi_post([first, overflow, b"never read"], headers)
    assert status == 413 and json.loads(body)["detail"]
    assert consumed == [0, 1] and delivered == [first]
    assert spools and all(s.closed and s._rolled for s in spools)
    assert calls == [] and main.rag_pipeline.list_documents() == []


@pytest.mark.parametrize("downstream_status", [400, 500])
def test_body_limit_stays_413_when_downstream_swallows_error(downstream_status):
    from app.request_limits import BodyLimitMiddleware
    from starlette.responses import JSONResponse
    async def swallowing(scope, receive, send):
        try:
            await receive()
        except Exception:
            await JSONResponse({"detail": "converted"}, status_code=downstream_status)(scope, receive, send)
    wrapped = BodyLimitMiddleware(swallowing, settings=lambda: Settings(max_http_body_bytes=1))
    status, body, consumed = asgi_post([b"xx", b"never"], app=wrapped)
    assert status == 413 and consumed == [0]


@pytest.mark.parametrize("headers", [[(b"content-length", b"-1")], [(b"content-length", b"bad")]])
def test_invalid_content_length_is_400_before_body(client, headers):
    assert asgi_post([b"unused"], headers)[::2] == (400, [])


@pytest.mark.parametrize("name,size", [("a.txt", 5), ("ş.txt", 6), ("şx.txt", 7)])
def test_filename_utf8_byte_boundary(client, monkeypatch, name, size):
    monkeypatch.setattr(main.rag_pipeline, "settings", replace(main.rag_pipeline.settings, max_filename_bytes=6))
    # Include the supported extension in the byte budget.
    response = client.post("/documents", content=multipart_body(filename=name), headers={"Content-Type": "multipart/form-data; boundary=edge"})
    assert response.status_code == (413 if size > 6 else 200)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_part_header_byte_boundary(client, monkeypatch, delta):
    header_size = len(b'Content-Disposition') + len(b'form-data; name="file"; filename="a.txt"') + 4
    monkeypatch.setattr(main.rag_pipeline, "settings", replace(main.rag_pipeline.settings, max_multipart_header_bytes=header_size-delta))
    response = client.post("/documents", content=multipart_body(), headers={"Content-Type": "multipart/form-data; boundary=edge"})
    assert response.status_code == (413 if delta > 0 else 200)


def test_extra_files_or_fields_are_rejected_and_spools_closed(client, monkeypatch):
    import starlette.formparsers as parser_module
    spools = []
    original = parser_module.SpooledTemporaryFile
    def tracked(*args, **kwargs):
        obj = original(*args, **kwargs)
        spools.append(obj)
        return obj
    monkeypatch.setattr(parser_module, "SpooledTemporaryFile", tracked)
    for files in [[("file", ("a.txt", b"one")), ("file", ("b.txt", b"two"))],
                  [("file", ("a.txt", b"one")), ("extra", (None, "bad"))]]:
        response = client.post("/documents", files=files)
        assert response.status_code == 400
    assert spools and all(s.closed for s in spools)
    assert main.rag_pipeline.list_documents() == []


def test_temporary_upload_file_cleaned_on_budget_error(client, monkeypatch, tmp_path):
    import tempfile
    created = []
    original = tempfile.NamedTemporaryFile
    def track(*args, **kwargs):
        f = original(*args, **kwargs, dir=tmp_path)
        created.append(f.name)
        return f
    monkeypatch.setattr(main.tempfile, "NamedTemporaryFile", track)
    monkeypatch.setattr(main.rag_pipeline, "settings", replace(main.rag_pipeline.settings, max_text_chars=2))
    response = client.post("/documents", files={"file": ("a.txt", b"abc")})
    assert response.status_code == 413
    assert created and all(not main.Path(p).exists() for p in created)
    assert main.rag_pipeline.list_documents() == []


def test_partial_spool_closed_on_truncated_body(client, monkeypatch):
    import starlette.formparsers as parser_module
    spools = []
    original = parser_module.SpooledTemporaryFile
    def track(*args, **kwargs):
        f = original(*args, **kwargs)
        spools.append(f)
        return f
    monkeypatch.setattr(parser_module, "SpooledTemporaryFile", track)
    # A final body without a terminating boundary must not become an indexed file.
    first = b'--edge\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\nabc'
    status, body, consumed = asgi_post([first], [CONTENT_TYPE])
    assert status in {400, 422}
    assert spools and all(s.closed for s in spools)
    assert main.rag_pipeline.list_documents() == []


def test_partial_spool_closed_on_client_disconnect(client, monkeypatch):
    import starlette.formparsers as parser_module
    from starlette.requests import ClientDisconnect
    spools = []
    original = parser_module.SpooledTemporaryFile
    def track(*args, **kwargs):
        f = original(*args, **kwargs)
        spools.append(f)
        return f
    monkeypatch.setattr(parser_module, "SpooledTemporaryFile", track)
    first = b'--edge\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\nabc'
    with pytest.raises(ClientDisconnect):
        asgi_post([first], [CONTENT_TYPE], disconnect=True)
    assert spools and all(s.closed for s in spools)
    assert main.rag_pipeline.list_documents() == []


def test_cancel_before_worker_starts_cleans_upload_resources(client, monkeypatch, tmp_path):
    import tempfile
    import starlette.formparsers as parser_module
    paths, spools = [], []
    original = tempfile.NamedTemporaryFile
    original_spool = parser_module.SpooledTemporaryFile
    def temporary(*args, **kwargs):
        f = original(*args, **kwargs, dir=tmp_path)
        paths.append(f.name)
        return f
    def spool(*args, **kwargs):
        f = original_spool(*args, **kwargs)
        spools.append(f)
        return f
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()
    monkeypatch.setattr(main.tempfile, "NamedTemporaryFile", temporary)
    monkeypatch.setattr(parser_module, "SpooledTemporaryFile", spool)
    monkeypatch.setattr(main, "run_in_threadpool", cancelled)
    with pytest.raises(asyncio.CancelledError):
        asgi_post([multipart_body()], [CONTENT_TYPE])
    assert paths and all(not main.Path(p).exists() for p in paths)
    assert spools and all(s.closed for s in spools)
    assert main.rag_pipeline.list_documents() == []


def test_multipart_header_limit_stops_before_spooling(client, monkeypatch):
    import starlette.formparsers as parser_module
    monkeypatch.setattr(main.rag_pipeline, "settings", replace(main.rag_pipeline.settings, max_multipart_header_bytes=10))
    def forbidden(*args, **kwargs):
        raise AssertionError("No file spool should be created")
    monkeypatch.setattr(parser_module, "SpooledTemporaryFile", forbidden)
    assert asgi_post([multipart_body()], [CONTENT_TYPE])[0] == 413


def test_missing_file_and_swagger_upload_contract(client):
    response = client.post("/documents", content=b"--edge--\r\n", headers={"Content-Type": "multipart/form-data; boundary=edge"})
    assert response.status_code == 422
    schema = client.get("/openapi.json").json()["paths"]["/documents"]["post"]["requestBody"]
    assert schema["content"]["multipart/form-data"]["schema"]["properties"]["file"]["format"] == "binary"


def test_temp_file_write_failure_does_not_leave_upload_file(client, monkeypatch, tmp_path):
    import tempfile
    original = tempfile.NamedTemporaryFile
    paths = []
    class BrokenWrite:
        def __init__(self, file):
            self.file = file
            self.name = file.name
        def __enter__(self): return self
        def __exit__(self, *args): self.file.close()
        def write(self, data): raise OSError("injected temporary-file write failure")
    def create(*args, **kwargs):
        file = original(*args, **kwargs, dir=tmp_path)
        paths.append(file.name)
        return BrokenWrite(file)
    monkeypatch.setattr(main.tempfile, "NamedTemporaryFile", create)
    response = client.post("/documents", files={"file": ("a.txt", b"abc")})
    assert response.status_code == 500  # An injected I/O error, not a limit rejection.
    assert paths and all(not main.Path(p).exists() for p in paths)
    assert main.rag_pipeline.list_documents() == []
>>>>>>> 29e1297 (Bound T02 input, document and store budgets)
