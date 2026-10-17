import numpy as np
import pytest

from app.rag.vector_store import DuplicateDocumentError, IncompatibleStoreError, VectorStore
from app.rag.config import InputLimitError


SIGNATURE = {"embedding_model_name": "model-a"}


def make_document(document_id, sha256, vectors):
    document = {"document_id": document_id, "filename": f"{document_id}.txt", "sha256": sha256}
    chunks = [
        {"text": f"{document_id} chunk {i}", "metadata": {"document_id": document_id, "chunk_id": f"{document_id}-{i}"}}
        for i in range(len(vectors))
    ]
    return document, chunks, np.array(vectors, dtype="float32")


def test_store_persists_documents_and_vectors(tmp_path):
    path = tmp_path / "store.npz"
    store = VectorStore(persist_path=path, signature=SIGNATURE)
    store.add_document(*make_document("doc-1", "hash-1", [[1.0, 0.0], [0.0, 1.0]]))

    reloaded = VectorStore(persist_path=path, signature=SIGNATURE)

    assert [document["document_id"] for document in reloaded.list_documents()] == ["doc-1"]
    assert reloaded.search([0.0, 1.0], top_k=1)[0]["chunk_id"] == "doc-1-1"
    assert not (tmp_path / "store.npz.tmp").exists()


def test_store_rejects_duplicate_content(tmp_path):
    store = VectorStore(persist_path=tmp_path / "store.npz")
    store.add_document(*make_document("doc-1", "same-hash", [[1.0, 0.0]]))

    with pytest.raises(DuplicateDocumentError) as error:
        store.add_document(*make_document("doc-2", "same-hash", [[0.0, 1.0]]))

    assert error.value.document["document_id"] == "doc-1"
    assert len(store.chunks) == 1


def test_delete_removes_only_that_documents_chunks(tmp_path):
    path = tmp_path / "store.npz"
    store = VectorStore(persist_path=path, signature=SIGNATURE)
    store.add_document(*make_document("doc-1", "hash-1", [[1.0, 0.0], [0.9, 0.1]]))
    store.add_document(*make_document("doc-2", "hash-2", [[0.0, 1.0]]))

    assert store.delete_document("doc-1") is True
    assert store.delete_document("doc-1") is False

    for current in (store, VectorStore(persist_path=path, signature=SIGNATURE)):
        assert [document["document_id"] for document in current.list_documents()] == ["doc-2"]
        results = current.search([1.0, 0.0], top_k=5)
        assert [result["chunk_id"] for result in results] == ["doc-2-0"]


def test_deleting_last_document_leaves_an_empty_store(tmp_path):
    path = tmp_path / "store.npz"
    store = VectorStore(persist_path=path)
    store.add_document(*make_document("doc-1", "hash-1", [[1.0, 0.0]]))

    store.delete_document("doc-1")
    reloaded = VectorStore(persist_path=path)

    assert reloaded.is_empty
    assert reloaded.list_documents() == []
    assert reloaded.search([1.0, 0.0]) == []
    # A document can be added again after the store was emptied.
    reloaded.add_document(*make_document("doc-1", "hash-1", [[1.0, 0.0]]))
    assert len(reloaded.chunks) == 1


def test_store_refuses_index_built_with_other_embedding_settings(tmp_path):
    path = tmp_path / "store.npz"
    VectorStore(persist_path=path, signature=SIGNATURE).add_document(
        *make_document("doc-1", "hash-1", [[1.0, 0.0]])
    )

    with pytest.raises(IncompatibleStoreError, match="model-a"):
        VectorStore(persist_path=path, signature={"embedding_model_name": "model-b"})


@pytest.mark.parametrize("count", [1, 2, 3])
def test_store_chunk_boundary_preserves_disk_on_rejection(tmp_path, count):
    path = tmp_path / "store.npz"
    store = VectorStore(persist_path=path, max_chunks=2)
    args = make_document("d", "h", [[1.0, 0.0]] * count)
    if count > 2:
        with pytest.raises(InputLimitError):
            store.add_document(*args)
        assert store.chunks == [] and store.list_documents() == []
        assert not path.exists()
    else:
        store.add_document(*args)
        assert len(VectorStore(persist_path=path).chunks) == count


@pytest.mark.parametrize("byte_count", [3, 4, 5])
def test_store_utf8_text_byte_boundary(tmp_path, byte_count):
    store = VectorStore(persist_path=tmp_path / "s.npz", max_text_bytes=4)
    args = make_document("d", "h", [[1.0, 0.0]])
    args[1][0]["text"] = "ş" + "x" * (byte_count - 2)
    if byte_count > 4:
        with pytest.raises(InputLimitError, match="UTF-8"):
            store.add_document(*args)
        assert store.is_empty and store.list_documents() == []
    else:
        store.add_document(*args)
        assert len(store.chunks[0]["text"].encode("utf-8")) == byte_count


@pytest.mark.parametrize("quota", ["chunks", "text"])
def test_concurrent_admission_cannot_exceed_store_quota(tmp_path, quota):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    path = tmp_path / "store.npz"
    store = VectorStore(persist_path=path, max_chunks=1 if quota == "chunks" else 10,
                        max_text_bytes=1 if quota == "text" else 100)
    barrier = threading.Barrier(2)
    def add(i):
        args = make_document(str(i), str(i), [[1.0, 0.0]])
        args[1][0]["text"] = "x"
        store.check_capacity(args[1])  # Both preflight checks pass before final admission.
        barrier.wait(timeout=5)
        try:
            store.add_document(*args)
            return "added"
        except InputLimitError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(add, [1, 2]))
    assert sorted(outcomes) == ["added", "rejected"]
    for current in [store, VectorStore(persist_path=path)]:
        assert len(current.chunks) == len(current.list_documents()) == 1
        assert current.index.ntotal == 1


def test_rejected_add_preserves_existing_disk_and_allows_delete(tmp_path):
    path = tmp_path / "store.npz"
    store = VectorStore(persist_path=path, max_chunks=1)
    store.add_document(*make_document("first", "a", [[1.0, 0.0]]))
    before = path.read_bytes()
    with pytest.raises(InputLimitError):
        store.add_document(*make_document("second", "b", [[0.0, 1.0]]))
    assert path.read_bytes() == before
    assert [d["document_id"] for d in store.list_documents()] == ["first"]
    assert store.delete_document("first")
    store.add_document(*make_document("second", "b", [[0.0, 1.0]]))
    assert len(store.chunks) == 1


def test_direct_add_also_obeys_quota():
    store = VectorStore(max_chunks=1)
    _, chunks, vectors = make_document("d", "h", [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(InputLimitError):
        store.add(chunks, vectors)
    assert store.is_empty
