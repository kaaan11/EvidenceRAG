import json
import os
import threading
from pathlib import Path
from dataclasses import dataclass
import logging
import tempfile
import uuid

import numpy as np
import faiss

from .config import InputLimitError


STORE_FORMAT_VERSION = 1


class DuplicateDocumentError(Exception):
    """Raised when a document with the same content is already indexed."""

    def __init__(self, document):
        super().__init__(
            f"This document is already indexed as '{document['filename']}'"
        )
        self.document = document


class IncompatibleStoreError(RuntimeError):
    """Raised when a saved store was built with different embedding settings."""


class StoreWriteError(RuntimeError):
    def __init__(self, operation_id):
        super().__init__(f"Store write failed before commit; operation {operation_id}")
        self.operation_id = operation_id
        self.commit_status = "not_committed"


class StoreCommitUncertainError(StoreWriteError):
    def __init__(self, operation_id):
        RuntimeError.__init__(self, f"Store commit is visible but durability is uncertain; reconcile operation {operation_id} before retry")
        self.operation_id = operation_id
        self.commit_status = "durability_uncertain"


@dataclass(frozen=True)
class _State:
    index: object
    chunks: list
    documents: dict
    operation_id: str | None = None


class VectorStore:
    def __init__(self, persist_path=None, signature=None, *, max_chunks=20000, max_text_bytes=64 * 1024 * 1024, write_guard=None, autoload=True):
        self._state = _State(None, [], {})
        self.write_guard = write_guard  # Service-scoped lease; standalone stores do not acquire leases.
        self.max_chunks = max_chunks
        self.max_text_bytes = max_text_bytes
        # Identifies how the stored embeddings were produced, so a saved store isn't
        # silently searched with vectors from a different embedding model.
        self.signature = signature or {}
        self.persist_path = Path(persist_path) if persist_path else None
        # Uploads, deletions and questions are served from different threads; the FAISS
        # index, the chunk list and the document registry must change together, so every
        # read and write holds this lock.
        self._lock = threading.Lock()

        if autoload and self.persist_path and self.persist_path.exists():
            self._load()

    def reload(self):
        """Refresh service state under its writer lease; standalone reads stay usable."""
        with self._lock:
            if self.write_guard: self.write_guard()
            if self.persist_path:
                if self.persist_path.exists():
                    self._load()
                else:
                    self._state = _State(None, [], {})

    @property
    def index(self): return self._state.index

    @property
    def chunks(self): return self._state.chunks

    @property
    def documents(self): return self._state.documents

    @property
    def operation_id(self): return self._state.operation_id

    @property
    def is_empty(self):
        return not self.chunks

    def add(self, chunks, embeddings):
        embeddings = self._prepare_embeddings(chunks, embeddings)
        if embeddings is None:
            return
        with self._lock:
            if self.write_guard: self.write_guard()
            self._check_capacity(chunks)
            self._commit(self._candidate_add(chunks, embeddings))

    def add_document(self, document, chunks, embeddings, *, max_chunks=None, max_text_bytes=None):
        embeddings = self._prepare_embeddings(chunks, embeddings)
        with self._lock:
            if self.write_guard: self.write_guard()
            existing = self._find_by_hash(document.get("sha256"))
            if existing:
                raise DuplicateDocumentError(existing)
            self._check_capacity(chunks, max_chunks=max_chunks, max_text_bytes=max_text_bytes)
            candidate = self._candidate_add(chunks, embeddings)
            candidate.documents[document["document_id"]] = dict(document)
            self._commit(candidate)

    def check_capacity(self, chunks, *, max_chunks=None, max_text_bytes=None):
        """Early advisory check; add_document rechecks under the mutation lock."""
        with self._lock:
            self._check_capacity(chunks, max_chunks=max_chunks, max_text_bytes=max_text_bytes)

    def _check_capacity(self, chunks, *, max_chunks=None, max_text_bytes=None):
        # Called under _lock before live state or disk is changed. T05 can retain
        # this check when introducing candidate-state persistence.
        chunk_limit = min(self.max_chunks, max_chunks) if max_chunks is not None else self.max_chunks
        text_limit = min(self.max_text_bytes, max_text_bytes) if max_text_bytes is not None else self.max_text_bytes
        if len(self.chunks) + len(chunks) > chunk_limit:
            raise InputLimitError(f"Store exceeds {chunk_limit} chunks")
        text_bytes = sum(len(chunk["text"].encode("utf-8")) for chunk in self.chunks)
        incoming_bytes = sum(len(chunk["text"].encode("utf-8")) for chunk in chunks)
        if text_bytes + incoming_bytes > text_limit:
            raise InputLimitError(f"Store text exceeds {text_limit} UTF-8 bytes")

    def find_document_by_hash(self, sha256):
        with self._lock:
            existing = self._find_by_hash(sha256)
            return dict(existing) if existing else None

    def list_documents(self):
        with self._lock:
            documents = [dict(document) for document in self.documents.values()]
        return sorted(documents, key=lambda document: document.get("uploaded_at") or "")

    def delete_document(self, document_id):
        with self._lock:
            if self.write_guard: self.write_guard()
            if document_id not in self.documents:
                return False

            keep = [
                position
                for position, chunk in enumerate(self.chunks)
                if chunk["metadata"].get("document_id") != document_id
            ]
            # IndexFlat has no stable ids, so rebuild it from the remaining vectors.
            embeddings = self._all_embeddings()[keep]
            chunks = [self.chunks[position] for position in keep]
            index = None
            if len(keep):
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)

            documents = {key: value for key,value in self.documents.items() if key != document_id}
            self._commit(_State(index, chunks, documents, uuid.uuid4().hex))
            return True

    def search(self, query_embedding, top_k=5):
        if self.is_empty or top_k <= 0:
            return []

        query_embedding = np.array(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        with self._lock:
            if self.index is None:
                return []
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
            matched = [
                (float(score), self.chunks[index])
                for score, index in zip(scores[0], indices[0])
                if index >= 0
            ]

        results = []
        for score, chunk in matched:
            metadata = dict(chunk["metadata"])
            results.append(
                {
                    "text": chunk["text"],
                    "score": score,
                    "metadata": metadata,
                    **metadata,
                }
            )
        return results

    @staticmethod
    def _prepare_embeddings(chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding")
        if not chunks:
            return None

        # Copy so normalizing in place doesn't modify the caller's array.
        embeddings = np.array(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _candidate_add(self, chunks, embeddings):
        index = faiss.clone_index(self.index) if self.index is not None else None
        if embeddings is not None and index is None:
            index = faiss.IndexFlatIP(embeddings.shape[1])
        elif embeddings is not None and index.d != embeddings.shape[1]:
            raise ValueError("Embedding dimension does not match the vector store")
        if embeddings is not None:
            index.add(embeddings)
        appended = [{"text": chunk["text"], "metadata": dict(chunk["metadata"])} for chunk in chunks]
        return _State(index, [*self.chunks, *appended], dict(self.documents), uuid.uuid4().hex)

    def _find_by_hash(self, sha256):
        if not sha256:
            return None
        return next(
            (document for document in self.documents.values() if document.get("sha256") == sha256),
            None,
        )

    def _all_embeddings(self):
        return self._vectors(self.index)

    @staticmethod
    def _vectors(index):
        if index is None or index.ntotal == 0:
            return np.empty((0, 0), dtype="float32")
        return index.reconstruct_n(0, index.ntotal)

    def _commit(self, candidate):
        if not self.persist_path:
            self._state = candidate
            return
        temporary_path = None
        try:
            meta = {"version": STORE_FORMAT_VERSION, "signature": self.signature,
                    "documents": list(candidate.documents.values()), "chunks": candidate.chunks,
                    "operation_id": candidate.operation_id}
            serialized = json.dumps(meta)
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.persist_path.parent,
                    prefix=f".{self.persist_path.name}.", suffix=".tmp", delete=False) as file:
                temporary_path = Path(file.name)
                np.savez(file, embeddings=self._vectors(candidate.index), meta=np.array(serialized))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.persist_path)  # Visibility commit point.
        except Exception as exc:
            raise StoreWriteError(candidate.operation_id) from exc
        finally:
            if temporary_path is not None:
                try: temporary_path.unlink(missing_ok=True)
                except OSError: logging.getLogger(__name__).error("Candidate temporary file cleanup failed")
        # One pointer publishes the coherent RAM state while the same lock is held.
        self._state = candidate
        try:
            self._sync_directory()
        except OSError as exc:
            raise StoreCommitUncertainError(candidate.operation_id) from exc

    def _sync_directory(self):
        # Windows supports pre-replace file fsync here, not POSIX directory fsync.
        if os.name == "nt": return
        descriptor = os.open(self.persist_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(descriptor)
        finally: os.close(descriptor)

    def _load(self):
        with np.load(self.persist_path, allow_pickle=False) as data:
            embeddings = data["embeddings"]
            meta = json.loads(str(data["meta"]))

        if meta.get("version") != STORE_FORMAT_VERSION:
            raise IncompatibleStoreError(
                f"Unsupported store format version {meta.get('version')} in {self.persist_path}"
            )
        if self.signature and meta.get("signature") != self.signature:
            raise IncompatibleStoreError(
                f"The saved index at {self.persist_path} was built with embedding settings "
                f"{meta.get('signature')}, but the current settings are {self.signature}. "
                "Restore the previous settings, or delete the index file to start over and "
                "upload the documents again."
            )
        if len(meta["chunks"]) != len(embeddings):
            raise IncompatibleStoreError(
                f"The saved index at {self.persist_path} is inconsistent: "
                f"{len(meta['chunks'])} chunks but {len(embeddings)} embeddings."
            )

        index = None
        if len(embeddings):
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        self._state = _State(index, meta["chunks"],
            {document["document_id"]: document for document in meta["documents"]}, meta.get("operation_id"))
