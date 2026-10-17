import hashlib
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

from .chunking import chunk_documents
from .config import InputLimitError, Settings
from .document_loader import load_document
from .embedding import embed_texts
from .generation import generate_answer
from .grounding import verify_answer
from .prompt import NO_EVIDENCE_MARKER, build_messages
from .retrieval import retrieve
from .runtime import get_runtime
from .vector_store import DuplicateDocumentError, VectorStore
from .diagnostics import trace_scope, phase, mark_reason, invoke_generator


INSUFFICIENT_EVIDENCE = (
    "Yüklenen dokümanlarda bu soruyu yanıtlamak için yeterli kanıt bulunamadı."
)


class RAGPipeline:
    def __init__(
        self,
        settings=None,
        embedding_function=embed_texts,
        generation_function=generate_answer,
        vector_store=None,
        runtime=None,
    ):
        self.settings = settings or Settings()
        self._default_generation = generation_function is generate_answer
        self.embedding_function = partial(embed_texts,revision=self.settings.embedding_model_revision) if embedding_function is embed_texts else embedding_function
        self.generation_function = partial(generate_answer,revision=self.settings.llm_model_revision) if generation_function is generate_answer else generation_function
        self.runtime = runtime or get_runtime()
        self.vector_store = vector_store or VectorStore(
            max_chunks=self.settings.max_store_chunks,
            max_text_bytes=self.settings.max_store_text_bytes,
        )

    def index_file(self, path, filename=None, source=None):
        if Path(path).stat().st_size > self.settings.max_upload_bytes:
            raise InputLimitError(f"File exceeds {self.settings.max_upload_bytes} bytes")
        return self.runtime.run_sync("index", self._index_file, path, filename, source)

    def _index_file(self, path, filename=None, source=None):
        filename = filename or Path(path).name
        sha256 = _file_sha256(path)
        # Checked before embedding so re-uploads don't pay for it; the store checks
        # again under its lock in case the same file is uploaded twice at once.
        existing = self.vector_store.find_document_by_hash(sha256)
        if existing:
            raise DuplicateDocumentError(existing)

        pages = load_document(
            path,
            filename=filename,
            source=source or filename,
            max_text_chars=self.settings.max_text_chars,
            max_pdf_pages=self.settings.max_pdf_pages,
        )
        chunks = chunk_documents(
            pages,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            max_chunks=self.settings.max_document_chunks,
        )
        self.vector_store.check_capacity(
            chunks, max_chunks=self.settings.max_store_chunks,
            max_text_bytes=self.settings.max_store_text_bytes,
        )
        prefix = self.settings.embedding_passage_prefix
        embeddings = self.embedding_function(
            [prefix + chunk["text"] for chunk in chunks],
            self.settings.embedding_model_name,
        )
        document = {
            "document_id": chunks[0]["metadata"]["document_id"],
            "filename": filename,
            "sha256": sha256,
            "pages": len({chunk["metadata"]["page"] for chunk in chunks} - {None}) or None,
            "chunks": len(chunks),
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.vector_store.add_document(
            document, chunks, embeddings, max_chunks=self.settings.max_store_chunks,
            max_text_bytes=self.settings.max_store_text_bytes,
        )
        return {
            "document_id": document["document_id"],
            "filename": filename,
            "chunks_indexed": len(chunks),
        }

    def list_documents(self):
        return self.vector_store.list_documents()

    def delete_document(self, document_id):
        return self.vector_store.delete_document(document_id)

    def ask(self, question):
        with trace_scope(self.settings, self.vector_store) as trace:
            if len(question) > self.settings.max_question_chars:
                mark_reason("input_limit")
                raise InputLimitError(f"Question exceeds {self.settings.max_question_chars} characters")
            result = self.runtime.run_sync("ask", self._ask, question)
            result["diagnostics"] = trace.snapshot()
            return result

    def _ask(self, question):
        def query_embedding(*args, **kwargs):
            with phase("query_embedding"):
                return self.embedding_function(*args, **kwargs)
        with phase("retrieval"):
            evidence = retrieve(
                question,
                self.vector_store,
                top_k=self.settings.top_k,
                relevance_threshold=self.settings.relevance_threshold,
                embedding_function=query_embedding,
                embedding_model_name=self.settings.embedding_model_name,
                query_prefix=self.settings.embedding_query_prefix,
            )
        if not evidence:
            mark_reason("retrieval_no_evidence")
            return _insufficient_evidence()

        messages = build_messages(question, evidence)
        generator = self.generation_function if self._default_generation else partial(invoke_generator, self.generation_function)
        answer = generator(
            messages,
            model_name=self.settings.llm_model_name,
            max_new_tokens=self.settings.max_new_tokens,
        )
        # The model signals unanswerable questions with a marker so the reply does not
        # depend on the model's ability to phrase a refusal in the question's language.
        if answer.strip().casefold() == NO_EVIDENCE_MARKER.casefold():
            mark_reason("model_no_evidence")
            return _insufficient_evidence()

        with phase("verification"):
            verification = verify_answer(answer, evidence, question)
        mark_reason("answered")
        return {
            "answer": answer,
            "sources": _sources_from(evidence, verification["citations"]),
            "verification": verification,
        }

    def preload_models(self):
        from .embedding import get_embedding_model
        from .generation import get_text_generator
        get_embedding_model(self.settings.embedding_model_name,revision=self.settings.embedding_model_revision)
        get_text_generator(self.settings.llm_model_name,revision=self.settings.llm_model_revision)


def _insufficient_evidence():
    return {"answer": INSUFFICIENT_EVIDENCE, "sources": [], "verification": None}


def _sources_from(evidence, citations):
    return [
        {
            "evidence_number": index,
            "cited": index in citations,
            "text": item.get("text", ""),
            "filename": item.get("filename"),
            "document_id": item.get("document_id"),
            "page": item.get("page"),
            "source": item.get("source"),
            "chunk_id": item.get("chunk_id"),
            "score": item.get("score"),
        }
        for index, item in enumerate(evidence, start=1)
    ]


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
