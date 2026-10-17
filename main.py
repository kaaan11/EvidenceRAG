import logging
import os
import tempfile
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# Load .env before any Settings object reads the environment.
load_dotenv()

from app.rag.config import InputLimitError, Settings
from app.access import LocalAccessMiddleware
from app.request_limits import BodyLimitMiddleware, bounded_upload
from app.rag.document_loader import DocumentError
from app.rag.generation import LLMGenerationError
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import DuplicateDocumentError, VectorStore, StoreWriteError
from app.rag.runtime import BusyError
from app.rag.storage_lease import WriterLease, WriterLeaseError, drain_service


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_directory = Path(__file__).parent
settings = Settings()
_service_lease = None


def _require_writer_lease():
    if _service_lease is None:
        raise WriterLeaseError("Service writer lifespan has not started")
    _service_lease.require_held()


rag_pipeline = RAGPipeline(
    settings=settings,
    vector_store=VectorStore(
        persist_path=project_directory / settings.index_dir / "store.npz",
        signature=settings.embedding_signature(),
        max_chunks=settings.max_store_chunks,
        max_text_bytes=settings.max_store_text_bytes,
        write_guard=_require_writer_lease,
    ),
)

@asynccontextmanager
async def lifespan(app):
    global _service_lease
    store = rag_pipeline.vector_store
    lease = WriterLease(store.persist_path.parent).acquire() if store.persist_path else None
    _service_lease = lease
    if lease is not None: store.write_guard = lease.require_held
    runtime = rag_pipeline.runtime
    app.state.service_accepting = True
    app.state.model_readiness = "lazy"
    try:
        if rag_pipeline.settings.preload_models:
            app.state.model_readiness = "loading"
            try:
                await runtime.run_async("load", rag_pipeline.preload_models)
                app.state.model_readiness = "ready"
            except Exception as exc:
                app.state.model_readiness = "failed"
                raise RuntimeError("Explicit model preload failed") from exc
        yield
    finally:
        app.state.service_accepting = False
        with runtime.condition:
            runtime.accepting = False
            runtime.condition.notify_all()
        # Submit synchronously before awaiting. Shielding task cancellation keeps
        # this watcher/native worker alive; registry/OS own lease until drain/exit.
        watcher = asyncio.get_running_loop().run_in_executor(None, drain_service, runtime, lease)
        await asyncio.shield(watcher)


app = FastAPI(title="EvidenceRAG", lifespan=lifespan)
app.state.model_readiness = "lazy"
app.state.service_accepting = False
app.add_middleware(BodyLimitMiddleware, settings=lambda: rag_pipeline.settings)
app.add_middleware(LocalAccessMiddleware)  # Outermost: reject before body parsing/spooling.
static_directory = project_directory / "static"
app.mount("/static", StaticFiles(directory=static_directory), name="static")


class QuestionRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def bounded_question(cls, value):
        if len(value) > rag_pipeline.settings.max_question_chars:
            raise ValueError(f"Question exceeds {rag_pipeline.settings.max_question_chars} characters")
        return value


@app.get("/")
def frontend():
    return FileResponse(static_directory / "index.html")


@app.get("/health")
def health_check():
    return {
        "status": "ok" if app.state.service_accepting else "not_ready",
        "indexed_chunks": len(rag_pipeline.vector_store.chunks),
        "model_readiness": app.state.model_readiness,
        "store_operation_id": rag_pipeline.vector_store.operation_id,
    }


@app.post("/documents", openapi_extra={"requestBody": {"required": True, "content": {
    "multipart/form-data": {"schema": {"type": "object", "required": ["file"],
        "properties": {"file": {"type": "string", "format": "binary"}}}}
}}})
async def upload_document(request: Request):
    _require_service_accepting()
    async with bounded_upload(request, rag_pipeline.settings) as file:
        return await _index_upload(file)


async def _index_upload(file):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Supported files are PDF, TXT, and MD")

    max_bytes = rag_pipeline.settings.max_upload_bytes
    # Read at most one byte past the limit so oversized uploads aren't loaded into memory.
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded document exceeds {max_bytes} bytes",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded document is empty")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(content)
        # Embedding is CPU-bound; run it off the event loop so other requests keep being served.
        return await rag_pipeline.runtime.run_async("index", rag_pipeline.index_file, temporary_path, filename=filename)
    except BusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
    except StoreWriteError as exc:
        return _store_error_response(exc)
    except InputLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (DocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(status_code=500, detail="Document indexing failed") from exc
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


@app.get("/documents")
def list_documents():
    return {"documents": rag_pipeline.list_documents()}


def _require_service_accepting():
    if not app.state.service_accepting:
        raise HTTPException(status_code=503, detail="Service is not accepting mutations/model jobs")


def _store_error_response(exc):
    return JSONResponse(status_code=500, content={"detail": str(exc),
        "operation_id": exc.operation_id, "commit_status": exc.commit_status})


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    _require_service_accepting()
    try:
        deleted = await rag_pipeline.runtime.run_async("index", rag_pipeline.delete_document, document_id)
    except StoreWriteError as exc:
        return _store_error_response(exc)
    except BusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After":"1"}) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document_id": document_id, "deleted": True}


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    _require_service_accepting()
    try:
        return await rag_pipeline.runtime.run_async("ask", rag_pipeline.ask, request.question)
    except BusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
    except InputLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
