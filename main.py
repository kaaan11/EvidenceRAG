import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# Load .env before any Settings object reads the environment.
load_dotenv()

from app.rag.config import InputLimitError, Settings
from app.request_limits import BodyLimitMiddleware, bounded_upload
from app.rag.document_loader import DocumentError
from app.rag.generation import LLMGenerationError
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import DuplicateDocumentError, VectorStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_directory = Path(__file__).parent
settings = Settings()
rag_pipeline = RAGPipeline(
    settings=settings,
    vector_store=VectorStore(
        persist_path=project_directory / settings.index_dir / "store.npz",
        signature=settings.embedding_signature(),
        max_chunks=settings.max_store_chunks,
        max_text_bytes=settings.max_store_text_bytes,
    ),
)

app = FastAPI(title="EvidenceRAG")
app.add_middleware(BodyLimitMiddleware, settings=lambda: rag_pipeline.settings)
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
        "status": "ok",
        "indexed_chunks": len(rag_pipeline.vector_store.chunks),
    }


@app.post("/documents", openapi_extra={"requestBody": {"required": True, "content": {
    "multipart/form-data": {"schema": {"type": "object", "required": ["file"],
        "properties": {"file": {"type": "string", "format": "binary"}}}}
}}})
async def upload_document(request: Request):
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
        return await run_in_threadpool(rag_pipeline.index_file, temporary_path, filename=filename)
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


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if not rag_pipeline.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document_id": document_id, "deleted": True}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        return rag_pipeline.ask(request.question)
    except InputLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
