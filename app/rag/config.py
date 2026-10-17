import os
import json
import re
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field


class InputLimitError(ValueError):
    """An explicit input or store budget was exceeded (not a server failure)."""


@lru_cache(maxsize=1)
def model_manifest():
    return json.loads((Path(__file__).resolve().parents[2] / "models.lock.json").read_text(encoding="utf-8"))["models"]


def effective_revision(kind, model_name, revision=None):
    if revision is None:
        settings = Settings()
        expected_name = settings.embedding_model_name if kind == "embedding" else settings.llm_model_name
        if model_name != expected_name:
            raise ValueError("Explicit immutable revision required for a different model repository")
        revision = settings.embedding_model_revision if kind == "embedding" else settings.llm_model_revision
    if not isinstance(revision,str) or not re.fullmatch(r"[0-9a-f]{40}",revision):
        raise ValueError("Model revision must be an immutable 40-character commit SHA")
    return revision


def _str_env(name, default):
    return lambda: os.getenv(name) or default


def _prefix_env(name, default):
    # Unlike other settings, an empty value is meaningful here: it disables the prefix.
    return lambda: os.getenv(name, default)


def _int_env(name, default):
    def read():
        value = os.getenv(name)
        try:
            return int(value) if value else default
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc

    return read


def _float_env(name, default):
    def read():
        value = os.getenv(name)
        return float(value) if value else default

    return read


def _bool_env(name):
    value = os.getenv(name, "false").lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true/false or 1/0")
    return value in {"true", "1"}


# Environment variables are read when Settings() is created, not when this module is
# imported, so values loaded from a .env file at startup are picked up.
@dataclass(frozen=True)
class Settings:
    access_profile: str = field(default_factory=_str_env("ACCESS_PROFILE", "local"))
    embedding_model_name: str = field(
        default_factory=_str_env("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
    )
    # E5 models expect these prefixes; set them to an empty string for models that don't.
    embedding_query_prefix: str = field(default_factory=_prefix_env("EMBEDDING_QUERY_PREFIX", "query: "))
    embedding_passage_prefix: str = field(
        default_factory=_prefix_env("EMBEDDING_PASSAGE_PREFIX", "passage: ")
    )
    llm_model_name: str = field(default_factory=_str_env("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"))
    embedding_model_revision: str | None = field(default_factory=_str_env("EMBEDDING_MODEL_REVISION", None))
    llm_model_revision: str | None = field(default_factory=_str_env("LLM_MODEL_REVISION", None))
    ingestion_version: int = field(default_factory=_int_env("INGESTION_VERSION", 1))
    chunker_strategy: str = field(default_factory=_str_env("CHUNKER_STRATEGY", "legacy_character"))
    chunk_size: int = field(default_factory=_int_env("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=_int_env("CHUNK_OVERLAP", 100))
    top_k: int = field(default_factory=_int_env("TOP_K", 5))
    relevance_threshold: float = field(default_factory=_float_env("RELEVANCE_THRESHOLD", 0.80))
    max_new_tokens: int = field(default_factory=_int_env("MAX_NEW_TOKENS", 256))
    max_upload_mb: int = field(default_factory=_int_env("MAX_UPLOAD_MB", 20))
    # Bytes override the legacy MiB setting when explicitly supplied.
    max_upload_bytes: int | None = field(
        default_factory=_int_env("MAX_UPLOAD_BYTES", None)
    )
    max_http_body_bytes: int = field(default_factory=_int_env("MAX_HTTP_BODY_BYTES", 20 * 1024 * 1024 + 65536))
    max_json_body_bytes: int = field(default_factory=_int_env("MAX_JSON_BODY_BYTES", 32768))
    max_multipart_header_bytes: int = field(default_factory=_int_env("MAX_MULTIPART_HEADER_BYTES", 8192))
    max_filename_bytes: int = field(default_factory=_int_env("MAX_FILENAME_BYTES", 512))
    max_question_chars: int = field(default_factory=_int_env("MAX_QUESTION_CHARS", 4000))
    max_pdf_pages: int = field(default_factory=_int_env("MAX_PDF_PAGES", 500))
    max_text_chars: int = field(default_factory=_int_env("MAX_TEXT_CHARS", 2000000))
    max_document_chunks: int = field(default_factory=_int_env("MAX_DOCUMENT_CHUNKS", 2000))
    max_store_chunks: int = field(default_factory=_int_env("MAX_STORE_CHUNKS", 20000))
    max_store_text_bytes: int = field(default_factory=_int_env("MAX_STORE_TEXT_BYTES", 64 * 1024 * 1024))
    max_active_heavy_jobs: int = field(default_factory=_int_env("MAX_ACTIVE_HEAVY_JOBS", 1))
    max_waiting_heavy_jobs: int = field(default_factory=_int_env("MAX_WAITING_HEAVY_JOBS", 2))
    ask_queue_capacity: int = field(default_factory=_int_env("ASK_QUEUE_CAPACITY", 2))
    index_queue_capacity: int = field(default_factory=_int_env("INDEX_QUEUE_CAPACITY", 2))
    model_queue_capacity: int = field(default_factory=_int_env("MODEL_QUEUE_CAPACITY", 2))
    heavy_wait_seconds: int = field(default_factory=_int_env("HEAVY_WAIT_SECONDS", 30))
    preload_models: bool = field(default_factory=lambda: _bool_env("PRELOAD_MODELS"))
    torch_num_threads: int | None = field(default_factory=_int_env("TORCH_NUM_THREADS", None))
    # Relative paths are resolved against the project root.
    index_dir: str = field(default_factory=_str_env("INDEX_DIR", "data/index"))

    def __post_init__(self):
        if self.access_profile != "local":
            raise ValueError("ACCESS_PROFILE supports only 'local'; shared/public access is not supported")
        for kind,name_field,revision_field in [
            ("embedding","embedding_model_name","embedding_model_revision"),
            ("llm","llm_model_name","llm_model_revision"),
        ]:
            revision=getattr(self,revision_field)
            if revision is None:
                recorded=model_manifest()[kind]
                if getattr(self,name_field)!=recorded["repository"]:
                    raise ValueError(f"{revision_field} is required for a custom model repository")
                revision=recorded["revision"]
                object.__setattr__(self,revision_field,revision)
            if not re.fullmatch(r"[0-9a-f]{40}",revision):
                raise ValueError(f"{revision_field} must be an immutable 40-character commit SHA")
        if self.chunker_strategy not in {"legacy_character","structured_v2"}:
            raise ValueError("Unsupported CHUNKER_STRATEGY")
        if self.chunker_strategy=="structured_v2" and self.ingestion_version!=2:
            raise ValueError("structured_v2 requires INGESTION_VERSION=2 and a separate rebuilt index")
        if isinstance(self.max_upload_mb, bool) or not isinstance(self.max_upload_mb, int) or self.max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be a positive integer")
        if self.max_upload_bytes is None:
            object.__setattr__(self, "max_upload_bytes", self.max_upload_mb * 1024 * 1024)
        for name in (
            "chunk_size", "top_k", "max_new_tokens", "max_upload_mb", "max_upload_bytes",
            "max_http_body_bytes", "max_json_body_bytes", "max_multipart_header_bytes",
            "max_filename_bytes", "max_question_chars", "max_pdf_pages", "max_text_chars",
            "max_document_chunks", "max_store_chunks", "max_store_text_bytes",
            "max_active_heavy_jobs", "max_waiting_heavy_jobs", "ask_queue_capacity",
            "index_queue_capacity", "model_queue_capacity", "heavy_wait_seconds",
            "ingestion_version",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.chunk_overlap, bool) or not isinstance(self.chunk_overlap, int) or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")
        if self.torch_num_threads is not None and (not isinstance(self.torch_num_threads, int) or self.torch_num_threads <= 0):
            raise ValueError("TORCH_NUM_THREADS must be a positive integer when set")

    def embedding_signature(self):
        """Settings that determine the stored vectors; changing any of them needs a re-index."""
        return {
            "schema_version": 2,
            "embedding_model_name": self.embedding_model_name,
            "embedding_model_revision": self.embedding_model_revision,
            "tokenizer": {"repository":self.embedding_model_name,"revision":self.embedding_model_revision},
            "embedding_query_prefix": self.embedding_query_prefix,
            "embedding_passage_prefix": self.embedding_passage_prefix,
            "normalization": "l2",
            "embedding_dimension": self.embedding_metadata().get("embedding_dimension"),
            "preprocessing_version": 1,
            "ingestion": {"strategy":self.chunker_strategy,"version":self.ingestion_version,
                          "chunk_size":self.chunk_size,"chunk_overlap":self.chunk_overlap},
        }

    def embedding_metadata(self):
        record=model_manifest()["embedding"]
        if (self.embedding_model_name,self.embedding_model_revision)==(record["repository"],record["revision"]):
            return {"embedding_dimension":record["embedding_dimension"],"max_sequence_length":record["max_sequence_length"]}
        return {}  # Unknown until an explicitly pinned custom model is actually loaded.

    def model_identity(self):
        return {"embedding": {"repository":self.embedding_model_name,"revision":self.embedding_model_revision,
                              "tokenizer_repository":self.embedding_model_name,"tokenizer_revision":self.embedding_model_revision},
                "llm": {"repository":self.llm_model_name,"revision":self.llm_model_revision,
                        "tokenizer_repository":self.llm_model_name,"tokenizer_revision":self.llm_model_revision},
                "index_signature":self.embedding_signature()}
