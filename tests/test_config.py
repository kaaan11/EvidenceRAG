from app.rag.config import Settings
import pytest


def test_settings_read_environment_when_created(monkeypatch):
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.5")
    monkeypatch.setenv("LLM_MODEL_NAME", "some/model")

    settings = Settings()

    assert settings.top_k == 7
    assert settings.relevance_threshold == 0.5
    assert settings.llm_model_name == "some/model"


def test_empty_embedding_prefix_disables_it(monkeypatch):
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "")
    monkeypatch.delenv("EMBEDDING_PASSAGE_PREFIX", raising=False)

    settings = Settings()

    assert settings.embedding_query_prefix == ""
    assert settings.embedding_passage_prefix == "passage: "


@pytest.mark.parametrize("name", [
    "max_upload_bytes", "max_http_body_bytes", "max_json_body_bytes", "max_multipart_header_bytes",
    "max_filename_bytes", "max_question_chars", "max_pdf_pages", "max_text_chars",
    "max_document_chunks", "max_store_chunks", "max_store_text_bytes", "max_upload_mb", "chunk_size",
])
@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_limits_fail_at_startup(name, value):
    with pytest.raises(ValueError, match=name):
        Settings(**{name: value})


@pytest.mark.parametrize("overlap", [-1, 800, 801])
def test_invalid_overlap_fails_at_startup(overlap):
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_overlap=overlap)


def test_upload_bytes_override_and_legacy_mib(monkeypatch):
    assert Settings(max_upload_mb=1).max_upload_bytes == 1024 * 1024
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "17")
    assert Settings(max_upload_mb=1).max_upload_bytes == 17


def test_invalid_limit_env_names_the_setting(monkeypatch):
    monkeypatch.setenv("MAX_TEXT_CHARS", "not-an-integer")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        Settings()


def test_invalid_budget_prevents_app_startup(monkeypatch):
    import os
    from pathlib import Path
    import subprocess
    import sys
    env = dict(os.environ, MAX_TEXT_CHARS="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run([sys.executable, "-c", "import main"],
                            cwd=Path(__file__).resolve().parents[1], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "max_text_chars must be a positive integer" in result.stderr


@pytest.mark.parametrize("setting", ["max_active_heavy_jobs","max_waiting_heavy_jobs","ask_queue_capacity","index_queue_capacity","model_queue_capacity","heavy_wait_seconds"])
def test_runtime_settings_are_finite_positive(setting):
    with pytest.raises(ValueError,match=setting): Settings(**{setting:0})


def test_preload_boolean_is_explicit(monkeypatch):
    monkeypatch.setenv("PRELOAD_MODELS","false")
    assert Settings().preload_models is False
    monkeypatch.setenv("PRELOAD_MODELS","true")
    assert Settings().preload_models is True
    monkeypatch.setenv("PRELOAD_MODELS","maybe")
    with pytest.raises(ValueError,match="PRELOAD_MODELS"): Settings()


def test_torch_threads_reject_nonpositive_setting():
    with pytest.raises(ValueError,match="TORCH_NUM_THREADS"): Settings(torch_num_threads=0)
