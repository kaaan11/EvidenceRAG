import importlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.mark.parametrize("module_name,getter,factory", [
    ("app.rag.embedding","get_embedding_model","_create_embedding_model"),
    ("app.rag.generation","get_text_generator","_create_text_generator"),
])
def test_model_loader_single_constructor_and_retry(monkeypatch,module_name,getter,factory):
    module=importlib.import_module(module_name)
    get=getattr(module,getter)
    from app.rag.config import model_manifest
    name=model_manifest()["embedding" if "embedding" in module_name else "llm"]["repository"]
    get.cache_clear()
    barrier=threading.Barrier(3)
    started=threading.Event(); release=threading.Event(); calls=[]; model=object()
    def create(*args):
        calls.append(args); started.set(); assert release.wait(5)
        return model
    monkeypatch.setattr(module,factory,create)
    def load(): barrier.wait(5); return get(name)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(load) for _ in range(2)]
        barrier.wait(5)
        try: assert started.wait(5)
        finally: release.set()
        assert [f.result(5) for f in futures]==[model,model]
    assert len(calls)==1
    get.cache_clear(); attempts=[]
    def retry(*args):
        attempts.append(args)
        if len(attempts)==1: raise RuntimeError("initialization failed")
        return model
    monkeypatch.setattr(module,factory,retry)
    with pytest.raises(RuntimeError): get(name)
    assert get(name) is model and len(attempts)==2
    get.cache_clear()


@pytest.mark.parametrize("module_name,getter,factory", [
    ("app.rag.embedding","get_embedding_model","_create_embedding_model"),
    ("app.rag.generation","get_text_generator","_create_text_generator"),
])
def test_loader_cache_key_separates_immutable_revisions(monkeypatch,module_name,getter,factory):
    module=importlib.import_module(module_name);get=getattr(module,getter);get.cache_clear()
    calls=[]
    def create(name,revision):
        value=object();calls.append((name,revision,value));return value
    monkeypatch.setattr(module,factory,create)
    first=get("synthetic/model",revision="1"*40)
    assert get("synthetic/model",revision="1"*40) is first
    assert get("synthetic/model",revision="2"*40) is not first
    assert len(calls)==2
    get.cache_clear()


def test_embedding_factory_forwards_revision_to_model_processor_and_config(monkeypatch):
    import sys
    from types import SimpleNamespace
    import app.rag.embedding as module
    from app.rag.config import model_manifest
    record=model_manifest()["embedding"];calls=[]
    model=SimpleNamespace(get_sentence_embedding_dimension=lambda:384,max_seq_length=512)
    def create(name,**kwargs):calls.append((name,kwargs));return model
    monkeypatch.setitem(sys.modules,"sentence_transformers",SimpleNamespace(SentenceTransformer=create))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace())
    monkeypatch.setattr(module,"configure_torch_threads",lambda *args:None)
    assert module._create_embedding_model(record["repository"],record["revision"]) is model
    assert calls==[(record["repository"],{"revision":record["revision"],
        "processor_kwargs":{"revision":record["revision"]},"config_kwargs":{"revision":record["revision"]}})]


def test_loaded_model_metadata_mismatch_is_rejected(monkeypatch):
    import sys
    from types import SimpleNamespace
    import app.rag.embedding as module
    from app.rag.config import model_manifest
    record=model_manifest()["embedding"]
    bad=SimpleNamespace(get_sentence_embedding_dimension=lambda:12,max_seq_length=512)
    monkeypatch.setitem(sys.modules,"sentence_transformers",SimpleNamespace(SentenceTransformer=lambda *args,**kwargs:bad))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace())
    monkeypatch.setattr(module,"configure_torch_threads",lambda *args:None)
    with pytest.raises(ValueError,match="dimension"):
        module._create_embedding_model(record["repository"],record["revision"])
