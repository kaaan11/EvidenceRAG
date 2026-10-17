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
    get.cache_clear()
    barrier=threading.Barrier(3)
    started=threading.Event(); release=threading.Event(); calls=[]; model=object()
    def create(*args):
        calls.append(args); started.set(); assert release.wait(5)
        return model
    monkeypatch.setattr(module,factory,create)
    def load(): barrier.wait(5); return get("test-model")
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
    with pytest.raises(RuntimeError): get("test-model")
    assert get("test-model") is model and len(attempts)==2
    get.cache_clear()
