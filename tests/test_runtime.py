import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.rag.runtime import BusyError, HeavyRuntime, SingleFlightCache


async def event_ready(event):
    deadline=time.monotonic()+5
    while not event.is_set():
        assert time.monotonic()<deadline
        await asyncio.sleep(0.005)


def test_mixed_jobs_obey_global_capacity_and_nested_ask_does_not_deadlock():
    runtime=HeavyRuntime(capacity=1,max_waiting=3,per_kind=3,wait_seconds=5)
    start=threading.Barrier(5); entered=threading.Event(); release=threading.Event()
    active=0; peak=0; lock=threading.Lock()
    def job(kind):
        start.wait(5)
        def native():
            nonlocal active,peak
            with lock: active+=1; peak=max(peak,active)
            entered.set(); assert release.wait(5)
            result=runtime.run_sync("generation",lambda:kind)  # Already owns the permit.
            with lock: active-=1
            return result
        return runtime.run_sync(kind,native)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(job,k) for k in ["ask","index","embedding","load"]]
        start.wait(5)
        try:
            assert entered.wait(5)
            deadline=time.monotonic()+5
            while runtime.snapshot()["waiting"]<3 and time.monotonic()<deadline: time.sleep(0.005)
            assert runtime.snapshot()=={"active":1,"waiting":3,"accepting":True}
        finally: release.set()
        assert sorted(f.result(5) for f in futures)==["ask","embedding","index","load"]
    assert peak==1 and runtime.snapshot()["active"]==0


def test_queued_cancel_removes_job_and_native_cancel_retains_permit():
    async def scenario():
        runtime=HeavyRuntime(max_waiting=1,wait_seconds=1)
        entered=threading.Event(); release=threading.Event(); calls=[]
        def native(): entered.set(); assert release.wait(5); return "done"
        task=asyncio.create_task(runtime.run_async("index",native))
        await event_ready(entered)
        queued=asyncio.create_task(runtime.run_async("ask",lambda:calls.append("wrong")))
        await asyncio.sleep(0.02)
        assert runtime.snapshot()["waiting"]==1
        queued.cancel()
        with pytest.raises(asyncio.CancelledError): await queued
        assert runtime.snapshot()["waiting"]==0
        task.cancel(); await asyncio.sleep(0.02); task.cancel()
        assert runtime.snapshot()["active"]==1 and not task.done()
        runtime.max_waiting=0
        try:
            with pytest.raises(BusyError): await runtime.run_async("ask",lambda:None)
        finally: release.set()
        with pytest.raises(asyncio.CancelledError): await task
        assert calls==[] and runtime.snapshot()["active"]==0
        assert await runtime.run_async("ask",lambda:42)==42
    asyncio.run(scenario())


def test_queue_full_timeout_and_exception_recover():
    async def scenario():
        runtime=HeavyRuntime(max_waiting=1,wait_seconds=0.04)
        entered=threading.Event(); release=threading.Event()
        def hold(): entered.set(); assert release.wait(5)
        first=asyncio.create_task(runtime.run_async("ask",hold)); await event_ready(entered)
        second=asyncio.create_task(runtime.run_async("index",lambda:None)); await asyncio.sleep(0.01)
        try:
            with pytest.raises(BusyError,match="full"): await runtime.run_async("load",lambda:None)
            with pytest.raises(BusyError,match="expired"): await second
        finally: release.set()
        await first
        def fail(): raise ValueError("native failed")
        with pytest.raises(ValueError): await runtime.run_async("ask",fail)
        assert await runtime.run_async("ask",lambda:"recovered")=="recovered"
        assert runtime.snapshot()["active"]==runtime.snapshot()["waiting"]==0
    asyncio.run(scenario())


def test_api_busy_has_retry_after_and_health_list_do_not_wait(monkeypatch):
    import httpx2
    import main
    runtime=HeavyRuntime(max_waiting=0,wait_seconds=1)
    entered=threading.Event(); release=threading.Event()
    monkeypatch.setattr(main.rag_pipeline,"runtime",runtime)
    def hold(question): entered.set(); assert release.wait(5); return {"answer":"done","sources":[],"verification":None}
    monkeypatch.setattr(main.rag_pipeline,"ask",hold)
    async def scenario():
        transport=httpx2.ASGITransport(app=main.app,client=("127.0.0.1",1))
        async with main.lifespan(main.app), httpx2.AsyncClient(transport=transport,base_url="http://127.0.0.1") as client:
            first=asyncio.create_task(client.post("/ask",json={"question":"ok"}))
            await event_ready(entered)
            try:
                busy=await client.post("/ask",json={"question":"ok"})
                assert busy.status_code==503 and busy.headers["retry-after"]=="1"
                assert (await client.get("/health")).status_code==200
                assert (await client.get("/documents")).status_code==200
            finally: release.set()
            assert (await first).status_code==200
    asyncio.run(scenario())


def test_default_startup_is_lazy_and_explicit_preload_failure_visible(monkeypatch):
    import main
    calls=[]
    monkeypatch.setattr(main.app.state,"model_readiness",main.app.state.model_readiness)
    monkeypatch.setattr(main.rag_pipeline,"settings",replace(main.rag_pipeline.settings,preload_models=False))
    monkeypatch.setattr(main.rag_pipeline,"preload_models",lambda:calls.append("loaded"))
    with TestClient(main.app,base_url="http://127.0.0.1",client=("127.0.0.1",1)) as client:
        assert client.get("/health").json()["model_readiness"]=="lazy"
    assert calls==[]
    monkeypatch.setattr(main.rag_pipeline,"settings",replace(main.rag_pipeline.settings,preload_models=True))
    with TestClient(main.app,base_url="http://127.0.0.1",client=("127.0.0.1",1)) as client:
        assert client.get("/health").json()["model_readiness"]=="ready"
    assert calls==["loaded"]
    def fail(): raise RuntimeError("test preload failure")
    monkeypatch.setattr(main.rag_pipeline,"preload_models",fail)
    with pytest.raises(RuntimeError,match="Explicit model preload failed"):
        with TestClient(main.app,base_url="http://127.0.0.1",client=("127.0.0.1",1)): pass
    assert main.app.state.model_readiness=="failed"


def test_torch_thread_setting_is_opt_in_and_applied_once(monkeypatch):
    import app.rag.runtime as module
    calls=[]
    class Torch:
        def set_num_threads(self,n): calls.append(n)
    monkeypatch.setattr(module,"_torch_configured",False)
    monkeypatch.delenv("TORCH_NUM_THREADS",raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS","7")
    module.configure_torch_threads(Torch())
    assert calls==[]
    monkeypatch.setattr(module,"_torch_configured",False)
    monkeypatch.setenv("TORCH_NUM_THREADS","3")
    module.configure_torch_threads(Torch()); module.configure_torch_threads(Torch())
    assert calls==[3]


def test_running_upload_cancel_keeps_file_until_worker_finishes(monkeypatch):
    from pathlib import Path
    import httpx2
    import main
    runtime=HeavyRuntime(max_waiting=0,wait_seconds=1)
    entered=threading.Event(); release=threading.Event(); paths=[]
    monkeypatch.setattr(main.rag_pipeline,"runtime",runtime)
    def index(path,filename=None):
        paths.append(Path(path)); entered.set(); assert release.wait(5)
        assert Path(path).exists()
        return {"document_id":"synthetic","filename":filename,"chunks_indexed":1}
    monkeypatch.setattr(main.rag_pipeline,"index_file",index)
    async def scenario():
        transport=httpx2.ASGITransport(app=main.app,client=("127.0.0.1",1))
        async with main.lifespan(main.app), httpx2.AsyncClient(transport=transport,base_url="http://127.0.0.1") as client:
            task=asyncio.create_task(client.post("/documents",files={"file":("a.txt",b"abc")}))
            await event_ready(entered)
            task.cancel(); await asyncio.sleep(0.02)
            try:
                assert paths[0].exists() and runtime.snapshot()["active"]==1
                assert (await client.post("/ask",json={"question":"ok"})).status_code==503
            finally: release.set()
            with pytest.raises(asyncio.CancelledError): await task
            assert not paths[0].exists() and runtime.snapshot()["active"]==0
    asyncio.run(scenario())
