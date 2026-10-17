import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from app.rag.storage_lease import WriterLease, WriterLeaseError


ROOT=Path(__file__).resolve().parents[1]
ATTEMPT='''
import sys
from app.rag.storage_lease import WriterLease,WriterLeaseError
try: lease=WriterLease(sys.argv[1]).acquire()
except WriterLeaseError:
    print('busy',flush=True);sys.exit(9)
lease.release();print('acquired',flush=True)
'''


def attempt(directory):
    return subprocess.run([sys.executable,"-u","-c",ATTEMPT,str(directory)],cwd=ROOT,
                          capture_output=True,text=True,timeout=15)


def test_two_processes_exclude_same_index_and_os_releases_on_exit(tmp_path):
    holder='''
import sys
from app.rag.storage_lease import WriterLease
lease=WriterLease(sys.argv[1]).acquire()
print('ready',flush=True)
sys.stdin.readline()
lease.release()
'''
    process=subprocess.Popen([sys.executable,"-u","-c",holder,str(tmp_path/"index")],cwd=ROOT,
                             stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(process.stdout.readline)
            try: assert future.result(10).strip()=="ready"
            except BaseException:
                process.kill(); process.wait(5); raise
        blocked=attempt(tmp_path/"index")
        assert blocked.returncode==9 and "busy" in blocked.stdout
        assert attempt(tmp_path/"other").returncode==0
        process.kill(); process.wait(10)
        assert (tmp_path/"index/.writer.lock").exists()  # Existence is not lock ownership.
        assert attempt(tmp_path/"index").returncode==0
        assert not (tmp_path/"index/store.npz").exists()
    finally:
        if process.poll() is None: process.kill();process.wait(10)
        for f in [process.stdin,process.stdout,process.stderr]: f.close()


def test_same_process_service_lease_conflict_and_different_directory(tmp_path):
    first=WriterLease(tmp_path/"one").acquire()
    try:
        with pytest.raises(WriterLeaseError): WriterLease(tmp_path/"one").acquire()
        other=WriterLease(tmp_path/"two").acquire();other.release()
    finally: first.release()
    again=WriterLease(tmp_path/"one").acquire();again.release()


def test_service_handover_reads_latest_commit_under_lease(tmp_path):
    """Another service can import before its predecessor's final commit."""
    import numpy as np
    from app.rag.config import Settings
    from app.rag.vector_store import VectorStore
    path=tmp_path/"store.npz"
    signature=Settings().embedding_signature()
    lease=WriterLease(tmp_path).acquire()
    store=VectorStore(path,signature,write_guard=lease.require_held)
    def add(name):
        store.add_document({"document_id":name,"filename":name+".txt","sha256":name},
            [{"text":name,"metadata":{"document_id":name}}],np.ones((1,384),dtype="float32"))
    child_code='''
import asyncio,json,sys
import numpy as np
import main
print('imported',flush=True)
sys.stdin.readline()
async def run():
    async with main.lifespan(main.app):
        store=main.rag_pipeline.vector_store
        before=sorted(store.documents)
        store.add_document({'document_id':'C','filename':'C.txt','sha256':'C'},
            [{'text':'C','metadata':{'document_id':'C'}}],np.ones((1,384),dtype='float32'))
        print(json.dumps({'before':before,'after':sorted(store.documents)}),flush=True)
asyncio.run(run())
'''
    process=None
    try:
        add("A")
        env=dict(os.environ,INDEX_DIR=str(tmp_path),PRELOAD_MODELS="false")
        process=subprocess.Popen([sys.executable,"-u","-c",child_code],cwd=ROOT,env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            imported=pool.submit(process.stdout.readline)
            try: assert imported.result(15).strip()=="imported"
            except BaseException:
                process.kill();process.wait(5);raise
        add("B")
        assert sorted(VectorStore(path,signature).documents)==["A","B"]
        lease.release()
        stdout,stderr=process.communicate("\n",timeout=20)
        assert process.returncode==0,stderr
        assert json.loads(stdout)=={"before":["A","B"],"after":["A","B","C"]}
        loaded=VectorStore(path,signature)
        assert sorted(loaded.documents)==["A","B","C"]
        assert loaded.index.ntotal==3
    finally:
        lease.release()
        if process is not None:
            if process.poll() is None: process.kill();process.communicate(timeout=5)
            for file in [process.stdin,process.stdout,process.stderr]: file.close()


def test_service_load_failure_releases_lease_before_readiness(tmp_path,monkeypatch):
    import main
    from app.rag.pipeline import RAGPipeline
    from app.rag.runtime import HeavyRuntime
    from app.rag.vector_store import VectorStore
    path=tmp_path/"store.npz"
    path.write_bytes(b"invalid index")
    store=VectorStore(persist_path=path,autoload=False)
    pipeline=RAGPipeline(vector_store=store,runtime=HeavyRuntime())
    monkeypatch.setattr(main,"rag_pipeline",pipeline)
    async def run():
        with pytest.raises(ValueError):
            async with main.lifespan(main.app): pytest.fail("Corrupt index was accepted")
        assert not main.app.state.service_accepting
        assert attempt(tmp_path).returncode==0
        assert path.read_bytes()==b"invalid index"
    asyncio.run(run())


def test_cancelled_client_and_shutdown_hold_os_lease_until_native_commit(tmp_path,monkeypatch):
    import httpx2
    import numpy as np
    import main
    from app.rag.config import Settings
    from app.rag.pipeline import RAGPipeline
    from app.rag.runtime import HeavyRuntime
    from app.rag.vector_store import VectorStore
    path=tmp_path/"index/store.npz"
    entered=threading.Event();release=threading.Event()
    runtime=HeavyRuntime(wait_seconds=2)
    def embed(texts,model_name):
        entered.set(); assert release.wait(15)
        return np.array([[1.,0.] for _ in texts],dtype="float32")
    pipeline=RAGPipeline(settings=Settings(),vector_store=VectorStore(persist_path=path),
                         embedding_function=embed,runtime=runtime)
    monkeypatch.setattr(main,"rag_pipeline",pipeline)
    async def scenario():
        manager=main.lifespan(main.app)
        await manager.__aenter__()
        lease=main._service_lease
        transport=httpx2.ASGITransport(app=main.app,client=("127.0.0.1",1))
        async with httpx2.AsyncClient(transport=transport,base_url="http://127.0.0.1") as client:
            task=asyncio.create_task(client.post("/documents",files={"file":("a.txt",b"native data")}))
            deadline=time.monotonic()+5
            while not entered.is_set():
                assert time.monotonic()<deadline
                await asyncio.sleep(0.005)
            task.cancel()
            shutdown=asyncio.create_task(manager.__aexit__(None,None,None))
            await asyncio.sleep(0.03)
            shutdown.cancel()
            with pytest.raises(asyncio.CancelledError): await shutdown
            try:
                assert lease.held and runtime.snapshot()["active"]==1
                # OS check from a different process, not just an in-process flag.
                assert attempt(path.parent).returncode==9
                assert (await client.post("/ask",json={"question":"no new work"})).status_code==503
            finally: release.set()
            with pytest.raises(asyncio.CancelledError): await task
            deadline=time.monotonic()+5
            while lease.held:
                assert time.monotonic()<deadline
                await asyncio.sleep(0.01)
            assert attempt(path.parent).returncode==0
        loaded=VectorStore(persist_path=path)
        assert len(loaded.documents)==len(loaded.chunks)==loaded.index.ntotal==1
    asyncio.run(scenario())
