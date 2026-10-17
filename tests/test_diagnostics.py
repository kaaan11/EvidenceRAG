import asyncio
from dataclasses import replace
import json
import logging
from types import SimpleNamespace
import threading
import time
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from app.rag.config import Settings, InputLimitError
from app.rag.diagnostics import Trace, trace_scope, current, phase, provenance, static_identity
from app.rag.generation import generate_answer, LLMGenerationError
from app.rag.pipeline import RAGPipeline
from app.rag.runtime import HeavyRuntime, BusyError


def evidence(monkeypatch):
    monkeypatch.setattr("app.rag.pipeline.retrieve",lambda *a,**k:[{"text":"Source 2024","score":1.0}])


def test_refusal_reasons_and_existing_fields(monkeypatch):
    pipeline=RAGPipeline(generation_function=lambda *a,**k:"NO_EVIDENCE")
    empty=pipeline.ask("question")
    assert empty["verification"] is None and empty["sources"]==[]
    assert empty["diagnostics"]["reason"]=="retrieval_no_evidence"
    assert empty["diagnostics"]["generation_invoked"] is False
    assert empty["diagnostics"]["timings_ms"]["generation"] is None
    evidence(monkeypatch)
    refused=pipeline.ask("question")
    assert refused["answer"]==empty["answer"] and refused["verification"] is None
    assert refused["diagnostics"]["reason"]=="model_no_evidence"
    assert refused["diagnostics"]["generation_invoked"] is True


@pytest.mark.parametrize("failure",["loader","budget","callable"])
def test_invocation_tracks_actual_callable_entry(monkeypatch,failure):
    import app.rag.generation as module
    def broken(*a,**k): raise RuntimeError("private content and path")
    broken.tokenizer=SimpleNamespace(model_max_length=10,apply_chat_template=lambda *a,**k:[1]*9)
    if failure=="loader": monkeypatch.setattr(module,"get_text_generator",lambda *a,**k: (_ for _ in ()).throw(RuntimeError("load failed")))
    if failure=="callable": broken.tokenizer.model_max_length=100
    with trace_scope(Settings()) as trace:
        with pytest.raises((LLMGenerationError,InputLimitError)):
            generate_answer([],max_new_tokens=2,text_generator=None if failure=="loader" else broken)
        assert trace.generation_invoked is (failure=="callable")
        assert (trace.timings["generation"] is not None) is (failure=="callable")
        assert trace.reason==("input_limit" if failure=="budget" else "model_error")


def test_monotonic_zero_unrun_overlap_and_total():
    clock=[0.0]
    trace=Trace(Settings(),clock=lambda:clock[0])
    with trace.phase("queue"): pass
    with trace.phase("retrieval"):
        clock[0]=1
        with trace.phase("query_embedding"): clock[0]=3
        clock[0]=4
    values=trace.snapshot()["timings_ms"]
    assert values["queue"]==0 and values["generation"] is None
    assert values["query_embedding"]==2000 and values["retrieval"]==values["total"]==4000
    assert values["query_embedding"]+values["retrieval"]>values["total"]


def test_cold_factory_time_separate_from_warm_cache():
    from app.rag.runtime import SingleFlightCache
    cache=SingleFlightCache();clock=[0.0]
    def factory(): clock[0]+=2;return object()
    with trace_scope(Settings()) as cold:
        cold.clock=lambda:clock[0];cold.started=0
        model=cache.get("model",factory)
    assert cold.timings["model_load"]==2000 and not cold.generation_invoked
    with trace_scope(Settings()) as warm:
        assert cache.get("model",factory) is model
    assert warm.timings["model_load"] is None


def test_answer_and_99_second_refusal_both_count_as_invocations(monkeypatch):
    evidence(monkeypatch)
    durations=[]
    for seconds,answer in [(1,"Answer [Kanıt 1]"),(99,"NO_EVIDENCE")]:
        clock=[0.0]
        def generator(*args,**kwargs): clock[0]+=seconds;return answer
        with trace_scope(Settings()) as trace:
            trace.clock=lambda:clock[0];trace.started=0
            result=RAGPipeline(generation_function=generator).ask("question")
        diag=result["diagnostics"]
        assert diag["generation_invoked"] is True
        durations.append(diag["timings_ms"]["generation"])
    assert durations==[1000,99000] and sum(durations)/len(durations)==50000


def test_provenance_determinism_revision_prompt_change_and_cached_sources(monkeypatch):
    import app.rag.diagnostics as module
    settings=Settings()
    assert provenance(settings)==provenance(settings)
    assert provenance(settings)!=provenance(replace(settings,llm_model_revision="a"*40))
    before=provenance(settings)
    snapshot=static_identity()
    monkeypatch.setattr(module,"static_identity",lambda:{**snapshot,"prompt_sha256":"changed"})
    assert provenance(settings)!=before
    assert snapshot is static_identity()  # cached file/version inventory
    assert "question" not in json.dumps(before)


def test_async_worker_context_cancellation_and_queue_states():
    runtime=HeavyRuntime(capacity=1,max_waiting=1,wait_seconds=1)
    entered=threading.Event();release=threading.Event();seen=[]
    def work():
        seen.append(current().request_id);entered.set();assert release.wait(10)
    async def wait_for(predicate):
        deadline=time.monotonic()+5
        while not predicate():
            assert time.monotonic()<deadline
            await asyncio.sleep(.005)
    async def run():
        with trace_scope(Settings()) as running:
            task=asyncio.create_task(runtime.run_async("ask",work))
        await wait_for(entered.is_set)
        with trace_scope(Settings()) as queued:
            waiting=asyncio.create_task(runtime.run_async("ask",lambda:pytest.fail("Queued work ran")))
        await wait_for(lambda:runtime.snapshot()["waiting"]==1)
        with trace_scope(Settings()) as rejected:
            with pytest.raises(BusyError): await runtime.run_async("ask",lambda:None)
        assert rejected.admission_state=="queue_rejected" and not rejected.generation_invoked
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError): await waiting
        assert queued.admission_state=="queue_cancelled"
        task.cancel();await asyncio.sleep(.02)
        try:
            assert running.admission_state=="client_cancelled_running"
            assert runtime.snapshot()["active"]==1 and not task.done()
        finally: release.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert runtime.snapshot()["active"]==0 and running.reason=="cancelled"
        assert seen==[running.request_id]
    asyncio.run(run())


@pytest.mark.parametrize("error,status",[(LLMGenerationError,503),(RuntimeError,500)])
def test_api_error_diagnostics_and_no_raw_content_in_logs(monkeypatch,caplog,error,status):
    import main
    from fastapi.testclient import TestClient
    evidence(monkeypatch)
    secret="SECRET-question-document-answer-key-path"
    def broken(*args,**kwargs):raise error(secret)
    pipeline=RAGPipeline(generation_function=broken)
    monkeypatch.setattr(main,"rag_pipeline",pipeline)
    caplog.set_level(logging.INFO)
    with TestClient(main.app,base_url="http://127.0.0.1",client=("127.0.0.1",1)) as client:
        response=client.post("/ask",json={"question":secret})
        assert response.status_code==status and secret not in response.text
        diag=response.json()["diagnostics"]
        assert diag["reason"]=="model_error" and diag["generation_invoked"] is True
        invalid=client.post("/ask",json={"question":"x"*4001})
        assert invalid.status_code==422 and invalid.json()["diagnostics"]["generation_invoked"] is False
    assert secret not in caplog.text
    assert "request_diagnostics" in caplog.text


def test_ui_reason_labels_legacy_output_and_busy_message():
    node=os.getenv("EVIDENCERAG_NODE_BINARY") or shutil.which("node")
    if not node: pytest.skip("Node required for UI regression")
    script=r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
function el(){return {children:[],textContent:'',value:'',style:{},classList:{add(){},remove(){},toggle(){}},querySelector(){return el()},addEventListener(){},setAttribute(){},append(...x){this.children.push(...x)},appendChild(x){this.children.push(x)},replaceChildren(){this.children=[]}}}
const els=new Map(),document={getElementById(id){if(!els.has(id))els.set(id,el());return els.get(id)},createElement:el};
const c=vm.createContext({document,window:{addEventListener(){}},fetch:async()=>({ok:false,status:503,json:async()=>({diagnostics:{reason:'busy'}})})});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),c);
function texts(e){return [e.textContent,...e.children.flatMap(texts)].join(' ')}
c.renderAnswer('answer',[],null,{reason:'model_no_evidence'});assert.match(texts(els.get('answerCard')),/Model yeterli/);
c.renderAnswer('answer',[],null,{reason:'retrieval_no_evidence'});assert.match(texts(els.get('answerCard')),/Uygun kaynak/);
c.renderAnswer('legacy',[],null);assert.match(texts(els.get('answerCard')),/legacy/);
els.get('question').value='question';
c.askQuestion().then(()=>{assert.match(texts(els.get('askFeedback')),/Servis meşgul/)}).catch(e=>{console.error(e);process.exitCode=1});
'''
    result=subprocess.run([node,"-e",script,str(Path(__file__).resolve().parents[1]/"static/app.js")],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
