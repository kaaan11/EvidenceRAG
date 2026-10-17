from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from app.rag.config import Settings
from app.rag.pipeline import RAGPipeline
from app.rag.runtime import HeavyRuntime
from app.rag.vector_store import VectorStore


def embed(texts,name):return np.array([[1.]+[0.]*383 for _ in texts],dtype='float32')


def fake_generator(messages,**kwargs):
    return ('2025' if '2025' in messages[-1]['content'] else '2024')+' [Kanıt 1]'


def pipeline(path,*,settings=None,generation=fake_generator,runtime=None):
    settings=settings or Settings(relevance_threshold=.1)
    return RAGPipeline(settings=settings,embedding_function=embed,generation_function=generation,
        vector_store=VectorStore(path,settings.embedding_signature(),autoload=False),runtime=runtime or HeavyRuntime())


def client():return TestClient(main.app,base_url='http://127.0.0.1',client=('127.0.0.1',1))


def pdf_bytes(path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject,DictionaryObject,NameObject
    writer=PdfWriter();page=writer.add_blank_page(width=100,height=100)
    font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
    page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font})})
    stream=DecodedStreamObject();stream.set_data(b'BT /F1 12 Tf 10 10 Td (Fixture PDF year 2024.) Tj ET')
    page[NameObject('/Contents')]=writer._add_object(stream);writer.write(path);return path.read_bytes()


def test_upload_list_ask_warning_duplicate_delete_and_restart_native(tmp_path,monkeypatch):
    fixture=Path(__file__).parent/'fixtures/end_to_end'
    files={'report.txt':(fixture/'report.txt').read_bytes(),'report.md':(fixture/'report.md').read_bytes(),'report.pdf':pdf_bytes(tmp_path/'source.pdf')}
    path=tmp_path/'index/store.npz';first=pipeline(path);monkeypatch.setattr(main,'rag_pipeline',first)
    with client() as api:
        ids={}
        for filename,data in files.items():
            response=api.post('/documents',files={'file':(filename,data)})
            assert response.status_code==200;ids[filename]=response.json()['document_id']
        assert len(api.get('/documents').json()['documents'])==3
        answer=api.post('/ask',json={'question':'Which source year?'}).json()
        assert answer['answer']=='2024 [Kanıt 1]' and answer['diagnostics']['generation_invoked']
        assert answer['sources'] and answer['verification']['unsupported_numbers']==[]
        warning=api.post('/ask',json={'question':'What about 2025?'}).json()
        assert warning['verification']['question_only_numbers']==['2025']
        assert api.post('/documents',files={'file':('duplicate.txt',files['report.txt'])}).status_code==409
        assert api.delete('/documents/'+ids['report.txt']).status_code==200
        assert {d['document_id'] for d in api.get('/documents').json()['documents']}=={ids['report.md'],ids['report.pdf']}
    saved=path.read_bytes();second=pipeline(path);monkeypatch.setattr(main,'rag_pipeline',second)
    with client() as api:
        assert {d['document_id'] for d in api.get('/documents').json()['documents']}=={ids['report.md'],ids['report.pdf']}
        assert api.post('/ask',json={'question':'Source year?'}).status_code==200
    assert path.read_bytes()==saved  # restart did not rebuild/relabel the committed index


def test_access_limit_and_precommit_failure_keep_committed_service_state(tmp_path,monkeypatch):
    import app.rag.vector_store as module
    current=pipeline(tmp_path/'store.npz',settings=Settings(relevance_threshold=.1,max_upload_bytes=4))
    monkeypatch.setattr(main,'rag_pipeline',current)
    with client() as api:
        assert api.post('/documents',files={'file':('one.txt',b'abc')}).status_code==200
        before=current.vector_store.persist_path.read_bytes();docs=api.get('/documents').json()
        assert api.post('/documents',files={'file':('deny.txt',b'abd')},headers={'Origin':'null'}).status_code==403
        assert api.post('/documents',files={'file':('large.txt',b'abcde')}).status_code==413
        original=module.np.savez
        def fail(*args,**kwargs):raise OSError('synthetic precommit disk fault')
        monkeypatch.setattr(module.np,'savez',fail)
        response=api.post('/documents',files={'file':('two.txt',b'abd')})
        assert response.status_code==500 and response.json()['commit_status']=='not_committed'
        assert api.get('/documents').json()==docs and current.vector_store.persist_path.read_bytes()==before
        monkeypatch.setattr(module.np,'savez',original)
        assert api.post('/documents',files={'file':('two.txt',b'abd')}).status_code==200
        assert len(api.get('/documents').json()['documents'])==2


def test_incompatible_startup_preserves_backup_and_rollback_directory(tmp_path,monkeypatch):
    from app.rag.storage_lease import WriterLease
    from app.rag.vector_store import IncompatibleStoreError
    old_settings=Settings(relevance_threshold=.1);path=tmp_path/'store.npz'
    store=VectorStore(path,old_settings.embedding_signature())
    store.add_document({'document_id':'old','filename':'old.txt','sha256':'original'},
        [{'text':'Event year 2024','metadata':{'document_id':'old','filename':'old.txt','chunk_id':'old-0'}}],embed(['event'],'fake'))
    before=path.read_bytes();backup=tmp_path/'backup.npz';backup.write_bytes(before)
    new_settings=Settings(chunker_strategy='structured_v2',ingestion_version=2,relevance_threshold=.1)
    monkeypatch.setattr(main,'rag_pipeline',pipeline(path,settings=new_settings))
    with pytest.raises(IncompatibleStoreError):
        with client():pytest.fail('Incompatible startup accepted')
    assert not main.app.state.service_accepting and path.read_bytes()==backup.read_bytes()==before
    lease=WriterLease(tmp_path).acquire();lease.release()
    monkeypatch.setattr(main,'rag_pipeline',pipeline(path,settings=old_settings))
    with client() as api:
        assert api.get('/documents').json()['documents'][0]['document_id']=='old'
        assert api.post('/ask',json={'question':'Year?'}).status_code==200
    assert path.read_bytes()==backup.read_bytes()==before


def test_model_loader_prompt_budget_and_busy_api_contract(tmp_path,monkeypatch):
    import app.rag.generation as module
    from app.rag.generation import generate_answer
    runtime=HeavyRuntime(max_waiting=0)
    current=pipeline(tmp_path/'store.npz',generation=generate_answer,runtime=runtime)
    monkeypatch.setattr(main,'rag_pipeline',current);monkeypatch.setattr(module,'get_runtime',lambda:runtime)
    def missing(*args,**kwargs):raise RuntimeError('private loader detail')
    monkeypatch.setattr(module,'get_text_generator',missing)
    with client() as api:
        assert api.post('/documents',files={'file':('source.txt',b'2024 source')}).status_code==200
        error=api.post('/ask',json={'question':'Year?'})
        assert error.status_code==503 and error.json()['diagnostics']['generation_invoked'] is False
        assert 'private' not in error.text
        calls=[]
        def generator(*args,**kwargs):calls.append(1);return [{'generated_text':'not allowed'}]
        generator.tokenizer=SimpleNamespace(model_max_length=5,apply_chat_template=lambda *a,**k:[1]*4)
        monkeypatch.setattr(module,'get_text_generator',lambda *a,**k:generator)
        limit=api.post('/ask',json={'question':'Year?'})
        assert limit.status_code==413 and not calls and limit.json()['diagnostics']['generation_invoked'] is False
        entered=threading.Event();release=threading.Event()
        def work():entered.set();assert release.wait(5)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(runtime.run_sync,'ask',work)
            try:
                assert entered.wait(5)
                busy=api.post('/ask',json={'question':'Year?'})
                assert busy.status_code==503 and busy.headers['Retry-After']=='1'
                assert busy.json()['diagnostics']['reason']=='busy'
                assert api.get('/health').status_code==200
            finally:release.set()
            future.result(5)


@pytest.mark.parametrize('key,value',[('jobs',100000),('timeout_seconds',0),('max_active',20),('mode','remote')])
def test_capacity_profile_rejects_unbounded_or_remote_requests(key,value):
    import json
    from experiments.capacity_check import validate_profile
    profile=json.loads((Path(__file__).resolve().parents[1]/'experiments/capacity_profiles/local_fixture.json').read_text())
    with pytest.raises(ValueError):validate_profile({**profile,key:value})


@pytest.mark.real_model
def test_explicit_cache_only_real_model_smoke(tmp_path):
    """Separate opt-in; absence of cache must never become fabricated success."""
    from huggingface_hub import scan_cache_dir
    settings=Settings(relevance_threshold=-1,max_new_tokens=16)
    try:cache=scan_cache_dir()
    except Exception:pytest.fail('Immutable local model cache absent; populate only under a separately authorized budget')
    for name,revision in [(settings.embedding_model_name,settings.embedding_model_revision),(settings.llm_model_name,settings.llm_model_revision)]:
        matches=[rev for repo in cache.repos if repo.repo_id==name for rev in repo.revisions if rev.commit_hash==revision]
        assert matches and any(f.file_name.endswith(('.safetensors','.bin')) for f in matches[0].files),'Required immutable model weights absent'
    source=tmp_path/'source.txt';source.write_text('The fixture event year is 2024.',encoding='utf-8')
    real=RAGPipeline(settings=settings);real.index_file(source)
    response=real.ask('What is the fixture event year?')
    assert response['diagnostics']['generation_invoked'] is True
    assert response['diagnostics']['reason'] in {'answered','model_no_evidence'}
