"""Bounded in-process ASGI/native-store fixture; never targets remote services."""
import argparse
import asyncio
import ctypes
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import threading
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def peak_rss_bytes():
    if os.name=='nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
                ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
        fn=ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not fn(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(counters),counters.cb):raise OSError('Cannot read process RSS')
        return counters.PeakWorkingSetSize
    import resource
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform=='darwin' else peak*1024


def validate_profile(profile):
    if profile.get('mode')!='native_asgi_fake_models':raise ValueError('Only bounded local fake-model fixture mode is supported')
    bounds={'jobs':(1,32),'max_active':(1,1),'max_waiting':(0,4),'work_ms':(1,100),
        'timeout_seconds':(1,60),'max_rss_mib':(64,2048),'max_index_bytes':(1024,10485760),'payload_bytes':(32,4096)}
    for key,(low,high) in bounds.items():
        value=profile.get(key)
        if type(value) is not int or not low<=value<=high:raise ValueError('Profile value outside bounded range: '+key)


async def exercise(profile):
    validate_profile(profile)
    import httpx2
    import numpy as np
    import main
    from app.rag.config import Settings
    from app.rag.pipeline import RAGPipeline
    from app.rag.runtime import HeavyRuntime
    from app.rag.vector_store import VectorStore
    if peak_rss_bytes()>profile['max_rss_mib']*1024**2:raise RuntimeError('Initial fixture RSS exceeds profile budget')
    active=0;high_water=0;lock=threading.Lock()
    def generator(*args,**kwargs):
        nonlocal active,high_water
        with lock:active+=1;high_water=max(high_water,active)
        try:
            time.sleep(profile['work_ms']/1000)
            return 'Fixture event year is 2024. [Kanıt 1]'
        finally:
            with lock:active-=1
    settings=Settings(relevance_threshold=.1)
    runtime=HeavyRuntime(profile['max_active'],profile['max_waiting'],profile['max_waiting'],profile['timeout_seconds'])
    original=main.rag_pipeline
    with tempfile.TemporaryDirectory(prefix='evidencerag-capacity-') as directory:
        store=VectorStore(Path(directory)/'store.npz',settings.embedding_signature(),autoload=False)
        pipeline=RAGPipeline(settings=settings,runtime=runtime,vector_store=store,
            embedding_function=lambda texts,name:np.array([[1.]+[0.]*383 for _ in texts],dtype='float32'),generation_function=generator)
        main.rag_pipeline=pipeline
        try:
            async with main.lifespan(main.app):
                transport=httpx2.ASGITransport(app=main.app,client=('127.0.0.1',1))
                async with httpx2.AsyncClient(transport=transport,base_url='http://127.0.0.1') as client:
                    payload=b'Fixture event year is 2024. '+b'x'*(profile['payload_bytes']-28)
                    setup=time.monotonic()
                    upload=await client.post('/documents',files={'file':('fixture.txt',payload)})
                    if upload.status_code!=200:raise RuntimeError('Synthetic index setup failed')
                    setup_ms=(time.monotonic()-setup)*1000
                    async def one(index):
                        started=time.monotonic();response=await client.post('/ask',json={'question':'Fixture event year?'})
                        data=response.json()
                        return {'index':index,'status':response.status_code,'request_ms':(time.monotonic()-started)*1000,
                            'diagnostics':data.get('diagnostics')}
                    started=time.monotonic()
                    rows=await asyncio.wait_for(asyncio.gather(*(one(i) for i in range(profile['jobs']))),profile['timeout_seconds'])
                    elapsed=time.monotonic()-started
                    health=(await client.get('/health')).json()
            owned_bytes=sum(p.stat().st_size for p in Path(directory).iterdir() if p.is_file())
        finally:main.rag_pipeline=original
    peak=peak_rss_bytes();ok=[r for r in rows if r['status']==200];busy=[r for r in rows if r['status']==503 and r['diagnostics']['reason']=='busy']
    within=peak<=profile['max_rss_mib']*1024**2 and owned_bytes<=profile['max_index_bytes'] and high_water<=profile['max_active']
    return {'schema_version':1,'execution_kind':'native_asgi_fake_models_no_network_no_model_weights',
        'status':'passed' if within and len(ok)+len(busy)==len(rows) else 'failed','profile':profile,
        'machine':{'platform':platform.platform(),'python':platform.python_version(),'processor':platform.processor(),'logical_cpus':os.cpu_count()},
        'requests':len(rows),'completed':len(ok),'busy_rejected':len(busy),'observed_max_active':high_water,
        'fixture_elapsed_seconds':elapsed,'fixture_requests_per_second':len(ok)/elapsed,
        'fixture_index_setup_ms':setup_ms,'cold_model_load_ms':None,
        'fixture_median_request_ms':statistics.median(r['request_ms'] for r in ok) if ok else None,
        'p95_ms':None,'p95_note':'Small fixture sample; no stable p95 claim',
        'process_peak_rss_bytes':peak,'owned_index_bytes':owned_bytes,'healthy_during_service':health['status']=='ok',
        'rows':rows,'real_model_calls':0,'quality_and_model_capacity_gate':'not_evaluated'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();profile=json.loads(args.profile.read_text(encoding='utf-8'));validate_profile(profile)
    result=asyncio.run(exercise(profile));args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
