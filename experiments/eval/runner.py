"""Local evaluation runner; source validation and model execution stay separate."""
import argparse
import dataclasses
from datetime import datetime,timezone
import json
import math
from pathlib import Path

from app.rag.config import Settings
from app.rag.diagnostics import provenance,trace_scope,phase
from app.rag.pipeline import RAGPipeline,INSUFFICIENT_EVIDENCE
from app.rag.retrieval import retrieve
from app.rag.vector_store import VectorStore
from experiments.eval.scoring import (ANSWERABLE,sha,legacy_qrels,validate_oracle,validate_splits,
    retrieval_row,retrieval_metrics,threshold_sweep,choose_threshold,lexical_match,answer_metrics)

PROJECT_DIRECTORY=Path(__file__).resolve().parents[2]


def resolve_dataset(path):
    path=Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIRECTORY/path).resolve()


def load_inputs(path,corpus_root=None):
    path=resolve_dataset(path);dataset=json.loads(path.read_text(encoding='utf-8'))
    root=Path(corpus_root).resolve() if corpus_root else PROJECT_DIRECTORY
    corpus={}
    for item in dataset.get('documents',[]):
        file=(root/item).resolve()
        if file.name in corpus:raise ValueError('Ambiguous corpus document basename')
        raw=file.read_bytes();text=raw.decode('utf-8').replace('\r\n','\n').replace('\r','\n')
        corpus[file.name]={'text':text,'sha256':sha(text.encode()),'file_sha256':sha(raw),'path':str(file)}
    questions=dataset.get('questions',[]);validate_splits(questions)
    manifest_path=path.parent/'legacy_manifest.json'
    if path==PROJECT_DIRECTORY/'experiments/eval/dataset.json' and manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['dataset_sha256']!=sha(path.read_bytes()):raise ValueError('Legacy manifest dataset hash mismatch')
        for q in questions:q.update({k:v for k,v in manifest['memberships'][q['id']].items() if k in {'split','fact_group'}})
    validate_splits(questions)
    qrels={q['id']:q.get('qrels',legacy_qrels(q,corpus)) for q in questions}
    errors={q['id']:validate_oracle(qrels[q['id']],corpus,q.get('category') in ANSWERABLE) for q in questions}
    return path,dataset,corpus,qrels,{k:v for k,v in errors.items() if v}


def _retrieval_row(question,pipeline,groups,corpus,budget):
    s=pipeline.settings
    with trace_scope(s,pipeline.vector_store) as trace:
        def embedding(*args,**kwargs):
            with phase('query_embedding'):return pipeline.embedding_function(*args,**kwargs)
        with phase('retrieval'):
            results=retrieve(question['question'],pipeline.vector_store,top_k=s.top_k,relevance_threshold=float('-inf'),
                embedding_function=embedding,embedding_model_name=s.embedding_model_name,query_prefix=s.embedding_query_prefix)
        raw_diagnostics=trace.snapshot()
    chunks=[{**c,**c.get('metadata',{})} for c in pipeline.vector_store.chunks]
    row=retrieval_row(question,groups,corpus,chunks,results,s.relevance_threshold,budget)
    row['raw_retrieval_diagnostics']=raw_diagnostics
    return row


def _answer_row(row,pipeline):
    with trace_scope(pipeline.settings,pipeline.vector_store) as trace:
        try:
            result=pipeline.ask(row['question'])
            row.update(answer=result['answer'],verification=result['verification'],refused=result['answer']==INSUFFICIENT_EVIDENCE,
                diagnostics=result.get('diagnostics',trace.snapshot()),status='completed')
            row['lexical_match']=lexical_match(result['answer'],row['expected_answer']) if row['category'] in ANSWERABLE else None
            annotation=row.get('rubric')
            if annotation and (not annotation.get('origin') or annotation.get('answer_sha256')!=sha(result['answer'].encode())):
                row['rubric']=None;row['rubric_annotation_status']='unverified_answer_identity_or_origin'
        except Exception as exc:
            row.update(status='error',error={'type':type(exc).__name__},diagnostics=trace.snapshot(),lexical_match=False)


def run_evaluation(dataset_path,settings=None,*,pipeline=None,llm=False,corpus_root=None,
        context_budget_chars=16000,preflight_only=False,tune=False,output_path=None,execution_kind='real_model',frozen_selection=None):
    settings=settings or Settings()
    if not isinstance(settings.relevance_threshold,(float,int)) or not math.isfinite(settings.relevance_threshold):raise ValueError('Finite threshold required')
    if context_budget_chars<=0:raise ValueError('Context budget must be positive')
    path,dataset,corpus,qrels,errors=load_inputs(dataset_path,corpus_root)
    identity={'schema_version':2,'dataset_sha256':sha(path.read_bytes()),
        'corpus':{k:{'source_sha256':v['sha256'],'file_sha256':v['file_sha256']} for k,v in sorted(corpus.items())},
        'qrels_sha256':sha(qrels),'diagnostics_identity':provenance(settings),
        'split_membership':[{k:q.get(k) for k in ('id','split','fact_group')} for q in dataset.get('questions',[])],
        'evaluator_source_sha256':sha(Path(__file__).read_bytes()+(Path(__file__).parent/'scoring.py').read_bytes()),
        'settings':dataclasses.asdict(settings),'context_budget_chars':context_budget_chars}
    tuning_identity={**identity,'settings':{k:v for k,v in identity['settings'].items() if k!='relevance_threshold'}}
    if frozen_selection:
        artifact=json.loads(Path(frozen_selection).read_text())
        if artifact['tuning_fingerprint']!=sha(tuning_identity):raise ValueError('Threshold input identity mismatch')
        if not isinstance(artifact.get('selected_threshold'),(float,int)) or not math.isfinite(artifact['selected_threshold']):raise ValueError('No feasible frozen threshold selection')
        settings=dataclasses.replace(settings,relevance_threshold=artifact['selected_threshold']);identity['settings']=dataclasses.asdict(settings)
    report={'schema_version':2,'created_at':datetime.now(timezone.utc).isoformat(),'dataset':str(path),
        'fingerprint':sha(identity),'identity':identity,'tuning_fingerprint':sha(tuning_identity),'execution_kind':execution_kind,
        'status':'failed' if errors else 'completed','data_gate':'failed' if errors else 'source_labels_valid',
        'quality_promotion_gate':'blocked_oracle_invalid' if errors else 'not_evaluated','model_resistance':'unmeasured','oracle_errors':errors,
        'preflight_questions':len(dataset.get('questions',[])),
        'questions':[],'index_errors':[],'mode':'preflight' if preflight_only else ('tuning' if tune else 'evaluation')}
    def checkpoint():
        if output_path:
            target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if preflight_only:checkpoint();return report
    pipeline=pipeline or RAGPipeline(settings=settings,vector_store=VectorStore(signature=settings.embedding_signature(),max_chunks=settings.max_store_chunks,max_text_bytes=settings.max_store_text_bytes))
    if frozen_selection:pipeline.settings=settings
    for name,source in corpus.items():
        try:pipeline.index_file(source['path'])
        except Exception as exc:report['index_errors'].append({'document':name,'type':type(exc).__name__});report['status']='failed'
    for q in dataset.get('questions',[]):
        if tune and q.get('split','legacy_regression')!='calibration':continue
        try:row=_retrieval_row(q,pipeline,qrels[q['id']],corpus,context_budget_chars)
        except Exception as exc:
            row={'id':q['id'],'question':q['question'],'category':q.get('category','uncategorized'),'split':q.get('split','legacy_regression'),
                'rank':None,'top_score':None,'relevant_score':None,'expected_answer':q.get('answer',[]),
                'oracle_invalid':bool(errors.get(q['id'])),'status':'error','error':{'type':type(exc).__name__}}
        if llm and not row.get('error'):_answer_row(row,pipeline)
        if row.get('error'):report['status']='failed'
        report['questions'].append(row);checkpoint()
    rows=report['questions'];report['retrieval']=retrieval_metrics(rows)
    report['index_observation']={'signature':pipeline.vector_store.signature,'effective_dimension':pipeline.vector_store.index.d if pipeline.vector_store.index is not None else None,
        'operation_id':pipeline.vector_store.operation_id,'chunks':len(pipeline.vector_store.chunks)}
    report['threshold_sweep']=threshold_sweep([r for r in rows if r.get('split')=='calibration'],settings.relevance_threshold)
    if tune:
        calibration=[r for r in rows if r.get('split')=='calibration' and not r.get('error')]
        if not calibration:
            report['status']='failed';report['tuning_error']='No calibration rows; legacy data is not silently repurposed'
        else:
            try:report['selected_threshold']=choose_threshold(calibration,settings.relevance_threshold,dataset.get('threshold_objective',{}))
            except ValueError as exc:report['status']='failed';report['tuning_error']=str(exc)
            report['threshold_objective']=dataset.get('threshold_objective')
    if llm:report['answers']=answer_metrics(rows)
    report['split_counts']={split:sum(r.get('split','legacy_regression')==split for r in rows) for split in ('legacy_regression','calibration','holdout')}
    report['quality_promotion_gate']='blocked_oracle_invalid' if errors else 'not_evaluated'
    if any(r.get('coverage_unverified') for r in rows):report['quality_promotion_gate']='blocked_coverage_unverified'
    checkpoint();return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='experiments/eval/dataset.json');p.add_argument('--corpus-root',type=Path)
    p.add_argument('--llm',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--tune',action='store_true')
    p.add_argument('--threshold',type=float);p.add_argument('--top-k',type=int);p.add_argument('--context-budget-chars',type=int,default=16000)
    p.add_argument('--frozen-selection',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();settings=Settings()
    settings=dataclasses.replace(settings,**{k:v for k,v in {'relevance_threshold':args.threshold,'top_k':args.top_k}.items() if v is not None})
    output=args.output or PROJECT_DIRECTORY/'experiments/results'/f'{datetime.now():%Y%m%d-%H%M%S-%f}.json'
    report=run_evaluation(args.dataset,settings,llm=args.llm,corpus_root=args.corpus_root,context_budget_chars=args.context_budget_chars,
        preflight_only=args.preflight_only,tune=args.tune,output_path=output,frozen_selection=args.frozen_selection,
        execution_kind='source_preflight' if args.preflight_only else 'real_model')
    print(json.dumps({'status':report['status'],'data_gate':report['data_gate'],'quality_promotion_gate':report['quality_promotion_gate'],
        'retrieval':report.get('retrieval'),'answers':report.get('answers'),'report':str(output)},ensure_ascii=False,indent=2))
    return 0 if report['status']=='completed' else 1
