import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.rag.config import Settings
from app.rag.pipeline import RAGPipeline
from experiments.eval.scoring import *
from experiments.eval.runner import load_inputs,resolve_dataset,run_evaluation


def oracle(text='alpha beta gamma'):
    corpus={'a.txt':{'text':text,'sha256':sha(text.encode())}}
    def span(start,end):return {'document':'a.txt','sha256':corpus['a.txt']['sha256'],'start':start,'end':end,'text':text[start:end]}
    return corpus,span


def chunk(span,score=1):
    return {'text':span['text'],'filename':span['document'],'source_sha256':span['sha256'],
        'char_start':span['start'],'char_end':span['end'],'score':score}


def question(qid='q',**kwargs):return {'id':qid,'question':'What fact?','category':'answerable','answer':[['alpha']],**kwargs}


def test_paraphrase_or_and_group_completion_rank():
    corpus,span=oracle();a,b=span(0,5),span(11,16);results=[chunk(a),chunk(b)]
    groups=[[a,b]]
    first=retrieval_row(question(),groups,corpus,results,results,.8,100)
    second=retrieval_row(question('para',question='Same fact in another language'),groups,corpus,results,results,.8,100)
    assert first['rank']==second['rank']==2
    assert group_rank(groups,results[:1],100) is None
    assert group_rank([[a],[b]],results[:1],100)==1
    assert group_rank(groups,results,5) is None


def test_oracle_invalid_vs_chunk_coverage_and_retrieval_miss():
    corpus,span=oracle();gold=span(0,5);good=chunk(gold)
    bad={**gold,'document':'wrong.txt'}
    assert retrieval_row(question(),[[bad]],corpus,[good],[good],.8,100)['oracle_invalid']
    for changes in ({'sha256':'wrong'},{'text':'not in source'},{'start':-1}):
        assert validate_oracle([[{**gold,**changes}]],corpus)
    lost=retrieval_row(question(),[[gold]],corpus,[],[],.8,100)
    assert lost['chunk_coverage_failure'] and not lost['oracle_invalid'] and not lost['retrieval_miss']
    missed=retrieval_row(question(),[[gold]],corpus,[good],[],.8,100)
    assert missed['retrieval_miss'] and not missed['chunk_coverage_failure']
    oversized=retrieval_row(question(),[[gold]],corpus,[good],[good],.8,4)
    assert oversized['chunk_coverage_failure'] and oversized['rank'] is None


def test_split_span_can_be_covered_by_selected_group_and_search_exhaustion_visible():
    corpus,span=oracle('alpha beta');groups=[[span(0,10)]]
    parts=[chunk(span(0,5)),chunk(span(5,10))]
    assert chunk_coverage(groups,parts,10) is True
    assert group_rank(groups,parts,10)==2
    assert group_rank(groups,parts[:1],10) is None
    assert chunk_coverage(groups,parts,10,max_states=0) is None


def test_raw_rank_does_not_depend_on_threshold():
    corpus,span=oracle();gold=span(0,5);results=[chunk(gold,.78)]
    low=retrieval_row(question(),[[gold]],corpus,results,results,.7,100)
    high=retrieval_row(question(),[[gold]],corpus,results,results,.8,100)
    assert low['rank']==high['rank']==1
    assert low['gold_retained'] and not high['gold_retained']


def test_p11_threshold_does_not_hide_tr_en_or_related_failures():
    rows=[{'category':c,'top_score':top,'relevant_score':relevant,'split':'calibration'} for c,top,relevant in [
        ('answerable',.90,.90),('answerable',None,None),('answerable_english',.78,.78),
        ('unanswerable_related',.95,None),('unanswerable_offtopic',.70,None)]]
    item=next(r for r in threshold_sweep(rows,.8)['sweep'] if r['threshold']==.8)
    assert item['answerable_kept']==.5 and item['english_kept']==0 and item['related_rejected']==0
    assert choose_threshold(rows,.8,{'name':'balanced_category_retention_rejection','constraints':{'answerable_kept':1,'english_kept':1}}) is None


def test_threshold_zero_negative_none_empty_categories():
    rows=[{'category':'answerable','top_score':0.,'relevant_score':0.}]
    zero=next(x for x in threshold_sweep(rows,0)['sweep'] if x['threshold']==0)
    assert zero['answerable_kept']==1 and zero['english_kept'] is None
    negative=next(x for x in threshold_sweep([{'category':'answerable','top_score':-.2,'relevant_score':-.2}],-.3)['sweep'] if x['threshold']==-.3)
    assert negative['answerable_kept']==1
    assert threshold_sweep([],0)['sweep'][0]['answerable_kept'] is None
    assert retrieval_metrics([])['answerable']['hit_at_k'] is None
    assert answer_metrics([])['generation_invocations']['median_seconds'] is None


def test_calibration_and_holdout_fact_leak_rejected():
    questions=[question('tr',split='calibration',fact_group='fact'),question('en',split='holdout',fact_group='fact')]
    with pytest.raises(ValueError):validate_splits(questions)
    with pytest.raises(ValueError):choose_threshold([{'split':'holdout'}],.8,{'name':'balanced_category_retention_rejection'})
    with pytest.raises(ValueError):validate_splits([question(split='holdout')])


def write_dataset(tmp_path,questions=None):
    (tmp_path/'a.txt').write_text('alpha beta gamma',encoding='utf-8')
    dataset={'documents':['a.txt'],'questions':questions if questions is not None else [question(document='a.txt',evidence='alpha')]}
    path=tmp_path/'dataset.json';path.write_text(json.dumps(dataset),encoding='utf-8');return path


def test_relative_absolute_external_paths_and_empty_missing_category(tmp_path,monkeypatch):
    import experiments.eval.runner as runner
    dataset=write_dataset(tmp_path)
    monkeypatch.setattr(runner,'PROJECT_DIRECTORY',tmp_path)
    assert resolve_dataset('dataset.json')==dataset
    assert load_inputs(dataset,tmp_path)[0]==dataset
    assert not load_inputs(dataset,tmp_path)[4]
    outside=tmp_path/'external';outside.mkdir()
    other=outside/'private.json';other.write_text(dataset.read_text())
    report=run_evaluation(other,corpus_root=tmp_path,preflight_only=True,output_path=outside/'report.json')
    assert report['status']=='completed' and Path(report['dataset'])==other
    empty=write_dataset(tmp_path,[])
    assert run_evaluation(empty,corpus_root=tmp_path,preflight_only=True)['quality_promotion_gate']=='not_evaluated'
    assert retrieval_row({'id':'unknown','question':'?','category':'uncategorized'},[],{},[],[],.8,100)['category']=='uncategorized'


def test_generation_median_includes_refusal_error_excludes_loader_queue():
    def row(reason,invoked,ms,error=False):
        return {'category':'answerable','rank':None,'diagnostics':{'reason':reason,'generation_invoked':invoked,'timings_ms':{'generation':ms}},'error':error}
    rows=[row('answered',True,1000),row('model_no_evidence',True,99000),row('retrieval_no_evidence',False,None)]
    assert answer_metrics(rows)['generation_invocations']['median_seconds']==50
    rows += [row('model_error',True,2000,True),row('model_error',False,None,True),row('busy',False,None,True)]
    metrics=answer_metrics(rows)['generation_invocations']
    assert metrics['count']==3 and metrics['errors']==1 and metrics['model_refusals']==1


def test_partial_error_preserves_all_rows_and_actual_invocation(tmp_path):
    from app.rag.generation import LLMGenerationError
    path=write_dataset(tmp_path,[question('bad',document='a.txt',evidence='alpha'),question('good',document='a.txt',evidence='alpha')])
    calls=[]
    def generate(*a,**k):
        calls.append(1)
        if len(calls)==1:raise LLMGenerationError('private')
        return 'alpha [Kanıt 1]'
    pipeline=RAGPipeline(settings=Settings(relevance_threshold=.1),embedding_function=lambda texts,name:np.array([[1.,0.] for _ in texts],dtype='float32'),generation_function=generate)
    output=tmp_path/'partial.json'
    report=run_evaluation(path,pipeline.settings,pipeline=pipeline,llm=True,corpus_root=tmp_path,output_path=output,execution_kind='fixture_native_faiss_fake_models')
    assert report['status']=='failed' and len(report['questions'])==2
    assert report['questions'][0]['diagnostics']['generation_invoked'] is True
    assert report['questions'][1]['lexical_match'] is True
    assert len(json.loads(output.read_text())['questions'])==2


def test_lexical_negation_and_independent_rubric_fixture():
    assert lexical_match('22 değil, 31',[['22']]) is True
    path=Path(__file__).resolve().parents[1]/'experiments/eval/rubric_fixtures.json'
    fixture=json.loads(path.read_text(encoding='utf-8'))
    negation=next(r for r in fixture['cases'] if r['id']=='negation')
    assert negation['rubric']['decision']=='wrong' and negation['rubric']['origin']=='fixture_author'
    assert all(c['rubric'].get('reason') for c in fixture['cases'])


def test_fingerprint_stable_time_changes_corpus_and_prompt_change(tmp_path,monkeypatch):
    import experiments.eval.runner as runner
    path=write_dataset(tmp_path)
    one=run_evaluation(path,corpus_root=tmp_path,preflight_only=True)
    two=run_evaluation(path,corpus_root=tmp_path,preflight_only=True)
    assert one['fingerprint']==two['fingerprint']
    (tmp_path/'a.txt').write_text('changed alpha beta gamma')
    three=run_evaluation(path,corpus_root=tmp_path,preflight_only=True)
    assert three['fingerprint']!=one['fingerprint']
    original=runner.provenance
    monkeypatch.setattr(runner,'provenance',lambda settings:{**original(settings),'prompt_sha256':'changed'})
    assert run_evaluation(path,corpus_root=tmp_path,preflight_only=True)['fingerprint']!=three['fingerprint']
    with pytest.raises(ValueError):compare_reports(one,three)


def test_declared_chunking_ablation_rejects_other_changes(tmp_path):
    path=write_dataset(tmp_path)
    before=run_evaluation(path,corpus_root=tmp_path,preflight_only=True)
    after=copy.deepcopy(before);after['identity']['settings']['chunker_strategy']='structured_v2';after['fingerprint']=sha(after['identity'])
    manifest={'variable':'chunking','hypothesis':'Preserve source structure within same context budget'}
    assert compare_reports(before,after,manifest)['ablation']==manifest
    after['identity']['settings']['embedding_query_prefix']='changed'
    with pytest.raises(ValueError):compare_reports(before,after,manifest)


def test_ten_source_valid_questions_keep_denominator_when_chunker_loses_one():
    corpus,span=oracle();gold=span(0,5);good=chunk(gold)
    rows=[retrieval_row(question(str(i)),[[gold]],corpus,[good] if i else [],[good] if i else [],.8,100) for i in range(10)]
    for row in rows:row['lexical_match']=not row['chunk_coverage_failure']
    assert retrieval_metrics(rows)['answerable']['count']==10
    assert retrieval_metrics(rows)['answerable']['hit_at_k']==.9
    assert answer_metrics(rows)['answerable']['end_to_end_lexical_with_coverage_rate']==.9


def test_tuning_never_evaluates_holdout_and_frozen_evaluation_preserves_labels(tmp_path):
    questions=[question('cal',document='a.txt',evidence='alpha',split='calibration',fact_group='cal-fact'),
        question('held',document='a.txt',evidence='gamma',answer=[['gamma']],split='holdout',fact_group='held-fact')]
    path=write_dataset(tmp_path,questions)
    data=json.loads(path.read_text());data['threshold_objective']={'name':'balanced_category_retention_rejection','constraints':{'answerable_kept':1}}
    path.write_text(json.dumps(data));before=path.read_bytes()
    def pipeline():return RAGPipeline(settings=Settings(relevance_threshold=.8),embedding_function=lambda texts,name:np.array([[1.,0.] for _ in texts],dtype='float32'))
    first=pipeline();tuned=run_evaluation(path,first.settings,pipeline=first,corpus_root=tmp_path,tune=True,execution_kind='fixture')
    assert [q['id'] for q in tuned['questions']]==['cal']
    artifact=tmp_path/'threshold.json';artifact.write_text(json.dumps(tuned))
    second=pipeline();evaluated=run_evaluation(path,second.settings,pipeline=second,corpus_root=tmp_path,frozen_selection=artifact,execution_kind='fixture')
    assert evaluated['mode']=='evaluation' and len(evaluated['questions'])==2
    assert path.read_bytes()==before
