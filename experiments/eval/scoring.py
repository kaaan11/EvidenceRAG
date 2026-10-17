"""Source-level qrels and transparent metrics. No model/judge calls."""
import hashlib
import copy
import json
import math
import re
import statistics

ANSWERABLE = ("answerable", "answerable_english")
UNANSWERABLE = ("unanswerable_related", "unanswerable_offtopic")


def sha(value):
    if not isinstance(value,bytes): value=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def share(values):
    values=list(values)
    return sum(values)/len(values) if values else None


def contains(text,expected):
    if not expected: return False
    if expected[0].isdigit(): return re.search(rf"(?<![\d.,]){re.escape(expected)}(?![.,]?\d)",text) is not None
    return expected.casefold() in text.casefold()


def lexical_match(answer,groups):
    return bool(groups) and all(any(contains(answer,alt) for alt in group) for group in groups)


def legacy_qrels(question,corpus):
    """Preserve legacy source anchors, expanded to their source paragraph, not new gold."""
    if not question.get("document"): return []
    doc=question["document"];source=corpus.get(doc)
    if source is None: return [[{"document":doc,"start":-1,"end":-1,"text":"","sha256":"missing"}]]
    text=source["text"];anchor=question.get("evidence","")
    at=text.find(anchor) if anchor else -1
    if at<0: return [[{"document":doc,"start":-1,"end":-1,"text":anchor,"sha256":source["sha256"]}]]
    start=text.rfind("\n\n",0,at)+2 if "\n\n" in text[:at] else 0
    end=text.find("\n\n",at+len(anchor));end=len(text) if end<0 else end
    return [[{"document":doc,"start":start,"end":end,"text":text[start:end],"sha256":source["sha256"]}]]


def validate_oracle(groups,corpus,answerable=True):
    if not groups: return [] if not answerable else ["missing_qrels"]
    errors=[]
    for alternatives in groups:
        if not alternatives: errors.append("empty_group")
        for span in alternatives:
            source=corpus.get(span.get("document"));start=span.get("start");end=span.get("end")
            if source is None: errors.append("wrong_document");continue
            if span.get("sha256")!=source["sha256"]: errors.append("source_hash_mismatch");continue
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(source["text"]):
                errors.append("invalid_span");continue
            if source["text"][start:end]!=span.get("text"): errors.append("span_not_in_source")
    return errors


def _metadata(chunk): return {**chunk.get("metadata",{}),**{k:v for k,v in chunk.items() if k!="metadata"}}


def span_covered(span,chunks):
    intervals=[]
    for chunk in chunks:
        data=_metadata(chunk)
        if data.get("filename")!=span["document"]: continue
        start,end=data.get("char_start"),data.get("char_end")
        if isinstance(start,int) and isinstance(end,int):
            # Source checksum/version prevents offsets from another representation.
            if data.get("source_sha256")==span["sha256"]: intervals.append((start,end))
        elif " ".join(span["text"].split()) in " ".join(data.get("text","").split()):
            return True  # Legacy chunks lack offsets; only whole-span textual coverage.
    cursor=span["start"]
    for start,end in sorted(intervals):
        if start>cursor: break
        if end>cursor: cursor=end
        if cursor>=span["end"]:return True
    return False


def group_hit(groups,chunks):
    return any(all(span_covered(span,chunks) for span in group) for group in groups)


def select_context(chunks,budget_chars):
    selected=[];used=0
    for chunk in chunks:
        size=len(chunk.get("text",""))
        if used+size<=budget_chars:
            selected.append(chunk);used+=size
    return selected


def group_rank(groups,results,budget_chars):
    """Earliest prefix completing any sufficient AND group within whole-chunk budget."""
    for rank in range(1,len(results)+1):
        if group_hit(groups,select_context(results[:rank],budget_chars)): return rank
    return None


def chunk_coverage(groups,chunks,budget_chars,max_states=10000):
    """Bounded exact sufficient-group packing. Search exhaustion is unknown, not miss."""
    unknown=False
    for group in groups:
        atoms=[]
        for span in group:
            cuts={span["start"],span["end"]}
            for chunk in chunks:
                data=_metadata(chunk)
                if data.get("filename")==span["document"] and data.get("source_sha256")==span["sha256"]:
                    cuts.update(x for x in [data.get("char_start"),data.get("char_end")] if isinstance(x,int) and span["start"]<x<span["end"])
            cuts=sorted(cuts)
            atoms.extend({**span,"start":a,"end":b,"text":span["text"][a-span["start"]:b-span["start"]]} for a,b in zip(cuts,cuts[1:]))
        if len(atoms)>4096:unknown=True;continue
        target=(1<<len(atoms))-1
        options=[]
        for chunk in chunks:
            mask=sum(1<<i for i,atom in enumerate(atoms) if span_covered(atom,[chunk]))
            if mask:options.append((mask,len(chunk.get("text",""))))
        stack=[(0,0)];best={0:0};steps=0
        while stack:
            mask,cost=stack.pop();steps+=1
            if mask==target:return True
            if steps>max_states:unknown=True;break
            needed=(target^mask)&-(target^mask)
            for bits,size in options:
                new=mask|bits;total=cost+size
                if bits&needed and total<=budget_chars and total<best.get(new,float('inf')):
                    best[new]=total;stack.append((new,total))
    return None if unknown else False


def retrieval_row(question,groups,corpus,chunks,results,threshold,budget_chars):
    errors=validate_oracle(groups,corpus,question.get("category") in ANSWERABLE)
    answerable=question.get("category") in ANSWERABLE
    coverage=chunk_coverage(groups,chunks,budget_chars) if answerable and not errors else None
    raw_rank=group_rank(groups,results,budget_chars) if coverage else None
    accepted=[r for r in results if r.get("score") is not None and r["score"]>=threshold]
    accepted=select_context(accepted,budget_chars)
    return {"id":question["id"],"question":question["question"],"category":question.get("category","uncategorized"),
        "split":question.get("split","legacy_regression"),"fact_group":question.get("fact_group",question["id"]),
        "oracle_invalid":bool(errors),"oracle_errors":errors,"chunk_coverage_failure":coverage is False,
        "coverage_unverified":answerable and not errors and coverage is None,
        "rank":raw_rank,"top_score":results[0].get("score") if results else None,
        "relevant_score":results[raw_rank-1].get("score") if raw_rank else None,
        "accepted_evidence":bool(accepted),"gold_retained":group_hit(groups,accepted) if answerable and not errors else None,
        "retrieval_miss":bool(coverage and raw_rank is None),"expected_answer":question.get("answer",[]),
        "rubric":question.get("rubric"),"qrels":groups,"raw_results":results,"accepted_context":accepted}


def retrieval_metrics(rows):
    metrics={}
    for category in ANSWERABLE:
        selected=[r for r in rows if r["category"]==category and not r.get("oracle_invalid")]
        ranks=[r["rank"] for r in selected]
        metrics[category]={"count":len(selected),"hit_at_1":share(r==1 for r in ranks),
            "hit_at_k":share(r is not None for r in ranks),"group_mrr":share(1/r if r else 0 for r in ranks),
            "chunk_coverage_failures":sum(r.get("chunk_coverage_failure",False) for r in selected),
            "coverage_unverified":sum(r.get("coverage_unverified",False) for r in selected),
            "gold_retained":share(r.get("gold_retained",False) for r in selected)}
    return metrics


def threshold_sweep(rows,current):
    scores={float(r[k]) for r in rows for k in ("top_score","relevant_score") if r.get(k) is not None and math.isfinite(r[k])}
    candidates=sorted({float(current),*scores,*[math.nextafter(s,math.inf) for s in scores]})
    sweep=[]
    for threshold in candidates:
        values={"threshold":threshold}
        for category,key,positive in [("answerable","answerable_kept",True),("answerable_english","english_kept",True),
            ("unanswerable_related","related_rejected",False),("unanswerable_offtopic","offtopic_rejected",False)]:
            selected=[r for r in rows if r["category"]==category and not r.get("oracle_invalid")]
            values[key]=share((r.get("relevant_score") is not None and r["relevant_score"]>=threshold) if positive
                else (r.get("top_score") is None or r["top_score"]<threshold) for r in selected)
        sweep.append(values)
    return {"selection":"descriptive_only","recommended_threshold":None,"sweep":sweep}


def choose_threshold(rows,current,objective):
    if any(r.get("split")!="calibration" for r in rows):raise ValueError("Only calibration rows may select a threshold")
    if objective.get("name")!="balanced_category_retention_rejection":raise ValueError("Predeclared objective required")
    candidates=[]
    for item in threshold_sweep(rows,current)["sweep"]:
        if all(item.get(key) is not None and item[key]>=minimum for key,minimum in objective.get("constraints",{}).items()):
            rates=[v for k,v in item.items() if k!="threshold" and v is not None]
            if rates:candidates.append((sum(rates)/len(rates),-abs(item["threshold"]-current),item["threshold"]))
    return max(candidates)[2] if candidates else None


def validate_splits(questions):
    memberships={}
    for q in questions:
        group=q.get("fact_group",q["id"]);split=q.get("split","legacy_regression")
        if split in {"calibration","holdout"} and not q.get("fact_group"):raise ValueError("Explicit fact groups required for calibration/holdout")
        memberships.setdefault(group,set()).add(split)
    if any("calibration" in splits and "holdout" in splits for splits in memberships.values()):
        raise ValueError("Fact/paraphrase group leaks between calibration and holdout")
    ids=[q["id"] for q in questions]
    if len(set(ids))!=len(ids):raise ValueError("Duplicate question IDs")


def answer_metrics(rows):
    metrics={}
    for category in ANSWERABLE+UNANSWERABLE+("uncategorized",):
        selected=[r for r in rows if r["category"]==category and not r.get("oracle_invalid")]
        lexical=[r for r in selected if r.get("lexical_match") is not None]
        metrics[category]={"count":len(selected),"lexical_match_rate":share(bool(r.get("lexical_match")) for r in lexical),
            "end_to_end_lexical_with_coverage_rate":share(bool(r.get("lexical_match")) and not r.get("chunk_coverage_failure") and not r.get("error") for r in lexical),
            "refusals":sum(bool(r.get("refused")) for r in selected),"errors":sum(bool(r.get("error")) for r in selected),
            "semantic_rubric_rated":sum(r.get("rubric") is not None for r in selected)}
    invoked=[r for r in rows if r.get("diagnostics",{}).get("generation_invoked")]
    def duration(row):return row["diagnostics"].get("timings_ms",{}).get("generation")
    durations=[duration(r)/1000 for r in invoked if duration(r) is not None]
    metrics["generation_invocations"]={"count":len(invoked),"timed_count":len(durations),
        "median_seconds":statistics.median(durations) if durations else None,
        "errors":sum(bool(r.get("error")) for r in invoked),
        "model_refusals":sum(r["diagnostics"].get("reason")=="model_no_evidence" for r in invoked)}
    return metrics


def compare_reports(before,after,ablation=None):
    if before["fingerprint"]!=after["fingerprint"]:
        if not ablation or ablation.get('variable')!='chunking' or not ablation.get('hypothesis'):
            raise ValueError("Incompatible data/model/prompt/chunker identity; declare an ablation and compare raw rows explicitly")
        a,b=before['identity'],after['identity']
        def invariant(identity):
            value=copy.deepcopy(identity)
            for key in ['chunk_size','chunk_overlap','chunker_strategy','ingestion_version']:value['settings'].pop(key,None)
            diag=value['diagnostics_identity'];diag.pop('identity_sha256',None)
            for key in ['index_signature','configured_index_signature']:
                if key in diag:diag[key].pop('ingestion',None)
            return value
        if invariant(a)!=invariant(b):raise ValueError('Non-chunking input changed during ablation')
        if [r['id'] for r in before.get('questions',[])]!=[r['id'] for r in after.get('questions',[])]:raise ValueError('Ablation denominator/IDs changed')
    return {"before":before.get("retrieval"),"after":after.get("retrieval"),"ablation":ablation,
        'before_rows':before.get('questions'),'after_rows':after.get('questions')}
