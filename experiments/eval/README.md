# Evaluation schema 2

Run from any working directory with the evaluator's absolute path, or from the
repository using `python experiments/evaluate.py`. Relative dataset paths resolve
against the repository; absolute/external private dataset paths are supported.
Document paths retain the legacy repository-root rule. Use `--corpus-root PATH`
explicitly for an external corpus; document basenames must be unique. Reports are
local and may include raw questions, answers and retrieved text: do not publish
private reports as CI artifacts. Checked-in rubric/baseline fixtures are synthetic.

`python experiments/evaluate.py --preflight-only --output <private-output.json>`
validates source labels without model downloads. Normal retrieval and `--llm`
execute actual configured adapters and require the T00 cache/budget policy. A
fixture-injected pipeline is labeled separately and is never model quality evidence.

## Labels, splits and metrics

The existing 41 IDs/data remain unchanged. `legacy_manifest.json` records exposure
to earlier tuning and keeps all members in `legacy_regression`. It is not a new
holdout. Its SHA, membership and fact groups are validated. No calibration/holdout
results or real-model baseline have been measured in this delivery.

Schema-2 questions can provide `qrels`: an OR list of sufficient AND groups. Each
group is a list of `{document, sha256, start, end, text}` source spans. Offsets are
Unicode character positions, `[start,end)`, in UTF-8 decoded text with CRLF/CR
normalized to LF; sha256 hashes that exact representation. The original file hash
is recorded separately. Document/hash/span errors are `oracle_invalid` and block
promotion. Legacy `evidence` anchors are expanded to their exact source paragraph;
this preserves compatibility but does not constitute independent semantic gold review.

All valid questions remain in the primary denominator. `chunk_coverage_failure`
means no sufficient chunk group fits the declared character context budget.
`retrieval_miss` means a sufficient group exists but raw top-k did not complete it.
Coverage packing is bounded to 10,000 states/4,096 span atoms; exhausted search is
`coverage_unverified`, never silently called a data error or retrieval success.
Whole chunks are selected in ranked order while skipping chunks that do not fit.
Legacy chunks use whole-span textual containment; offset-bearing chunks require
matching source hashes and complete interval coverage. PDF qrels require an explicit
page-text representation; this text-corpus runner does not invent PDF byte spans.

Raw `hit_at_1`, `hit_at_k`, `group_mrr` precede thresholds. Group rank is the first
ranked prefix that completes any sufficient AND group within the budget. It is
not a single-chunk MRR claim. Raw chunks and accepted context are retained per row.
Threshold-retained gold, no accepted evidence, coverage failure and miss remain
separate. Empty categories return count 0 and null rates; None is not score 0.

`lexical_match` tests alternative-string groups, not factual truth. For example
`22 değil, 31` can match lexical 22 and be manually rated wrong. Optional rubric
annotations are separate and their origin must be recorded; no automatic semantic
judge is installed. `rubric_fixtures.json` fixes numeric, negation, entity, citation,
refusal and T03 injection expectations; model resistance remains unmeasured.

Calibration/holdout questions need explicit `split` and `fact_group`. The same
fact/paraphrase cannot cross those splits. `--tune` runs calibration only, using a
dataset `threshold_objective` with name `balanced_category_retention_rejection`
and explicit constraints such as `{"answerable_kept":1,"english_kept":1}`.
Candidate thresholds come from observed scores, their next representable values
and the current threshold. No universal .70–.90 range or perfect separation claim.
No feasible candidate returns null. Save the tuning report, then evaluate with
`--frozen-selection <report.json>`; non-threshold input identity must match.
Legacy regression alone cannot select a new default threshold.

## Timing, errors and reproducibility

Rows use T07 `diagnostics.generation_invoked` and `timings_ms.generation` in ms.
Both model answers and model refusals count. Started generator failures count in
a separate error group. Retrieval-only, loader and queue rejections do not count.
Synthetic 1s answer + 99s model refusal therefore has median 50s. Raw retrieval
and later ask are separate probes; their diagnostics are not summed as one request.
Monotonic phase durations can overlap as specified by T07.

Each question failure is retained by type without private exception text and the
remaining questions run. Checkpoints retain partial rows; failed runs never claim
success. Fingerprints include data/corpus/qrel/split, model/tokenizer/prompt/source,
runtime profile and settings identities, excluding timestamps, durations and random
operation/request IDs. Actual index dimension/signature/operation ID are observed
separately. `compare_reports` rejects incompatible inputs unless a chunking ablation
manifest explicitly declares its hypothesis and the unchanged source/model/budget
invariants pass. T09 must keep legacy default until independent quality/cost gates
are measured; fixtures provide mechanics only.
