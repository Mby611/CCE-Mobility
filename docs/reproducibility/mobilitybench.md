# MobilityBench Official Continuation

## Independent Upstream Runtime

Clone [MobilityBench](https://github.com/AMAP-ML/MobilityBench) into `third_party/MobilityBench`, checkout the desired upstream revision, and install its own package in a separate environment according to its README:

```bash
git clone https://github.com/AMAP-ML/MobilityBench third_party/MobilityBench
python3.12 -m venv third_party/MobilityBench/.venv
third_party/MobilityBench/.venv/bin/python -m pip install -e third_party/MobilityBench
third_party/MobilityBench/.venv/bin/python -m pip install -e '.[mobilitybench]'
```

The second install makes CCE modules available inside the upstream runtime. It does not merge that environment with the core environment or SkillOpt. Pin a clean revision before configuration; [source revisions](upstream-versions.md) distinguish inspected source references from live acceptance.

## Official Dataset and Sandbox

Obtain query CSV and replay sandbox from [official HF](https://huggingface.co/datasets/GD-ML/MobilityBench). Prepared train/validation/test splits are seeded, disjoint local subsets, not benchmark-provided official split labels.

```bash
cce prepare-data --benchmark mobilitybench --output data/mobilitybench \
  --train-size 60 --validation-size 100 --test-size 500
```

If the downloaded revision contains multiple full query CSVs, preparation refuses to choose silently. Pass the exact file and sandbox explicitly:
```bash
cce prepare-data --benchmark mobilitybench \
  --source /path/to/official_full_query.csv \
  --sandbox-source /path/to/official_sandbox \
  --output data/mobilitybench \
  --train-size 60 --validation-size 100 --test-size 500
```
Preparation validates unique case IDs, nonempty queries and sandbox files, saves split memberships/hashes, and records remote revision or local source hashes. Sandbox files stay under ignored `data/mobilitybench/sandbox/`. The formal training schedule draws disjoint tasks across cycles; three cycles of train20 require at least 60 training tasks.

## Configure and Run the Complete Flow

Use an OpenAI-compatible acting endpoint in the local model config. Agent and WM share a model; Judge is independently configured. Then:

```bash
cce prepare-config --benchmark mobilitybench \
  --models configs/local/models.json --data-dir data/mobilitybench \
  --repo-path third_party/MobilityBench \
  --mb-python third_party/MobilityBench/.venv/bin/python \
  --output configs/generated/mobilitybench.json \
  --output-root outputs/mobilitybench --name mobilitybench_cce
cce check-models configs/generated/mobilitybench.json
cce train configs/generated/mobilitybench.json
cce validate configs/generated/mobilitybench.json
cce test configs/generated/mobilitybench.json
cce evaluate-official configs/generated/mobilitybench.json
```

Training first generates official plan-and-execute factual traces, converts them to CCE records, freezes the upstream prompt bundle and runtime contract, then runs real reconstructed-state continuation. Every cycle evaluates pre/candidate/final memories on identical cases; only final F→F outcomes update world memory and ontology.

Validation and test generate their own official factual inputs and fixed executable challenge sets, then run selected-memory Planner–Worker–Reporter continuations. Validation selects among completed training cycles. Test writes `test_manifest.json` and does not update memory.

## Auditing and Output Scope

Each factual manifest reports `planner_output_present`, `worker_output_avg_len`, `reporter_output_present`, `error_count`, raw-result hashes, dataset and converted-record hashes. Empty Reporter output, missing Worker tool events, coverage errors and unfinished continuation errors stop the flow.

The official evaluator is invoked from the separate upstream runtime and saves factual metrics in `official/`; a metric exception is a failed stage, not a zero score. These official metrics describe the official factual trace population. CCE continuation recovery metrics remain separate. Do not present the input factual scores as method-specific improvement without independently executing method-conditioned factual traces.

Compatibility code in `patches/mobilitybench/` passes explicit model-body options and controls async-client lifetime. It does not silently enable reasoning-off for every provider or force mandatory tool calls. Upstream prompt text and executed changes are hashed in the continuation protocol.

No-key synthetic adapter tests verify code paths only. A current provider, upstream installation, sandbox revision and paid three-task acceptance run are required to verify the real chain. See [paid-smoke.md](paid-smoke.md). Historical posthoc tables and fresh full-continuation panels are not interchangeable.

中文提示：MB 使用官方 sandbox、Planner–Worker–Reporter 和状态重建 continuation，不是纯文本 posthoc。上游环境单独安装；训练三 cycle 每轮 20 个不重复任务时，训练池至少 60 条。官方 factual 与动态恢复单独产出、单独解释。
