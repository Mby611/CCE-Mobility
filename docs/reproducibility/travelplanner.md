# TravelPlanner Reproduction

## Official Inputs

Clone [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner) into `third_party/TravelPlanner`. Choose and checkout an upstream commit before preparation; `prepare-config --code-revision COMMIT` verifies it, or omission records current HEAD. Tracked upstream modifications are rejected. See [source revisions and compatibility scope](upstream-versions.md).

```bash
git clone https://github.com/OSU-NLP-Group/TravelPlanner third_party/TravelPlanner
cce prepare-data --benchmark travelplanner --output data/travelplanner
```

The official [HF dataset](https://huggingface.co/datasets/osunlp/TravelPlanner) contains train/validation/test CSV queries and corresponding reference JSONL. Each reference row is a resource dictionary, not necessarily a query-bearing object. Preparation checks aligned counts and cross-split query identities, copies the official files and writes `split_manifest.json`. To reuse an existing download:
```bash
cce prepare-data --benchmark travelplanner --source /path/to/official_data --output data/travelplanner
```
Pass `--revision HF_COMMIT` to fix a known data revision. Network downloads resolve mutable names to an immutable commit on the first run and reuse cache afterward. Local source provenance uses file hashes, not an invented remote revision.

Download the official environment database using the link in the upstream README, including [the official database archive](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link). Extract it under `third_party/TravelPlanner/database/` so that `database/flights/clean_Flights_2022.csv` exists. The dataset query/reference files alone are not the environment database. This repository does not redistribute either.

## Configure and Run

Follow [provider setup](../setup/providers.md), then:
```bash
cce prepare-config --benchmark travelplanner \
  --models configs/local/models.json --data-dir data/travelplanner \
  --repo-path third_party/TravelPlanner \
  --output configs/generated/travelplanner.json \
  --output-root outputs/travelplanner --name travelplanner_cce
cce check-models configs/generated/travelplanner.json
cce train configs/generated/travelplanner.json
cce validate configs/generated/travelplanner.json
cce test configs/generated/travelplanner.json
cce evaluate-official configs/generated/travelplanner.json
```

All stages read one generated experiment config. Public defaults use train45, val180, test1000, seed42 and three matched strict cycles. See [budget profiles](budgets.md) for archived adaptive phase lengths and caps; they are not silently interpreted as cycle counts.

Training saves each `full/round_N/round_manifest.json`, frozen interventions, factual contexts, pre/candidate/final records, outcome buckets and Agent/World/ontology memories. Validation uses independent hold-out tasks, writes `validation/matrix_report.json` and `selected_checkpoint.json`. Test freezes the selected Agent memory and challenge set, and writes `test_manifest.json`. The separate `official/` directory contains exported factual plans, evaluator logs, scores, detailed constraint results and protocol hashes.

Generate a new experiment for custom counts:
```bash
cce prepare-config --benchmark travelplanner \
  --models configs/local/models.json --data-dir data/travelplanner \
  --repo-path third_party/TravelPlanner --train-tasks 3 --cycles 1 \
  --max-phase-updates 2 --validation-tasks 3 --test-tasks 3 \
  --output configs/generated/tp_small.json --output-root outputs/tp_small
```

## Official Test Compatibility

The upstream evaluator has a test scoring branch but some revisions omit the test query-loading branch. The adapter loads the pinned local test CSV, keeps supplied constraint fields, and deterministically derives missing compatibility labels from the query. It does not rewrite upstream commonsense/hard constraint functions or modify the upstream checkout.

For a subset, denominators are configured task count, eight commonsense checks per task and budget plus applicable local constraints. The adapter records its hash and upstream evaluator hash; unsupported upstream loader layouts fail explicitly. This is disclosed **offline compatibility evaluation**, not submission to an official leaderboard. It requires `.[travelplanner]` and the official database.

## Comparing Methods

Use identical frozen challenge JSONL plus `.manifest.json` sidecars via `--frozen-validation` and `--frozen-test` in each generated config. Keep the same Agent/Judge protocol and data revisions. Do not compare fresh WM-generated sets as if they were paired. Baselines and complementarity have a [separate guide](../baselines.md).

中文提示：先准备官方 query/reference 数据和环境 database，再配置自己的服务。训练、选点、测试分别运行；测试只读验证选点。公共新实验的冻结集不等于论文历史冻结集，不能据此宣称精确复现数值。
