# Three-Task Paid Acceptance

These commands are provided for reader/server acceptance. They were **not executed against paid services during release preparation**. Protocol mocks and synthetic smoke do not establish live model or upstream sandbox acceptance. Use your own local keys, explicitly available models, downloaded official inputs, and isolated upstream runtimes.

## TravelPlanner

```bash
cce prepare-config --benchmark travelplanner \
  --models configs/local/models.json --data-dir data/travelplanner \
  --repo-path third_party/TravelPlanner \
  --train-tasks 3 --cycles 1 --max-phase-updates 2 \
  --validation-tasks 3 --test-tasks 3 --seed 42 \
  --output configs/generated/tp_acceptance.json \
  --output-root outputs/tp_acceptance --name tp_acceptance
cce check-models configs/generated/tp_acceptance.json
cce train configs/generated/tp_acceptance.json
cce validate configs/generated/tp_acceptance.json
cce test configs/generated/tp_acceptance.json
cce evaluate-official configs/generated/tp_acceptance.json
```

## MobilityBench

Prepare at least three distinct tasks in each split and nonempty official sandbox files, then:

```bash
cce prepare-config --benchmark mobilitybench \
  --models configs/local/models.json --data-dir data/mobilitybench \
  --repo-path third_party/MobilityBench \
  --mb-python third_party/MobilityBench/.venv/bin/python \
  --train-tasks 3 --cycles 1 --max-phase-updates 2 \
  --validation-tasks 3 --test-tasks 3 --seed 42 \
  --output configs/generated/mb_acceptance.json \
  --output-root outputs/mb_acceptance --name mb_acceptance
cce check-models configs/generated/mb_acceptance.json
cce train configs/generated/mb_acceptance.json
cce validate configs/generated/mb_acceptance.json
cce test configs/generated/mb_acceptance.json
cce evaluate-official configs/generated/mb_acceptance.json
```

## Acceptance Checklist

Require a complete round manifest, consistent pre/candidate/final challenge keys, F→F-only world input, no hidden recovery fields in acting prompts, validation-only selection, unchanged test memory hash and complete requested coverage. Inspect model errors and Judge fallbacks, not just exit status. MB also requires actual Worker tool calls and Reporter output. Official evaluator exceptions fail acceptance.

Run benchmarks in different output roots to avoid overlapping writers. Do not reuse or overwrite completed paper runs. An acceptance run measures workflow validity, not method quality or statistical significance.
