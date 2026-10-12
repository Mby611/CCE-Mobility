# Baselines and Ablations

## Implementations and Protocols

| Method | Implementation | Evaluation updates |
| --- | --- | --- |
| No evolution | Same task adapter, no learned memory | None |
| [Official SkillOpt](https://github.com/microsoft/SkillOpt) | Upstream trainer with the supplied task adapter | Artifact frozen after training/selection |
| [Reflexion](https://github.com/noahshinn/reflexion) | Task-adapted reflection memory in `methods/baselines.py` | Online, after each scored case |
| [Voyager](https://github.com/MineDojo/Voyager) | Task-adapted reusable skill memory in `methods/baselines.py` | Online, after each scored case |

Reflexion and Voyager are adapted to mobility planning and recovery in this repository. These implementations do not reproduce the original upstream environments. Report their online information access and evaluation order separately from frozen methods. Baselines use the same configured Agent, WM and Judge endpoints; do not transfer a learned artifact between backbone models as a matched baseline.

## TravelPlanner Comparison

Complete the [CCE workflow](reproducibility/travelplanner.md), then generate named baseline configurations on its unchanged held-out challenges:

```bash
uv run python -m cce_mobility.workflows.public.baselines \
  --experiment configs/generated/tp.json \
  --output configs/generated/tp_baselines
uv run cce run-config configs/generated/tp_baselines/no_evolution.json
uv run cce run-config configs/generated/tp_baselines/reflexion.json
uv run cce run-config configs/generated/tp_baselines/voyager.json
```

These runs execute their own factual planning and retain those outputs. They do not relabel CCE's shared no-evolution factual contexts as method-specific official results. Match `(sample_id, intervention_id)` and disclose any differing factual context when interpreting paired recovery comparisons.

## Official SkillOpt

Clone SkillOpt into `third_party/SkillOpt`, record its checked-out commit, and follow its own dependency instructions in a separate environment. It must provide `skillopt.model.qwen_backend`, `scripts/train.py`, and the trainer adapter registry; incompatible upstream interfaces fail explicitly. CCE does not vendor or silently replace that trainer. Put the CCE `src` directory on that runtime's `PYTHONPATH` and install the adapter's Pydantic, YAML, pandas and numpy dependencies without downgrading the main CCE runtime.

Materialize **training and validation only**, never test feedback:

```bash
uv run python -m cce_mobility.workflows.baselines.materialize_official_skillopt_dataset \
  --train-interventions outputs/tp/full/round_1/frozen_cycle_interventions.jsonl \
  --val-interventions outputs/tp/challenges/validation.jsonl \
  --repo-path third_party/TravelPlanner --data-root data/travelplanner \
  --output-dir outputs/skillopt/travelplanner_dataset
```

Use `configs/baselines/travelplanner_skillopt.example.yaml` as a starting point, copying it to `configs/local/skillopt.yaml`. Fill in your explicitly available models, endpoint and Judge fields. Keys are read from `local_secrets.py`; the wrapper bridges them only inside the trainer process. Start it with the **separate SkillOpt Python executable**, not an assumed compatible main environment:

```bash
PYTHONPATH=src third_party/SkillOpt/.venv/bin/python \
  -m cce_mobility.workflows.baselines.run_official_skillopt \
  --skillopt-root third_party/SkillOpt --config configs/local/skillopt.yaml
uv run python -m cce_mobility.workflows.public.baselines \
  --experiment configs/generated/tp.json --output configs/generated/tp_baselines \
  --skillopt-artifact outputs/skillopt/travelplanner/best_skill.md
uv run cce run-config configs/generated/tp_baselines/official_skillopt.json
```

For SkillOpt + CCE, create a new experiment with `cce prepare-config --base-agent-guidance outputs/skillopt/travelplanner/best_skill.md` plus the same data/model arguments as Quick Start. The artifact's hash is locked; residual recovery memory is trained on top of that frozen guidance. Reuse validation/test challenges with `--frozen-validation` and `--frozen-test`. Do not reuse the original CCE memory as a substitute for this residual training.

## MobilityBench Baselines

Use official factual traces and frozen challenges from the [MB workflow](reproducibility/mobilitybench.md). The module `cce_mobility.workflows.mobilitybench.run_mobilitybench_official_continuation` accepts the same locked runtime/prompt/case inputs with empty Agent memory for no-evolution. For online adaptations, `cce_mobility.workflows.mobilitybench.run_mobilitybench_official_prequential` accepts `--method reflexion` or `--method voyager`; invoke it with `python -m MODULE --help` and provide the same factual/case inputs. This posthoc baseline interface is not the CCE continuation training loop. Its result scope must be disclosed separately.

The official SkillOpt MB adapter is `src/cce_mobility/integrations/skillopt_mobilitybench.py`; its example YAML is in `configs/baselines/` and its dataset materializer is `cce_mobility.workflows.baselines.materialize_official_skillopt_mobilitybench_dataset`. Its upstream trainer environment is likewise separate. These public provider combinations have not all been live-tested; connectivity and paid smoke are required before formal runs.

## Mechanism Ablations

Use `cce prepare-config --variant agent-only` or `--variant without-memory-curation` with a separate output directory. Agent-only freezes World memory/ontology. Without memory curation directly accepts candidate Agent edits, but retains the same final reevaluation and F-to-F World routing. Keep task budget, seed, service settings, initial state and held-out challenges matched. Neither ablation is a separate historical method version.

See [budgets](reproducibility/budgets.md), [metrics](metrics.md) and [protocol](protocol.md) for budget and interpretation boundaries.
