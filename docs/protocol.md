# Experiment Protocol

## Final CCE

One paired cycle freezes the current World state, factual contexts, executable challenge identities, and recovery specifications. It evaluates the current Agent, generates candidate recovery skills, and compares candidate memory on the same challenges. Paired memory curation retains, repairs, or rejects modifications with support-sensitive outcome evidence; successful pre-update cases are retention anchors.

After curation, the final Agent memory is frozen and re-evaluated on the **same frozen challenge set**, not a newly sampled set. Classify paired final outcomes as S→S, F→S, S→F, or F→F. Only F→F records drive World-memory/spec/ontology learning. S→F is logged as Agent regression and never routed to WM. Agent acts on query, observable history and tool results, not Judge rationales, privileged sandbox state, or hidden recovery contracts.

World memory influences subsequent training challenges, not online actions. Agent memory is used through agent-owned top-2 retrieval. Optional memory repair encourages verified alternatives without benchmark-specific hard-coded flight/hotel rules.

## Named Ablations

- `full`: Agent and World update, paired curation, final same-challenge evaluation.
- `agent-only`: Agent updates with curation; World memory and ontology stay byte-identical.
- `without-memory-curation`: candidate Agent changes are directly accepted; final paired re-evaluation and F→F filtering remain.

Use separate output roots and the same data/model/budget settings. For held-out comparisons, pass the same `--frozen-validation` and `--frozen-test` artifact paths with their manifest sidecars. Changing the challenge identities invalidates a paired comparison.

## Selection and Test

The selection score is
`Recovery - 0.7*Low - 0.8*Unavailable + 0.2*CF-Final`.
TP ties use Recovery, lower Unavailable, lower Low, then CF-Final. MB uses Recovery then earlier cycle. Validation writes selection split, rule, metrics, memory path/hash and configuration hash. Test refuses a missing or changed validation selection; it never selects using test metrics.

Hold-out challenges are locked before candidate evaluation. No method or world-memory updates occur in test. Source, data, protocol config and selected artifacts are hashed. One process owns each experiment root. Resuming reuses complete outputs; changed inputs require a new output root. Unresolved model/Judge errors must not enter a claimed clean comparison.

## Reporting Boundary

TP dynamic recovery is CCE's intervened-world evaluation. Official TP factual metrics come from a separate factual planning execution with the selected guidance, using upstream constraint functions and a disclosed offline loader/denominator compatibility adapter.

MB uses upstream sandbox execution, frozen official prompt bundles, tool-result state reconstruction, and real Planner–Worker–Reporter continuation. Its standard official factual metrics are evaluated on the input official factual traces; dynamic continuation scores are reported separately. These metrics are not a combined leaderboard, and posthoc text generation is not full continuation.

## Exact Reproduction

Public downloads and current APIs permit workflow reproduction, not guaranteed numerical repetition. Historical challenge sets, raw traces, selected memories and private model snapshots are not bundled. A freshly generated challenge set is recorded as `historical_paper_artifact=false`; it must not be renamed or described as the original paper set. The historical MB main panel is development-heldout500; the fresh formal continuation test is a separate panel. Do not merge them.

The matched strict paired cycle profile is explicitly distinguished from archived adaptive phase budgets in [budgets.md](reproducibility/budgets.md).
