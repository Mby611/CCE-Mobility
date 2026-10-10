# Method

See the final [experiment protocol](protocol.md) for executable counterfactual challenges, same-challenge Agent updates, paired memory curation, F→F-only World updates and top-2 retrieval.

The unified public entry is `cce`. Internal schema revisions are serialization formats, not method generations. Named options `agent-only` and `without-memory-curation` implement ablations without duplicated method code.

[Budget units](reproducibility/budgets.md) distinguish phase updates, evaluations and cycles. [Metrics](metrics.md) distinguish dynamic recovery from official factual quality.
