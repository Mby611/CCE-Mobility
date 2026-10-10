# Training Budget Profiles and Sources

The archived experiment JSONs and the paper's matched mechanism experiments have different budgets. Their units must not be conflated.

| Profile | Training tasks | Budget | Source |
| --- | ---: | --- | --- |
| Archived adaptive TP memory training | 45 | Agent phase 3–8 updates, World phase 2–6 updates, patience 2; at most 3 cycles and 42 phase updates; early stop after 2 unproductive cycles | Archived training configuration inspected during release preparation |
| Matched paired TP curation experiment | 45 | 3 cycles; each has pre/candidate/final evaluation, curation and residual World update | Archived formal curation configuration |
| TP world-evolution mechanism panel | 45 | Reported final round-4 comparison; validation panel 149 tasks / 247 challenges | Main-paper Table 3 |
| Formal MB continuation mechanism | 20 per cycle | 3 cycles with disjoint task schedule, 3 Agent evaluations per cycle | Supplied formal continuation runner/config and paper mechanism panel |

The unified `cce train` default is the **strict same-challenge paired-cycle** profile, not the archived online adaptive runner. It implements the final residual/curation protocol. The archived 42-update cap and phase lengths are supplied as provenance in `configs/ablations/training_budgets.json`; they are not claimed to execute automatically in the strict three-cycle profile.

`--cycles` controls complete paired cycles. `--max-phase-updates` bounds Agent+World memory updates (two per complete full cycle) and must be at least twice the requested cycle count. Pre/candidate/final evaluations are not three cycles. Changing caps, task counts or seed writes a new effective configuration and requires a new output root.

The archive source file hashes are stored with the profile to make extraction auditable without publishing private configs, keys or local paths. Exact historical trained checkpoints and challenge sets are not included. A strict new run must not be described as the identical historical training schedule or as exact numerical reproduction.
