# Metrics and Denominators

CCE and official benchmark metrics measure different populations.

| Metric | Calculation | Direction |
| --- | --- | --- |
| Recovery | Seven Judge dimensions in [0,1]; case score is four times their mean, rounded to three decimals; average valid scored counterfactual cases | Higher |
| Low / Poor Recovery | Valid scored counterfactual cases with raw Recovery strictly below 2, divided by scored cases | Lower |
| Unavailable (TP) | Applicable intervened cases whose final structured itinerary still selects an unavailable resource ID, divided by cases carrying such IDs | Lower |
| CF-Final (TP) | Complete local commonsense and applicable hard constraints pass in the changed world, divided by valid constraint evaluations | Higher |
| CF-Final (MB) | Benchmark-specific continuation recovery acceptance, including parseability and applicable hard checks | Higher |
| TP Delivery | Nonempty evaluable factual plans divided by factual tasks | Higher |
| TP CS Micro | Passed commonsense checks divided by configured check instances | Higher |
| TP CS Macro | Tasks passing every commonsense check divided by factual tasks | Higher |
| TP Hard Micro | Passed hard checks divided by applicable hard-check instances | Higher |
| TP Hard Macro | Tasks passing all applicable hard constraints divided by factual tasks | Higher |
| TP Final | Tasks passing both complete commonsense and complete hard constraints divided by factual tasks | Higher |
| MB DR / FPR | Official delivery / fully successful factual responses; use upstream evaluator definitions | Higher |
| MB TS-R | Upstream Tool Selection Recall; the released implementation uses one minus tool redundancy | Higher |
| MB DEC-R | Coverage of required planning steps in the official evaluator | Higher |
| MB IT / OT | Mean input/output tokens per factual task | Lower cost |

README Recovery percentages are raw Recovery / 4 * 100; it is a score, not accuracy. Low and CF-Final small tables remain fractions. A raw score exactly 2 is not Low. Training failure is a different predicate: score below 3 or failed final acceptance. Read the saved protocol before combining runs.

Denominators may differ: score coverage, constraint coverage, cases with unavailable IDs, factual tasks and model calls are not interchangeable. Report configured task/challenge counts, effective denominators, parse rates, LLM errors and Judge fallback counts. Cases are not independent when multiple interventions share a factual task; paper claims should prefer task-clustered paired intervals.

中文说明：Recovery 是每个反事实案例的 0–4 分数均值，不是 test1000 的任务成功率；同一任务可能有多条挑战。Low 严格小于 2；Unavailable 不把仅提到被拒绝资源计为使用。官方 factual 与动态 counterfactual 指标必须分开报告。
