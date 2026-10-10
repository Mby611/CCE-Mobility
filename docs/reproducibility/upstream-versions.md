# Upstream Revisions

Always use a clean upstream checkout and record a full commit hash. `cce prepare-config --code-revision COMMIT` verifies the checkout and locks that revision in the experiment. It never applies undocumented changes to an upstream repository. Dataset revisions are locked independently by `prepare-data`.

The following public revisions were used as source references during packaging:

| Dependency | Public source revision | Scope |
| --- | --- | --- |
| [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner/commit/e52c87f4ac348a3410c46dc3553c519db5ec5e23) | `e52c87f4ac348a3410c46dc3553c519db5ec5e23` | Original evaluator API; test-loader compatibility is supplied by CCE without modifying constraint functions |
| [MobilityBench](https://github.com/AMAP-ML/MobilityBench) | `addac8657fc0369c3868dfd42c08e6ee42708ae2` | Upstream origin/main recorded in the inspected local clone; the live sandbox/runtime acceptance for this exact public checkout remains to be run |
| [SkillOpt](https://github.com/microsoft/SkillOpt/commit/6940e46f4e0e537a1d7ca8432f24248d0d3550f5) | `6940e46f4e0e537a1d7ca8432f24248d0d3550f5` | Upstream trainer source reference; install its independent runtime |

These are source references, not claims of paid end-to-end acceptance or exact historical model reproduction. Do not use a private mirror's orphan commit as a public upstream revision.

For TravelPlanner:

```bash
git -C third_party/TravelPlanner checkout e52c87f4ac348a3410c46dc3553c519db5ec5e23
cce prepare-config --benchmark travelplanner \
  --models configs/local/models.json --data-dir data/travelplanner \
  --repo-path third_party/TravelPlanner \
  --code-revision e52c87f4ac348a3410c46dc3553c519db5ec5e23 \
  --output configs/generated/travelplanner.json \
  --output-root outputs/travelplanner --name travelplanner_cce
```

Use a new output directory when changing revisions. A missing commit, dirty upstream source, changed model config, or mutated dataset fails explicitly. Newer upstream commits may require compatibility verification; no automatic upgrade is performed mid-experiment.
