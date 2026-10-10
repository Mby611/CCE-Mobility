# Development

## Layout

```text
CCE-Mobility/
  README.md                 English paper entry
  README.zh-CN.md            Chinese entry with identical commands/tables
  assets/                   Paper motivation and method PNGs
  configs/
    models/                 Public service examples without keys
    travelplanner/          TP method templates
    mobilitybench/          MB continuation templates
    baselines/              Upstream/adapter baseline templates
    ablations/              Named experiment options and budget provenance
    smoke/                  Synthetic no-key configs
  src/cce_mobility/          Shared method code and installed workflows
    cli.py                  Unified cce command
    workflows/              TP, MB, baseline and analysis implementations
  scripts/                  Upstream MobilityBench Shell launcher only
  tools/                    Explicit release allowlist and audit/export tools
  tests/
  docs/
  data/                     Ignored official downloads
  third_party/              Ignored upstream checkouts/runtimes
  outputs/                  Ignored experiment artifacts
```

## Local Checks

```bash
python -m pip install -e '.[dev,travelplanner,mobilitybench,data]'
cce --help
cce run-config configs/smoke/travelplanner_offline.json
cce run-config configs/smoke/mobilitybench_mock.json
python -m pytest -q
ruff check .
python tools/audit_publication.py
python tools/build_public_candidate.py --output-dir /tmp/cce-source-check
python tools/build_public_candidate.py --verify-dir /tmp/cce-source-check
```

## Entry Points

Training, checkpoint selection, frozen testing and official evaluation use the `cce`
CLI described in the README. Python workflows live only inside the installed package;
there are no mirrored Python entry points under `scripts/`. Invoke advanced helpers
as modules, for example:

```bash
python -m cce_mobility.workflows.public.baselines --help
python -m cce_mobility.workflows.travelplanner.analyze_travelplanner_paired_significance --help
python -m cce_mobility.workflows.mobilitybench.prepare_mobilitybench_continuation_protocol --help
```

The only script is `scripts/run_mobilitybench_official_smoke.sh`, which launches the
upstream MobilityBench CLI in its separate runtime. It is not a second method
implementation. `tools/` contains repository maintenance commands, not experiment
workflow copies.

The exporter uses an explicit allowlist, scans sensitive content, permits reviewed paper PNGs only, refuses symlinks and existing destinations, and writes SHA-256 hashes. It does not copy Git history, local keys, runtime configs, datasets or output artifacts. Core source fingerprints exclude the ignored local key file; generated configs and model settings remain outside tracked-source integrity scope.

CI runs Python 3.12 tests, lint, offline TP/MB smoke, link/table checks and clean export. Network/provider/upstream runtime acceptance is separately documented; CI does not use secrets or rerun formal experiments.

The publication audit scans the worktree, exact Git index and every locally reachable
commit, including deleted files. It rejects files outside the explicit allowlist,
credentials, internal addresses, unreviewed data/binaries and unreviewed image metadata. Only
the named synthetic fixtures and reviewed paper PNGs are exempt from the data/binary
restriction. Findings contain file locations and rule names, never matched values.
A shallow clone fails the history check; fetch its full history before auditing.
Initial commits containing only `README.md`, `LICENSE` and `.gitignore` can precede
the source allowlist. Their contents and commit metadata are still scanned; this
exception never permits source files, datasets or credentials without a manifest.
Ignored local keys, datasets and outputs are excluded from publication, not certified
safe to share. Do not force-add them or upload the whole working directory as an archive.
Pattern checks do not establish code ownership or third-party redistribution rights.
Corporate author/committer emails are reported separately for publication privacy review.
To also match opaque keys from a local key file without importing it or printing values:

```bash
python tools/audit_publication.py --known-secrets-file src/cce_mobility/local_secrets.py
```

Use `cce_mobility` imports and `cce` CLI. Do not add private gateway defaults, personal paths or research iteration labels to public files. Schema revision identifiers describe file formats, not method variants.
