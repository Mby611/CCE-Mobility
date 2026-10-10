from __future__ import annotations

import argparse
from importlib import import_module
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TP_IRRELEVANT_TRACKED_PREFIXES = ("src/cce_mobility/integrations/mobilitybench_",)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resume TravelPlanner memory-curation validation or test after an "
            "unrelated Git commit, while enforcing the original locked protocol."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("val", "test"), required=True)
    args = parser.parse_args()

    runner = _load_runner()
    runner.load_dotenv(PROJECT_ROOT / ".env")
    config = runner._read_json(args.config)
    runner._validate_config(config)
    root = runner._project_path(config["output_root"])
    preflight_path = root / "preflight.json"
    if not preflight_path.exists():
        raise FileNotFoundError(f"Locked training preflight not found: {preflight_path}")

    preflight = runner._read_json(preflight_path)
    gate = validate_resume_gate(runner, config=config, preflight=preflight, root=root)
    stage_root = root / ("validation" if args.stage == "val" else "test")
    stage_root.mkdir(parents=True, exist_ok=True)
    runner._write_json(stage_root / "resume_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)

    if args.stage == "val":
        runner.run_validation(config, arms=runner.ARMS)
    else:
        runner.run_test(config)
    return 0


def validate_resume_gate(
    runner: ModuleType,
    *,
    config: dict[str, Any],
    preflight: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    if preflight.get("schema_version") != runner.SCHEMA_VERSION:
        raise ValueError("Locked preflight schema does not match the current runner")
    if preflight.get("training_protocol") != runner.TRAINING_PROTOCOL:
        raise ValueError("Locked training protocol does not match the current runner")
    if preflight.get("method_version") != runner.METHOD_VERSION:
        raise ValueError("Locked method version does not match the current runner")

    expected_config_hash = runner._stable_hash(config)
    if preflight.get("config_sha256") != expected_config_hash:
        raise ValueError(
            "Config changed since training: "
            f"locked={preflight.get('config_sha256')} current={expected_config_hash}"
        )

    current_provenance = runner._code_provenance()
    runner._require_committed_clean_code(current_provenance)
    locked_provenance = preflight.get("formal_code_provenance") or {}
    compatibility_mode, changed_tracked_files = _source_compatibility(
        locked_provenance,
        current_provenance,
        project_root=runner.PROJECT_ROOT,
    )

    checked_artifacts: list[str] = []
    for section in ("initial_state", "required_inputs"):
        for name, artifact in (preflight.get(section) or {}).items():
            _assert_artifact_unchanged(runner, artifact, label=f"{section}.{name}")
            checked_artifacts.append(f"{section}.{name}")

    cycles = int(config["cycles"])
    runner._require_training_complete(root, cycles=cycles, arms=runner.ARMS)
    manifest_count = 0
    training_commits: set[str] = set()
    for arm in runner.ARMS:
        for cycle in range(1, cycles + 1):
            path = root / arm / f"round_{cycle}" / "round_manifest.json"
            manifest = runner._read_json(path)
            runner._validate_completed_manifest(manifest)
            manifest_provenance = manifest.get("formal_code_provenance") or {}
            _assert_source_exact(locked_provenance, manifest_provenance)
            commit = str(manifest_provenance.get("git_commit") or "")
            if commit:
                training_commits.add(commit)
            manifest_count += 1

    return {
        "schema_version": "travelplanner-memory-curation-resume-gate-v1",
        "status": "passed",
        "reason": "tp_dependency_scope_and_locked_artifacts_unchanged",
        "source_compatibility_mode": compatibility_mode,
        "changed_tracked_files_outside_tp_scope": changed_tracked_files,
        "locked_git_commit": locked_provenance.get("git_commit"),
        "current_git_commit": current_provenance.get("git_commit"),
        "git_commit_changed": (
            locked_provenance.get("git_commit") != current_provenance.get("git_commit")
        ),
        "formal_source_sha256": current_provenance.get("formal_source_sha256"),
        "config_sha256": expected_config_hash,
        "checked_artifacts": checked_artifacts,
        "validated_training_manifests": manifest_count,
        "training_manifest_commits": sorted(training_commits),
    }


def _assert_source_exact(locked: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("formal_source_sha256", "formal_file_count"):
        if locked.get(key) != current.get(key):
            raise ValueError(
                f"Formal experiment source changed since training ({key}): "
                f"locked={locked.get(key)} current={current.get(key)}"
            )


def _source_compatibility(
    locked: dict[str, Any],
    current: dict[str, Any],
    *,
    project_root: Path,
) -> tuple[str, list[str]]:
    if all(
        locked.get(key) == current.get(key) for key in ("formal_source_sha256", "formal_file_count")
    ):
        return "exact_runtime_formal_source", []

    locked_commit = str(locked.get("git_commit") or "")
    current_commit = str(current.get("git_commit") or "")
    if not locked_commit or not current_commit:
        raise ValueError(
            "Formal source hash changed and Git commits are unavailable for a "
            "dependency-scoped comparison"
        )
    changed = _tracked_formal_changes(
        project_root,
        locked_commit=locked_commit,
        current_commit=current_commit,
    )
    unsafe = [path for path in changed if not _is_tp_irrelevant_tracked_path(path)]
    if unsafe:
        raise ValueError(
            "TravelPlanner experiment dependencies changed since training: " + ", ".join(unsafe)
        )
    return "tp_dependencies_unchanged_ignored_runtime_or_mb_only_drift", changed


def _tracked_formal_changes(
    project_root: Path,
    *,
    locked_commit: str,
    current_commit: str,
) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "diff",
            "--name-only",
            f"{locked_commit}..{current_commit}",
            "--",
            "src/cce_mobility",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "Could not compare locked and current formal sources: " + completed.stderr.strip()
        )
    return sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _is_tp_irrelevant_tracked_path(path: str) -> bool:
    return path.startswith(TP_IRRELEVANT_TRACKED_PREFIXES)


def _assert_artifact_unchanged(
    runner: ModuleType,
    artifact: dict[str, Any],
    *,
    label: str,
) -> None:
    path = Path(str(artifact.get("path") or ""))
    if not path.exists():
        raise FileNotFoundError(f"Locked artifact is missing: {label}={path}")
    current = runner._artifact(path)
    if current != artifact:
        raise ValueError(
            f"Locked artifact changed since training: {label}; locked={artifact} current={current}"
        )


def _load_runner() -> ModuleType:
    return import_module(
        "cce_mobility.workflows.travelplanner.run_travelplanner_memory_curation_ablation"
    )


if __name__ == "__main__":
    raise SystemExit(main())
