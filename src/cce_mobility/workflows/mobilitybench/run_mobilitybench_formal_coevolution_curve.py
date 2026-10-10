from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.integrations.mobilitybench_continuation import (  # noqa: E402
    CONTINUATION_PROTOCOL,
    MobilityBenchContinuationCase,
    file_sha256,
    formal_experiment_code_provenance,
    stable_hash,
    validate_continuation_case,
)
from cce_mobility.methods.formal_coevolution import (  # noqa: E402
    FORMAL_METHOD_VERSION,
    FORMAL_TRAINING_PROTOCOL,
)


CURVE_CONFIG_SCHEMA = "mobilitybench-formal-coevolution-curve-v1"
OUTPUT_FILES = {
    "points": "mb_coevolution_main_curve.csv",
    "cases": "mb_coevolution_case_scores.jsonl",
    "deltas": "mb_coevolution_paired_deltas.csv",
    "audit": "mb_coevolution_protocol_audit.json",
    "manifest": "mb_coevolution_artifact_manifest.json",
    "readme": "README_ZH.md",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, run, and export the seven-point formal MobilityBench curve."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("prepare", "eval", "export", "all"), default="prepare")
    parser.add_argument("--only", default="", help="Optional comma-separated point names.")
    args = parser.parse_args()
    config = _read_json(args.config)
    _validate_config(config)
    if args.stage == "export":
        points, lock = load_locked_export_inputs(config)
        print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)
        export(config, points)
        return 0
    points, lock = prepare(config, config_path=args.config)
    print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "prepare":
        return 0
    selected = {item.strip() for item in args.only.split(",") if item.strip()}
    if args.stage in {"eval", "all"}:
        evaluate(config, points, selected_names=selected)
    if args.stage in {"export", "all"}:
        export(config, points)
    return 0


def load_locked_export_inputs(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the completed curve without rerunning pre-evaluation provenance gates."""

    collection = _project_path(config["collection_dir"])
    lock_path = collection / "protocol_lock.json"
    index_path = collection / "checkpoint_eval_index.json"
    for path in (lock_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(f"Locked curve export input missing: {path}")

    lock = _read_json(lock_path)
    index = _read_json(index_path)
    points = index.get("points") or []
    names = [str(point.get("name") or "") for point in points]
    expected_names = [str(name) for name in lock.get("point_names") or []]
    checks = {
        "lock_schema": lock.get("schema_version") == CURVE_CONFIG_SCHEMA,
        "lock_status": lock.get("status") == "locked_before_formal_evaluation",
        "config_hash": lock.get("config_sha256") == stable_hash(config),
        "point_count": len(points) == int(lock.get("expected_points", -1)) == 7,
        "point_names": names == expected_names and len(set(names)) == len(names),
        "case_count": int(lock.get("expected_cases_per_point", -1))
        == int(config["expected_count"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Locked curve export inputs failed: {failed}")

    locked_hashes = lock.get("eval_config_sha256") or {}
    for point in points:
        name = str(point["name"])
        eval_path = Path(str(point["eval_config_path"])).expanduser()
        if not eval_path.is_file():
            raise FileNotFoundError(f"Locked evaluation config missing: {eval_path}")
        eval_config = _read_json(eval_path)
        if stable_hash(eval_config) != locked_hashes.get(name):
            raise ValueError(f"Locked evaluation config hash differs: {eval_path}")
        if int(eval_config.get("expected_count", -1)) != int(config["expected_count"]):
            raise ValueError(f"Locked evaluation denominator differs: {eval_path}")
        point["eval_config"] = eval_config
        for key in ("agent_memory", "world_memory", "world_ontology"):
            artifact = point.get(key) or {}
            artifact_path = Path(str(artifact.get("path") or "")).expanduser()
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Locked checkpoint artifact missing: {artifact_path}")
            if file_sha256(artifact_path) != artifact.get("sha256"):
                raise ValueError(f"Locked checkpoint artifact hash differs: {artifact_path}")
    return points, lock


def prepare(
    config: dict[str, Any], *, config_path: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    collection = _project_path(config["collection_dir"])
    collection.mkdir(parents=True, exist_ok=True)
    cases_path = _project_path(config["continuation_cases_path"])
    input_audit_path = _project_path(config["input_audit_path"])
    cases_audit_path = cases_path.with_suffix(".audit.json")
    prompt_bundle = _project_path(config["official_prompt_bundle_path"])
    train_root = _project_path(config["train_output_root"])
    for path in (
        cases_path,
        cases_audit_path,
        input_audit_path,
        prompt_bundle,
        train_root / "preflight.json",
    ):
        if not path.exists():
            raise FileNotFoundError(f"Formal curve input missing: {path}")
    cases = _load_cases(cases_path)
    if len(cases) != int(config["expected_count"]):
        raise ValueError(f"Expected {config['expected_count']} frozen cases, got {len(cases)}")
    input_audit, cases_audit = _validate_input_audits(
        config=config,
        cases_path=cases_path,
        input_audit_path=input_audit_path,
        cases_audit_path=cases_audit_path,
    )
    train_preflight = _read_json(train_root / "preflight.json")
    formal_code = formal_experiment_code_provenance(PROJECT_ROOT)
    allow_qoder_only_transition = all(
        config[role]["provider"] == "aihub" for role in ("agent", "judge")
    )
    training_code_transition = _require_matching_formal_code(
        formal_code,
        train_preflight,
        allow_qoder_only_transition=allow_qoder_only_transition,
    )

    initial = train_root / "initial_state"
    points: list[dict[str, Any]] = [
        _point_spec(
            config,
            name="no_evolution_round0",
            series="no_evolution",
            curve_role="shared_round0_anchor",
            round_number=0,
            raw_iteration=0,
            agent_update_count=0,
            training_rollout_count=0,
            agent_memory=initial / "agent_memory.jsonl",
            world_memory=initial / "world_memory.jsonl",
            world_ontology=initial / "world_ontology.jsonl",
            update_accepted=False,
            agent_update_accept_count=0,
            world_memory_changed=False,
            world_ontology_changed=False,
        )
    ]
    checkpoint_index: list[dict[str, Any]] = []
    round_manifests: dict[tuple[str, int], dict[str, Any]] = {}
    for round_number in range(1, int(config["cycles"]) + 1):
        for arm, series, role in (
            ("full", "full_coevolution", "full_world_agent_coevolution"),
            ("agent_only", "agent_only_frozen_world", "agent_only_with_frozen_world"),
        ):
            round_dir = train_root / arm / f"round_{round_number}"
            manifest_path = round_dir / "round_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Training checkpoint manifest missing: {manifest_path}")
            manifest = _read_json(manifest_path)
            checkpoint_code_transition = _validate_training_manifest(
                manifest,
                arm=arm,
                round_number=round_number,
                expected_code=formal_code,
                allow_qoder_only_transition=allow_qoder_only_transition,
            )
            round_manifests[(arm, round_number)] = manifest
            point = _point_spec(
                config,
                name=f"{series}_round{round_number}",
                series=series,
                curve_role=role,
                round_number=round_number,
                raw_iteration=int(manifest["raw_iteration"]),
                agent_update_count=int(manifest["agent_update_attempt_count"]),
                training_rollout_count=int(manifest["cumulative_training_rollout_count"]),
                agent_memory=round_dir / "agent_memory.jsonl",
                world_memory=round_dir / "world_memory.jsonl",
                world_ontology=round_dir / "world_ontology.jsonl",
                update_accepted=bool(manifest["agent_update_accepted"]),
                agent_update_accept_count=int(manifest["agent_update_accept_count"]),
                world_memory_changed=bool(manifest["world_memory_changed"]),
                world_ontology_changed=bool(manifest["world_ontology_changed"]),
            )
            points.append(point)
            checkpoint_index.append(
                {
                    "name": point["name"],
                    "series": series,
                    "completed_cycle": round_number,
                    "raw_iteration": manifest["raw_iteration"],
                    "phase": manifest["phase"],
                    "agent_update_attempt_count": manifest["agent_update_attempt_count"],
                    "agent_update_accept_count": manifest["agent_update_accept_count"],
                    "cumulative_training_rollout_count": manifest[
                        "cumulative_training_rollout_count"
                    ],
                    "agent_update_accepted": manifest["agent_update_accepted"],
                    "world_memory_changed": manifest["world_memory_changed"],
                    "world_ontology_changed": manifest["world_ontology_changed"],
                    "world_update_mode": manifest["world_update_mode"],
                    "agent_memory": point["agent_memory"],
                    "world_memory": point["world_memory"],
                    "world_ontology": point["world_ontology"],
                    "artifact_commit": manifest["artifact_commit"],
                    "training_code_transition": checkpoint_code_transition,
                }
            )

    arm_audit = _validate_curve_arm_invariants(
        train_root=train_root,
        manifests=round_manifests,
        cycles=int(config["cycles"]),
        train_task_count=int(config["train_task_count"]),
    )
    lock = {
        "status": "locked_before_formal_evaluation",
        "schema_version": CURVE_CONFIG_SCHEMA,
        "training_protocol": FORMAL_TRAINING_PROTOCOL,
        "method_version": FORMAL_METHOD_VERSION,
        "continuation_protocol": CONTINUATION_PROTOCOL,
        "config_source": str(config_path.resolve()) if config_path else "programmatic",
        "config_sha256": stable_hash(config),
        "artifact_commit": _git_commit(),
        "formal_code_provenance": formal_code,
        "training_code_transition": training_code_transition,
        "seed": config["seed"],
        "model": config["agent"]["model"],
        "judge": config["judge"]["model"],
        "evaluation_identity": config["evaluation_identity"],
        "evaluation_identity_note": config["evaluation_identity_note"],
        "expected_points": 7,
        "expected_cases_per_point": config["expected_count"],
        "expected_recovery_rollouts": 7 * int(config["expected_count"]),
        "primary_comparison": "full_coevolution_round3 - agent_only_frozen_world_round3",
        "selection_after_results_forbidden": True,
        "continuation_cases": _artifact(cases_path),
        "continuation_cases_audit": _artifact(cases_audit_path),
        "input_audit": _artifact(input_audit_path),
        "input_audit_summary": {
            "status": input_audit["status"],
            "evaluation_identity": input_audit["evaluation_identity"],
            "independent_heldout_claim_allowed": input_audit["independent_heldout_claim_allowed"],
        },
        "continuation_cases_audit_summary": {
            "case_count": cases_audit["case_count"],
            "trigger_resolution_counts": cases_audit["trigger_resolution_counts"],
            "agent_payload_privileged_field_count": cases_audit[
                "agent_payload_privileged_field_count"
            ],
            "agent_visible_context_key_counts": cases_audit["agent_visible_context_key_counts"],
            "agent_visible_context_unexpected_keys": cases_audit[
                "agent_visible_context_unexpected_keys"
            ],
        },
        "official_prompt_bundle": _artifact(prompt_bundle),
        "checkpoint_index": checkpoint_index,
        "comparison_arm_audit": arm_audit,
        "point_names": [point["name"] for point in points],
        "eval_config_sha256": {
            point["name"]: stable_hash(point["eval_config"]) for point in points
        },
        "estimated_cost": _estimate_cost(config),
    }
    for point in points:
        path = Path(point["eval_config_path"])
        if path.exists():
            locked = _read_json(path)
            if stable_hash(
                {k: v for k, v in locked.items() if k != "artifact_commit"}
            ) != stable_hash(
                {k: v for k, v in point["eval_config"].items() if k != "artifact_commit"}
            ):
                raise ValueError(f"Existing point evaluation config differs: {path}")
            point["eval_config"] = locked
    lock["eval_config_sha256"] = {
        point["name"]: stable_hash(point["eval_config"]) for point in points
    }
    previous_lock = collection / "protocol_lock.json"
    if previous_lock.exists():
        old = _read_json(previous_lock)
        lock = _reuse_protocol_lock(
            lock,
            old,
            formal_code=formal_code,
            allow_qoder_only_transition=allow_qoder_only_transition,
        )
    else:
        _write_json(previous_lock, lock)
    for point in points:
        _write_locked_json(Path(point["eval_config_path"]), point["eval_config"])
    _write_locked_json(collection / "checkpoint_eval_index.json", {"points": points})
    return points, lock


def _reuse_protocol_lock(
    candidate: dict[str, Any],
    frozen: dict[str, Any],
    *,
    formal_code: dict[str, Any],
    allow_qoder_only_transition: bool,
) -> dict[str, Any]:
    # The lock records the code transition at preparation time. Later policy-only
    # commits change that diagnostic metadata, not the locked evaluation inputs.
    _require_matching_formal_code(
        formal_code,
        {"formal_code_provenance": frozen.get("formal_code_provenance")},
        allow_qoder_only_transition=allow_qoder_only_transition,
    )
    candidate["artifact_commit"] = frozen.get("artifact_commit")
    candidate["formal_code_provenance"] = frozen.get("formal_code_provenance")
    candidate["training_code_transition"] = frozen.get("training_code_transition")
    frozen_checkpoints = {row["name"]: row for row in frozen.get("checkpoint_index", [])}
    for row in candidate["checkpoint_index"]:
        if row["name"] in frozen_checkpoints:
            row["training_code_transition"] = frozen_checkpoints[row["name"]].get(
                "training_code_transition"
            )
    if stable_hash(candidate) != stable_hash(frozen):
        changed = sorted(
            key
            for key in candidate.keys() | frozen.keys()
            if stable_hash(candidate.get(key)) != stable_hash(frozen.get(key))
        )
        raise ValueError(
            "Existing protocol_lock.json differs in locked inputs; "
            f"changed_fields={changed}. Keep the original lock and inspect these fields."
        )
    return frozen


def _validate_curve_arm_invariants(
    *,
    train_root: Path,
    manifests: dict[tuple[str, int], dict[str, Any]],
    cycles: int,
    train_task_count: int,
) -> dict[str, Any]:
    initial_world = file_sha256(train_root / "initial_state" / "world_memory.jsonl")
    initial_ontology = file_sha256(train_root / "initial_state" / "world_ontology.jsonl")
    checks: dict[str, bool] = {}
    for round_number in range(1, cycles + 1):
        full = manifests[("full", round_number)]
        agent_only = manifests[("agent_only", round_number)]
        prefix = f"round_{round_number}"
        checks[f"{prefix}_same_factual_tasks"] = (
            full["inputs"]["round_factual_records"]["sha256"]
            == agent_only["inputs"]["round_factual_records"]["sha256"]
        )
        checks[f"{prefix}_agent_update_attempts_aligned"] = (
            int(full["agent_update_attempt_count"])
            == int(agent_only["agent_update_attempt_count"])
            == round_number
        )
        expected_cumulative = round_number * train_task_count * 3
        checks[f"{prefix}_training_rollouts_aligned"] = (
            int(full["cumulative_training_rollout_count"])
            == int(agent_only["cumulative_training_rollout_count"])
            == expected_cumulative
        )
        checks[f"{prefix}_agent_only_world_memory_frozen"] = (
            agent_only["outputs"]["world_memory"]["sha256"] == initial_world
        )
        checks[f"{prefix}_agent_only_world_ontology_frozen"] = (
            agent_only["outputs"]["world_ontology"]["sha256"] == initial_ontology
        )
        checks[f"{prefix}_regressions_excluded_from_world"] = not bool(
            full["paired_outcomes"].get("success_to_failure_routed_to_world")
            or agent_only["paired_outcomes"].get("success_to_failure_routed_to_world")
        )
        if round_number == 1:
            checks["round_1_shared_initial_challenges"] = (
                full["inputs"]["interventions"]["sha256"]
                == agent_only["inputs"]["interventions"]["sha256"]
            )
        else:
            full_previous = manifests[("full", round_number - 1)]
            agent_only_previous = manifests[("agent_only", round_number - 1)]
            previous_world_identical = all(
                full_previous["outputs"][key]["sha256"]
                == agent_only_previous["outputs"][key]["sha256"]
                for key in ("world_memory", "world_ontology")
            )
            checks[f"{prefix}_same_world_implies_same_challenges"] = (
                not previous_world_identical
                or full["inputs"]["interventions"]["sha256"]
                == agent_only["inputs"]["interventions"]["sha256"]
            )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Formal Full/Agent-only arm invariant failed: {failed}")
    return {
        "status": "pass",
        "checks": checks,
        "agent_only_world_memory_sha256": initial_world,
        "agent_only_world_ontology_sha256": initial_ontology,
        "later_round_challenges_may_differ_by_world_memory": True,
        "total_cost_equality_claimed": False,
    }


def evaluate(
    config: dict[str, Any],
    points: list[dict[str, Any]],
    *,
    selected_names: set[str],
) -> None:
    env = os.environ.copy()
    env.update(
        {str(key): str(value) for key, value in config.get("official_runtime_env", {}).items()}
    )
    pythonpath = [str(PROJECT_ROOT / "patches" / "mobilitybench"), str(PROJECT_ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    for point in points:
        if selected_names and point["name"] not in selected_names:
            continue
        report_path = Path(point["eval_config"]["output_dir"]) / "report.json"
        if report_path.exists() and _report_terminal(
            _read_json(report_path), int(config["expected_count"])
        ):
            print(f"Reuse terminal point: {point['name']}", flush=True)
            continue
        _run_command(
            [
                str(Path(config["mobilitybench_python"]).expanduser()),
                "-m",
                "cce_mobility.workflows.mobilitybench.run_mobilitybench_official_continuation",
                point["eval_config_path"],
            ],
            cwd=Path(config["mobilitybench_root"]).expanduser(),
            env=env,
        )


def export(config: dict[str, Any], points: list[dict[str, Any]]) -> None:
    collection = _project_path(config["collection_dir"])
    bootstrap_samples = int(config.get("bootstrap_samples", 10_000))
    records_by_point: dict[str, dict[str, dict[str, Any]]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for point in points:
        report_path = Path(point["eval_config"]["output_dir"]) / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Evaluation report missing: {report_path}")
        report = _read_json(report_path)
        if not _report_terminal(report, int(config["expected_count"])):
            raise ValueError(f"Point is incomplete or has retryable errors: {point['name']}")
        reports[point["name"]] = report
        records_by_point[point["name"]] = _load_record_rows(_project_path(report["records_path"]))

    audit = _protocol_audit(config, points, records_by_point, reports=reports)
    point_rows = [
        _point_row(
            point,
            reports[point["name"]],
            records_by_point[point["name"]],
            bootstrap_samples=bootstrap_samples,
        )
        for point in points
    ]
    comparisons = _comparison_specs(points)
    delta_rows = [
        _paired_delta_row(
            name=name,
            left_name=left,
            right_name=right,
            comparison_role=role,
            left=records_by_point[left],
            right=records_by_point[right],
            bootstrap_samples=bootstrap_samples,
        )
        for name, left, right, role in comparisons
    ]
    case_rows = _case_score_rows(points, records_by_point)

    _write_csv(collection / OUTPUT_FILES["points"], point_rows)
    _write_jsonl(collection / OUTPUT_FILES["cases"], case_rows)
    _write_csv(collection / OUTPUT_FILES["deltas"], delta_rows)
    _write_json(collection / OUTPUT_FILES["audit"], audit)
    _write_json(
        collection / OUTPUT_FILES["manifest"],
        _artifact_manifest(config, points, reports),
    )
    (collection / OUTPUT_FILES["readme"]).write_text(
        _render_readme(config, point_rows, delta_rows, audit),
        encoding="utf-8",
    )
    print(f"Exported formal curve package: {collection}", flush=True)


def _point_spec(
    config: dict[str, Any],
    *,
    name: str,
    series: str,
    curve_role: str,
    round_number: int,
    raw_iteration: int,
    agent_update_count: int,
    training_rollout_count: int,
    agent_memory: Path,
    world_memory: Path,
    world_ontology: Path,
    update_accepted: bool,
    agent_update_accept_count: int,
    world_memory_changed: bool,
    world_ontology_changed: bool,
) -> dict[str, Any]:
    for path in (agent_memory, world_memory, world_ontology):
        if not path.exists():
            raise FileNotFoundError(f"Curve checkpoint artifact missing: {path}")
    collection = _project_path(config["collection_dir"])
    eval_config_path = collection / "eval_configs" / f"{name}.json"
    output_dir = collection / "eval_runs" / name
    eval_config = {
        "schema_version": "mobilitybench-continuation-eval-v1",
        "method_version": FORMAL_METHOD_VERSION,
        "experiment_name": name,
        "output_dir": str(output_dir),
        "continuation_cases_path": str(_project_path(config["continuation_cases_path"])),
        "official_config_path": str(_project_path(config["official_config_path"])),
        "official_prompt_bundle_path": str(_project_path(config["official_prompt_bundle_path"])),
        "official_model_alias": config["official_model_alias"],
        "expected_count": int(config["expected_count"]),
        "max_case_attempts": int(config.get("max_case_attempts", 2)),
        "agent": config["agent"],
        "judge": config["judge"],
        "agent_memory_path": str(agent_memory),
        "world_memory_path": str(world_memory),
        "world_ontology_path": str(world_ontology),
        "retrieval": config["retrieval"],
        "training_series": series,
        "completed_cycle": round_number,
        "raw_iteration": raw_iteration,
        "agent_update_count": agent_update_count,
        "training_rollout_count": training_rollout_count,
        "artifact_commit": _git_commit(),
        "seed": config["seed"],
    }
    return {
        "name": name,
        "series": series,
        "curve_role": curve_role,
        "completed_cycle": round_number,
        "raw_iteration": raw_iteration,
        "agent_update_count": agent_update_count,
        "training_rollout_count": training_rollout_count,
        "update_accepted": update_accepted,
        "agent_update_accept_count": agent_update_accept_count,
        "world_memory_changed": world_memory_changed,
        "world_ontology_changed": world_ontology_changed,
        "agent_memory": _artifact(agent_memory),
        "world_memory": _artifact(world_memory),
        "world_ontology": _artifact(world_ontology),
        "eval_config_path": str(eval_config_path),
        "eval_config": eval_config,
    }


def _protocol_audit(
    config: dict[str, Any],
    points: list[dict[str, Any]],
    records_by_point: dict[str, dict[str, dict[str, Any]]],
    *,
    reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    anchor_name = points[0]["name"]
    anchor = records_by_point[anchor_name]
    keys = set(anchor)
    mismatches: list[dict[str, Any]] = []
    invariant_fields = (
        "case_hash",
        "checkpoint_position",
        "factual_source_hash",
        "intervention_source_hash",
        "recovery_contract_hash",
        "continuation_cases_sha256",
        "official_prompt_bundle_sha256",
        "official_runtime_contract_sha256",
        "official_config_sha256",
        "official_tool_schema_sha256",
        "official_model_config_sha256",
        "official_mobilitybench_code_sha256",
        "agent_model_config_sha256",
        "judge_model_config_sha256",
        "judge_prompt_sha256",
        "retrieval_config_sha256",
    )
    from cce_mobility.integrations.mobilitybench_continuation import (
        verify_scoped_provenance_transition,
    )

    runtime_transitions: dict[str, dict[str, Any]] = {}
    for point in points:
        for row in records_by_point[point["name"]].values():
            metadata = row.get("metadata", {})
            source_hash = metadata.get("cce_runtime_code_sha256")
            commit = metadata.get("cce_runtime_git_commit")
            key = f"{commit}:{source_hash}"
            if key not in runtime_transitions:
                try:
                    runtime_transitions[key] = verify_scoped_provenance_transition(
                        PROJECT_ROOT,
                        {"git_commit": commit, "runtime_source_sha256": source_hash},
                        scope="runtime",
                        allow_qoder_only_transition=all(
                            config[role].get("provider") == "aihub" for role in ("agent", "judge")
                        ),
                    )
                except ValueError as exc:
                    # Evaluation already persisted the exact source commit/hash in
                    # every record. Export remains reproducible even when later
                    # unrelated repository history prevents reconstructing it.
                    runtime_transitions[key] = {
                        "equivalence": "recorded_runtime_not_reverified_posthoc",
                        "recorded_commit": commit,
                        "recorded_source_sha256": source_hash,
                        "reason": str(exc),
                    }
    for point in points[1:]:
        name = point["name"]
        rows = records_by_point[name]
        if set(rows) != keys:
            mismatches.append(
                {
                    "point": name,
                    "field": "pair_keys",
                    "missing": sorted(keys - set(rows))[:10],
                    "extra": sorted(set(rows) - keys)[:10],
                }
            )
            continue
        for pair_key in sorted(keys):
            left = anchor[pair_key].get("metadata", {})
            right = rows[pair_key].get("metadata", {})
            for field in invariant_fields:
                if left.get(field) != right.get(field):
                    mismatches.append(
                        {
                            "point": name,
                            "pair_key": pair_key,
                            "field": field,
                            "anchor": left.get(field),
                            "value": right.get(field),
                        }
                    )
                    if len(mismatches) >= 50:
                        break
            if len(mismatches) >= 50:
                break
    prompt_diff_sources = Counter()
    for point in points:
        for row in records_by_point[point["name"]].values():
            retrieval = row.get("metadata", {}).get("agent_skill_retrieval", {})
            prompt_diff_sources[
                "selected_memory_guidance"
                if retrieval.get("selected_skill_ids")
                else "no_skill_selected"
            ] += 1
    retry_audit = {
        point["name"]: _attempt_summary(
            _project_path(point["eval_config"]["output_dir"]) / "attempts.jsonl"
        )
        for point in points
    }
    quality = {
        name: {
            "clean": bool(report.get("metrics", {}).get("clean")),
            "agent_error_count": int(report.get("metrics", {}).get("agent_error_count") or 0),
            "judge_error_count": int(report.get("metrics", {}).get("judge_error_count") or 0),
            "parse_error_count": int(report.get("metrics", {}).get("parse_error_count") or 0),
            "retry_budget_exhausted_case_count": int(
                report.get("metrics", {}).get("retry_budget_exhausted_case_count") or 0
            ),
        }
        for name, report in (reports or {}).items()
    }
    has_recorded_errors = any(not item["clean"] for item in quality.values())
    audit = {
        "status": (
            "fail"
            if mismatches
            else "pass_with_recorded_errors"
            if has_recorded_errors
            else "pass_clean"
        ),
        "protocol": CONTINUATION_PROTOCOL,
        "method_version": FORMAL_METHOD_VERSION,
        "evaluation_identity": config["evaluation_identity"],
        "evaluation_identity_note": config["evaluation_identity_note"],
        "point_count": len(points),
        "expected_point_count": 7,
        "fixed_denominator": int(config["expected_count"]),
        "pair_key_count": len(keys),
        "invariant_fields": list(invariant_fields),
        "runtime_source_transitions": runtime_transitions,
        "mismatches": mismatches,
        "runtime_prompt_difference_policy": (
            "Base official prompts are identical. Runtime worker/reporter prompts may differ "
            "only through Agent-owned retrieval from the point-specific Agent memory."
        ),
        "runtime_prompt_difference_sources": dict(prompt_diff_sources),
        "retry_audit": retry_audit,
        "point_quality": quality,
        "judge_model": config["judge"]["model"],
        "agent_model": config["agent"]["model"],
        "bootstrap": {
            "samples": int(config.get("bootstrap_samples", 10_000)),
            "unit": "task_cluster",
            "single_training_seed": True,
            "cross_seed_stability_claim_allowed": False,
        },
    }
    if mismatches:
        raise ValueError(f"Cross-point protocol mismatch: {mismatches[:3]}")
    return audit


def _point_row(
    point: dict[str, Any],
    report: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    grouped = _scores_by_task(records)
    attempts = _attempt_summary(_project_path(report["records_path"]).parent / "attempts.jsonl")
    mean, low, high = _cluster_bootstrap_stat(
        grouped,
        statistic=lambda values: statistics.fmean(values),
        samples=bootstrap_samples,
        seed=42,
    )
    metrics = report["metrics"]
    return {
        "series": point["series"],
        "curve_role": point["curve_role"],
        "completed_round": point["completed_cycle"],
        "raw_iteration": point["raw_iteration"],
        "agent_update_count": point["agent_update_count"],
        "agent_update_accept_count": point["agent_update_accept_count"],
        "cumulative_training_rollout_count": point["training_rollout_count"],
        "agent_update_accepted": point["update_accepted"],
        "world_memory_changed": point["world_memory_changed"],
        "world_ontology_changed": point["world_ontology_changed"],
        "recovery_avg": round(mean, 6),
        "recovery_ci95_low": round(low, 6),
        "recovery_ci95_high": round(high, 6),
        "fixed_denominator": int(metrics.get("expected_count") or len(records)),
        "observed_record_count": len(records),
        "task_count": len(grouped),
        "coverage_rate": metrics.get("coverage_rate"),
        "low_score_rate": metrics.get("counterfactual_low_score_rate"),
        "hard_pass_rate": metrics.get("counterfactual_hard_pass_rate"),
        "final_pass_rate": metrics.get("counterfactual_final_pass_rate"),
        "unavailable_usage_rate": metrics.get("counterfactual_unavailable_usage_rate", 0.0),
        "agent_error_count": metrics.get("agent_error_count", 0),
        "judge_error_count": metrics.get("judge_error_count", 0),
        "parse_error_count": metrics.get("parse_error_count", 0),
        "missing_count": metrics.get("missing_count", 0),
        **attempts,
        "retry_reason_counts": json.dumps(
            attempts["retry_reason_counts"], ensure_ascii=False, sort_keys=True
        ),
        "agent_memory_sha256": point["agent_memory"]["sha256"],
        "world_memory_sha256": point["world_memory"]["sha256"],
        "world_ontology_sha256": point["world_ontology"]["sha256"],
    }


def _comparison_specs(points: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
    names = {point["name"] for point in points}
    specs: list[tuple[str, str, str, str]] = []
    anchor = "no_evolution_round0"
    for round_number in (1, 2, 3):
        full = f"full_coevolution_round{round_number}"
        agent = f"agent_only_frozen_world_round{round_number}"
        specs.extend(
            [
                (
                    f"full_minus_agent_only_round{round_number}",
                    full,
                    agent,
                    "primary" if round_number == 3 else "same_round_diagnostic",
                ),
                (f"full_round{round_number}_minus_round0", full, anchor, "vs_no_evolution"),
                (
                    f"agent_only_round{round_number}_minus_round0",
                    agent,
                    anchor,
                    "vs_no_evolution",
                ),
            ]
        )
        if round_number > 1:
            specs.extend(
                [
                    (
                        f"full_round{round_number}_minus_round{round_number - 1}",
                        full,
                        f"full_coevolution_round{round_number - 1}",
                        "adjacent_round_diagnostic",
                    ),
                    (
                        f"agent_only_round{round_number}_minus_round{round_number - 1}",
                        agent,
                        f"agent_only_frozen_world_round{round_number - 1}",
                        "adjacent_round_diagnostic",
                    ),
                ]
            )
    missing = {name for spec in specs for name in spec[1:3] if name not in names}
    if missing:
        raise ValueError(f"Comparison point missing: {sorted(missing)}")
    return specs


def _paired_delta_row(
    *,
    name: str,
    left_name: str,
    right_name: str,
    comparison_role: str,
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError(f"Paired keys differ for {name}")
    grouped: dict[str, list[float]] = defaultdict(list)
    improved = worse = same = 0
    for pair_key in sorted(left):
        delta = _record_score(left[pair_key]) - _record_score(right[pair_key])
        task_id = str(left[pair_key]["sample"]["sample_id"])
        grouped[task_id].append(delta)
        improved += delta > 0
        worse += delta < 0
        same += delta == 0
    mean, low, high = _cluster_bootstrap_stat(
        grouped,
        statistic=lambda values: statistics.fmean(values),
        samples=bootstrap_samples,
        seed=43,
    )
    return {
        "comparison": name,
        "comparison_role": comparison_role,
        "left": left_name,
        "right": right_name,
        "paired_case_count": len(left),
        "task_cluster_count": len(grouped),
        "mean_delta": round(mean, 6),
        "delta_ci95_low": round(low, 6),
        "delta_ci95_high": round(high, 6),
        "improved": improved,
        "worse": worse,
        "same": same,
    }


def _case_score_rows(
    points: list[dict[str, Any]],
    records_by_point: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in points:
        for pair_key, record in sorted(records_by_point[point["name"]].items()):
            metadata = record.get("metadata", {})
            hard = metadata.get("counterfactual_hard_metrics", {})
            retrieval = metadata.get("agent_skill_retrieval", {})
            counterfactual_raw = (record.get("counterfactual") or {}).get("raw") or {}
            intervention = record.get("intervention") or {}
            rows.append(
                {
                    "point": point["name"],
                    "series": point["series"],
                    "completed_round": point["completed_cycle"],
                    "task_id": record["sample"]["sample_id"],
                    "intervention_id": intervention.get("intervention_id"),
                    "pair_key": pair_key,
                    "checkpoint_position": metadata.get("checkpoint_position"),
                    "dimension": intervention.get("dimension"),
                    "recovery_score": _record_score(record),
                    "hard_pass": hard.get("hard_pass"),
                    "final_pass": hard.get("final_pass"),
                    "uses_unavailable_identifier": hard.get("uses_unavailable_identifier"),
                    "agent_error": hard.get("llm_error"),
                    "agent_error_message": hard.get("llm_error_message"),
                    "retrieval_error": retrieval.get("error"),
                    "judge_error": metadata.get("judge_error"),
                    "judge_error_message": metadata.get("judge_error_message"),
                    "parse_error": not bool(hard.get("parse_success")),
                    "official_planner_present": hard.get("official_planner_present"),
                    "official_worker_tool_call_present": hard.get(
                        "official_worker_tool_call_present"
                    ),
                    "official_reporter_present": hard.get("official_reporter_present"),
                    "termination_reason": hard.get("termination_reason"),
                    "attempt": metadata.get("attempt"),
                    "case_hash": metadata.get("case_hash"),
                    "agent_memory_sha256": metadata.get("agent_memory_sha256"),
                    "world_memory_sha256": metadata.get("world_memory_sha256"),
                    "runtime_prompt_hashes": metadata.get("runtime_prompt_hashes"),
                    "official_tool_call_count": len(counterfactual_raw.get("tool_call_log") or []),
                    "source_records_path": str(
                        _project_path(point["eval_config"]["output_dir"] + "/records.jsonl")
                    ),
                }
            )
    return rows


def _cluster_bootstrap_stat(
    grouped: dict[str, list[float]],
    *,
    statistic: Callable[[list[float]], float],
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not grouped:
        return 0.0, 0.0, 0.0
    task_ids = sorted(grouped)
    observed_values = [value for task_id in task_ids for value in grouped[task_id]]
    observed = statistic(observed_values)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(task_ids) for _ in task_ids]
        values = [value for task_id in selected for value in grouped[task_id]]
        estimates.append(statistic(values))
    estimates.sort()
    lower = estimates[int(0.025 * samples)]
    upper = estimates[min(samples - 1, int(0.975 * samples))]
    return observed, lower, upper


def _scores_by_task(records: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records.values():
        grouped[str(record["sample"]["sample_id"])].append(_record_score(record))
    return grouped


def _record_score(record: dict[str, Any]) -> float:
    return float((record.get("counterfactual_score") or {}).get("total") or 0.0)


def _load_record_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pair_key = str(row.get("metadata", {}).get("pair_key") or "")
            if not pair_key:
                raise ValueError(f"Record missing pair_key: {path}")
            if pair_key in rows:
                raise ValueError(f"Duplicate pair_key {pair_key}: {path}")
            rows[pair_key] = row
    return rows


def _attempt_summary(path: Path) -> dict[str, Any]:
    attempts_by_pair: Counter[str] = Counter()
    retry_reasons: Counter[str] = Counter()
    if not path.exists():
        return {
            "attempt_log_present": False,
            "attempt_count": 0,
            "retried_pair_count": 0,
            "failed_attempt_count": 0,
            "retry_reason_counts": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pair_key = str(row.get("pair_key") or "")
            if pair_key:
                attempts_by_pair[pair_key] += 1
            reason = str(row.get("retry_reason") or "")
            if reason:
                retry_reasons[reason] += 1
    retried_pairs = sum(count > 1 for count in attempts_by_pair.values())
    return {
        "attempt_log_present": True,
        "attempt_count": sum(attempts_by_pair.values()),
        "retried_pair_count": retried_pairs,
        "failed_attempt_count": sum(retry_reasons.values()),
        "retry_reason_counts": dict(sorted(retry_reasons.items())),
    }


def _load_cases(path: Path) -> list[MobilityBenchContinuationCase]:
    cases: list[MobilityBenchContinuationCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                case = MobilityBenchContinuationCase.model_validate_json(line)
                validate_continuation_case(case)
                cases.append(case)
    return cases


def _validate_input_audits(
    *,
    config: dict[str, Any],
    cases_path: Path,
    input_audit_path: Path,
    cases_audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_audit = _read_json(input_audit_path)
    cases_audit = _read_json(cases_audit_path)
    expected = int(config["expected_count"])
    eval_factual = input_audit.get("inputs", {}).get("eval_factual", {})
    interventions = input_audit.get("inputs", {}).get("frozen_interventions", {})
    fallback_count = int(
        cases_audit.get("trigger_resolution_counts", {}).get("first_worker_event_fallback", 0)
    )
    max_fallback_count = int(config.get("max_fallback_trigger_count", 0))
    checks = {
        "input_status": input_audit.get("status") == "pass",
        "evaluation_identity": input_audit.get("evaluation_identity")
        == config["evaluation_identity"],
        "factual_reusable": bool(
            input_audit.get("factual_source_reusable_for_continuation_binding")
        ),
        "case_protocol": cases_audit.get("protocol") == CONTINUATION_PROTOCOL,
        "case_count": int(cases_audit.get("case_count", -1)) == expected,
        "case_artifact_hash": cases_audit.get("cases_sha256") == file_sha256(cases_path),
        "factual_hash": cases_audit.get("factual_records_sha256") == eval_factual.get("sha256"),
        "intervention_hash": cases_audit.get("frozen_interventions_sha256")
        == interventions.get("sha256"),
        "agent_payload_clean": int(cases_audit.get("agent_payload_privileged_field_count", -1))
        == 0,
        "agent_context_whitelist": not bool(
            cases_audit.get("agent_visible_context_unexpected_keys", ["audit_field_missing"])
        ),
        "fallback_trigger_within_preregistered_limit": (fallback_count <= max_fallback_count),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Formal curve input audit failed: "
            f"checks={failed}, input_audit={input_audit_path}, "
            f"cases_audit={cases_audit_path}"
        )
    return input_audit, cases_audit


def _validate_training_manifest(
    manifest: dict[str, Any],
    *,
    arm: str,
    round_number: int,
    expected_code: dict[str, Any] | None = None,
    allow_qoder_only_transition: bool = False,
) -> dict[str, Any] | None:
    expected_mode = "enabled" if arm == "full" else "frozen"
    artifacts = [
        item
        for section in ("inputs", "outputs")
        for item in (manifest.get(section) or {}).values()
        if isinstance(item, dict) and item.get("path") and item.get("sha256")
    ]
    checks = {
        "status": manifest.get("status") == "complete",
        "protocol": manifest.get("training_protocol") == FORMAL_TRAINING_PROTOCOL,
        "continuation": manifest.get("continuation_protocol") == CONTINUATION_PROTOCOL,
        "arm": manifest.get("arm") == arm,
        "round": int(manifest.get("completed_cycle", -1)) == round_number,
        "raw_iteration": int(manifest.get("raw_iteration", -1)) == 2 * round_number,
        "world_mode": manifest.get("world_update_mode") == expected_mode,
        "regression_not_world": not bool(
            manifest.get("paired_outcomes", {}).get("success_to_failure_routed_to_world")
        ),
        "artifact_integrity": bool(artifacts)
        and all(
            Path(str(item["path"])).is_file()
            and file_sha256(Path(str(item["path"]))) == item["sha256"]
            for item in artifacts
        ),
    }
    transition = None
    if expected_code is not None:
        from cce_mobility.integrations.mobilitybench_continuation import (
            verify_scoped_provenance_transition,
        )

        manifest_code = manifest.get("formal_code_provenance") or {}
        kwargs = {"allow_qoder_only_transition": True} if allow_qoder_only_transition else {}
        transition = verify_scoped_provenance_transition(
            PROJECT_ROOT, manifest_code, scope="formal", **kwargs
        )
        checks.update(
            {
                "formal_source": transition["current_source_sha256"]
                == expected_code.get("formal_source_sha256"),
                "formal_code_clean": not bool(manifest_code.get("relevant_dirty")),
            }
        )
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid {arm} round {round_number} manifest checks: {failed}")
    return transition


def _require_matching_formal_code(
    current: dict[str, Any],
    preflight: dict[str, Any],
    *,
    allow_qoder_only_transition: bool = False,
) -> dict[str, Any]:
    frozen = preflight.get("formal_code_provenance") or {}
    from cce_mobility.integrations.mobilitybench_continuation import (
        verify_scoped_provenance_transition,
    )

    kwargs = {"allow_qoder_only_transition": True} if allow_qoder_only_transition else {}
    transition = verify_scoped_provenance_transition(PROJECT_ROOT, frozen, scope="formal", **kwargs)
    checks = {
        "current_commit_known": current.get("git_commit") not in {"", "unknown"},
        "current_code_clean": not bool(current.get("relevant_dirty")),
        "verified_current_source": current.get("formal_source_sha256")
        == transition["current_source_sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Formal curve code provenance differs from training preflight: "
            f"checks={failed}, current={current}, frozen={frozen}"
        )
    return transition


def _report_complete_clean(report: dict[str, Any], expected: int) -> bool:
    metrics = report.get("metrics", {})
    return bool(
        metrics.get("clean")
        and int(metrics.get("completed_count", -1)) == expected
        and int(metrics.get("missing_count", -1)) == 0
        and int(metrics.get("duplicate_pair_key_count", -1)) == 0
    )


def _report_terminal(report: dict[str, Any], expected: int) -> bool:
    metrics = report.get("metrics", {})
    return bool(
        "retryable_error_count" in metrics
        and int(metrics.get("completed_count", -1)) == expected
        and int(metrics.get("missing_count", -1)) == 0
        and int(metrics.get("duplicate_pair_key_count", -1)) == 0
        and int(metrics.get("retryable_error_count", 0)) == 0
    )


def _artifact_manifest(
    config: dict[str, Any],
    points: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    paths = {
        _project_path(config["continuation_cases_path"]),
        _project_path(config["continuation_cases_path"]).with_suffix(".audit.json"),
        _project_path(config["input_audit_path"]),
        _project_path(config["official_prompt_bundle_path"]),
        _project_path(config["official_config_path"]),
        _project_path(config["collection_dir"]) / "protocol_lock.json",
        _project_path(config["train_output_root"]) / "preflight.json",
        _project_path(config["train_output_root"]) / "training_task_schedule.json",
    }
    optional_artifact_omissions: list[dict[str, Any]] = []
    for point in points:
        eval_root = _project_path(reports[point["name"]]["records_path"]).parent
        paths.update(
            {
                Path(point["agent_memory"]["path"]),
                Path(point["world_memory"]["path"]),
                Path(point["world_ontology"]["path"]),
                Path(point["eval_config_path"]),
                eval_root / "records.jsonl",
                eval_root / "attempts.jsonl",
                eval_root / "report.json",
                eval_root / "protocol_config.json",
            }
        )
        if point["completed_cycle"]:
            round_root = (
                _project_path(config["train_output_root"])
                / ("full" if point["series"] == "full_coevolution" else "agent_only")
                / f"round_{point['completed_cycle']}"
            )
            round_paths, round_omissions = _round_training_artifacts(round_root)
            paths.update(round_paths)
            optional_artifact_omissions.extend(round_omissions)
    missing = [str(path) for path in sorted(paths, key=str) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required formal-curve artifact missing: " + ", ".join(missing))
    return {
        "artifact_commit": _git_commit(),
        "files": [_artifact(path) for path in sorted(paths, key=str)],
        "optional_artifact_omissions": optional_artifact_omissions,
    }


def _round_training_artifacts(
    round_root: Path,
) -> tuple[set[Path], list[dict[str, Any]]]:
    """Resolve round artifacts while preserving byte-identical challenge reuse."""

    summary_path = round_root / "interventions.summary.json"
    paths = {
        round_root / "round_manifest.json",
        round_root / "interventions.jsonl",
        summary_path,
        round_root / "continuation_cases.jsonl",
    }
    generation_config = round_root / "challenge_generation_config.json"
    if generation_config.is_file():
        paths.add(generation_config)
        return paths, []

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Challenge-generation config and reuse summary are both missing: {round_root}"
        )
    summary = _read_json(summary_path)
    reuse_mode = str(summary.get("reuse_mode") or "")
    allowed_reuse_modes = {
        "byte_identical_shared_round1",
        "byte_identical_full_challenges_same_world_state",
    }
    if reuse_mode not in allowed_reuse_modes:
        raise FileNotFoundError(
            "Challenge-generation config is missing without a verified reuse mode: "
            f"{generation_config}"
        )
    source = summary.get("source_interventions") or {}
    source_path = Path(str(source.get("path") or "")).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Reused challenge source is missing: {source_path}")
    expected_sha256 = str(source.get("sha256") or "")
    actual_sha256 = file_sha256(source_path)
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise ValueError(f"Reused challenge source hash differs: {source_path}")
    output_path = round_root / "interventions.jsonl"
    if not output_path.is_file() or file_sha256(output_path) != actual_sha256:
        raise ValueError(
            f"Reused challenge output is not byte-identical to its source: {output_path}"
        )
    paths.add(source_path)
    return paths, [
        {
            "path": str(generation_config.resolve()),
            "reason": "challenge_generation_reused_byte_identically",
            "reuse_mode": reuse_mode,
            "source_interventions": _artifact(source_path),
        }
    ]


def _estimate_cost(config: dict[str, Any]) -> dict[str, Any]:
    n = int(config["expected_count"])
    worker_calls = float(config.get("assumed_worker_llm_calls_per_rollout", 2.0))
    official_calls_per_rollout = 2 + worker_calls
    formal_rollouts = 7 * n
    official_calls = int(formal_rollouts * official_calls_per_rollout)
    retrieval_calls_upper = 6 * n
    judge_calls = formal_rollouts
    token_estimate = config.get("token_estimate_per_rollout") or {
        "official_planner_worker_reporter": 9000,
        "retrieval": 2300,
        "judge": 3000,
    }
    official_tokens = int(token_estimate.get("official_planner_worker_reporter", 9000))
    retrieval_tokens = int(token_estimate.get("retrieval", 2300))
    judge_tokens = int(token_estimate.get("judge", 3000))
    world_generation_tokens = int(token_estimate.get("world_generation", 3000))
    world_update_tokens = int(token_estimate.get("world_update", 3000))
    formal_tokens_upper = int(
        formal_rollouts * (official_tokens + judge_tokens)
        + retrieval_calls_upper * retrieval_tokens
    )
    cycles = int(config.get("cycles", 3))
    train_tasks = int(config.get("train_task_count", 20))
    train_rollouts = 2 * cycles * train_tasks * 3
    training_official_calls = int(train_rollouts * official_calls_per_rollout)
    training_wm_generation_calls = train_tasks * (1 + 2 * (cycles - 1))
    training_retrieval_calls_upper = train_rollouts
    training_judge_calls = train_rollouts
    training_world_update_calls_upper = cycles
    training_tokens_upper = int(
        train_rollouts * (official_tokens + judge_tokens)
        + training_retrieval_calls_upper * retrieval_tokens
        + training_wm_generation_calls * world_generation_tokens
        + training_world_update_calls_upper * world_update_tokens
    )

    source_factual_tasks = int(
        config.get(
            "source_factual_task_count_estimate",
            cycles * train_tasks
            + (n if config.get("evaluation_identity", "").startswith("fresh_") else 0),
        )
    )
    source_factual_official_calls = int(source_factual_tasks * official_calls_per_rollout)
    source_factual_tokens = source_factual_tasks * official_tokens
    frozen_panel_wm_calls = int(
        config.get(
            "frozen_panel_wm_generation_call_estimate",
            n if config.get("evaluation_identity", "").startswith("fresh_") else 0,
        )
    )
    frozen_panel_tokens = frozen_panel_wm_calls * world_generation_tokens
    smoke_cases = int(config.get("protocol_smoke_case_count_estimate", 0))
    smoke_official_calls = int(smoke_cases * official_calls_per_rollout)
    smoke_judge_calls = smoke_cases
    smoke_tokens = smoke_cases * (official_tokens + judge_tokens)
    setup_tokens_upper = source_factual_tokens + frozen_panel_tokens + smoke_tokens
    all_stage_tokens_upper = setup_tokens_upper + training_tokens_upper + formal_tokens_upper
    all_stage_calls_upper = (
        source_factual_official_calls
        + frozen_panel_wm_calls
        + smoke_official_calls
        + smoke_judge_calls
        + training_official_calls
        + training_retrieval_calls_upper
        + training_judge_calls
        + training_wm_generation_calls
        + training_world_update_calls_upper
        + official_calls
        + retrieval_calls_upper
        + judge_calls
    )
    return {
        "assumptions": {
            "average_worker_llm_calls_per_official_graph": worker_calls,
            "official_llm_calls_per_recovery_rollout": official_calls_per_rollout,
            "token_estimate_per_call_or_rollout": {
                "official_planner_worker_reporter": official_tokens,
                "retrieval": retrieval_tokens,
                "judge": judge_tokens,
                "world_generation": world_generation_tokens,
                "world_update": world_update_tokens,
            },
            "paid_call_counts_are_estimates_or_upper_bounds": True,
        },
        "setup": {
            "source_factual_task_count": source_factual_tasks,
            "source_factual_official_llm_calls_estimate": source_factual_official_calls,
            "frozen_panel_wm_generation_calls_estimate": frozen_panel_wm_calls,
            "protocol_smoke_case_count": smoke_cases,
            "protocol_smoke_official_llm_calls_estimate": smoke_official_calls,
            "protocol_smoke_judge_calls": smoke_judge_calls,
            "token_upper_estimate": setup_tokens_upper,
        },
        "training": {
            "recovery_rollouts": train_rollouts,
            "agent_official_llm_calls_estimate": training_official_calls,
            "agent_retrieval_calls_upper_bound": training_retrieval_calls_upper,
            "judge_calls": training_judge_calls,
            "wm_challenge_generation_calls_estimate": training_wm_generation_calls,
            "wm_world_update_calls_full_only_upper_bound": (training_world_update_calls_upper),
            "updater_llm_calls": 0,
            "token_upper_estimate": training_tokens_upper,
        },
        "formal_evaluation": {
            "recovery_rollouts": formal_rollouts,
            "agent_official_llm_calls_estimate": official_calls,
            "agent_retrieval_calls_upper_bound": retrieval_calls_upper,
            "judge_calls": judge_calls,
            "token_upper_estimate": formal_tokens_upper,
        },
        "all_stages": {
            "api_calls_upper_estimate": all_stage_calls_upper,
            "token_upper_estimate": all_stage_tokens_upper,
        },
        "formal_recovery_rollouts": formal_rollouts,
        "formal_agent_official_llm_calls_estimate": official_calls,
        "formal_agent_retrieval_calls_upper_bound": retrieval_calls_upper,
        "formal_judge_calls": judge_calls,
        "formal_token_upper_estimate": formal_tokens_upper,
        "training_recovery_rollouts": train_rollouts,
        "training_agent_official_llm_calls_estimate": training_official_calls,
        "training_agent_retrieval_calls_upper_bound": training_retrieval_calls_upper,
        "training_judge_calls": training_judge_calls,
        "training_wm_challenge_generation_calls_estimate": training_wm_generation_calls,
        "training_wm_world_update_calls_full_only_upper_bound": (training_world_update_calls_upper),
        "training_updater_llm_calls": 0,
        "note": (
            "A recovery rollout invokes planner, one or more workers, reporter, Judge, and "
            "optionally retrieval. World challenge generation and World-memory update are "
            "additional calls. Therefore 3500 rollouts are not 3500 API calls. Replace all "
            "estimates with logged role-level usage after execution."
        ),
    }


def _render_readme(
    config: dict[str, Any],
    points: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    lines = [
        "# MobilityBench 正式共进化轮次数据包",
        "",
        f"- 协议：`{CONTINUATION_PROTOCOL}`",
        f"- 训练：`{FORMAL_TRAINING_PROTOCOL}`",
        f"- 评测身份：`{config['evaluation_identity']}`",
        f"- 身份说明：{config['evaluation_identity_note']}",
        f"- 基座模型：`{config['agent']['model']}`",
        f"- Judge：`{config['judge']['model']}`",
        f"- 单训练 seed：`{config['seed']}`",
        f"- 固定分母：{config['expected_count']}",
        f"- 协议审计：`{audit['status']}`",
        "",
        "## 曲线点",
        "",
        "| series | round | recovery | 95% CI | errors |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for point in points:
        errors = (
            int(point["agent_error_count"])
            + int(point["judge_error_count"])
            + int(point["parse_error_count"])
        )
        lines.append(
            f"| {point['series']} | {point['completed_round']} | {point['recovery_avg']:.4f} | "
            f"[{point['recovery_ci95_low']:.4f}, {point['recovery_ci95_high']:.4f}] | {errors} |"
        )
    primary = next(row for row in deltas if row["comparison_role"] == "primary")
    lines.extend(
        [
            "",
            "## 预注册主要比较",
            "",
            (
                f"最终轮 Full - Agent-only = {primary['mean_delta']:.4f}, "
                f"95% CI [{primary['delta_ci95_low']:.4f}, {primary['delta_ci95_high']:.4f}]。"
            ),
            "",
            "## 口径",
            "",
            "这是 official plan-and-execute 的状态重建续跑，不是 bitwise LangGraph resume，也不是一次性 posthoc 回答。",
            "误差条是在固定案例上的 task-cluster bootstrap，不代表跨训练 seed 稳定性。",
            "官方 factual planning 指标与本数据包中的 CCE dynamic recovery 指标必须分开报告。",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != CURVE_CONFIG_SCHEMA:
        raise ValueError(f"Expected schema_version={CURVE_CONFIG_SCHEMA}")
    required = {
        "collection_dir",
        "train_output_root",
        "continuation_cases_path",
        "input_audit_path",
        "official_prompt_bundle_path",
        "official_config_path",
        "official_model_alias",
        "mobilitybench_root",
        "mobilitybench_python",
        "expected_count",
        "cycles",
        "seed",
        "agent",
        "judge",
        "retrieval",
        "evaluation_identity",
        "evaluation_identity_note",
        "method_version",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Formal curve config missing keys: {missing}")
    if int(config["cycles"]) != 3 or int(config["seed"]) != 42:
        raise ValueError("Formal minimum curve is locked to three cycles and seed=42")
    if config["method_version"] != FORMAL_METHOD_VERSION:
        raise ValueError(f"Expected method_version={FORMAL_METHOD_VERSION}")
    if int(config.get("bootstrap_samples", 10_000)) != 10_000:
        raise ValueError("Formal curve requires exactly 10000 bootstrap samples")
    max_fallback = int(config.get("max_fallback_trigger_count", 0))
    if max_fallback < 0 or max_fallback > int(config["expected_count"]):
        raise ValueError("max_fallback_trigger_count must be within the panel size")
    forbidden = [
        key for key in _flatten_keys(config) if "api_key" in key and not key.endswith("api_key_env")
    ]
    if forbidden:
        raise ValueError(f"Literal API key fields are forbidden: {forbidden}")


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_flatten_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            keys.extend(_flatten_keys(child, f"{prefix}[{index}]"))
    return keys


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": file_sha256(path),
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_locked_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path)
        if stable_hash(existing) != stable_hash(payload):
            raise ValueError(f"Existing locked JSON differs; use a new collection: {path}")
        return
    _write_json(path, payload)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
