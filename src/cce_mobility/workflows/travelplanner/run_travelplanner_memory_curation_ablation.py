from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.analysis.tables import write_matrix_tables  # noqa: E402
from cce_mobility.core.config import ExperimentConfig, ModelConfig  # noqa: E402
from cce_mobility.core.env import load_dotenv  # noqa: E402
from cce_mobility.core.llm import ChatClient  # noqa: E402
from cce_mobility.core.schema import (  # noqa: E402
    EvaluationRecord,
    InterventionCandidate,
)
from cce_mobility.interventions.ontology import (  # noqa: E402
    registry_from_seed_dimensions,
)
from cce_mobility.methods.formal_coevolution import (  # noqa: E402
    build_agent_candidate_memory,
    classify_paired_outcomes,
    curation_decision,
    update_world_from_persistent_failures,
)
from cce_mobility.runners.evaluation import EvaluationRunner  # noqa: E402
from cce_mobility.skills import SkillRegistry, SkillType  # noqa: E402
from cce_mobility.skills.lifecycle import (  # noqa: E402
    gate_memory_with_paired_outcome_evidence,
)


SCHEMA_VERSION = "travelplanner-memory-curation-ablation-v1"
TRAINING_PROTOCOL = "tp-solo-same-challenge-paired-curation-v1"
METHOD_VERSION = "cce_paired_memory_curation_ablation_v1"
ARMS = ("full", "without_memory_curation", "agent_only")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal TravelPlanner paired-memory-curation ablation. "
            "The only arm-level switch is whether raw Agent-memory candidates "
            "pass the paired lifecycle gate before final re-evaluation."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "val", "test", "all"),
        default="prepare",
    )
    parser.add_argument("--arm", choices=(*ARMS, "both"), default="both")
    parser.add_argument("--through-cycle", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    config = _read_json(args.config)
    _validate_config(config)
    preflight = prepare(config, config_path=args.config)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "prepare":
        return 0

    cycles = int(config["cycles"])
    through_cycle = args.through_cycle or cycles
    if through_cycle < 1 or through_cycle > cycles:
        raise ValueError("--through-cycle must be between 1 and configured cycles")
    arms = ARMS if args.arm == "both" else (args.arm,)

    if args.stage in {"train", "all"}:
        for cycle in range(1, through_cycle + 1):
            for arm in arms:
                run_training_cycle(config, arm=arm, cycle=cycle)
    if args.stage in {"val", "all"}:
        if through_cycle != cycles:
            raise ValueError("Validation requires every configured training cycle")
        run_validation(config, arms=arms)
    if args.stage in {"test", "all"}:
        if args.arm != "both":
            raise ValueError("Final test requires both validation-selected arms")
        run_test(config)
    return 0


def prepare(config: dict[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    root = _project_path(config["output_root"])
    initial = root / "initial_state"
    initial.mkdir(parents=True, exist_ok=True)
    agent_memory = initial / "agent_memory.jsonl"
    world_memory = initial / "world_memory.jsonl"
    ontology_path = initial / "world_ontology.jsonl"
    if not agent_memory.exists():
        SkillRegistry().save_jsonl(agent_memory)
    if not world_memory.exists():
        SkillRegistry().save_jsonl(world_memory)
    if not ontology_path.exists():
        registry_from_seed_dimensions(config["train_intervention"]["dimensions"]).save_jsonl(
            ontology_path
        )

    required = {
        "benchmark_repo": _project_path(config["benchmark"]["repo_path"]),
        "train_data": _project_path(config["benchmark"]["train_data_path"]),
        "validation_data": _project_path(config["benchmark"]["validation_data_path"]),
        "test_data": _project_path(config["benchmark"]["test_data_path"]),
    }
    if not config.get("public_profile"):
        required.update(
            {
                "validation_interventions": _project_path(
                    config["validation"]["frozen_intervention_path"]
                ),
                "test_interventions": _project_path(config["test"]["frozen_intervention_path"]),
            }
        )
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(
            "TravelPlanner curation-ablation inputs missing: "
            + json.dumps(missing, ensure_ascii=False, indent=2)
        )
    train_count = _jsonl_count(required["train_data"])
    if train_count < int(config["train_task_count"]):
        raise ValueError(
            f"train_task_count={config['train_task_count']} exceeds train rows={train_count}"
        )
    val_challenges = (
        _frozen_challenge_count(required["validation_interventions"])
        if "validation_interventions" in required
        else 0
    )
    test_challenges = (
        _frozen_challenge_count(required["test_interventions"])
        if "test_interventions" in required
        else 0
    )
    if val_challenges != int(config["validation"]["expected_counterfactual_count"]):
        raise ValueError(
            "Validation frozen challenge count mismatch: "
            f"expected={config['validation']['expected_counterfactual_count']} "
            f"actual={val_challenges}"
        )
    if test_challenges != int(config["test"]["expected_counterfactual_count"]):
        raise ValueError(
            "Test frozen challenge count mismatch: "
            f"expected={config['test']['expected_counterfactual_count']} "
            f"actual={test_challenges}"
        )

    provenance = _code_provenance()
    if not config.get("public_profile"):
        _require_committed_clean_code(provenance)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "config_source": str(config_path.resolve()) if config_path else "programmatic",
        "config_sha256": _stable_hash(config),
        "seed": int(config["seed"]),
        "cycles": int(config["cycles"]),
        "arms": list(ARMS),
        "single_experimental_switch": "paired_memory_curation",
        "train_task_count_per_cycle": int(config["train_task_count"]),
        "task_schedule": "repeat_same_train_pool",
        "formal_code_provenance": provenance,
        "initial_state": {
            "agent_memory": _artifact(agent_memory),
            "world_memory": _artifact(world_memory),
            "world_ontology": _artifact(ontology_path),
        },
        "required_inputs": {name: _artifact(path) for name, path in required.items()},
        "budget_contract": {
            "agent_evaluations_per_cycle": ["pre", "raw_candidate", "final"],
            "matched_between_arms": True,
            "same_challenge_within_cycle": True,
            "same_factual_context_within_cycle": True,
            "final_re_evaluation_in_both_arms": True,
            "failure_to_failure_world_routing_in_both_arms": True,
            "success_to_failure_world_routing_in_both_arms": False,
            "round1_common_random_artifacts": True,
        },
    }
    preflight_path = root / "preflight.json"
    if preflight_path.exists():
        previous_payload = _read_json(preflight_path)
        old_source = previous_payload["formal_code_provenance"]
        if old_source["formal_source_sha256"] != provenance["formal_source_sha256"]:
            raise ValueError("Training source changed; use a new output directory")
        payload["formal_code_provenance"] = old_source
    _write_locked_json(preflight_path, payload)
    return payload


def run_training_cycle(config: dict[str, Any], *, arm: str, cycle: int) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"Unsupported arm: {arm}")
    root = _project_path(config["output_root"])
    cycle_dir = root / arm / f"round_{cycle}"
    manifest_path = cycle_dir / "round_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        _validate_completed_manifest(manifest)
        print(f"Reuse complete {arm} round {cycle}: {manifest_path}", flush=True)
        return manifest
    cycle_dir.mkdir(parents=True, exist_ok=True)
    previous = _previous_state(root, arm=arm, cycle=cycle)

    shared = _round1_shared_artifacts(root) if cycle == 1 else None
    if arm != "full" and cycle == 1 and shared and all(path.exists() for path in shared.values()):
        if not shared or not all(path.exists() for path in shared.values()):
            raise FileNotFoundError(
                "Run full round 1 first so the no-curation arm can "
                "reuse byte-identical pre/candidate evidence."
            )
        pre_report = _read_json(shared["pre_report"])
        pre_report["records_path"] = str(shared["pre_records"].resolve())
        pre_report["_source_report_path"] = str(shared["pre_report"].resolve())
        pre_records_path = shared["pre_records"]
        factual_contexts = shared["factual_contexts"]
        interventions_path = shared["interventions"]
        candidate_memory = cycle_dir / "agent_memory_raw_candidate.jsonl"
        _copy_locked(shared["candidate_memory"], candidate_memory)
        candidate_report = _read_json(shared["candidate_report"])
        candidate_report["records_path"] = str(shared["candidate_records"].resolve())
        candidate_report["_source_report_path"] = str(shared["candidate_report"].resolve())
        _write_json(
            cycle_dir / "candidate_build.json",
            {
                "reuse_mode": "byte_identical_full_round_1",
                "source_candidate_memory": _artifact(shared["candidate_memory"]),
                "output_candidate_memory": _artifact(candidate_memory),
            },
        )
        common_random_artifact_source = "full_round_1"
    else:
        pre_memory = cycle_dir / "pre_combined_memory.jsonl"
        _merge_memories(previous["agent_memory"], previous["world_memory"], pre_memory)
        pre_config = _training_eval_config(
            config,
            arm=arm,
            cycle=cycle,
            label="pre",
            combined_memory=pre_memory,
        )
        pre_report = _run_or_resume(
            pre_config,
            expected_samples=int(config["train_task_count"]),
            quality=config["quality_gate"],
        )
        pre_records_path = Path(pre_report["records_path"])
        factual_contexts = Path(pre_report["records_path"]).parent / "factual_contexts.jsonl"
        interventions_path = cycle_dir / "frozen_cycle_interventions.jsonl"
        _extract_frozen_interventions(pre_records_path, interventions_path)
        if _jsonl_count(interventions_path) < int(
            int(config["train_task_count"])
            * float(config["quality_gate"]["min_counterfactual_coverage"])
        ):
            raise RuntimeError(
                f"Too few valid training challenges in {interventions_path}: "
                f"{_jsonl_count(interventions_path)}"
            )

        pre_records = _load_records(pre_records_path)
        candidate_memory = cycle_dir / "agent_memory_raw_candidate.jsonl"
        candidate_build = build_agent_candidate_memory(
            previous["agent_memory"], pre_records, candidate_memory
        )
        candidate_combined = cycle_dir / "candidate_combined_memory.jsonl"
        _merge_memories(candidate_memory, previous["world_memory"], candidate_combined)
        candidate_config = _training_eval_config(
            config,
            arm=arm,
            cycle=cycle,
            label="raw_candidate",
            combined_memory=candidate_combined,
            frozen_interventions=interventions_path,
            frozen_factual_contexts=factual_contexts,
        )
        candidate_report = _run_or_resume(
            candidate_config,
            expected_samples=int(config["train_task_count"]),
            expected_counterfactual=_jsonl_count(interventions_path),
            quality=config["quality_gate"],
        )
        _write_json(cycle_dir / "candidate_build.json", candidate_build)
        common_random_artifact_source = "generated_for_current_state"
        if arm == "full" and cycle == 1:
            shared = _round1_shared_artifacts(root)
            shared["root"].mkdir(parents=True, exist_ok=True)
            _copy_locked(pre_records_path.parent / "report.json", shared["pre_report"])
            _copy_locked(pre_records_path, shared["pre_records"])
            _copy_locked(factual_contexts, shared["factual_contexts"])
            _copy_locked(interventions_path, shared["interventions"])
            _copy_locked(candidate_memory, shared["candidate_memory"])
            _copy_locked(
                Path(candidate_report["records_path"]).parent / "report.json",
                shared["candidate_report"],
            )
            _copy_locked(Path(candidate_report["records_path"]), shared["candidate_records"])

    pre_records = _load_records(pre_records_path)
    candidate_records_path = Path(candidate_report["records_path"])
    candidate_records = _load_records(candidate_records_path)
    paired_pre_records, paired_candidate_records = _assert_same_paired_keys(
        pre_records,
        candidate_records,
        label="pre/raw_candidate",
    )

    checkpoint_agent = cycle_dir / "agent_memory.jsonl"
    if arm != "without_memory_curation":
        lifecycle_summary = gate_memory_with_paired_outcome_evidence(
            memory_path=candidate_memory,
            baseline_records_path=pre_records_path,
            target_records_path=candidate_records_path,
            output_path=checkpoint_agent,
            repair_mode="verification_constrained",
            **_lifecycle_kwargs(config["lifecycle"]),
        )
        curation_mode = "paired_promote_repair_deprecate"
    else:
        _copy_locked(candidate_memory, checkpoint_agent)
        lifecycle_summary = _accept_all_summary(
            candidate_memory=candidate_memory,
            output_memory=checkpoint_agent,
            baseline_records=paired_pre_records,
            candidate_records=paired_candidate_records,
        )
        _write_json(cycle_dir / "accept_all_summary.json", lifecycle_summary)
        curation_mode = "accept_all_raw_candidates"

    final_combined = cycle_dir / "final_combined_memory.jsonl"
    _merge_memories(checkpoint_agent, previous["world_memory"], final_combined)
    final_config = _training_eval_config(
        config,
        arm=arm,
        cycle=cycle,
        label="final",
        combined_memory=final_combined,
        frozen_interventions=interventions_path,
        frozen_factual_contexts=factual_contexts,
    )
    final_report = _run_or_resume(
        final_config,
        expected_samples=int(config["train_task_count"]),
        expected_counterfactual=_jsonl_count(interventions_path),
        quality=config["quality_gate"],
    )
    final_records_path = Path(final_report["records_path"])
    final_records = _load_records(final_records_path)
    paired_pre_records, paired_final_records = _assert_same_paired_keys(
        pre_records,
        final_records,
        label="pre/final",
    )
    outcomes = classify_paired_outcomes(paired_pre_records, paired_final_records)
    final_diagnostic = curation_decision(
        paired_pre_records,
        paired_final_records,
        thresholds=config["acceptance_diagnostic"],
    )
    regression_path = cycle_dir / "success_to_failure_regressions.jsonl"
    persistent_path = cycle_dir / "failure_to_failure_residuals.jsonl"
    _write_records(regression_path, outcomes["agent_regression_records"])
    _write_records(persistent_path, outcomes["persistent_failure_records"])

    checkpoint_world = cycle_dir / "world_memory.jsonl"
    checkpoint_ontology = cycle_dir / "world_ontology.jsonl"
    world_client = ChatClient(ModelConfig.model_validate(config["world_model"]))
    if arm == "agent_only":
        _copy_locked(previous["world_memory"], checkpoint_world)
        _copy_locked(previous["world_ontology"], checkpoint_ontology)
        world_update = {"enabled": False, "reason": "agent_only_world_frozen"}
    else:
        world_update = update_world_from_persistent_failures(
            persistent_records=outcomes["persistent_failure_records"],
            current_memory_path=previous["world_memory"],
            current_ontology_path=previous["world_ontology"],
            output_memory_path=checkpoint_world,
            output_ontology_path=checkpoint_ontology,
            world_client=world_client,
            ontology_limit=int(config.get("ontology_discovery_limit", 8)),
        )
        world_update["enabled"] = True
    world_update["input_transition"] = "failure_to_failure_only"
    world_update["transport"] = world_client.transport_metrics()

    manifest = {
        "status": "complete",
        "schema_version": "travelplanner-memory-curation-round-v1",
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "arm": arm,
        "single_experimental_switch": {
            "name": "paired_memory_curation",
            "enabled": arm != "without_memory_curation",
            "mode": curation_mode,
        },
        "completed_cycle": cycle,
        "seed": int(config["seed"]),
        "common_random_artifact_source": common_random_artifact_source,
        "final_re_evaluation_performed": True,
        "agent_memory_changed": _sha256(checkpoint_agent) != _sha256(previous["agent_memory"]),
        "world_memory_changed": _sha256(checkpoint_world) != _sha256(previous["world_memory"]),
        "world_ontology_changed": _sha256(checkpoint_ontology)
        != _sha256(previous["world_ontology"]),
        "inputs": {
            "previous_agent_memory": _artifact(previous["agent_memory"]),
            "previous_world_memory": _artifact(previous["world_memory"]),
            "previous_world_ontology": _artifact(previous["world_ontology"]),
            "cycle_interventions": _artifact(interventions_path),
            "factual_contexts": _artifact(factual_contexts),
            "raw_candidate_memory": _artifact(candidate_memory),
        },
        "outputs": {
            "agent_memory": _artifact(checkpoint_agent),
            "world_memory": _artifact(checkpoint_world),
            "world_ontology": _artifact(checkpoint_ontology),
            "persistent_failures": _artifact(persistent_path),
            "agent_regressions": _artifact(regression_path),
        },
        "evaluations": {
            "pre": _report_reference(pre_report),
            "raw_candidate": _report_reference(candidate_report),
            "final": _report_reference(final_report),
        },
        "candidate_build": _read_json(cycle_dir / "candidate_build.json"),
        "lifecycle": lifecycle_summary,
        "final_outcome_diagnostic": final_diagnostic,
        "paired_outcomes": {
            "counts": outcomes["counts"],
            "rates": outcomes["rates"],
            "world_update_input": "failure_to_failure_only",
            "success_to_failure_routed_to_world": False,
        },
        "world_update": world_update,
        "formal_code_provenance": _code_provenance(),
    }
    _write_json(manifest_path, manifest)
    print(
        f"Completed {arm} round {cycle}: "
        f"F->F={outcomes['counts']['failure_to_failure']} "
        f"S->F={outcomes['counts']['success_to_failure']}",
        flush=True,
    )
    return manifest


def run_validation(config: dict[str, Any], *, arms: tuple[str, ...] = ARMS) -> dict[str, Any]:
    root = _project_path(config["output_root"])
    _require_training_complete(root, cycles=int(config["cycles"]), arms=arms)
    stage_root = root / "validation"
    frozen_interventions = _project_path(config["validation"]["frozen_intervention_path"])
    expected_samples = int(config["validation"]["sample_count"])
    expected_counterfactual = int(config["validation"]["expected_counterfactual_count"])

    baseline_config = _holdout_eval_config(
        config,
        split="validation",
        label="no_evolution",
        method="no_evolution",
        frozen_interventions=frozen_interventions,
    )
    baseline_report = _run_or_resume(
        baseline_config,
        expected_samples=expected_samples,
        expected_counterfactual=expected_counterfactual,
        quality=config["quality_gate"],
    )
    factual_contexts = Path(baseline_report["records_path"]).parent / "factual_contexts.jsonl"

    reports = [_named_report("no_evolution", baseline_report)]
    candidates_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    for cycle in range(1, int(config["cycles"]) + 1):
        for arm in arms:
            memory = root / arm / f"round_{cycle}" / "agent_memory.jsonl"
            run_name = f"{arm}_round_{cycle}"
            eval_config = _holdout_eval_config(
                config,
                split="validation",
                label=run_name,
                method="agent_skill_only",
                frozen_interventions=frozen_interventions,
                frozen_factual_contexts=factual_contexts,
                agent_memory=memory,
            )
            report = _run_or_resume(
                eval_config,
                expected_samples=expected_samples,
                expected_counterfactual=expected_counterfactual,
                quality=config["quality_gate"],
            )
            named = _named_report(run_name, report)
            named["arm"] = arm
            named["completed_cycle"] = cycle
            named["agent_memory_path"] = str(memory.resolve())
            named["agent_memory_sha256"] = _sha256(memory)
            named["selection_score"] = _selection_score(report["metrics"])
            reports.append(named)
            candidates_by_arm[arm].append(named)

    selected: dict[str, Any] = {}
    for arm, rows in candidates_by_arm.items():
        if not rows:
            continue
        best = max(rows, key=_selection_key)
        selected[arm] = {
            "run_name": best["name"],
            "completed_cycle": best["completed_cycle"],
            "agent_memory_path": best["agent_memory_path"],
            "agent_memory_sha256": best["agent_memory_sha256"],
            "selection_score": best["selection_score"],
            "validation_metrics": best["metrics"],
            "validation_records_path": best["records_path"],
        }

    matrix = {
        "matrix_name": f"{config['experiment_name']}_validation_selection",
        "schema_version": "travelplanner-memory-curation-validation-v1",
        "training_protocol": TRAINING_PROTOCOL,
        "frozen_intervention_path": str(frozen_interventions.resolve()),
        "frozen_intervention_sha256": _sha256(frozen_interventions),
        "frozen_factual_context_path": str(factual_contexts.resolve()),
        "frozen_factual_context_sha256": _sha256(factual_contexts),
        "selection_rule": {
            "score": (
                "recovery_avg - 0.7*low_score_rate - "
                "0.8*unavailable_usage_rate + 0.2*final_pass_rate"
            ),
            "tie_breakers": [
                "higher_recovery_avg",
                "lower_unavailable_usage_rate",
                "lower_low_score_rate",
                "higher_final_pass_rate",
            ],
        },
        "runs": reports,
        "comparison": _comparison_rows(reports),
    }
    stage_root.mkdir(parents=True, exist_ok=True)
    report_path = stage_root / "matrix_report.json"
    _write_json(report_path, matrix)
    matrix["tables"] = write_matrix_tables(report_path)
    _write_json(report_path, matrix)
    selection_payload = {
        "schema_version": "travelplanner-memory-curation-selected-checkpoints-v1",
        "selection_split": "validation",
        "selected": selected,
    }
    _write_json(stage_root / "selected_checkpoints.json", selection_payload)
    print(json.dumps(selection_payload, ensure_ascii=False, indent=2), flush=True)
    return {"matrix": matrix, "selected": selection_payload}


def run_test(config: dict[str, Any], *, arms: tuple[str, ...] = ARMS) -> dict[str, Any]:
    root = _project_path(config["output_root"])
    selection_path = root / "validation" / "selected_checkpoints.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run --stage val before --stage test")
    selection = _read_json(selection_path).get("selected") or {}
    if not set(arms).issubset(selection):
        raise ValueError(f"Test requires validation selections for {ARMS}, got {sorted(selection)}")
    stage_root = root / "test"
    frozen_interventions = _project_path(config["test"]["frozen_intervention_path"])
    expected_samples = int(config["test"]["sample_count"])
    expected_counterfactual = int(config["test"]["expected_counterfactual_count"])

    baseline_config = _holdout_eval_config(
        config,
        split="test",
        label="no_evolution",
        method="no_evolution",
        frozen_interventions=frozen_interventions,
    )
    baseline_report = _run_or_resume(
        baseline_config,
        expected_samples=expected_samples,
        expected_counterfactual=expected_counterfactual,
        quality=config["quality_gate"],
    )
    factual_contexts = Path(baseline_report["records_path"]).parent / "factual_contexts.jsonl"
    reports = [_named_report("no_evolution", baseline_report)]
    for arm in arms:
        chosen = selection[arm]
        memory = Path(chosen["agent_memory_path"])
        if not memory.exists() or _sha256(memory) != chosen["agent_memory_sha256"]:
            raise ValueError(f"Selected checkpoint missing or mutated for {arm}: {memory}")
        run_name = f"{arm}_selected_round_{chosen['completed_cycle']}"
        eval_config = _holdout_eval_config(
            config,
            split="test",
            label=run_name,
            method="agent_skill_only",
            frozen_interventions=frozen_interventions,
            frozen_factual_contexts=factual_contexts,
            agent_memory=memory,
        )
        report = _run_or_resume(
            eval_config,
            expected_samples=expected_samples,
            expected_counterfactual=expected_counterfactual,
            quality=config["quality_gate"],
        )
        named = _named_report(run_name, report)
        named.update(
            {
                "arm": arm,
                "validation_selected_cycle": chosen["completed_cycle"],
                "agent_memory_path": str(memory.resolve()),
                "agent_memory_sha256": _sha256(memory),
            }
        )
        reports.append(named)

    matrix = {
        "matrix_name": f"{config['experiment_name']}_test1000_memory_curation",
        "schema_version": "travelplanner-memory-curation-test-v1",
        "training_protocol": TRAINING_PROTOCOL,
        "selection_source": str(selection_path.resolve()),
        "frozen_intervention_path": str(frozen_interventions.resolve()),
        "frozen_intervention_sha256": _sha256(frozen_interventions),
        "frozen_factual_context_path": str(factual_contexts.resolve()),
        "frozen_factual_context_sha256": _sha256(factual_contexts),
        "claim_scope": (
            "Held-out dynamic recovery only. Official factual planning is reported "
            "separately and is not inferred from frozen factual contexts."
        ),
        "runs": reports,
        "comparison": _comparison_rows(reports),
        "paired_comparisons": _paired_report_comparisons(reports),
    }
    stage_root.mkdir(parents=True, exist_ok=True)
    report_path = stage_root / "matrix_report.json"
    _write_json(report_path, matrix)
    matrix["tables"] = write_matrix_tables(report_path)
    _write_json(report_path, matrix)
    print(f"Wrote held-out test report: {report_path}", flush=True)
    return matrix


def _training_eval_config(
    config: dict[str, Any],
    *,
    arm: str,
    cycle: int,
    label: str,
    combined_memory: Path,
    frozen_interventions: Path | None = None,
    frozen_factual_contexts: Path | None = None,
) -> dict[str, Any]:
    cycle_dir = _project_path(config["output_root"]) / arm / f"round_{cycle}"
    dynamic = frozen_interventions is None
    extra = {
        "freeze_method": True,
        "save_skill_memory": False,
        "resume": True,
        "load_skill_memory_path": str(combined_memory.resolve()),
        "load_intervention_ontology_path": str(
            _previous_state(_project_path(config["output_root"]), arm=arm, cycle=cycle)[
                "world_ontology"
            ].resolve()
        ),
        "seed_ontology_dimensions": config["train_intervention"]["dimensions"],
        "ontology_prompt_limit": int(config.get("ontology_prompt_limit", 10)),
        "agent_owned_retrieval": True,
        "agent_retrieval_max_skills": int(config["retrieval"]["max_skills"]),
        "agent_retrieval_index_limit": int(config["retrieval"]["index_limit"]),
        "skip_factual_judge": True,
        "formal_training_protocol": TRAINING_PROTOCOL,
        "formal_method_version": METHOD_VERSION,
        "curation_ablation_arm": arm,
        "curation_ablation_cycle": cycle,
        "curation_ablation_eval": label,
    }
    if frozen_interventions is not None:
        extra["frozen_intervention_path"] = str(frozen_interventions.resolve())
    if config.get("base_agent_guidance"):
        extra["base_agent_guidance_path"] = config["base_agent_guidance"]["path"]
    if frozen_factual_contexts is not None:
        extra["frozen_factual_context_path"] = str(frozen_factual_contexts.resolve())
    return {
        "experiment_name": f"eval_{label}",
        "output_dir": str(cycle_dir.resolve()),
        "method": "ontology_phase_switching",
        "iterations": 1,
        "seed": int(config["seed"]) + cycle,
        "benchmark": {
            "name": "travelplanner",
            "repo_path": str(_project_path(config["benchmark"]["repo_path"])),
            "data_path": str(_project_path(config["benchmark"]["train_data_path"])),
            "split": "train",
            "limit": int(config["train_task_count"]),
            "extra": {"planner_mode": "adaptive_rule"},
        },
        "agent": config["agent"],
        "world_model": config["world_model"],
        "judge": config["judge"],
        "intervention": {
            **config["train_intervention"],
            "policy": "ontology" if dynamic else "frozen",
        },
        "extra": extra,
    }


def _holdout_eval_config(
    config: dict[str, Any],
    *,
    split: str,
    label: str,
    method: str,
    frozen_interventions: Path,
    frozen_factual_contexts: Path | None = None,
    agent_memory: Path | None = None,
) -> dict[str, Any]:
    split_config = config[split]
    stage_root = _project_path(config["output_root"]) / split
    extra: dict[str, Any] = {
        "frozen_intervention_path": str(frozen_interventions.resolve()),
        "freeze_method": True,
        "save_skill_memory": False,
        "resume": True,
        "agent_owned_retrieval": bool(agent_memory),
        "agent_retrieval_max_skills": int(config["retrieval"]["max_skills"]),
        "agent_retrieval_index_limit": int(config["retrieval"]["index_limit"]),
        "skip_factual_judge": True,
        "formal_training_protocol": TRAINING_PROTOCOL,
        "formal_method_version": METHOD_VERSION,
        "holdout_split": split,
    }
    if frozen_factual_contexts is not None:
        extra["frozen_factual_context_path"] = str(frozen_factual_contexts.resolve())
    if agent_memory is not None:
        extra["load_skill_memory_path"] = str(agent_memory.resolve())
    if config.get("base_agent_guidance"):
        extra["base_agent_guidance_path"] = config["base_agent_guidance"]["path"]
    return {
        "experiment_name": f"{config['experiment_name']}_{split}_{label}",
        "output_dir": str(stage_root.resolve()),
        "method": method,
        "iterations": 1,
        "seed": int(config["seed"]),
        "benchmark": {
            "name": "travelplanner",
            "repo_path": str(_project_path(config["benchmark"]["repo_path"])),
            "data_path": str(_project_path(config["benchmark"][f"{split}_data_path"])),
            "split": split,
            "limit": int(split_config["sample_count"]),
            "extra": {"planner_mode": "adaptive_rule"},
        },
        "agent": config["agent"],
        "world_model": config["world_model"],
        "judge": config["judge"],
        "intervention": {
            "policy": "frozen",
            "candidate_budget": 4,
            "selected_budget": int(split_config["selected_budget"]),
            "dimensions": [],
            "severity_grid": [1.0],
            "validity_threshold": 0.7,
            "frontier_low": 0.4,
            "frontier_high": 0.6,
        },
        "extra": extra,
    }


def _run_or_resume(
    config: dict[str, Any],
    *,
    expected_samples: int,
    quality: dict[str, Any],
    expected_counterfactual: int | None = None,
) -> dict[str, Any]:
    normalized = ExperimentConfig.model_validate(config).model_dump(mode="json")
    run_dir = Path(normalized["output_dir"]) / normalized["experiment_name"]
    config_path = run_dir / "protocol_config.json"
    _write_locked_json(config_path, normalized)
    report_path = run_dir / "report.json"
    if report_path.exists():
        report = _read_json(report_path)
        try:
            _assert_report_quality(
                report,
                expected_samples=expected_samples,
                expected_counterfactual=expected_counterfactual,
                quality=quality,
            )
        except (RuntimeError, ValueError):
            pass
        else:
            print(f"Reuse complete run: {run_dir}", flush=True)
            return report

    print(f"Run/resume: {normalized['experiment_name']}", flush=True)
    report = (
        EvaluationRunner(ExperimentConfig.model_validate(normalized)).run().model_dump(mode="json")
    )
    _assert_report_quality(
        report,
        expected_samples=expected_samples,
        expected_counterfactual=expected_counterfactual,
        quality=quality,
    )
    return report


def _assert_report_quality(
    report: dict[str, Any],
    *,
    expected_samples: int,
    quality: dict[str, Any],
    expected_counterfactual: int | None,
) -> None:
    metrics = report.get("metrics") or {}
    if int(metrics.get("sample_count", -1)) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} samples, got {metrics.get('sample_count')}"
        )
    records_path = Path(str(report.get("records_path") or ""))
    if not records_path.exists():
        raise FileNotFoundError(f"Evaluation records missing: {records_path}")
    records = _load_records(records_path)
    sample_ids = {record.sample.sample_id for record in records}
    if len(sample_ids) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} represented samples, got {len(sample_ids)}"
        )
    counterfactual_count = sum(record.intervention is not None for record in records)
    if expected_counterfactual is not None and counterfactual_count != expected_counterfactual:
        raise RuntimeError(
            f"Expected {expected_counterfactual} counterfactual records, got {counterfactual_count}"
        )
    if expected_counterfactual is None:
        coverage = counterfactual_count / expected_samples if expected_samples else 0.0
        if coverage < float(quality["min_counterfactual_coverage"]):
            raise RuntimeError(f"Counterfactual coverage {coverage:.4f} is below the formal gate")

    max_error = float(quality["max_llm_error_rate"])
    min_parse = float(quality["min_parse_success_rate"])
    checks = {
        "factual_llm_error_rate": float(metrics.get("factual_llm_error_rate", 1.0)) <= max_error,
        "counterfactual_llm_error_rate": float(metrics.get("counterfactual_llm_error_rate", 1.0))
        <= max_error,
        "factual_parse_success_rate": float(metrics.get("factual_parse_success_rate", 0.0))
        >= min_parse,
        "counterfactual_parse_success_rate": float(
            metrics.get("counterfactual_parse_success_rate", 0.0)
        )
        >= min_parse,
    }
    if expected_counterfactual == 0:
        checks = {key: value for key, value in checks.items() if key.startswith("factual_")}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Evaluation quality gate failed {failed}: "
            f"experiment={report.get('experiment_name')} metrics={metrics}"
        )


def _extract_frozen_interventions(records_path: Path, output_path: Path) -> None:
    candidates: list[InterventionCandidate] = []
    seen: set[tuple[str, str]] = set()
    for record in _load_records(records_path):
        if record.intervention is None:
            continue
        key = (record.sample.sample_id, record.intervention.intervention_id)
        if key in seen:
            raise ValueError(f"Duplicate generated intervention key: {key}")
        seen.add(key)
        candidates.append(
            InterventionCandidate(
                spec=record.intervention.model_copy(deep=True),
                recovery_spec=(
                    record.recovery_spec.model_copy(deep=True)
                    if record.recovery_spec is not None
                    else None
                ),
                validity_score=float(record.metadata.get("intervention_validity") or 0.0),
                utility=_optional_float(record.metadata.get("intervention_utility")),
                validation_notes=[
                    str(item) for item in record.metadata.get("validation_notes", [])
                ],
            )
        )
    payload = "".join(
        json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False) + "\n"
        for candidate in candidates
    )
    _write_locked_text(output_path, payload)


def _accept_all_summary(
    *,
    candidate_memory: Path,
    output_memory: Path,
    baseline_records: list[EvaluationRecord],
    candidate_records: list[EvaluationRecord],
) -> dict[str, Any]:
    registry = SkillRegistry.load_jsonl(output_memory)
    status_counts = Counter(skill.status.value for skill in registry.all())
    type_counts = Counter(skill.skill_type.value for skill in registry.all())
    paired = classify_paired_outcomes(baseline_records, candidate_records)
    return {
        "mode": "accept_all_raw_candidates",
        "paired_memory_curation_enabled": False,
        "candidate_memory_sha256": _sha256(candidate_memory),
        "output_memory_sha256": _sha256(output_memory),
        "byte_identical_to_candidate": _sha256(candidate_memory) == _sha256(output_memory),
        "skill_count": len(registry),
        "status_counts": dict(status_counts),
        "skill_type_counts": dict(type_counts),
        "raw_candidate_paired_outcomes": {
            "counts": paired["counts"],
            "rates": paired["rates"],
        },
        "decision": (
            "All raw candidate memory modifications are accepted without "
            "promote, repair, or deprecate decisions."
        ),
    }


def _previous_state(root: Path, *, arm: str, cycle: int) -> dict[str, Path]:
    state_dir = root / "initial_state" if cycle == 1 else root / arm / f"round_{cycle - 1}"
    paths = {
        "agent_memory": state_dir / "agent_memory.jsonl",
        "world_memory": state_dir / "world_memory.jsonl",
        "world_ontology": state_dir / "world_ontology.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Previous training state missing: " + ", ".join(missing))
    return paths


def _round1_shared_artifacts(root: Path) -> dict[str, Path]:
    shared = root / "shared" / "round_1"
    return {
        "root": shared,
        "pre_report": shared / "pre_report.json",
        "pre_records": shared / "pre_records.jsonl",
        "factual_contexts": shared / "factual_contexts.jsonl",
        "interventions": shared / "frozen_cycle_interventions.jsonl",
        "candidate_memory": shared / "agent_memory_raw_candidate.jsonl",
        "candidate_report": shared / "candidate_report.json",
        "candidate_records": shared / "candidate_records.jsonl",
    }


def _merge_memories(agent_memory: Path, world_memory: Path, output_path: Path) -> None:
    agent_registry = SkillRegistry.load_jsonl(agent_memory)
    world_registry = SkillRegistry.load_jsonl(world_memory)
    merged = SkillRegistry(
        [
            *[
                skill
                for skill in agent_registry.all()
                if skill.skill_type == SkillType.AGENT_RECOVERY
            ],
            *[
                skill
                for skill in world_registry.all()
                if skill.skill_type == SkillType.INTERVENTION_SPEC
            ],
        ]
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    merged.save_jsonl(temporary)
    _copy_locked(temporary, output_path)
    temporary.unlink(missing_ok=True)


def _assert_same_paired_keys(
    left: list[EvaluationRecord], right: list[EvaluationRecord], *, label: str
) -> tuple[list[EvaluationRecord], list[EvaluationRecord]]:
    left_factual_only = Counter(
        record.sample.sample_id for record in left if record.intervention is None
    )
    right_factual_only = Counter(
        record.sample.sample_id for record in right if record.intervention is None
    )
    if left_factual_only != right_factual_only:
        raise ValueError(
            f"Strict factual-only coverage pairing failed for {label}: "
            f"left={dict(left_factual_only)}, right={dict(right_factual_only)}"
        )

    left_counterfactual = [record for record in left if record.intervention is not None]
    right_counterfactual = [record for record in right if record.intervention is not None]
    try:
        classify_paired_outcomes(left_counterfactual, right_counterfactual)
    except ValueError as exc:
        raise ValueError(f"Strict same-challenge pairing failed for {label}: {exc}") from exc
    return left_counterfactual, right_counterfactual


def _lifecycle_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "promote_delta_threshold",
        "deprecate_delta_threshold",
        "min_support",
        "passive_notify_delta_threshold",
        "low_score_delta_threshold",
        "unavailable_usage_delta_repair_threshold",
        "unavailable_usage_delta_deprecate_threshold",
        "lifecycle_score_promote_threshold",
        "lifecycle_score_deprecate_threshold",
    }
    return {key: value for key, value in config.items() if key in allowed}


def _selection_score(metrics: dict[str, Any]) -> float:
    return round(
        float(metrics.get("counterfactual_recovery_avg") or 0.0)
        - 0.7 * float(metrics.get("counterfactual_low_score_rate") or 0.0)
        - 0.8 * float(metrics.get("counterfactual_unavailable_usage_rate") or 0.0)
        + 0.2 * float(metrics.get("counterfactual_final_pass_rate") or 0.0),
        8,
    )


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = row["metrics"]
    return (
        float(row["selection_score"]),
        float(metrics.get("counterfactual_recovery_avg") or 0.0),
        -float(metrics.get("counterfactual_unavailable_usage_rate") or 0.0),
        -float(metrics.get("counterfactual_low_score_rate") or 0.0),
        float(metrics.get("counterfactual_final_pass_rate") or 0.0),
    )


def _named_report(name: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "experiment_name": report["experiment_name"],
        "benchmark": report["benchmark"],
        "method": report["method"],
        "total_samples": report["total_samples"],
        "metrics": report["metrics"],
        "records_path": str(Path(report["records_path"]).resolve()),
    }


def _comparison_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": report["name"],
            "experiment_name": report["experiment_name"],
            "total_samples": report["total_samples"],
            **report["metrics"],
        }
        for report in reports
    ]


def _paired_report_comparisons(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {report["name"]: report for report in reports}
    targets = [name for name in indexed if name != "no_evolution"]
    pairs = [("no_evolution", target) for target in targets]
    paired_name = next(
        (name for name in targets if name.startswith("full_selected")),
        None,
    )
    raw_name = next(
        (name for name in targets if name.startswith("without_memory_curation_selected")),
        None,
    )
    if paired_name and raw_name:
        pairs.append((raw_name, paired_name))

    output = []
    for baseline_name, target_name in pairs:
        baseline = _load_records(Path(indexed[baseline_name]["records_path"]))
        target = _load_records(Path(indexed[target_name]["records_path"]))
        diagnostic = curation_decision(baseline, target, thresholds={})
        output.append(
            {
                "baseline": baseline_name,
                "target": target_name,
                "mean_recovery_delta": diagnostic["mean_recovery_delta"],
                "low_score_rate_delta": diagnostic["low_score_rate_delta"],
                "unavailable_usage_rate_delta": diagnostic["unavailable_usage_rate_delta"],
                "final_pass_rate_delta": diagnostic["final_pass_rate_delta"],
                "success_to_failure_rate": diagnostic["transition_rates"]["success_to_failure"],
                "transition_counts": diagnostic["transition_counts"],
            }
        )
    return output


def _report_reference(report: dict[str, Any]) -> dict[str, Any]:
    records_path = Path(report["records_path"])
    report_path = Path(report.get("_source_report_path") or records_path.parent / "report.json")
    return {
        "experiment_name": report["experiment_name"],
        "report_path": str(report_path.resolve()),
        "records_path": str(records_path.resolve()),
        "records_sha256": _sha256(records_path),
        "metrics": report.get("metrics", {}),
    }


def _require_training_complete(root: Path, *, cycles: int, arms: tuple[str, ...]) -> None:
    missing = [
        str(root / arm / f"round_{cycle}" / "round_manifest.json")
        for arm in arms
        for cycle in range(1, cycles + 1)
        if not (root / arm / f"round_{cycle}" / "round_manifest.json").exists()
    ]
    if missing:
        raise FileNotFoundError("Training rounds incomplete:\n- " + "\n- ".join(missing))


def _validate_completed_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "complete":
        raise ValueError("Existing round manifest is not complete")
    if not manifest.get("final_re_evaluation_performed"):
        raise ValueError("Existing round did not record final re-evaluation")
    if manifest.get("paired_outcomes", {}).get("world_update_input") != ("failure_to_failure_only"):
        raise ValueError("Existing round violates F->F World-update routing")
    for name in ("agent_memory", "world_memory", "world_ontology"):
        item = manifest.get("outputs", {}).get(name) or {}
        path = Path(str(item.get("path") or ""))
        if not path.exists() or _sha256(path) != item.get("sha256"):
            raise ValueError(f"Round output missing or mutated: {name} {path}")


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION}")
    required = {
        "experiment_name",
        "output_root",
        "cycles",
        "train_task_count",
        "seed",
        "benchmark",
        "agent",
        "world_model",
        "judge",
        "train_intervention",
        "retrieval",
        "lifecycle",
        "acceptance_diagnostic",
        "quality_gate",
        "validation",
        "test",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Curation-ablation config missing keys: {missing}")
    if not config.get("public_profile") and int(config["cycles"]) != 3:
        raise ValueError("The preregistered ablation uses exactly three cycles")
    if not config.get("public_profile") and int(config["seed"]) != 42:
        raise ValueError("The preregistered ablation uses seed=42")
    if int(config["train_intervention"].get("selected_budget", 0)) != 1:
        raise ValueError("Training selected_budget must be 1 for paired evidence")
    if int(config["validation"].get("selected_budget", 0)) != 2:
        raise ValueError("Validation selected_budget must match the frozen x2 set")
    if int(config["test"].get("selected_budget", 0)) != 2:
        raise ValueError("Test selected_budget must match the frozen x2 set")
    if config["agent"].get("model") != config["world_model"].get("model"):
        raise ValueError("Agent and World Model must use the same base model")
    if int(config["cycles"]) < 1 or int(config["train_task_count"]) < 1:
        raise ValueError("cycles and train_task_count must be positive")
    forbidden = [
        key for key in _flatten_keys(config) if "api_key" in key and not key.endswith("api_key_env")
    ]
    if forbidden:
        raise ValueError(f"Literal API-key fields are forbidden in tracked config: {forbidden}")


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


def _code_provenance() -> dict[str, Any]:
    files = sorted(
        path
        for path in (PROJECT_ROOT / "src" / "cce_mobility").rglob("*.py")
        if not path.name.startswith("local_secrets")
    )
    files.append(Path(__file__).resolve())
    status = _git_output(
        [
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src/cce_mobility",
        ]
    )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "git_commit": _git_output(["rev-parse", "HEAD"]) or "unknown",
        "relevant_dirty": bool(status),
        "relevant_status": status.splitlines(),
        "formal_source_sha256": digest.hexdigest(),
        "formal_file_count": len(set(files)),
    }


def _require_committed_clean_code(provenance: dict[str, Any]) -> None:
    if provenance.get("git_commit") in {"", "unknown"} or provenance.get("relevant_dirty"):
        raise RuntimeError(
            f"Formal TravelPlanner ablation requires committed clean experiment code: {provenance}"
        )


def _git_output(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_dir():
        digest = hashlib.sha256()
        total_bytes = 0
        file_count = 0
        for child in _stable_directory_files(path):
            relative = child.relative_to(path).as_posix()
            size = child.stat().st_size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
            total_bytes += size
            file_count += 1
        return {
            "path": str(path.resolve()),
            "kind": "directory",
            "file_count": file_count,
            "bytes": total_bytes,
            "sha256": digest.hexdigest(),
        }
    return {
        "path": str(path.resolve()),
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _stable_directory_files(path: Path) -> list[Path]:
    ignored_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
        "__pycache__",
    }
    return sorted(
        child
        for child in path.rglob("*")
        if child.is_file()
        and not any(part in ignored_parts for part in child.relative_to(path).parts)
    )


def _frozen_challenge_count(path: Path) -> int:
    keys: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = InterventionCandidate.model_validate_json(line)
            key = (candidate.spec.sample_id, candidate.spec.intervention_id)
            if key in keys:
                raise ValueError(f"Duplicate frozen challenge key in {path}: {key}")
            keys.add(key)
    return len(keys)


def _load_records(path: Path) -> list[EvaluationRecord]:
    with path.open("r", encoding="utf-8") as handle:
        return [EvaluationRecord.model_validate_json(line) for line in handle if line.strip()]


def _write_records(path: Path, records: list[EvaluationRecord]) -> None:
    payload = "".join(record.model_dump_json() + "\n" for record in records)
    _write_locked_text(path, payload)


def _jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_locked_json(path: Path, payload: Any) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _write_locked_text(path, rendered)


def _write_locked_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"Existing locked artifact differs: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def _copy_locked(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise ValueError(f"Existing copied artifact differs: {destination}")
        return
    shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
