from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.core.config import ExperimentConfig, ModelConfig  # noqa: E402
from cce_mobility.core.llm import ChatClient  # noqa: E402
from cce_mobility.core.schema import EvaluationRecord  # noqa: E402
from cce_mobility.integrations.mobilitybench_continuation import (  # noqa: E402
    CONTINUATION_PROTOCOL,
    SUPPORTED_TOOL_INTERVENTION_DIMENSIONS,
    file_sha256,
    formal_experiment_code_provenance,
    stable_hash,
)
from cce_mobility.interventions.ontology import registry_from_seed_dimensions  # noqa: E402
from cce_mobility.methods.formal_coevolution import (  # noqa: E402
    FORMAL_METHOD_VERSION,
    FORMAL_TRAINING_PROTOCOL,
    build_agent_candidate_memory,
    classify_paired_outcomes,
    curation_decision,
    update_world_from_persistent_failures,
)
from cce_mobility.skills.lifecycle import gate_memory_with_paired_outcome_evidence  # noqa: E402
from cce_mobility.skills.registry import SkillRegistry  # noqa: E402


TRAIN_CONFIG_SCHEMA = "mobilitybench-formal-coevolution-train-v1"
ARMS = ("full", "agent_only", "without_memory_curation")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train formal paired-residual MobilityBench co-evolution checkpoints."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("prepare", "train"), default="prepare")
    parser.add_argument("--arm", choices=(*ARMS, "both"), default="both")
    parser.add_argument("--through-cycle", type=int, default=0)
    args = parser.parse_args()
    config = _read_json(args.config)
    _validate_config(config)
    through_cycle = args.through_cycle or int(config["cycles"])
    if through_cycle < 1 or through_cycle > int(config["cycles"]):
        raise ValueError("--through-cycle must be between 1 and configured cycles")

    preflight = prepare(config, config_path=args.config)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "prepare":
        return 0

    arms = ARMS if args.arm == "both" else (args.arm,)
    # Interleave arms by cycle to reduce time-window/API drift. Full round 1 runs
    # first because Agent-only reuses that exact challenge artifact while the two
    # arms still have byte-identical Agent and World states.
    for cycle in range(1, through_cycle + 1):
        for arm in arms:
            run_cycle(config, arm=arm, cycle=cycle)
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
        registry_from_seed_dimensions(config["intervention"]["dimensions"]).save_jsonl(
            ontology_path
        )

    required = {
        "train_factual_records": _project_path(config["train_factual_records_path"]),
        "official_config": _project_path(config["official_config_path"]),
        "official_prompt_bundle": _project_path(config["official_prompt_bundle_path"]),
        "mobilitybench_python": Path(config["mobilitybench_python"]).expanduser(),
        "mobilitybench_root": Path(config["mobilitybench_root"]).expanduser(),
    }
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    if missing:
        raise FileNotFoundError("Formal training inputs missing: " + json.dumps(missing, indent=2))
    factual_count = _jsonl_count(required["train_factual_records"])
    schedule = _prepare_task_schedule(
        root=root,
        factual_records_path=required["train_factual_records"],
        cycles=int(config["cycles"]),
        tasks_per_cycle=int(config["train_task_count"]),
        seed=int(config["seed"]),
        mode=str(config.get("task_schedule_mode") or "disjoint_without_replacement"),
    )

    formal_code = formal_experiment_code_provenance(PROJECT_ROOT)
    if not config.get("public_profile"):
        _require_committed_clean_code(formal_code)
    artifact_commit = str(formal_code["git_commit"])
    payload = {
        "schema_version": TRAIN_CONFIG_SCHEMA,
        "training_protocol": FORMAL_TRAINING_PROTOCOL,
        "method_version": FORMAL_METHOD_VERSION,
        "continuation_protocol": CONTINUATION_PROTOCOL,
        "config_source": str(config_path.resolve()) if config_path else "programmatic",
        "config_sha256": stable_hash(config),
        "artifact_commit": artifact_commit,
        "formal_code_provenance": formal_code,
        "seed": config["seed"],
        "cycles": config["cycles"],
        "arms": list(ARMS),
        "train_factual_pool_count": factual_count,
        "train_task_count_per_cycle": int(config["train_task_count"]),
        "task_schedule": schedule,
        "initial_agent_memory": _artifact(agent_memory),
        "initial_world_memory": _artifact(world_memory),
        "initial_world_ontology": _artifact(ontology_path),
        "required_inputs": {
            name: _artifact(path) if path.is_file() else {"path": str(path)}
            for name, path in required.items()
        },
        "budget_contract": {
            "agent_evaluations_per_cycle": ["pre", "candidate", "curated"],
            "matched_between_full_and_agent_only": True,
            "challenge_artifacts_shared_when_world_state_identical": True,
            "world_generation_calls_matched": False,
            "world_generation_output_count_matched": True,
            "world_update_calls_matched": False,
            "world_update_calls_are_experimental_variable": True,
        },
    }
    preflight_path = root / "preflight.json"
    if preflight_path.exists():
        previous = _read_json(preflight_path)
        compatible = dict(payload)
        compatible["artifact_commit"] = previous.get("artifact_commit")
        compatible["formal_code_provenance"] = previous.get("formal_code_provenance")
        if stable_hash(previous) != stable_hash(compatible):
            raise ValueError("Existing preflight.json differs; use a new output_root")
        from cce_mobility.integrations.mobilitybench_continuation import (
            verify_scoped_provenance_transition,
        )

        if config.get("public_profile"):
            if (
                previous["formal_code_provenance"]["formal_source_sha256"]
                != payload["formal_code_provenance"]["formal_source_sha256"]
            ):
                raise ValueError("Formal source changed; use a new output_root")
            transition = {"equivalence": "byte_identical"}
        else:
            transition = verify_scoped_provenance_transition(
                PROJECT_ROOT, previous["formal_code_provenance"], scope="formal"
            )
        if transition["equivalence"] != "byte_identical":
            print(json.dumps({"preflight_source_transition": transition}, indent=2), flush=True)
        payload = previous
    else:
        _write_json(preflight_path, payload)
    return payload


def run_cycle(config: dict[str, Any], *, arm: str, cycle: int) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"Unsupported arm: {arm}")
    root = _project_path(config["output_root"])
    cycle_dir = root / arm / f"round_{cycle}"
    manifest_path = cycle_dir / "round_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        _validate_completed_manifest(manifest, cycle_dir)
        print(f"Reuse complete {arm} round {cycle}: {manifest_path}", flush=True)
        return manifest
    cycle_dir.mkdir(parents=True, exist_ok=True)

    previous = _previous_state(root, arm=arm, cycle=cycle)
    round_factual_records = _materialize_round_factual_records(
        root=root,
        source_path=_project_path(config["train_factual_records_path"]),
        cycle=cycle,
    )
    interventions_path = _round_interventions_path(root, arm=arm, cycle=cycle)
    intervention_summary_path = interventions_path.with_suffix(".summary.json")
    shared_source = _shared_full_challenges_if_world_identical(
        root=root,
        arm=arm,
        cycle=cycle,
        previous=previous,
    )
    if shared_source is not None:
        _reuse_challenges(
            source=shared_source,
            output=interventions_path,
            summary_path=intervention_summary_path,
            factual_records_path=round_factual_records,
            previous=previous,
            cycle=cycle,
        )
    else:
        _generate_round_challenges(
            config,
            arm=arm,
            cycle=cycle,
            world_memory_path=previous["world_memory"],
            ontology_path=previous["world_ontology"],
            factual_records_path=round_factual_records,
            output_path=interventions_path,
        )
        if cycle == 1 and arm == "full":
            shared = root / "shared" / "round_1" / "interventions.jsonl"
            shared.parent.mkdir(parents=True, exist_ok=True)
            if shared.exists() and file_sha256(shared) != file_sha256(interventions_path):
                raise ValueError(f"Locked shared round-1 challenges differ: {shared}")
            if not shared.exists():
                shutil.copy2(interventions_path, shared)
    _assert_jsonl_count(interventions_path, int(config["train_task_count"]))
    if not intervention_summary_path.exists():
        raise FileNotFoundError(
            f"Challenge-generation usage summary missing: {intervention_summary_path}"
        )

    continuation_cases = cycle_dir / "continuation_cases.jsonl"
    _prepare_continuation_cases(
        config,
        factual_records_path=round_factual_records,
        interventions_path=interventions_path,
        output_path=continuation_cases,
    )

    pre_config = _eval_config(
        config,
        arm=arm,
        cycle=cycle,
        label="pre",
        continuation_cases=continuation_cases,
        agent_memory=previous["agent_memory"],
        world_memory=previous["world_memory"],
        agent_update_count=cycle - 1,
        cumulative_rollouts=(cycle - 1) * int(config["train_task_count"]) * 3,
    )
    pre_report = _run_evaluation(config, pre_config, cycle_dir / "pre_eval_config.json")
    pre_records_path = _project_path(pre_report["records_path"])
    pre_records_all = _load_records(pre_records_path)
    minimum_valid_fraction = float(config.get("training_evidence_min_valid_fraction", 0.5))
    pre_records, pre_evidence = _select_training_evidence(
        pre_records_all,
        expected_count=int(config["train_task_count"]),
        minimum_valid_fraction=minimum_valid_fraction,
        label=f"{arm} round {cycle} pre",
    )
    evidence_dir = cycle_dir / "training_evidence"
    candidate_build_records_path = evidence_dir / "pre_for_candidate_build.jsonl"
    _write_records(candidate_build_records_path, pre_records)

    candidate_memory = cycle_dir / "agent_memory_candidate.jsonl"
    candidate_build = build_agent_candidate_memory(
        previous["agent_memory"], pre_records, candidate_memory
    )
    candidate_config = _eval_config(
        config,
        arm=arm,
        cycle=cycle,
        label="candidate",
        continuation_cases=continuation_cases,
        agent_memory=candidate_memory,
        world_memory=previous["world_memory"],
        agent_update_count=cycle,
        cumulative_rollouts=(cycle - 1) * int(config["train_task_count"]) * 3
        + 2 * int(config["train_task_count"]),
    )
    candidate_report = _run_evaluation(
        config, candidate_config, cycle_dir / "candidate_eval_config.json"
    )
    candidate_records_all = _load_records(_project_path(candidate_report["records_path"]))
    gate_pre_records, candidate_records, candidate_evidence = _paired_training_evidence(
        pre_records_all,
        candidate_records_all,
        expected_count=int(config["train_task_count"]),
        minimum_valid_fraction=minimum_valid_fraction,
        label=f"{arm} round {cycle} pre/candidate",
    )
    gate_pre_records_path = evidence_dir / "pre_for_candidate_gate.jsonl"
    candidate_records_path = evidence_dir / "candidate_for_lifecycle_gate.jsonl"
    _write_records(gate_pre_records_path, gate_pre_records)
    _write_records(candidate_records_path, candidate_records)

    curated_memory = cycle_dir / "agent_memory_curated.jsonl"
    if arm == "without_memory_curation":
        shutil.copy2(candidate_memory, curated_memory)
        gate_summary = {"enabled": False, "mode": "accept_all_raw_candidates"}
    else:
        gate_summary = gate_memory_with_paired_outcome_evidence(
            memory_path=candidate_memory,
            baseline_records_path=gate_pre_records_path,
            target_records_path=candidate_records_path,
            output_path=curated_memory,
            repair_mode="verification_constrained",
            **_lifecycle_kwargs(config.get("lifecycle") or {}),
        )
    curated_config = _eval_config(
        config,
        arm=arm,
        cycle=cycle,
        label="curated",
        continuation_cases=continuation_cases,
        agent_memory=curated_memory,
        world_memory=previous["world_memory"],
        agent_update_count=cycle,
        cumulative_rollouts=cycle * int(config["train_task_count"]) * 3,
    )
    curated_report = _run_evaluation(config, curated_config, cycle_dir / "curated_eval_config.json")
    curated_records_all = _load_records(_project_path(curated_report["records_path"]))
    acceptance_pre_records, curated_records, curated_evidence = _paired_training_evidence(
        pre_records_all,
        curated_records_all,
        expected_count=int(config["train_task_count"]),
        minimum_valid_fraction=minimum_valid_fraction,
        label=f"{arm} round {cycle} pre/curated",
    )
    acceptance_pre_records_path = evidence_dir / "pre_for_acceptance.jsonl"
    curated_records_path = evidence_dir / "curated_for_acceptance.jsonl"
    _write_records(acceptance_pre_records_path, acceptance_pre_records)
    _write_records(curated_records_path, curated_records)
    acceptance = curation_decision(
        acceptance_pre_records,
        curated_records,
        thresholds=config.get("acceptance") or {},
    )
    if arm == "without_memory_curation":
        acceptance["diagnostic_accepted"] = acceptance["accepted"]
        acceptance["accepted"] = True
        acceptance["mode"] = "accept_all_keep_final_reevaluation_and_failure_to_failure_filter"

    checkpoint_agent = cycle_dir / "agent_memory.jsonl"
    if acceptance["accepted"]:
        shutil.copy2(curated_memory, checkpoint_agent)
        post_records = curated_records
        post_records_path = curated_records_path
    else:
        shutil.copy2(previous["agent_memory"], checkpoint_agent)
        post_records = acceptance_pre_records
        post_records_path = acceptance_pre_records_path
    outcomes = classify_paired_outcomes(acceptance_pre_records, post_records)
    regression_path = cycle_dir / "agent_regressions.jsonl"
    persistent_path = cycle_dir / "persistent_failures.jsonl"
    _write_records(regression_path, outcomes["agent_regression_records"])
    _write_records(persistent_path, outcomes["persistent_failure_records"])

    checkpoint_world = cycle_dir / "world_memory.jsonl"
    checkpoint_ontology = cycle_dir / "world_ontology.jsonl"
    if arm != "agent_only":
        world_client = ChatClient(ModelConfig.model_validate(config["world_model"]))
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
        world_update["transport"] = world_client.transport_metrics()
    else:
        shutil.copy2(previous["world_memory"], checkpoint_world)
        shutil.copy2(previous["world_ontology"], checkpoint_ontology)
        world_update = {
            "enabled": False,
            "freeze_reason": "formal_agent_only_ablation",
            "persistent_failure_count": len(outcomes["persistent_failure_records"]),
            "world_memory_changed": False,
            "ontology_changed": False,
            "new_world_spec_ids": [],
            "new_ontology_ids": [],
        }

    manifest = {
        "status": "complete",
        "schema_version": "mobilitybench-formal-coevolution-round-v1",
        "training_protocol": FORMAL_TRAINING_PROTOCOL,
        "method_version": FORMAL_METHOD_VERSION,
        "continuation_protocol": CONTINUATION_PROTOCOL,
        "arm": arm,
        "world_update_mode": "frozen" if arm == "agent_only" else "enabled",
        "seed": config["seed"],
        "completed_cycle": cycle,
        "raw_iteration": 2 * cycle,
        "phase": "paired_agent_then_world",
        "agent_update_attempt_count": cycle,
        "agent_update_accept_count": _previous_accept_count(root, arm, cycle)
        + int(acceptance["accepted"]),
        "cumulative_training_rollout_count": cycle * int(config["train_task_count"]) * 3,
        "round_training_rollout_count": int(config["train_task_count"]) * 3,
        "challenge_generation_task_count": int(config["train_task_count"]),
        "agent_update_accepted": acceptance["accepted"],
        "agent_memory_changed": file_sha256(checkpoint_agent)
        != file_sha256(previous["agent_memory"]),
        "world_memory_changed": file_sha256(checkpoint_world)
        != file_sha256(previous["world_memory"]),
        "world_ontology_changed": file_sha256(checkpoint_ontology)
        != file_sha256(previous["world_ontology"]),
        "artifact_commit": _git_commit(),
        "formal_code_provenance": formal_experiment_code_provenance(PROJECT_ROOT),
        "inputs": {
            "previous_agent_memory": _artifact(previous["agent_memory"]),
            "previous_world_memory": _artifact(previous["world_memory"]),
            "previous_world_ontology": _artifact(previous["world_ontology"]),
            "round_factual_records": _artifact(round_factual_records),
            "interventions": _artifact(interventions_path),
            "challenge_generation_summary": _artifact(intervention_summary_path),
            "continuation_cases": _artifact(continuation_cases),
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
            "candidate": _report_reference(candidate_report),
            "curated": _report_reference(curated_report),
            "accepted_post_records_path": str(post_records_path),
            "accepted_post_records_sha256": file_sha256(post_records_path),
        },
        "training_evidence": {
            "policy": "exclude_execution_errors_after_bounded_retry_v1",
            "fixed_denominator_count": int(config["train_task_count"]),
            "minimum_valid_fraction": minimum_valid_fraction,
            "pre": pre_evidence,
            "pre_candidate_pair": candidate_evidence,
            "pre_curated_pair": curated_evidence,
            "artifacts": {
                "pre_for_candidate_build": _artifact(candidate_build_records_path),
                "pre_for_candidate_gate": _artifact(gate_pre_records_path),
                "candidate_for_lifecycle_gate": _artifact(candidate_records_path),
                "pre_for_acceptance": _artifact(acceptance_pre_records_path),
                "curated_for_acceptance": _artifact(curated_records_path),
            },
        },
        "candidate_build": candidate_build,
        "lifecycle_gate": gate_summary,
        "acceptance": acceptance,
        "paired_outcomes": {
            "counts": outcomes["counts"],
            "rates": outcomes["rates"],
            "world_update_input": "failure_to_failure_only",
            "success_to_failure_routed_to_world": False,
        },
        "world_update": world_update,
    }
    _write_json(manifest_path, manifest)
    print(
        f"Completed {arm} round {cycle}: agent_accept={acceptance['accepted']} "
        f"F->F={outcomes['counts']['failure_to_failure']} "
        f"S->F={outcomes['counts']['success_to_failure']}",
        flush=True,
    )
    return manifest


def _previous_state(root: Path, *, arm: str, cycle: int) -> dict[str, Path]:
    state_dir = root / "initial_state" if cycle == 1 else root / arm / f"round_{cycle - 1}"
    paths = {
        "agent_memory": state_dir / "agent_memory.jsonl",
        "world_memory": state_dir / "world_memory.jsonl",
        "world_ontology": state_dir / "world_ontology.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Previous round state missing: " + ", ".join(missing))
    return paths


def _round_interventions_path(root: Path, *, arm: str, cycle: int) -> Path:
    return root / arm / f"round_{cycle}" / "interventions.jsonl"


def _shared_full_challenges_if_world_identical(
    *,
    root: Path,
    arm: str,
    cycle: int,
    previous: dict[str, Path],
) -> Path | None:
    if arm != "agent_only":
        return None
    if cycle == 1:
        source = root / "shared" / "round_1" / "interventions.jsonl"
        if not source.exists():
            raise FileNotFoundError(
                "Round-1 shared challenges are missing. Run full round 1 before agent_only."
            )
        return source

    full_previous = _previous_state(root, arm="full", cycle=cycle)
    same_world = all(
        file_sha256(previous[key]) == file_sha256(full_previous[key])
        for key in ("world_memory", "world_ontology")
    )
    if not same_world:
        return None
    source = _round_interventions_path(root, arm="full", cycle=cycle)
    if not source.exists():
        raise FileNotFoundError(
            "Full challenges for an identical World state are missing. "
            f"Run full round {cycle} before agent_only: {source}"
        )
    return source


def _reuse_challenges(
    *,
    source: Path,
    output: Path,
    summary_path: Path,
    factual_records_path: Path,
    previous: dict[str, Path],
    cycle: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and file_sha256(output) != file_sha256(source):
        raise ValueError(
            "Existing Agent-only challenges differ from the byte-identical shared "
            f"artifact: {output}"
        )
    if not output.exists():
        shutil.copy2(source, output)
    reuse_summary = {
        "reuse_mode": (
            "byte_identical_shared_round1"
            if cycle == 1
            else "byte_identical_full_challenges_same_world_state"
        ),
        "source_interventions": _artifact(source),
        "output_sha256": file_sha256(output),
        "official_factual_records_sha256": file_sha256(factual_records_path),
        "world_memory": _artifact(previous["world_memory"]),
        "world_ontology": _artifact(previous["world_ontology"]),
        "world_model_transport": {
            "call_count": 0,
            "request_count": 0,
            "retry_count": 0,
            "error_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
    _write_locked_json(summary_path, reuse_summary)


def _generate_round_challenges(
    config: dict[str, Any],
    *,
    arm: str,
    cycle: int,
    world_memory_path: Path,
    ontology_path: Path,
    factual_records_path: Path,
    output_path: Path,
) -> None:
    cycle_dir = output_path.parent
    generation_config = {
        "experiment_name": f"{config['experiment_name']}_{arm}_round{cycle}_challenge_generation",
        "output_dir": str(cycle_dir),
        "method": "ontology_phase_switching",
        "iterations": 1,
        "seed": int(config["seed"]) + cycle,
        "benchmark": {
            "name": "mobilitybench",
            "data_path": str(factual_records_path),
            "split": "official_train",
            "limit": int(config["train_task_count"]),
        },
        "agent": config["agent"],
        "world_model": config["world_model"],
        "judge": config["judge"],
        "intervention": config["intervention"],
        "extra": {
            "freeze_method": True,
            "save_skill_memory": False,
            "load_skill_memory_path": str(world_memory_path),
            "load_intervention_ontology_path": str(ontology_path),
            "seed_ontology_dimensions": config["intervention"]["dimensions"],
            "formal_training_protocol": FORMAL_TRAINING_PROTOCOL,
            "formal_method_version": FORMAL_METHOD_VERSION,
            "arm": arm,
            "completed_cycle_before_generation": cycle - 1,
            "require_full_coverage": True,
            "refuse_overwrite": True,
        },
    }
    config_path = cycle_dir / "challenge_generation_config.json"
    _write_locked_json(config_path, generation_config)
    if output_path.exists():
        _assert_jsonl_count(output_path, int(config["train_task_count"]))
        summary_path = output_path.with_suffix(".summary.json")
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Existing interventions lack generation summary: {summary_path}"
            )
        _validate_challenge_reuse(
            summary_path=summary_path,
            generation_config=generation_config,
            output_path=output_path,
            factual_records_path=factual_records_path,
            world_memory_path=world_memory_path,
            ontology_path=ontology_path,
        )
        return
    _run_command(
        [
            sys.executable,
            "-m",
            "cce_mobility.workflows.mobilitybench.generate_mobilitybench_official_frozen_intervention_set",
            str(config_path),
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
    )


def _prepare_continuation_cases(
    config: dict[str, Any],
    *,
    factual_records_path: Path,
    interventions_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        _assert_jsonl_count(output_path, int(config["train_task_count"]))
        audit_path = output_path.with_suffix(".audit.json")
        if not audit_path.exists():
            raise FileNotFoundError(f"Existing continuation cases lack source audit: {audit_path}")
        audit = _read_json(audit_path)
        checks = {
            "case_artifact": audit.get("cases_sha256") == file_sha256(output_path),
            "factual_source": audit.get("factual_records_sha256")
            == file_sha256(factual_records_path),
            "intervention_source": audit.get("frozen_interventions_sha256")
            == file_sha256(interventions_path),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                f"Existing continuation cases fail source integrity checks {failed}: {output_path}"
            )
        return
    _run_command(
        [
            sys.executable,
            "-m",
            "cce_mobility.workflows.mobilitybench.prepare_mobilitybench_continuation_protocol",
            "cases",
            "--factual-records",
            str(factual_records_path),
            "--frozen-interventions",
            str(interventions_path),
            "--output",
            str(output_path),
            "--expected-count",
            str(config["train_task_count"]),
        ],
        cwd=PROJECT_ROOT,
    )


def _eval_config(
    config: dict[str, Any],
    *,
    arm: str,
    cycle: int,
    label: str,
    continuation_cases: Path,
    agent_memory: Path,
    world_memory: Path,
    agent_update_count: int,
    cumulative_rollouts: int,
) -> dict[str, Any]:
    output_dir = _project_path(config["output_root"]) / arm / f"round_{cycle}" / f"eval_{label}"
    return {
        "schema_version": "mobilitybench-continuation-eval-v1",
        "method_version": FORMAL_METHOD_VERSION,
        "experiment_name": f"{config['experiment_name']}_{arm}_round{cycle}_{label}",
        "output_dir": str(output_dir),
        "continuation_cases_path": str(continuation_cases),
        "official_config_path": str(_project_path(config["official_config_path"])),
        "official_prompt_bundle_path": str(_project_path(config["official_prompt_bundle_path"])),
        "official_model_alias": config["official_model_alias"],
        "expected_count": int(config["train_task_count"]),
        "max_case_attempts": int(config.get("max_case_attempts", 2)),
        "agent": config["agent"],
        "judge": config["judge"],
        "agent_memory_path": str(agent_memory),
        "world_memory_path": str(world_memory),
        "world_ontology_path": str(
            _previous_state(_project_path(config["output_root"]), arm=arm, cycle=cycle)[
                "world_ontology"
            ]
        ),
        "retrieval": config["retrieval"],
        "training_series": arm,
        "completed_cycle": cycle if label != "pre" else cycle - 1,
        "raw_iteration": 2 * cycle if label != "pre" else 2 * (cycle - 1),
        "agent_update_count": agent_update_count,
        "training_rollout_count": cumulative_rollouts,
        "artifact_commit": _git_commit(),
        "seed": config["seed"],
    }


def _run_evaluation(
    config: dict[str, Any], eval_config: dict[str, Any], eval_config_path: Path
) -> dict[str, Any]:
    if eval_config_path.exists():
        locked = _read_json(eval_config_path)
        if stable_hash({k: v for k, v in locked.items() if k != "artifact_commit"}) != stable_hash(
            {k: v for k, v in eval_config.items() if k != "artifact_commit"}
        ):
            raise ValueError(f"Existing locked evaluation config differs: {eval_config_path}")
        eval_config = locked
    else:
        _write_locked_json(eval_config_path, eval_config)
    report_path = _project_path(eval_config["output_dir"]) / "report.json"
    if report_path.exists():
        report = _read_json(report_path)
        if report.get("metrics", {}).get("clean"):
            return report
        if _report_has_only_retryable_execution_errors(report):
            repair_manifest = report_path.parent / "error_only_repairs" / "repair_manifest.json"
            if repair_manifest.exists():
                print(
                    "Reuse exhausted error-only repair budget and continue with "
                    f"degraded evidence: {repair_manifest}",
                    flush=True,
                )
                return report
            return _repair_exhausted_evaluation(
                config=config,
                eval_config=eval_config,
                eval_config_path=eval_config_path,
            )
    env = os.environ.copy()
    env.update(
        {str(key): str(value) for key, value in config.get("official_runtime_env", {}).items()}
    )
    pythonpath = [str(PROJECT_ROOT / "patches" / "mobilitybench"), str(PROJECT_ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if config.get("public_profile"):
        from cce_mobility.workflows.public.pipeline import _mb_env

        env.update(_mb_env(config))
    _run_command(
        [
            str(Path(config["mobilitybench_python"]).expanduser()),
            "-m",
            "cce_mobility.workflows.mobilitybench.run_mobilitybench_official_continuation",
            str(eval_config_path),
        ],
        cwd=Path(config["mobilitybench_root"]).expanduser(),
        env=env,
    )
    report = _read_json(report_path)
    if _report_has_only_retryable_execution_errors(report):
        return _repair_exhausted_evaluation(
            config=config,
            eval_config=eval_config,
            eval_config_path=eval_config_path,
        )
    return report


def _report_has_only_retryable_execution_errors(report: dict[str, Any]) -> bool:
    metrics = report.get("metrics") or {}
    return (
        int(metrics.get("completed_count") or 0) == int(metrics.get("expected_count") or -1)
        and int(metrics.get("duplicate_pair_key_count") or 0) == 0
        and int(metrics.get("judge_error_count") or 0) + int(metrics.get("agent_error_count") or 0)
        > 0
    )


def _repair_exhausted_evaluation(
    *,
    config: dict[str, Any],
    eval_config: dict[str, Any],
    eval_config_path: Path,
) -> dict[str, Any]:
    output_dir = _project_path(eval_config["output_dir"])
    records_path = output_dir / "records.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    repair_root = output_dir / "error_only_repairs"
    repair_root.mkdir(parents=True, exist_ok=True)
    source_cases_path = _project_path(eval_config["continuation_cases_path"])
    source_cases = _load_jsonl_payloads(source_cases_path)
    case_order = [str(row.get("pair_key") or "") for row in source_cases]
    if not all(case_order) or len(case_order) != len(set(case_order)):
        raise ValueError(f"Invalid pair keys in continuation cases: {source_cases_path}")

    initial_records = repair_root / "records.before_error_repairs.jsonl"
    initial_attempts = repair_root / "attempts.before_error_repairs.jsonl"
    if not initial_records.exists():
        shutil.copy2(records_path, initial_records)
    if attempts_path.exists() and not initial_attempts.exists():
        shutil.copy2(attempts_path, initial_attempts)

    env = os.environ.copy()
    env.update(
        {str(key): str(value) for key, value in config.get("official_runtime_env", {}).items()}
    )
    pythonpath = [
        str(PROJECT_ROOT / "patches" / "mobilitybench"),
        str(PROJECT_ROOT / "src"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if config.get("public_profile"):
        from cce_mobility.workflows.public.pipeline import _mb_env

        env.update(_mb_env(config))

    repair_summaries: list[dict[str, Any]] = []
    for repair_round in range(1, 4):
        base_records = _records_by_pair_key(records_path)
        dirty_keys = [
            pair_key
            for pair_key in case_order
            if _continuation_retry_reason(base_records[pair_key])
        ]
        if not dirty_keys:
            break
        round_dir = repair_root / f"round_{repair_round}"
        subset_path = round_dir / "continuation_cases.jsonl"
        repair_output = round_dir / "eval"
        repair_config_path = round_dir / "eval_config.json"
        if repair_config_path.exists():
            repair_config = _read_json(repair_config_path)
            if not subset_path.exists():
                raise FileNotFoundError(
                    f"Locked repair config lacks its case subset: {subset_path}"
                )
        else:
            _write_jsonl_payloads(
                subset_path,
                [row for row in source_cases if str(row.get("pair_key")) in set(dirty_keys)],
            )
            repair_config = {
                **eval_config,
                "experiment_name": (
                    f"{eval_config['experiment_name']}_error_only_repair_{repair_round}"
                ),
                "output_dir": str(repair_output),
                "continuation_cases_path": str(subset_path),
                "expected_count": len(dirty_keys),
            }
            _write_locked_json(repair_config_path, repair_config)
        repair_report_path = repair_output / "report.json"
        if not repair_report_path.exists():
            _run_command(
                [
                    str(Path(config["mobilitybench_python"]).expanduser()),
                    "-m",
                    "cce_mobility.workflows.mobilitybench.run_mobilitybench_official_continuation",
                    str(repair_config_path),
                ],
                cwd=Path(config["mobilitybench_root"]).expanduser(),
                env=env,
            )
        repair_report = _read_json(repair_report_path)
        merge_summary = _merge_error_only_repair(
            records_path=records_path,
            attempts_path=attempts_path,
            repair_records_path=_project_path(repair_report["records_path"]),
            repair_attempts_path=repair_output / "attempts.jsonl",
            source_cases_path=source_cases_path,
            eval_config=eval_config,
            repair_config=repair_config,
            repair_round=repair_round,
        )
        repair_summaries.append(merge_summary)

    _run_command(
        [
            str(Path(config["mobilitybench_python"]).expanduser()),
            "-m",
            "cce_mobility.workflows.mobilitybench.run_mobilitybench_official_continuation",
            str(eval_config_path),
        ],
        cwd=Path(config["mobilitybench_root"]).expanduser(),
        env=env,
    )
    report_path = output_dir / "report.json"
    report = _read_json(report_path)
    report.setdefault("metrics", {}).update(
        {
            "error_only_repair_policy": "execution_errors_only_no_score_based_retry",
            "error_only_repair_round_count": len(repair_summaries),
            "error_only_repair_summaries": repair_summaries,
        }
    )
    _write_json(report_path, report)
    _write_json(
        repair_root / "repair_manifest.json",
        {
            "schema_version": "mobilitybench-formal-error-only-repair-v1",
            "policy": "execution_errors_only_no_score_based_retry",
            "max_additional_rounds": 3,
            "attempts_per_round": int(eval_config.get("max_case_attempts", 2)),
            "base_eval_config": _artifact(eval_config_path),
            "source_records": _artifact(initial_records),
            "source_attempts": (_artifact(initial_attempts) if initial_attempts.exists() else None),
            "repair_summaries": repair_summaries,
            "final_records": _artifact(records_path),
            "final_attempts": _artifact(attempts_path),
            "final_clean": bool(report.get("metrics", {}).get("clean")),
        },
    )
    return report


def _merge_error_only_repair(
    *,
    records_path: Path,
    attempts_path: Path,
    repair_records_path: Path,
    repair_attempts_path: Path,
    source_cases_path: Path,
    eval_config: dict[str, Any],
    repair_config: dict[str, Any],
    repair_round: int,
) -> dict[str, Any]:
    base_records = _records_by_pair_key(records_path)
    repair_records = _records_by_pair_key(repair_records_path)
    base_attempts = _load_jsonl_payloads(attempts_path) if attempts_path.exists() else []
    offsets: dict[str, int] = {}
    for row in base_attempts:
        pair_key = str(row.get("pair_key") or "")
        offsets[pair_key] = max(offsets.get(pair_key, 0), int(row.get("attempt") or 0))
    source_case_hash = file_sha256(source_cases_path)
    base_config_hash = stable_hash(eval_config)
    repair_config_hash = stable_hash(repair_config)
    replaced: list[str] = []
    for pair_key, repair_record in repair_records.items():
        if pair_key not in base_records:
            raise ValueError(f"Repair produced unknown pair key: {pair_key}")
        if _continuation_retry_reason(repair_record):
            continue
        original_reason = _continuation_retry_reason(base_records[pair_key])
        if not original_reason:
            continue
        metadata = repair_record.setdefault("metadata", {})
        metadata["continuation_cases_sha256"] = source_case_hash
        metadata["eval_config_sha256"] = base_config_hash
        metadata["attempt"] = offsets.get(pair_key, 0) + int(metadata.get("attempt") or 0)
        metadata["error_only_repair"] = {
            "policy": "execution_errors_only_no_score_based_retry",
            "repair_round": repair_round,
            "original_retry_reason": original_reason,
            "repair_eval_config_sha256": repair_config_hash,
            "repair_records_path": str(repair_records_path),
        }
        base_records[pair_key] = repair_record
        replaced.append(pair_key)

    case_order = [str(row.get("pair_key") or "") for row in _load_jsonl_payloads(source_cases_path)]
    _write_jsonl_payloads(records_path, [base_records[key] for key in case_order])

    repair_attempts = (
        _load_jsonl_payloads(repair_attempts_path) if repair_attempts_path.exists() else []
    )
    already_merged = {
        (str(row.get("pair_key") or ""), int(row.get("repair_round") or 0)) for row in base_attempts
    }
    appended = 0
    for row in repair_attempts:
        pair_key = str(row.get("pair_key") or "")
        if (pair_key, repair_round) in already_merged:
            continue
        row = json.loads(json.dumps(row))
        row["attempt"] = offsets.get(pair_key, 0) + int(row.get("attempt") or 0)
        row["repair_round"] = repair_round
        row["repair_source"] = str(repair_attempts_path)
        raw_record = row.get("record")
        if isinstance(raw_record, dict):
            metadata = raw_record.setdefault("metadata", {})
            metadata["attempt"] = row["attempt"]
            metadata["continuation_cases_sha256"] = source_case_hash
            metadata["eval_config_sha256"] = base_config_hash
            metadata["repair_eval_config_sha256"] = repair_config_hash
        base_attempts.append(row)
        appended += 1
    _write_jsonl_payloads(attempts_path, base_attempts)

    remaining = sorted(
        pair_key for pair_key, record in base_records.items() if _continuation_retry_reason(record)
    )
    return {
        "repair_round": repair_round,
        "requested_pair_count": len(repair_records),
        "replaced_clean_pair_count": len(replaced),
        "replaced_pair_keys": sorted(replaced),
        "appended_attempt_count": appended,
        "remaining_error_pair_count": len(remaining),
        "remaining_error_pair_keys": remaining,
        "repair_report_path": str(repair_records_path.parent / "report.json"),
    }


def _records_by_pair_key(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl_payloads(path):
        pair_key = str((row.get("metadata") or {}).get("pair_key") or "")
        if not pair_key or pair_key in records:
            raise ValueError(f"Invalid or duplicate pair key in {path}: {pair_key!r}")
        records[pair_key] = row
    return records


def _continuation_retry_reason(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    if (metadata.get("agent_skill_retrieval") or {}).get("error"):
        return "agent_retrieval_api_or_parse_error"
    hard = metadata.get("counterfactual_hard_metrics") or {}
    if hard.get("llm_error"):
        return "agent_api_or_execution_error"
    if not hard.get("parse_success"):
        return "agent_empty_or_parse_error"
    if not hard.get("official_planner_present"):
        return "official_planner_missing"
    if not hard.get("official_worker_tool_call_present"):
        return "official_worker_tool_call_missing"
    if not hard.get("official_reporter_present"):
        return "official_reporter_missing"
    if metadata.get("judge_error"):
        return "judge_api_or_parse_error"
    return ""


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


def _previous_accept_count(root: Path, arm: str, cycle: int) -> int:
    if cycle <= 1:
        return 0
    manifest = _read_json(root / arm / f"round_{cycle - 1}" / "round_manifest.json")
    return int(manifest.get("agent_update_accept_count", 0))


def _validate_completed_manifest(manifest: dict[str, Any], cycle_dir: Path) -> None:
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete round manifest: {cycle_dir}")
    for key in ("agent_memory", "world_memory", "world_ontology"):
        item = manifest.get("outputs", {}).get(key) or {}
        path = Path(item.get("path") or "")
        if not path.exists() or file_sha256(path) != item.get("sha256"):
            raise ValueError(f"Round artifact missing or mutated: {key} {path}")


def _select_training_evidence(
    records: list[EvaluationRecord],
    *,
    expected_count: int,
    minimum_valid_fraction: float,
    label: str,
) -> tuple[list[EvaluationRecord], dict[str, Any]]:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must be in (0, 1]")
    if len(records) > expected_count:
        raise RuntimeError(
            f"Training evidence exceeds the fixed denominator for {label}: "
            f"observed={len(records)} expected={expected_count}"
        )

    indexed: dict[tuple[str, str], EvaluationRecord] = {}
    valid: list[EvaluationRecord] = []
    excluded: list[dict[str, str]] = []
    for record in records:
        key = _training_record_key(record)
        if key in indexed:
            raise RuntimeError(f"Duplicate training evidence key for {label}: {key}")
        indexed[key] = record
        reason = _continuation_retry_reason(record.model_dump(mode="json"))
        if reason:
            excluded.append(
                {
                    "pair_key": str(record.metadata.get("pair_key") or ""),
                    "task_id": key[0],
                    "intervention_id": key[1],
                    "reason": reason,
                }
            )
        else:
            valid.append(record)

    minimum_valid_count = max(1, int(expected_count * minimum_valid_fraction + 0.999999))
    missing_count = expected_count - len(records)
    audit = {
        "label": label,
        "fixed_denominator_count": expected_count,
        "observed_record_count": len(records),
        "valid_record_count": len(valid),
        "valid_fraction": round(len(valid) / expected_count, 6),
        "minimum_valid_count": minimum_valid_count,
        "minimum_valid_fraction": minimum_valid_fraction,
        "missing_record_count": missing_count,
        "excluded_error_record_count": len(excluded),
        "excluded_error_records": excluded,
        "score_based_exclusion": False,
        "status": "clean" if len(valid) == expected_count else "degraded_but_usable",
    }
    if len(valid) < minimum_valid_count:
        raise RuntimeError(
            f"Insufficient usable training evidence for {label}: "
            f"valid={len(valid)}/{expected_count}, required={minimum_valid_count}; "
            f"execution_errors={len(excluded)} missing={missing_count}"
        )
    if audit["status"] != "clean":
        print(
            f"Proceed with degraded {label} training evidence: "
            f"valid={len(valid)}/{expected_count}, execution_errors={len(excluded)}, "
            f"missing={missing_count}, minimum={minimum_valid_count}",
            flush=True,
        )
    return valid, audit


def _paired_training_evidence(
    baseline_records: list[EvaluationRecord],
    target_records: list[EvaluationRecord],
    *,
    expected_count: int,
    minimum_valid_fraction: float,
    label: str,
) -> tuple[list[EvaluationRecord], list[EvaluationRecord], dict[str, Any]]:
    baseline_valid, baseline_audit = _select_training_evidence(
        baseline_records,
        expected_count=expected_count,
        minimum_valid_fraction=minimum_valid_fraction,
        label=f"{label} baseline",
    )
    target_valid, target_audit = _select_training_evidence(
        target_records,
        expected_count=expected_count,
        minimum_valid_fraction=minimum_valid_fraction,
        label=f"{label} target",
    )
    baseline_index = {_training_record_key(record): record for record in baseline_valid}
    target_index = {_training_record_key(record): record for record in target_valid}
    common_keys = [
        _training_record_key(record)
        for record in baseline_valid
        if _training_record_key(record) in target_index
    ]
    minimum_valid_count = max(1, int(expected_count * minimum_valid_fraction + 0.999999))
    if len(common_keys) < minimum_valid_count:
        raise RuntimeError(
            f"Insufficient paired training evidence for {label}: "
            f"paired={len(common_keys)}/{expected_count}, required={minimum_valid_count}"
        )
    baseline_only = sorted(set(baseline_index) - set(target_index))
    target_only = sorted(set(target_index) - set(baseline_index))
    audit = {
        "label": label,
        "fixed_denominator_count": expected_count,
        "paired_valid_count": len(common_keys),
        "paired_valid_fraction": round(len(common_keys) / expected_count, 6),
        "minimum_valid_count": minimum_valid_count,
        "minimum_valid_fraction": minimum_valid_fraction,
        "baseline": baseline_audit,
        "target": target_audit,
        "baseline_only_valid_keys": [list(key) for key in baseline_only],
        "target_only_valid_keys": [list(key) for key in target_only],
        "score_based_exclusion": False,
        "status": "clean" if len(common_keys) == expected_count else "degraded_but_usable",
    }
    if audit["status"] != "clean":
        print(
            f"Proceed with degraded paired evidence for {label}: "
            f"paired={len(common_keys)}/{expected_count}, minimum={minimum_valid_count}",
            flush=True,
        )
    return (
        [baseline_index[key] for key in common_keys],
        [target_index[key] for key in common_keys],
        audit,
    )


def _training_record_key(record: EvaluationRecord) -> tuple[str, str]:
    intervention_id = record.intervention.intervention_id if record.intervention else ""
    if not intervention_id:
        raise RuntimeError(f"Training evidence lacks an intervention id: {record.sample.sample_id}")
    return record.sample.sample_id, intervention_id


def _report_reference(report: dict[str, Any]) -> dict[str, Any]:
    records_path = _project_path(report["records_path"])
    return {
        "experiment_name": report["experiment_name"],
        "report_path": str(records_path.parent / "report.json"),
        "records_path": str(records_path),
        "records_sha256": file_sha256(records_path),
        "metrics": report.get("metrics", {}),
    }


def _load_records(path: Path) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(EvaluationRecord.model_validate_json(line))
    return records


def _write_records(path: Path, records: list[EvaluationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != TRAIN_CONFIG_SCHEMA:
        raise ValueError(f"Expected schema_version={TRAIN_CONFIG_SCHEMA}")
    required = {
        "experiment_name",
        "output_root",
        "cycles",
        "seed",
        "train_factual_records_path",
        "train_task_count",
        "mobilitybench_root",
        "mobilitybench_python",
        "official_config_path",
        "official_prompt_bundle_path",
        "official_model_alias",
        "agent",
        "world_model",
        "judge",
        "retrieval",
        "intervention",
        "method_version",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Formal training config missing keys: {missing}")
    if not config.get("public_profile") and int(config["cycles"]) != 3:
        raise ValueError("The preregistered minimum curve uses exactly three cycles")
    if not config.get("public_profile") and int(config["seed"]) != 42:
        raise ValueError("This single-seed protocol is preregistered with seed=42")
    if config["method_version"] != FORMAL_METHOD_VERSION:
        raise ValueError(f"Expected method_version={FORMAL_METHOD_VERSION}")
    if int(config["intervention"].get("selected_budget", 0)) != 1:
        raise ValueError("Formal train20 protocol requires selected_budget=1")
    unsupported_dimensions = sorted(
        set(config["intervention"].get("dimensions") or []) - SUPPORTED_TOOL_INTERVENTION_DIMENSIONS
    )
    if unsupported_dimensions:
        raise ValueError(
            f"Formal continuation cannot execute intervention dimensions: {unsupported_dimensions}"
        )
    if config.get("task_schedule_mode", "disjoint_without_replacement") not in {
        "disjoint_without_replacement",
        "repeat_fixed_pool",
    }:
        raise ValueError("Unsupported task_schedule_mode")
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
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def _require_committed_clean_code(provenance: dict[str, Any]) -> None:
    if provenance.get("git_commit") in {"", "unknown"} or provenance.get("relevant_dirty"):
        raise RuntimeError(
            f"Formal MobilityBench training requires committed clean experiment code: {provenance}"
        )


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _assert_jsonl_count(path: Path, expected: int) -> None:
    actual = _jsonl_count(path)
    if actual != expected:
        raise ValueError(f"Expected {expected} rows in {path}, got {actual}")


def _jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _prepare_task_schedule(
    *,
    root: Path,
    factual_records_path: Path,
    cycles: int,
    tasks_per_cycle: int,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    rows = _load_jsonl_payloads(factual_records_path)
    sample_ids = [_sample_id(row) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample ids in train factual pool: {factual_records_path}")
    if mode == "disjoint_without_replacement":
        required = cycles * tasks_per_cycle
        if len(rows) < required:
            raise ValueError(
                f"Formal disjoint schedule needs at least {required} factual records, got {len(rows)}"
            )
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        assignments = {
            str(cycle): [
                sample_ids[index]
                for index in indices[(cycle - 1) * tasks_per_cycle : cycle * tasks_per_cycle]
            ]
            for cycle in range(1, cycles + 1)
        }
    elif mode == "repeat_fixed_pool":
        if len(rows) != tasks_per_cycle:
            raise ValueError(
                f"repeat_fixed_pool expects exactly {tasks_per_cycle} records, got {len(rows)}"
            )
        assignments = {str(cycle): list(sample_ids) for cycle in range(1, cycles + 1)}
    else:
        raise ValueError(f"Unsupported task_schedule_mode: {mode}")
    payload = {
        "schema_version": "mobilitybench-formal-training-task-schedule-v1",
        "mode": mode,
        "seed": seed,
        "cycles": cycles,
        "tasks_per_cycle": tasks_per_cycle,
        "source_path": str(factual_records_path.resolve()),
        "source_sha256": file_sha256(factual_records_path),
        "source_count": len(rows),
        "assignments": assignments,
        "assignment_sha256": stable_hash(assignments),
    }
    schedule_path = root / "training_task_schedule.json"
    if schedule_path.exists():
        previous = _read_json(schedule_path)
        if stable_hash(previous) != stable_hash(payload):
            raise ValueError("Existing training_task_schedule.json differs; use a new output_root")
    else:
        _write_json(schedule_path, payload)
    return payload


def _materialize_round_factual_records(
    *,
    root: Path,
    source_path: Path,
    cycle: int,
) -> Path:
    schedule = _read_json(root / "training_task_schedule.json")
    selected_ids = list(schedule["assignments"][str(cycle)])
    rows = {_sample_id(row): row for row in _load_jsonl_payloads(source_path)}
    missing = [sample_id for sample_id in selected_ids if sample_id not in rows]
    if missing:
        raise ValueError(f"Scheduled factual records are missing: {missing[:10]}")
    output = root / "shared" / f"round_{cycle}" / "factual_records.jsonl"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for sample_id in selected_ids:
                handle.write(json.dumps(rows[sample_id], ensure_ascii=False) + "\n")
    _assert_jsonl_count(output, len(selected_ids))
    materialized_ids = [_sample_id(row) for row in _load_jsonl_payloads(output)]
    if materialized_ids != selected_ids:
        raise ValueError(f"Materialized round factual schedule mismatch: {output}")
    return output


def _load_jsonl_payloads(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl_payloads(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _sample_id(row: dict[str, Any]) -> str:
    sample = row.get("sample") or {}
    sample_id = str(sample.get("sample_id") or "")
    if not sample_id:
        raise ValueError("Factual record missing sample.sample_id")
    return sample_id


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_locked_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path)
        if stable_hash(existing) != stable_hash(payload):
            raise ValueError(f"Existing locked JSON differs; use a new output root: {path}")
        return
    _write_json(path, payload)


def _validate_challenge_reuse(
    *,
    summary_path: Path,
    generation_config: dict[str, Any],
    output_path: Path,
    factual_records_path: Path,
    world_memory_path: Path,
    ontology_path: Path,
) -> None:
    summary = _read_json(summary_path)
    checks = {
        "config": summary.get("config_sha256")
        == stable_hash(ExperimentConfig.model_validate(generation_config).model_dump(mode="json")),
        "output": summary.get("output_sha256") == file_sha256(output_path),
        "factual_source": summary.get("official_factual_records_sha256")
        == file_sha256(factual_records_path),
        "world_memory": (summary.get("world_memory") or {}).get("sha256")
        == file_sha256(world_memory_path),
        "world_ontology": (summary.get("world_ontology") or {}).get("sha256")
        == file_sha256(ontology_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"Existing frozen challenges fail source integrity checks {failed}: {output_path}"
        )


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
