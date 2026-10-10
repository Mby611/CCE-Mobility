from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from cce_mobility.workflows.mobilitybench import (
    run_mobilitybench_formal_coevolution_curve as formal_mb_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_repeated_prepare_keeps_original_lock_after_verified_code_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = {
        "status": "locked_before_formal_evaluation",
        "artifact_commit": "original",
        "formal_code_provenance": {"git_commit": "original"},
        "training_code_transition": {"current_commit": "original"},
        "checkpoint_index": [
            {
                "name": "full_coevolution_round1",
                "agent_memory": {"sha256": "memory-hash"},
                "training_code_transition": {"current_commit": "original"},
            }
        ],
        "eval_config_sha256": {"full_coevolution_round1": "config-hash"},
    }
    candidate = deepcopy(frozen)
    candidate["artifact_commit"] = "later"
    candidate["formal_code_provenance"] = {"git_commit": "later"}
    candidate["training_code_transition"] = {"current_commit": "later"}
    candidate["checkpoint_index"][0]["training_code_transition"] = {"current_commit": "later"}

    def verified(current: dict, preflight: dict, *, allow_qoder_only_transition: bool) -> dict:
        assert current["git_commit"] == "later"
        assert preflight["formal_code_provenance"]["git_commit"] == "original"
        assert allow_qoder_only_transition
        return {"equivalence": "policy_only_ast_verified"}

    monkeypatch.setattr(formal_mb_curve, "_require_matching_formal_code", verified)
    restored = formal_mb_curve._reuse_protocol_lock(
        candidate, frozen, formal_code={"git_commit": "later"}, allow_qoder_only_transition=True
    )
    assert restored is frozen
    assert restored["artifact_commit"] == "original"

    changed = deepcopy(candidate)
    changed["checkpoint_index"][0]["agent_memory"]["sha256"] = "other-memory"
    with pytest.raises(ValueError, match="changed_fields=\\['checkpoint_index'\\]"):
        formal_mb_curve._reuse_protocol_lock(
            changed,
            frozen,
            formal_code={"git_commit": "later"},
            allow_qoder_only_transition=True,
        )


def test_export_loads_locked_inputs_without_replaying_training_provenance(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"
    eval_dir = collection / "eval_configs"
    eval_dir.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text("\n", encoding="utf-8")
    artifact = {
        "path": str(checkpoint),
        "sha256": formal_mb_curve.file_sha256(checkpoint),
    }
    config = {"collection_dir": str(collection), "expected_count": 500}
    points = []
    hashes = {}
    names = ["no_evolution_round0"] + [
        f"{series}_round{round_number}"
        for round_number in (1, 2, 3)
        for series in ("full_coevolution", "agent_only_frozen_world")
    ]
    for name in names:
        eval_config = {
            "experiment_name": name,
            "expected_count": 500,
            "output_dir": str(collection / "eval_runs" / name),
        }
        eval_path = eval_dir / f"{name}.json"
        eval_path.write_text(json.dumps(eval_config), encoding="utf-8")
        hashes[name] = formal_mb_curve.stable_hash(eval_config)
        points.append(
            {
                "name": name,
                "eval_config_path": str(eval_path),
                "eval_config": eval_config,
                "agent_memory": artifact,
                "world_memory": artifact,
                "world_ontology": artifact,
            }
        )
    lock = {
        "schema_version": formal_mb_curve.CURVE_CONFIG_SCHEMA,
        "status": "locked_before_formal_evaluation",
        "config_sha256": formal_mb_curve.stable_hash(config),
        "expected_points": 7,
        "expected_cases_per_point": 500,
        "point_names": names,
        "eval_config_sha256": hashes,
        "formal_code_provenance": {"unavailable": "must not be consulted"},
    }
    (collection / "protocol_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    (collection / "checkpoint_eval_index.json").write_text(
        json.dumps({"points": points}), encoding="utf-8"
    )

    loaded_points, loaded_lock = formal_mb_curve.load_locked_export_inputs(config)

    assert loaded_lock == lock
    assert [point["name"] for point in loaded_points] == names
    assert loaded_points[-1]["eval_config"]["expected_count"] == 500


def test_round_artifacts_accept_verified_byte_identical_challenge_reuse(
    tmp_path: Path,
) -> None:
    round_root = tmp_path / "agent_only" / "round_1"
    round_root.mkdir(parents=True)
    source = tmp_path / "shared" / "round_1" / "interventions.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"case_id":"1"}\n', encoding="utf-8")
    output = round_root / "interventions.jsonl"
    output.write_bytes(source.read_bytes())
    for name in ("round_manifest.json", "continuation_cases.jsonl"):
        (round_root / name).write_text("{}\n", encoding="utf-8")
    (round_root / "interventions.summary.json").write_text(
        json.dumps(
            {
                "reuse_mode": "byte_identical_shared_round1",
                "source_interventions": {
                    "path": str(source),
                    "sha256": formal_mb_curve.file_sha256(source),
                },
            }
        ),
        encoding="utf-8",
    )

    paths, omissions = formal_mb_curve._round_training_artifacts(round_root)

    assert source in paths
    assert round_root / "challenge_generation_config.json" not in paths
    assert omissions == [
        {
            "path": str((round_root / "challenge_generation_config.json").resolve()),
            "reason": "challenge_generation_reused_byte_identically",
            "reuse_mode": "byte_identical_shared_round1",
            "source_interventions": formal_mb_curve._artifact(source),
        }
    ]


def test_round_artifacts_reject_missing_generation_config_without_reuse(
    tmp_path: Path,
) -> None:
    round_root = tmp_path / "full" / "round_1"
    round_root.mkdir(parents=True)
    (round_root / "interventions.summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="without a verified reuse mode"):
        formal_mb_curve._round_training_artifacts(round_root)


def test_comparison_plan_has_preregistered_round3_primary() -> None:
    points = [{"name": "no_evolution_round0"}]
    for round_number in (1, 2, 3):
        points.extend(
            [
                {"name": f"full_coevolution_round{round_number}"},
                {"name": f"agent_only_frozen_world_round{round_number}"},
            ]
        )

    comparisons = formal_mb_curve._comparison_specs(points)

    assert len(comparisons) == 13
    assert [item for item in comparisons if item[3] == "primary"] == [
        (
            "full_minus_agent_only_round3",
            "full_coevolution_round3",
            "agent_only_frozen_world_round3",
            "primary",
        )
    ]


def test_cluster_bootstrap_resamples_tasks_not_individual_interventions() -> None:
    grouped = {"task_a": [0.0, 4.0], "task_b": [2.0]}

    first = formal_mb_curve._cluster_bootstrap_stat(
        grouped,
        statistic=lambda values: sum(values) / len(values),
        samples=10_000,
        seed=42,
    )
    second = formal_mb_curve._cluster_bootstrap_stat(
        grouped,
        statistic=lambda values: sum(values) / len(values),
        samples=10_000,
        seed=42,
    )

    assert first == second
    assert first[0] == 2.0


def test_input_audit_requires_matching_factual_and_intervention_hashes(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text("case artifact\n", encoding="utf-8")
    cases_audit = cases.with_suffix(".audit.json")
    input_audit = tmp_path / "input_audit.json"
    config = {
        "expected_count": 1,
        "evaluation_identity": "fresh_disjoint_evaluation_candidate",
    }
    cases_audit.write_text(
        json.dumps(
            {
                "protocol": formal_mb_curve.CONTINUATION_PROTOCOL,
                "case_count": 1,
                "cases_sha256": formal_mb_curve.file_sha256(cases),
                "factual_records_sha256": "factual-hash",
                "frozen_interventions_sha256": "intervention-hash",
                "agent_payload_privileged_field_count": 0,
                "agent_visible_context_unexpected_keys": [],
                "trigger_resolution_counts": {"target_and_tool": 1},
            }
        ),
        encoding="utf-8",
    )
    input_audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "evaluation_identity": "fresh_disjoint_evaluation_candidate",
                "factual_source_reusable_for_continuation_binding": True,
                "inputs": {
                    "eval_factual": {"sha256": "factual-hash"},
                    "frozen_interventions": {"sha256": "wrong-hash"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="intervention_hash"):
        formal_mb_curve._validate_input_audits(
            config=config,
            cases_path=cases,
            input_audit_path=input_audit,
            cases_audit_path=cases_audit,
        )


def test_input_audit_allows_only_preregistered_fallback_count(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text("case artifact\n", encoding="utf-8")
    cases_audit = cases.with_suffix(".audit.json")
    input_audit = tmp_path / "input_audit.json"
    config = {
        "expected_count": 1,
        "evaluation_identity": "fresh_disjoint_evaluation_candidate",
        "max_fallback_trigger_count": 1,
    }
    cases_audit.write_text(
        json.dumps(
            {
                "protocol": formal_mb_curve.CONTINUATION_PROTOCOL,
                "case_count": 1,
                "cases_sha256": formal_mb_curve.file_sha256(cases),
                "factual_records_sha256": "factual-hash",
                "frozen_interventions_sha256": "intervention-hash",
                "agent_payload_privileged_field_count": 0,
                "agent_visible_context_unexpected_keys": [],
                "trigger_resolution_counts": {"first_worker_event_fallback": 1},
            }
        ),
        encoding="utf-8",
    )
    input_audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "evaluation_identity": "fresh_disjoint_evaluation_candidate",
                "factual_source_reusable_for_continuation_binding": True,
                "inputs": {
                    "eval_factual": {"sha256": "factual-hash"},
                    "frozen_interventions": {"sha256": "intervention-hash"},
                },
            }
        ),
        encoding="utf-8",
    )

    formal_mb_curve._validate_input_audits(
        config=config,
        cases_path=cases,
        input_audit_path=input_audit,
        cases_audit_path=cases_audit,
    )
    config["max_fallback_trigger_count"] = 0
    with pytest.raises(ValueError, match="fallback_trigger_within_preregistered_limit"):
        formal_mb_curve._validate_input_audits(
            config=config,
            cases_path=cases,
            input_audit_path=input_audit,
            cases_audit_path=cases_audit,
        )


def test_attempt_summary_preserves_failed_retry_denominator(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "pair_key": "a",
                        "attempt": 1,
                        "retry_reason": "agent_api_or_execution_error",
                    }
                ),
                json.dumps({"pair_key": "a", "attempt": 2, "retry_reason": ""}),
                json.dumps({"pair_key": "b", "attempt": 1, "retry_reason": ""}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = formal_mb_curve._attempt_summary(attempts)

    assert summary["attempt_count"] == 3
    assert summary["retried_pair_count"] == 1
    assert summary["failed_attempt_count"] == 1
    assert summary["retry_reason_counts"] == {"agent_api_or_execution_error": 1}


def test_arm_invariants_require_frozen_world_and_matched_agent_budget(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial_state"
    initial.mkdir()
    (initial / "world_memory.jsonl").write_text("", encoding="utf-8")
    (initial / "world_ontology.jsonl").write_text("seed\n", encoding="utf-8")
    world_hash = formal_mb_curve.file_sha256(initial / "world_memory.jsonl")
    ontology_hash = formal_mb_curve.file_sha256(initial / "world_ontology.jsonl")

    def manifest(arm: str, round_number: int) -> dict:
        return {
            "agent_update_attempt_count": round_number,
            "cumulative_training_rollout_count": round_number * 60,
            "inputs": {
                "round_factual_records": {"sha256": f"factual-{round_number}"},
                "interventions": {
                    "sha256": "shared-round1" if round_number == 1 else f"{arm}-round{round_number}"
                },
            },
            "outputs": {
                "world_memory": {
                    "sha256": world_hash if arm == "agent_only" else f"full-{round_number}"
                },
                "world_ontology": {
                    "sha256": ontology_hash
                    if arm == "agent_only"
                    else f"full-ontology-{round_number}"
                },
            },
            "paired_outcomes": {"success_to_failure_routed_to_world": False},
        }

    manifests = {
        (arm, round_number): manifest(arm, round_number)
        for round_number in (1, 2, 3)
        for arm in ("full", "agent_only")
    }

    audit = formal_mb_curve._validate_curve_arm_invariants(
        train_root=tmp_path,
        manifests=manifests,
        cycles=3,
        train_task_count=20,
    )

    assert audit["status"] == "pass"
    manifests[("agent_only", 2)]["outputs"]["world_memory"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="agent_only_world_memory_frozen"):
        formal_mb_curve._validate_curve_arm_invariants(
            train_root=tmp_path,
            manifests=manifests,
            cycles=3,
            train_task_count=20,
        )


def test_arm_invariants_share_challenges_when_previous_world_is_identical(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial_state"
    initial.mkdir()
    (initial / "world_memory.jsonl").write_text("", encoding="utf-8")
    (initial / "world_ontology.jsonl").write_text("seed\n", encoding="utf-8")
    world_hash = formal_mb_curve.file_sha256(initial / "world_memory.jsonl")
    ontology_hash = formal_mb_curve.file_sha256(initial / "world_ontology.jsonl")

    def manifest(arm: str, round_number: int) -> dict:
        return {
            "agent_update_attempt_count": round_number,
            "cumulative_training_rollout_count": round_number * 60,
            "inputs": {
                "round_factual_records": {"sha256": f"factual-{round_number}"},
                "interventions": {
                    "sha256": "shared-round1" if round_number == 1 else f"{arm}-round{round_number}"
                },
            },
            "outputs": {
                "world_memory": {"sha256": world_hash},
                "world_ontology": {"sha256": ontology_hash},
            },
            "paired_outcomes": {"success_to_failure_routed_to_world": False},
        }

    manifests = {
        (arm, round_number): manifest(arm, round_number)
        for round_number in (1, 2, 3)
        for arm in ("full", "agent_only")
    }

    with pytest.raises(ValueError, match="same_world_implies_same_challenges"):
        formal_mb_curve._validate_curve_arm_invariants(
            train_root=tmp_path,
            manifests=manifests,
            cycles=3,
            train_task_count=20,
        )


def test_terminal_report_allows_fixed_denominator_exhausted_errors() -> None:
    report = {
        "metrics": {
            "completed_count": 500,
            "missing_count": 0,
            "duplicate_pair_key_count": 0,
            "retryable_error_count": 0,
            "retry_budget_exhausted_case_count": 2,
            "clean": False,
        }
    }

    assert formal_mb_curve._report_terminal(report, 500) is True


def test_posthoc_export_records_unreconstructable_runtime_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cce_mobility.integrations import mobilitybench_continuation as provenance

    def unavailable(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("historical hash unavailable")

    monkeypatch.setattr(provenance, "verify_scoped_provenance_transition", unavailable)
    output = tmp_path / "eval"
    point = {
        "name": "no_evolution_round0",
        "eval_config": {"output_dir": str(output)},
    }
    metadata = {
        "pair_key": "task::intervention",
        "cce_runtime_git_commit": "old-commit",
        "cce_runtime_code_sha256": "old-hash",
        "agent_skill_retrieval": {},
    }
    audit = formal_mb_curve._protocol_audit(
        {
            "evaluation_identity": "frozen_eval",
            "evaluation_identity_note": "locked",
            "expected_count": 1,
            "agent": {"model": "agent", "provider": "aihub"},
            "judge": {"model": "judge", "provider": "aihub"},
        },
        [point],
        {"no_evolution_round0": {"task::intervention": {"metadata": metadata}}},
        reports={"no_evolution_round0": {"metrics": {"clean": True}}},
    )

    transition = audit["runtime_source_transitions"]["old-commit:old-hash"]
    assert transition["equivalence"] == "recorded_runtime_not_reverified_posthoc"
    assert transition["reason"] == "historical hash unavailable"


def test_curve_locked_json_refuses_mutation(tmp_path: Path) -> None:
    path = tmp_path / "eval_config.json"
    formal_mb_curve._write_locked_json(path, {"version": 1})
    formal_mb_curve._write_locked_json(path, {"version": 1})

    with pytest.raises(ValueError, match="Existing locked JSON differs"):
        formal_mb_curve._write_locked_json(path, {"version": 2})
