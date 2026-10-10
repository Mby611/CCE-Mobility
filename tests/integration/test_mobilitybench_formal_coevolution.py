from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cce_mobility.core.config import ExperimentConfig
from cce_mobility.core.schema import (
    BenchmarkName,
    EvaluationRecord,
    InterventionSpec,
    JudgeScore,
    TaskSample,
)
from cce_mobility.methods.formal_coevolution import (
    build_agent_candidate_memory,
    classify_paired_outcomes,
    curation_decision,
    update_world_from_persistent_failures,
)
from cce_mobility.interventions.ontology import registry_from_seed_dimensions
from cce_mobility.skills import SkillRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = (
    PROJECT_ROOT
    / "src/cce_mobility/workflows/mobilitybench"
    / "run_mobilitybench_formal_coevolution_train.py"
)
SPEC = importlib.util.spec_from_file_location("formal_mb_train", TRAIN_PATH)
assert SPEC and SPEC.loader
formal_mb_train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formal_mb_train)


def test_paired_outcomes_route_only_persistent_failures_to_world() -> None:
    pre = [
        _record("ff", 1.0, final_pass=False),
        _record("fs", 1.0, final_pass=False),
        _record("sf", 4.0, final_pass=True),
        _record("ss", 4.0, final_pass=True),
    ]
    post = [
        _record("ff", 1.5, final_pass=False),
        _record("fs", 3.5, final_pass=True),
        _record("sf", 2.5, final_pass=True),
        _record("ss", 3.8, final_pass=True),
    ]

    outcome = classify_paired_outcomes(pre, post)

    assert outcome["counts"] == {
        "failure_to_failure": 1,
        "failure_to_success": 1,
        "success_to_failure": 1,
        "success_to_success": 1,
    }
    assert [row.sample.sample_id for row in outcome["persistent_failure_records"]] == ["ff"]
    assert [row.sample.sample_id for row in outcome["agent_regression_records"]] == ["sf"]


def test_agent_update_gate_rejects_excess_success_to_failure_regression() -> None:
    pre = [_record("a", 4.0), _record("b", 4.0)]
    post = [_record("a", 1.0, final_pass=False), _record("b", 4.0)]

    decision = curation_decision(
        pre,
        post,
        thresholds={"max_success_to_failure_rate": 0.1},
    )

    assert decision["accepted"] is False
    assert decision["checks"]["regression_rate"] is False


def test_three_cycle_task_schedule_is_disjoint_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "factual.jsonl"
    source.write_text(
        "".join(json.dumps({"sample": {"sample_id": f"mb_{index}"}}) + "\n" for index in range(60)),
        encoding="utf-8",
    )
    root = tmp_path / "run"

    first = formal_mb_train._prepare_task_schedule(
        root=root,
        factual_records_path=source,
        cycles=3,
        tasks_per_cycle=20,
        seed=42,
        mode="disjoint_without_replacement",
    )
    second = formal_mb_train._prepare_task_schedule(
        root=root,
        factual_records_path=source,
        cycles=3,
        tasks_per_cycle=20,
        seed=42,
        mode="disjoint_without_replacement",
    )

    sets = [set(first["assignments"][str(cycle)]) for cycle in (1, 2, 3)]
    assert first == second
    assert all(len(values) == 20 for values in sets)
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]


def test_formal_agent_memory_uses_benchmark_agnostic_recovery_text(tmp_path: Path) -> None:
    initial = tmp_path / "initial.jsonl"
    output = tmp_path / "candidate.jsonl"
    SkillRegistry().save_jsonl(initial)

    build_agent_candidate_memory(initial, [_record("failed", 1.0)], output)

    rendered = output.read_text(encoding="utf-8").lower()
    assert "benchmark_agnostic_recovery_v1" in rendered
    for legacy_term in ("itinerary", "flight", "hotel", "restaurant", "meal slot"):
        assert legacy_term not in rendered


def test_formal_world_memory_uses_persistent_failure_template(tmp_path: Path) -> None:
    initial_memory = tmp_path / "initial_world.jsonl"
    initial_ontology = tmp_path / "initial_ontology.jsonl"
    output_memory = tmp_path / "world.jsonl"
    output_ontology = tmp_path / "ontology.jsonl"
    SkillRegistry().save_jsonl(initial_memory)
    registry_from_seed_dimensions(["tool_empty_result"]).save_jsonl(initial_ontology)

    update_world_from_persistent_failures(
        persistent_records=[_record("failed", 1.0)],
        current_memory_path=initial_memory,
        current_ontology_path=initial_ontology,
        output_memory_path=output_memory,
        output_ontology_path=output_ontology,
        world_client=None,
    )

    rendered = output_memory.read_text(encoding="utf-8").lower()
    assert "benchmark_agnostic_world_spec_v1" in rendered
    assert "persistent failure evidence only" in rendered
    for legacy_term in ("flight", "hotel", "restaurant"):
        assert legacy_term not in rendered


def test_frozen_challenge_reuse_requires_matching_sources(tmp_path: Path) -> None:
    factual = tmp_path / "factual.jsonl"
    world = tmp_path / "world.jsonl"
    ontology = tmp_path / "ontology.jsonl"
    output = tmp_path / "interventions.jsonl"
    for path, text in (
        (factual, "factual\n"),
        (world, "world\n"),
        (ontology, "ontology\n"),
        (output, "intervention\n"),
    ):
        path.write_text(text, encoding="utf-8")
    generation_config = {
        "experiment_name": "formal_round",
        "benchmark": {"name": "mobilitybench", "data_path": str(factual), "limit": 1},
        "method": "ontology_phase_switching",
        "agent": {},
        "world_model": {},
        "judge": {},
        "intervention": {"selected_budget": 1},
        "extra": {
            "load_skill_memory_path": str(world),
            "load_intervention_ontology_path": str(ontology),
        },
    }
    normalized = ExperimentConfig.model_validate(generation_config).model_dump(mode="json")
    summary = {
        "config_sha256": formal_mb_train.stable_hash(normalized),
        "output_sha256": formal_mb_train.file_sha256(output),
        "official_factual_records_sha256": formal_mb_train.file_sha256(factual),
        "world_memory": {"sha256": formal_mb_train.file_sha256(world)},
        "world_ontology": {"sha256": formal_mb_train.file_sha256(ontology)},
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    formal_mb_train._validate_challenge_reuse(
        summary_path=summary_path,
        generation_config=generation_config,
        output_path=output,
        factual_records_path=factual,
        world_memory_path=world,
        ontology_path=ontology,
    )

    factual.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="factual_source"):
        formal_mb_train._validate_challenge_reuse(
            summary_path=summary_path,
            generation_config=generation_config,
            output_path=output,
            factual_records_path=factual,
            world_memory_path=world,
            ontology_path=ontology,
        )


def test_locked_json_refuses_mutation(tmp_path: Path) -> None:
    path = tmp_path / "locked.json"
    formal_mb_train._write_locked_json(path, {"version": 1})
    formal_mb_train._write_locked_json(path, {"version": 1})

    with pytest.raises(ValueError, match="Existing locked JSON differs"):
        formal_mb_train._write_locked_json(path, {"version": 2})


def test_error_only_repair_preserves_attempts_and_is_idempotent(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"
    repair_records_path = tmp_path / "repair_records.jsonl"
    repair_attempts_path = tmp_path / "repair_attempts.jsonl"
    cases_path = tmp_path / "cases.jsonl"
    dirty = _continuation_payload("dirty", clean=False, attempt=4)
    clean = _continuation_payload("clean", clean=True, attempt=1)
    repaired = _continuation_payload("dirty", clean=True, attempt=2)
    formal_mb_train._write_jsonl_payloads(records_path, [dirty, clean])
    formal_mb_train._write_jsonl_payloads(
        attempts_path,
        [{"pair_key": "dirty", "attempt": attempt, "record": dirty} for attempt in range(1, 5)],
    )
    formal_mb_train._write_jsonl_payloads(repair_records_path, [repaired])
    formal_mb_train._write_jsonl_payloads(
        repair_attempts_path,
        [
            {"pair_key": "dirty", "attempt": 1, "record": dirty},
            {"pair_key": "dirty", "attempt": 2, "record": repaired},
        ],
    )
    formal_mb_train._write_jsonl_payloads(
        cases_path,
        [{"pair_key": "dirty"}, {"pair_key": "clean"}],
    )
    eval_config = {"experiment_name": "base", "seed": 42}
    repair_config = {**eval_config, "experiment_name": "repair"}

    first = formal_mb_train._merge_error_only_repair(
        records_path=records_path,
        attempts_path=attempts_path,
        repair_records_path=repair_records_path,
        repair_attempts_path=repair_attempts_path,
        source_cases_path=cases_path,
        eval_config=eval_config,
        repair_config=repair_config,
        repair_round=1,
    )

    merged = formal_mb_train._records_by_pair_key(records_path)
    attempts = formal_mb_train._load_jsonl_payloads(attempts_path)
    assert first["replaced_pair_keys"] == ["dirty"]
    assert first["appended_attempt_count"] == 2
    assert not formal_mb_train._continuation_retry_reason(merged["dirty"])
    assert merged["dirty"]["metadata"]["attempt"] == 6
    assert merged["dirty"]["metadata"]["error_only_repair"]["original_retry_reason"]
    assert [row["attempt"] for row in attempts] == [1, 2, 3, 4, 5, 6]
    assert attempts[-1]["repair_round"] == 1

    second = formal_mb_train._merge_error_only_repair(
        records_path=records_path,
        attempts_path=attempts_path,
        repair_records_path=repair_records_path,
        repair_attempts_path=repair_attempts_path,
        source_cases_path=cases_path,
        eval_config=eval_config,
        repair_config=repair_config,
        repair_round=1,
    )
    assert second["appended_attempt_count"] == 0
    assert len(formal_mb_train._load_jsonl_payloads(attempts_path)) == 6


def test_error_only_repair_uses_execution_errors_not_budget_labels() -> None:
    metrics = {
        "completed_count": 20,
        "expected_count": 20,
        "duplicate_pair_key_count": 0,
        "retry_budget_exhausted_case_count": 2,
        "retryable_error_count": 0,
        "agent_error_count": 2,
        "judge_error_count": 0,
    }
    assert formal_mb_train._report_has_only_retryable_execution_errors({"metrics": metrics})
    assert formal_mb_train._report_has_only_retryable_execution_errors(
        {
            "metrics": {
                **metrics,
                "retry_budget_exhausted_case_count": 0,
                "retryable_error_count": 3,
            }
        }
    )
    assert not formal_mb_train._report_has_only_retryable_execution_errors(
        {"metrics": {**metrics, "agent_error_count": 0}}
    )


def test_run_evaluation_reuses_exhausted_repair_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "eval"
    repair_root = output / "error_only_repairs"
    repair_root.mkdir(parents=True)
    config_path = tmp_path / "eval_config.json"
    eval_config = {
        "output_dir": str(output),
        "artifact_commit": "old",
    }
    config_path.write_text(json.dumps(eval_config), encoding="utf-8")
    report = {
        "metrics": {
            "clean": False,
            "completed_count": 20,
            "expected_count": 20,
            "duplicate_pair_key_count": 0,
            "agent_error_count": 3,
            "judge_error_count": 0,
        }
    }
    (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (repair_root / "repair_manifest.json").write_text("{}", encoding="utf-8")

    def unexpected_repair(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"repair should not run again: {kwargs}")

    monkeypatch.setattr(
        formal_mb_train,
        "_repair_exhausted_evaluation",
        unexpected_repair,
    )

    assert formal_mb_train._run_evaluation({}, eval_config, config_path) == report


def test_training_evidence_excludes_execution_errors_and_keeps_fixed_denominator() -> None:
    records = [_record(f"case_{index}", 3.0) for index in range(20)]
    for record in records[:3]:
        record.metadata["counterfactual_hard_metrics"]["llm_error"] = True

    valid, audit = formal_mb_train._select_training_evidence(
        records,
        expected_count=20,
        minimum_valid_fraction=0.5,
        label="round 2 pre",
    )

    assert len(valid) == 17
    assert audit["fixed_denominator_count"] == 20
    assert audit["valid_record_count"] == 17
    assert audit["excluded_error_record_count"] == 3
    assert audit["status"] == "degraded_but_usable"
    assert {row["task_id"] for row in audit["excluded_error_records"]} == {
        "case_0",
        "case_1",
        "case_2",
    }


def test_training_evidence_stops_only_below_progress_threshold() -> None:
    records = [_record(f"case_{index}", 3.0) for index in range(20)]
    for record in records[:11]:
        record.metadata["counterfactual_hard_metrics"]["llm_error"] = True

    with pytest.raises(RuntimeError, match="Insufficient usable training evidence"):
        formal_mb_train._select_training_evidence(
            records,
            expected_count=20,
            minimum_valid_fraction=0.5,
            label="round 2 pre",
        )


def test_paired_training_evidence_uses_clean_intersection() -> None:
    baseline = [_record(f"case_{index}", 3.0) for index in range(20)]
    target = [_record(f"case_{index}", 3.1) for index in range(20)]
    baseline[0].metadata["counterfactual_hard_metrics"]["llm_error"] = True
    target[1].metadata["judge_error"] = True

    paired_baseline, paired_target, audit = formal_mb_train._paired_training_evidence(
        baseline,
        target,
        expected_count=20,
        minimum_valid_fraction=0.5,
        label="pre/candidate",
    )

    assert len(paired_baseline) == len(paired_target) == 18
    assert [record.sample.sample_id for record in paired_baseline] == [
        record.sample.sample_id for record in paired_target
    ]
    assert {record.sample.sample_id for record in paired_baseline}.isdisjoint({"case_0", "case_1"})
    assert audit["paired_valid_count"] == 18
    assert audit["status"] == "degraded_but_usable"


def _continuation_payload(pair_key: str, *, clean: bool, attempt: int) -> dict:
    return {
        "metadata": {
            "pair_key": pair_key,
            "attempt": attempt,
            "judge_error": False,
            "agent_skill_retrieval": {},
            "counterfactual_hard_metrics": {
                "llm_error": not clean,
                "parse_success": clean,
                "official_planner_present": clean,
                "official_worker_tool_call_present": clean,
                "official_reporter_present": clean,
            },
        }
    }


def _record(sample_id: str, score: float, *, final_pass: bool = True) -> EvaluationRecord:
    intervention_id = f"int_{sample_id}"
    return EvaluationRecord(
        sample=TaskSample(
            sample_id=sample_id,
            benchmark=BenchmarkName.MOBILITY_BENCH,
            query="Recover.",
        ),
        intervention=InterventionSpec(
            intervention_id=intervention_id,
            sample_id=sample_id,
            benchmark=BenchmarkName.MOBILITY_BENCH,
            dimension="tool_empty_result",
            intervention_type="tool_result",
        ),
        counterfactual_score=JudgeScore(
            sample_id=sample_id,
            intervention_id=intervention_id,
            total=score,
        ),
        metadata={
            "pair_key": f"{sample_id}::{intervention_id}",
            "judge_error": False,
            "counterfactual_hard_metrics": {
                "final_pass": final_pass,
                "uses_unavailable_identifier": False,
                "llm_error": False,
                "parse_success": True,
                "official_planner_present": True,
                "official_worker_tool_call_present": True,
                "official_reporter_present": True,
            },
        },
    )
