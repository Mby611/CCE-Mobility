from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from cce_mobility.core.schema import (
    AgentTrajectory,
    BenchmarkName,
    EvaluationRecord,
    ExpectedRecoverySpec,
    InterventionCandidate,
    InterventionSpec,
    RunMode,
    TaskSample,
    ToolEvent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "src/cce_mobility/workflows/baselines"
    / "materialize_official_skillopt_mobilitybench_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_mb_skillopt", SCRIPT_PATH)
assert SPEC is not None
materialize_mb_skillopt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(materialize_mb_skillopt)


def test_materialize_official_skillopt_mobilitybench_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_factual = tmp_path / "train_factual.jsonl"
    val_factual = tmp_path / "val_factual.jsonl"
    train_interventions = tmp_path / "train_interventions.jsonl"
    val_interventions = tmp_path / "val_interventions.jsonl"
    output_dir = tmp_path / "skillopt_dataset"

    _write_jsonl(train_factual, [_record("train_case")])
    _write_jsonl(val_factual, [_record("val_case")])
    _write_jsonl(train_interventions, [_candidate("train_case")])
    _write_jsonl(val_interventions, [_candidate("val_case")])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_official_skillopt_mobilitybench_dataset.py",
            "--train-factual-records",
            str(train_factual),
            "--train-interventions",
            str(train_interventions),
            "--val-factual-records",
            str(val_factual),
            "--val-interventions",
            str(val_interventions),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert materialize_mb_skillopt.main() == 0

    train_items = json.loads((output_dir / "train" / "items.json").read_text())
    val_items = json.loads((output_dir / "val" / "items.json").read_text())
    test_items = json.loads((output_dir / "test" / "items.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert [item["id"] for item in train_items] == ["train_case::cf_train_case"]
    assert [item["id"] for item in val_items] == ["val_case::cf_val_case"]
    assert test_items == []
    assert train_items[0]["factual"]["final_response"] == "factual answer"
    assert train_items[0]["task_type"] == "tool_result_missing"
    assert manifest["counts"] == {"train": 1, "val": 1, "test": 0}


def _record(sample_id: str) -> EvaluationRecord:
    sample = TaskSample(
        sample_id=sample_id,
        benchmark=BenchmarkName.MOBILITY_BENCH,
        query="Find a nearby POI.",
    )
    factual = AgentTrajectory(
        sample_id=sample_id,
        run_mode=RunMode.FACTUAL,
        events=[
            ToolEvent(
                step=0,
                tool_name="search_around_poi",
                observation="one result",
                metadata={"official_role": "worker"},
            )
        ],
        final_response="factual answer",
    )
    return EvaluationRecord(sample=sample, factual=factual)


def _candidate(sample_id: str) -> InterventionCandidate:
    intervention = InterventionSpec(
        intervention_id=f"cf_{sample_id}",
        sample_id=sample_id,
        benchmark=BenchmarkName.MOBILITY_BENCH,
        dimension="tool_empty_result",
        domain_operator="tool_result_missing",
        capability_targets=["tool_robustness"],
        intervention_type="tool_result",
        condition_text="The latest tool result is empty.",
    )
    recovery = ExpectedRecoverySpec(
        intervention_id=intervention.intervention_id,
        sample_id=sample_id,
        capability_targets=["tool_robustness"],
        summary="Handle the empty tool result.",
        must_do=["Ask for a nearby alternative check."],
        must_not_do=["Invent an unavailable POI."],
    )
    return InterventionCandidate(
        spec=intervention,
        recovery_spec=recovery,
        validity_score=1.0,
    )


def _write_jsonl(path: Path, rows: list[EvaluationRecord | InterventionCandidate]) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )
