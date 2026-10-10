from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "src/cce_mobility/workflows/mobilitybench"
    / "run_mobilitybench_official_prequential.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_mobilitybench_official_prequential",
    SCRIPT_PATH,
)
assert SPEC is not None
run_mobilitybench_official_prequential = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_mobilitybench_official_prequential)


def test_write_single_case_dataset_preserves_header_and_row(tmp_path: Path) -> None:
    output = tmp_path / "one.csv"
    run_mobilitybench_official_prequential._write_single_case_dataset(
        output,
        ["case_id", "query", "context"],
        {"case_id": "42", "query": "nearest bank", "context": "{'city':'gz'}"},
    )

    with output.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows == [{"case_id": "42", "query": "nearest bank", "context": "{'city':'gz'}"}]


def test_official_env_injects_prompt_and_paths(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Verify current observations.\n", encoding="utf-8")

    env = run_mobilitybench_official_prequential._official_env(
        mobilitybench_root=tmp_path / "MobilityBench",
        dataset_path=tmp_path / "case.csv",
        output_dir=tmp_path / "run",
        model="aihub-qwen35b",
        framework="plan_and_execute",
        official_config_path=tmp_path / "official_model_config.yaml",
        prompt_path=prompt,
        prompt_roles="worker,reporter",
        uv_cache_dir=str(tmp_path / ".uv-cache"),
    )

    assert env["MB_ROOT"] == str(tmp_path / "MobilityBench")
    assert env["DATASET"] == str(tmp_path / "case.csv")
    assert env["OUTPUT_DIR"] == str(tmp_path / "run")
    assert env["CCE_MOBILITYBENCH_SKILL_PROMPT_PATH"] == str(prompt)
    assert env["CCE_MOBILITYBENCH_SKILL_PROMPT_ROLES"] == "worker,reporter"


def test_dry_run_validates_prequential_inputs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    dataset = tmp_path / "official.csv"
    with dataset.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "query"])
        writer.writeheader()
        writer.writerow({"case_id": "1", "query": "nearby gas station"})
        writer.writerow({"case_id": "2", "query": "route to airport"})

    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text("", encoding="utf-8")
    config = tmp_path / "posthoc.json"
    config.write_text(
        json.dumps(
            {
                "experiment_name": "unit_posthoc",
                "output_dir": str(tmp_path / "runs"),
                "method": "official_posthoc_no_evolution",
                "iterations": 1,
                "benchmark": {
                    "name": "mobilitybench",
                    "data_path": str(tmp_path / "cce_factual_records.jsonl"),
                    "split": "unit",
                    "limit": 2,
                },
                "agent": {"model": "placeholder-agent", "provider": "placeholder"},
                "world_model": {"model": "placeholder-wm", "provider": "placeholder"},
                "judge": {"model": "placeholder-judge", "provider": "placeholder"},
                "intervention": {
                    "policy": "frozen",
                    "candidate_budget": 1,
                    "selected_budget": 1,
                    "validity_threshold": 0.7,
                },
                "extra": {"frozen_intervention_path": str(frozen)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mobilitybench_official_prequential.py",
            "--method",
            "reflexion_like",
            "--dataset",
            str(dataset),
            "--config",
            str(tmp_path / "official.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--mobilitybench-root",
            str(tmp_path / "MobilityBench"),
            "--posthoc-base-config",
            str(config),
            "--limit",
            "1",
            "--dry-run",
        ],
    )

    assert run_mobilitybench_official_prequential.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["method"] == "reflexion_like"
    assert payload["case_count"] == 1
    assert payload["prequential_update_rule"] == "score case t, then update memory for case t+1"


def test_posthoc_recovery_system_prompt_hides_evaluator_terms() -> None:
    prompt = run_mobilitybench_official_prequential.RECOVERY_SYSTEM_PROMPT.lower()

    assert "counterfactual" not in prompt
    assert "intervention" not in prompt
    assert "benchmark" not in prompt
    assert "evaluator" not in prompt
    assert "judge" not in prompt
