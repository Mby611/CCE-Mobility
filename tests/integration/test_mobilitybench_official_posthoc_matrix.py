from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_official_posthoc_matrix_runs_and_writes_report(tmp_path: Path) -> None:
    factual_path = tmp_path / "cce_factual_records.jsonl"
    factual_path.write_text(
        json.dumps(_official_factual_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    frozen_path = tmp_path / "frozen.jsonl"
    frozen_path.write_text(
        json.dumps(_frozen_candidate("mb_official_case1"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    matrix_path = tmp_path / "matrix.json"
    matrix = {
        "matrix_name": "pytest_official_posthoc_matrix",
        "output_dir": str(tmp_path / "matrices"),
        "base_config": {
            "output_dir": str(tmp_path / "runs"),
            "method": "official_posthoc_no_evolution",
            "iterations": 1,
            "benchmark": {
                "name": "mobilitybench",
                "data_path": str(factual_path),
                "split": "unit",
                "limit": 1,
            },
            "agent": {"model": "placeholder-agent", "provider": "placeholder"},
            "world_model": {"model": "placeholder-wm", "provider": "placeholder"},
            "judge": {"model": "placeholder-judge", "provider": "placeholder"},
            "intervention": {
                "policy": "frozen",
                "candidate_budget": 1,
                "selected_budget": 1,
                "severity_grid": [1.0],
                "validity_threshold": 0.7,
            },
            "extra": {
                "save_skill_memory": False,
                "frozen_intervention_path": str(frozen_path),
            },
        },
        "runs": [
            {
                "name": "no_evolution",
                "overrides": {"experiment_name": "pytest_matrix_no_evolution"},
            },
            {
                "name": "ours_v2_ontology",
                "overrides": {
                    "experiment_name": "pytest_matrix_ours_v2",
                    "method": "ontology_phase_switching",
                    "extra": {
                        "freeze_method": True,
                        "save_skill_memory": False,
                        "frozen_intervention_path": str(frozen_path),
                    },
                },
            },
        ],
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "src/cce_mobility/workflows/mobilitybench"
                / "run_mobilitybench_official_posthoc_matrix.py"
            ),
            str(matrix_path),
        ],
        check=True,
    )

    report_path = tmp_path / "matrices" / "pytest_official_posthoc_matrix" / "matrix_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert [run["name"] for run in report["runs"]] == ["no_evolution", "ours_v2_ontology"]
    assert report["runs"][0]["metrics"]["counterfactual_count"] == 1
    assert report["runs"][1]["metrics"]["posthoc_method_lifecycle_enabled"] is True
    assert (report_path.parent / "_configs" / "no_evolution.json").exists()
    assert (report_path.parent / "_configs" / "ours_v2_ontology.json").exists()


def _official_factual_record() -> dict:
    return {
        "sample": {
            "sample_id": "mb_official_case1",
            "benchmark": "mobilitybench",
            "query": "帮我导航到人民公园",
            "context": {},
            "ground_truth": {},
            "metadata": {"official_case_id": "case1"},
        },
        "factual": {
            "sample_id": "mb_official_case1",
            "run_mode": "factual",
            "events": [
                {
                    "step": 0,
                    "tool_name": "official_mobilitybench_planner",
                    "parameters": {"intent": "route_planning"},
                    "observation": "",
                    "action_text": "Plan a route to 人民公园.",
                    "success": True,
                    "metadata": {"official_role": "planner"},
                },
                {
                    "step": 1,
                    "tool_name": "route_planning",
                    "parameters": {"destination": "人民公园"},
                    "observation": "Route A is available.",
                    "action_text": "Use Route A.",
                    "success": True,
                    "metadata": {"official_role": "worker"},
                },
                {
                    "step": 2,
                    "tool_name": "official_mobilitybench_reporter",
                    "parameters": {},
                    "observation": "",
                    "action_text": "建议走 Route A。",
                    "success": True,
                    "metadata": {"official_role": "reporter"},
                },
            ],
            "final_response": "建议走 Route A。",
            "raw": {"source": "unit"},
        },
        "metadata": {"source": "unit"},
    }


def _frozen_candidate(sample_id: str) -> dict:
    return {
        "spec": {
            "intervention_id": "frozen_poi_closed_case1",
            "sample_id": sample_id,
            "benchmark": "mobilitybench",
            "dimension": "poi_closed",
            "domain": "mobility",
            "domain_operator": "selected_poi_unavailable",
            "capability_targets": ["change_detection", "local_substitution"],
            "intervention_type": "availability",
            "severity": 1.0,
            "condition_text": "人民公园当前临时关闭，需要选择附近可去的替代地点。",
            "diagnostic_goal": "Check whether the assistant avoids a now-unavailable POI.",
            "expected_recovery_actions": ["notice_change", "suggest_alternative"],
            "validity_constraints": ["The task remains recoverable."],
            "source": "unit_frozen",
        },
        "recovery_spec": {
            "intervention_id": "frozen_poi_closed_case1",
            "sample_id": sample_id,
            "capability_targets": ["change_detection", "local_substitution"],
            "failure_types": [],
            "recovery_contract": {},
            "summary": "人民公园当前临时关闭，需要选择附近可去的替代地点。",
            "changed_facts": ["人民公园当前临时关闭。"],
            "must_detect": ["poi_closed"],
            "must_do": ["Notice the changed environment condition.", "Suggest an alternative."],
            "must_not_do": ["Do not keep recommending 人民公园 as available."],
            "acceptable_recoveries": ["Suggest a nearby alternative POI."],
            "invalid_failures": ["Ignores the changed POI availability."],
            "rubric": {},
            "metadata": {"dimension": "poi_closed"},
        },
        "validity_score": 1.0,
        "utility": 0.5,
        "validation_notes": [],
    }
