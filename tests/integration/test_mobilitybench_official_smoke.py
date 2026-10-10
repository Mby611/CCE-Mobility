from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_prepare_mobilitybench_official_smoke(tmp_path: Path) -> None:
    source = tmp_path / "validation.csv"
    with source.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "query", "context", "tool_list"])
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "a",
                "query": "去人民公园",
                "context": "{}",
                "tool_list": "[]",
            }
        )
        writer.writerow(
            {
                "case_id": "b",
                "query": "查附近停车场",
                "context": "{}",
                "tool_list": "[]",
            }
        )
        writer.writerow(
            {
                "case_id": "c",
                "query": "查附近充电站",
                "context": "{}",
                "tool_list": "[]",
            }
        )

    output_dir = tmp_path / "official"
    subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "src/cce_mobility/workflows/mobilitybench"
                / "prepare_mobilitybench_official_smoke.py"
            ),
            "--source-csv",
            str(source),
            "--output-dir",
            str(output_dir),
            "--limit",
            "2",
            "--offset",
            "1",
            "--model-alias",
            "unit-model",
            "--model-name",
            "unit-model-name",
            "--base-url",
            "https://model.example.test/v1",
        ],
        check=True,
    )

    dataset_path = output_dir / "official_smoke_2.csv"
    config_path = output_dir / "official_model_config.yaml"
    manifest_path = output_dir / "official_smoke_manifest.json"

    assert dataset_path.exists()
    assert config_path.exists()
    assert manifest_path.exists()

    rows = list(csv.DictReader(dataset_path.open(encoding="utf-8")))
    assert [row["case_id"] for row in rows] == ["b", "c"]

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["models"]["unit-model"]["roles"]["planner"] == "unit-model-name"
    assert config["agent"]["default_framework"] == "plan_and_execute"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["offset"] == 1
    assert manifest["case_ids"] == ["b", "c"]
    assert manifest["source_sha256"]
    assert manifest["dataset_sha256"]
    assert "base_url" not in manifest
    assert str(tmp_path) not in json.dumps(manifest)


def test_official_smoke_requires_explicit_upstream_checkout() -> None:
    env = os.environ.copy()
    env.pop("MB_ROOT", None)
    result = subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "run_mobilitybench_official_smoke.sh"),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "Set MB_ROOT" in result.stderr


def test_export_agent_skill_prompt_sanitizes_evaluator_terms(tmp_path: Path) -> None:
    memory_path = tmp_path / "skill_memory.jsonl"
    memory_path.write_text(
        json.dumps(
            {
                "skill_id": "agent::tool_robustness::unit",
                "skill_type": "agent_recovery",
                "name": "Recover from counterfactual tool result gaps",
                "trigger_dimensions": ["tool_empty_result"],
                "capability_targets": ["tool_robustness"],
                "domain_operators": ["tool_result_missing"],
                "procedure": [
                    "When an intervention makes a tool result empty, verify current evidence."
                ],
                "success_criteria": ["Do not mention evaluator internals."],
                "status": "active",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "prompt.md"

    subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT / "src/cce_mobility/workflows/shared" / "export_agent_skill_prompt.py"
            ),
            "--memory-path",
            str(memory_path),
            "--output",
            str(output),
            "--strict-agent-visible",
        ],
        check=True,
    )

    text = output.read_text(encoding="utf-8")
    assert "tool_robustness" in text
    assert "environment update" in text
    assert "counterfactual" not in text.lower()
    assert "intervention" not in text.lower()
    assert "evaluator" not in text.lower()
    assert "applicability boundary" not in text.lower()


def test_export_agent_skill_prompt_adds_observable_boundary(tmp_path: Path) -> None:
    memory_path = tmp_path / "skill_memory.jsonl"
    memory_path.write_text(
        json.dumps(
            {
                "skill_id": "agent::tool_robustness::unit",
                "skill_type": "agent_recovery",
                "name": "Recover from tool result gaps",
                "trigger_dimensions": ["tool_empty_result"],
                "capability_targets": ["tool_robustness"],
                "domain_operators": ["tool_result_missing"],
                "procedure": [
                    "When a tool result is empty, verify current evidence before fallback."
                ],
                "success_criteria": ["Do not invent unverified alternatives."],
                "status": "active",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "prompt.md"

    subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT / "src/cce_mobility/workflows/shared" / "export_agent_skill_prompt.py"
            ),
            "--memory-path",
            str(memory_path),
            "--output",
            str(output),
            "--strict-agent-visible",
            "--boundary-mode",
            "observable_mobility",
        ],
        check=True,
    )

    text = output.read_text(encoding="utf-8")
    lower = text.lower()
    assert "applicability boundary for recovery skill use" in lower
    assert "route distance" in lower
    assert "preference-constrained planning" in lower
    assert "ignore it and solve from the current task state" in lower
    for forbidden in [
        "counterfactual",
        "intervention",
        "evaluator",
        "judge",
        "benchmark",
        "posthoc",
    ]:
        assert forbidden not in lower


def test_export_agent_skill_prompt_fails_on_missing_memory(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT / "src/cce_mobility/workflows/shared" / "export_agent_skill_prompt.py"
            ),
            "--memory-path",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(tmp_path / "prompt.md"),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "skill memory not found" in result.stderr


def test_mobilitybench_skill_prompt_patch_is_env_gated(tmp_path: Path, monkeypatch) -> None:
    patch_path = PROJECT_ROOT / "patches" / "mobilitybench" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("cce_mb_sitecustomize_unit", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.delenv("CCE_MOBILITYBENCH_SKILL_PROMPT_PATH", raising=False)
    assert module._official_skill_prompt_suffix(framework="plan_and_execute", role="worker") == ""

    prompt_file = tmp_path / "skill.md"
    prompt_file.write_text(
        "Use this counterfactual intervention note from the evaluator.",
        encoding="utf-8",
    )
    monkeypatch.setenv("CCE_MOBILITYBENCH_SKILL_PROMPT_PATH", str(prompt_file))
    monkeypatch.setenv("CCE_MOBILITYBENCH_SKILL_PROMPT_ROLES", "worker,reporter")

    suffix = module._official_skill_prompt_suffix(
        framework="plan_and_execute",
        role="worker",
    )
    assert "Reusable recovery guidance" in suffix
    assert "environment update" in suffix
    assert "counterfactual" not in suffix.lower()
    assert "intervention" not in suffix.lower()
    assert (
        module._official_skill_prompt_suffix(
            framework="plan_and_execute",
            role="planner",
        )
        == ""
    )


def test_summarize_mobilitybench_official_metrics(tmp_path: Path) -> None:
    official_eval = tmp_path / "evaluation_results.json"
    official_eval.write_text(
        json.dumps(
            {
                "tool": {
                    "aihub-qwen35b": {
                        "summary": {
                            "metric_name": "tool",
                            "average_score": 0.0,
                            "total_cases": 2,
                            "successful_cases": 2,
                            "failed_cases": 0,
                            "sub_scores": {"TS_P": 0.5, "TS_R": 1.0, "SC": 0.75},
                        }
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cce_report = tmp_path / "matrix_report.json"
    cce_report.write_text(
        json.dumps(
            {
                "matrix_name": "unit_matrix",
                "runs": [
                    {
                        "name": "cce",
                        "metrics": {
                            "counterfactual_recovery_avg": 3.8,
                            "counterfactual_low_score_rate": 0.01,
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"

    subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "src/cce_mobility/workflows/mobilitybench"
                / "summarize_mobilitybench_official_metrics.py"
            ),
            "--official-eval",
            f"base={official_eval}",
            "--cce-report",
            str(cce_report),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert (
        summary["official_factual_metrics"]["base"]["models"]["aihub-qwen35b"]["metrics"]["tool"][
            "sub_scores"
        ]["TS_P"]
        == 0.5
    )
    assert (
        summary["cce_counterfactual_metrics"]["runs"]["cce"]["counterfactual_recovery_avg"] == 3.8
    )
    assert "Official MobilityBench metrics" in output_md.read_text(encoding="utf-8")


def test_summarize_mobilitybench_official_run_flags_dirty_cases(tmp_path: Path) -> None:
    run_dir = tmp_path / "official_run"
    model_dir = run_dir / "aihub-qwen35b"
    model_dir.mkdir(parents=True)
    (model_dir / "raw_results.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "clean",
                    "model_name": "aihub-qwen35b",
                    "planner_output": {"intent": "route", "steps": ["drive"]},
                    "worker_output": [{"tool_calls": [{"id": "1", "name": "driving_route"}]}],
                    "reporter_output": "ok",
                    "error": None,
                },
                {
                    "case_id": "dirty",
                    "model_name": "aihub-qwen35b",
                    "planner_output": {},
                    "worker_output": [],
                    "reporter_output": "Allocated quota exceeded",
                    "error": None,
                },
                {
                    "case_id": "dirty_new_quota_format",
                    "model_name": "aihub-glm-5-2",
                    "planner_output": {},
                    "worker_output": [],
                    "reporter_output": ("You exceeded your current quota; type=insufficient_quota"),
                    "error": None,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = run_dir / "official_summary.json"

    subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "src/cce_mobility/workflows/mobilitybench"
                / "summarize_mobilitybench_official_run.py"
            ),
            str(run_dir),
            "--output",
            str(output),
        ],
        check=True,
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    model_summary = summary["models"]["aihub-qwen35b"]
    assert model_summary["dirty_case_count"] == 2
    assert model_summary["dirty_case_ids"] == ["dirty", "dirty_new_quota_format"]
    assert model_summary["dirty_case_reasons"]["dirty"] == [
        "quota",
        "missing_planner",
        "no_worker_tool_call",
    ]
    assert model_summary["dirty_case_reasons"]["dirty_new_quota_format"] == [
        "quota",
        "missing_planner",
        "no_worker_tool_call",
    ]


def test_summarize_mobilitybench_official_run_flags_tool_input_errors(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "official_run"
    model_dir = run_dir / "aihub-glm-5-2"
    model_dir.mkdir(parents=True)
    (model_dir / "raw_results.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "coord_parse",
                    "model_name": "aihub-glm-5-2",
                    "planner_output": {"intent": "route", "steps": ["drive"]},
                    "worker_output": [
                        {
                            "tool_calls": [{"id": "1", "name": "driving_route"}],
                            "tool_results": {
                                "1": (
                                    "输入参数错误: 起点坐标 parse error, "
                                    "cannot convert to number: (121.1, 37.1)"
                                )
                            },
                        }
                    ],
                    "reporter_output": "ok",
                    "error": None,
                },
                {
                    "case_id": "unknown_tool",
                    "model_name": "aihub-glm-5-2",
                    "planner_output": {"intent": "route", "steps": ["bus"]},
                    "worker_output": [
                        {
                            "tool_calls": [
                                {
                                    "id": "1",
                                    "name": "bus_route_strategy0(origin='x')",
                                }
                            ],
                        }
                    ],
                    "reporter_output": "[reporter] Unknown tool: bus_route_strategy0",
                    "error": None,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output = run_dir / "official_summary.json"
    subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "src/cce_mobility/workflows/mobilitybench"
                / "summarize_mobilitybench_official_run.py"
            ),
            str(run_dir),
            "--output",
            str(output),
        ],
        check=True,
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    model_summary = summary["models"]["aihub-glm-5-2"]
    assert model_summary["dirty_case_count"] == 2
    assert model_summary["dirty_case_reasons"]["coord_parse"] == ["tool_input_error"]
    assert model_summary["dirty_case_reasons"]["unknown_tool"] == ["unknown_tool"]
