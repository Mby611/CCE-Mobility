from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cce_mobility.analysis.tables import write_matrix_tables
from cce_mobility.core.config import ExperimentConfig
from cce_mobility.core.schema import ExperimentReport
from cce_mobility.runners.evaluation import EvaluationRunner


class MatrixRunSpec(BaseModel):
    name: str
    overrides: dict[str, Any] = Field(default_factory=dict)


class ExperimentMatrixConfig(BaseModel):
    matrix_name: str = "cce_matrix"
    output_dir: str = "runs/matrices"
    base_config: str | dict[str, Any]
    runs: list[MatrixRunSpec]

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentMatrixConfig":
        with path.open("r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


def run_matrix(path: Path, *, resume_existing: bool = False) -> dict[str, Any]:
    matrix = ExperimentMatrixConfig.from_json(path)
    base = _load_base_config(matrix.base_config, path.parent)
    reports = []

    for run_spec in matrix.runs:
        data = _deep_merge(copy.deepcopy(base), run_spec.overrides)
        data["experiment_name"] = data.get("experiment_name") or run_spec.name
        cfg = ExperimentConfig.model_validate(data)
        run_dir = Path(cfg.output_dir) / cfg.experiment_name
        report_path = run_dir / "report.json"
        records_path = run_dir / "records.jsonl"
        if resume_existing and report_path.exists():
            report = ExperimentReport.model_validate_json(report_path.read_text(encoding="utf-8"))
            if (
                report.experiment_name != cfg.experiment_name
                or report.total_samples != cfg.benchmark.limit
                or report.metrics.get("sample_count") != cfg.benchmark.limit
                or report.method != cfg.method
                or not records_path.is_file()
            ):
                raise ValueError(f"Existing matrix run is not complete: {report_path}")
            if _report_has_llm_errors(report):
                cfg.extra["resume"] = True
                report = EvaluationRunner(cfg).run()
        else:
            if resume_existing and records_path.is_file() and records_path.stat().st_size:
                cfg.extra["resume"] = True
            report = EvaluationRunner(cfg).run()
        reports.append(
            {
                "name": run_spec.name,
                "experiment_name": report.experiment_name,
                "benchmark": report.benchmark,
                "method": report.method,
                "total_samples": report.total_samples,
                "metrics": report.metrics,
                "records_path": str(Path(report.records_path).resolve())
                if report.records_path
                else "",
            }
        )

    summary = {
        "matrix_name": matrix.matrix_name,
        "runs": reports,
        "comparison": _comparison_table(reports),
    }
    output_dir = Path(matrix.output_dir) / matrix.matrix_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["tables"] = write_matrix_tables(output_dir / "matrix_report.json")
    (output_dir / "matrix_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _report_has_llm_errors(report: ExperimentReport) -> bool:
    return any(
        float(report.metrics.get(key, 0.0) or 0.0) > 0.0
        for key in ("factual_llm_error_rate", "counterfactual_llm_error_rate")
    )


def _load_base_config(base_config: str | dict[str, Any], matrix_dir: Path) -> dict[str, Any]:
    if isinstance(base_config, dict):
        return copy.deepcopy(base_config)
    path = Path(base_config)
    if not path.is_absolute():
        path = matrix_dir / path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _comparison_table(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "sample_count",
        "counterfactual_count",
        "expected_counterfactual_slots",
        "counterfactual_coverage_rate",
        "factual_judge_avg",
        "counterfactual_recovery_avg",
        "counterfactual_low_score_rate",
        "factual_final_pass_rate",
        "counterfactual_final_pass_rate",
        "final_pass_rate_delta",
        "counterfactual_unavailable_usage_rate",
        "counterfactual_llm_error_rate",
        "capability_coverage",
        "domain_operator_coverage",
        "dimension_coverage",
        "high_value_wm_challenge_rate",
        "invalid_or_unfair_wm_challenge_rate",
        "wm_score_bin_accuracy",
        "wm_failure_mode_hit_rate",
        "wm_failure_mode_precision",
        "wm_failure_mode_recall",
        "wm_prediction_avg_confidence",
        "agent_recovery_skill_count",
        "wm_intervention_spec_count",
        "skill_spec_total_count",
        "reflection_memory_count",
        "voyager_skill_count",
        "optimized_policy_signal_count",
        "skillos_agent_skill_count",
        "adaptive_wm_spec_update_count",
        "adaptive_agent_skill_update_count",
        "agent_owned_retrieval_enabled",
        "agent_owned_retrieval_case_count",
        "agent_owned_retrieval_selected_case_count",
        "agent_owned_retrieval_no_skill_rate",
        "agent_owned_retrieval_avg_selected_skills",
        "agent_owned_retrieval_avg_available_skills",
        "agent_owned_retrieval_avg_unfiltered_available_skills",
        "agent_owned_retrieval_filtered_case_rate",
        "agent_owned_retrieval_excluded_dimension_count",
        "agent_owned_retrieval_excluded_domain_operator_count",
        "agent_owned_retrieval_unique_selected_skill_count",
        "agent_owned_retrieval_agent_selected_case_count",
        "agent_owned_retrieval_agent_avg_selected_skills",
        "agent_owned_retrieval_selected_with_evidence_rate",
        "agent_owned_retrieval_selected_with_trigger_reason_rate",
        "agent_owned_retrieval_agent_selected_with_evidence_rate",
        "agent_owned_retrieval_agent_selected_with_trigger_reason_rate",
        "agent_owned_retrieval_ignored_reason_coverage",
        "agent_transport_backend",
        "agent_transport_cli_version",
        "agent_transport_call_count",
        "agent_transport_error_count",
        "agent_transport_input_tokens",
        "agent_transport_output_tokens",
        "agent_transport_credits",
        "agent_transport_duration_api_ms",
        "agent_transport_reported_model_usage_keys",
        "agent_transport_underlying_model_disclosed",
        "agent_transport_temperature_controlled",
        "world_model_transport_backend",
        "world_model_transport_cli_version",
        "world_model_transport_call_count",
        "world_model_transport_error_count",
        "world_model_transport_input_tokens",
        "world_model_transport_output_tokens",
        "world_model_transport_credits",
        "world_model_transport_duration_api_ms",
        "world_model_transport_reported_model_usage_keys",
        "world_model_transport_underlying_model_disclosed",
        "world_model_transport_temperature_controlled",
    ]
    rows = []
    for report in reports:
        metrics = report.get("metrics", {})
        row = {
            "name": report["name"],
            "experiment_name": report["experiment_name"],
            "total_samples": report.get("total_samples", ""),
        }
        row.update({key: metrics.get(key) for key in keys if key in metrics})
        rows.append(row)
    return rows
