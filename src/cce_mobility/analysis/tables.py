from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from cce_mobility.analysis.badcases import (
    extract_badcases,
    summarize_badcases,
    write_badcase_reports,
)
from cce_mobility.core.capabilities import infer_capability_targets, infer_domain_operator


DIMENSION_FIELDS = [
    "count",
    "counterfactual_recovery_avg",
    "counterfactual_low_score_rate",
    "counterfactual_final_pass_rate",
    "counterfactual_unavailable_usage_rate",
    "counterfactual_commonsense_pass_rate",
    "counterfactual_hard_pass_rate",
    "counterfactual_llm_error_rate",
]


ITERATION_FIELDS = [
    "count",
    "counterfactual_count",
    "counterfactual_recovery_avg",
    "counterfactual_low_score_rate",
    "counterfactual_final_pass_rate",
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
]


COVERAGE_FIELDS = [
    "count",
    "counterfactual_recovery_avg",
    "counterfactual_low_score_rate",
    "counterfactual_final_pass_rate",
    "counterfactual_unavailable_usage_rate",
    "counterfactual_llm_error_rate",
]


COMPARISON_FIELDS = [
    "name",
    "experiment_name",
    "total_samples",
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


def write_run_tables(report_path: Path) -> dict[str, str]:
    report = _read_json(report_path)
    records_path = Path(report["records_path"])
    if not records_path.is_absolute():
        records_path = report_path.parent / records_path.name
    records = _read_jsonl(records_path)

    output_dir = report_path.parent
    by_dimension = summarize_by_dimension(records)
    dimension_csv = output_dir / "by_dimension.csv"
    dimension_md = output_dir / "by_dimension.md"
    _write_csv(dimension_csv, by_dimension, ["dimension"] + DIMENSION_FIELDS)
    _write_markdown(dimension_md, by_dimension, ["dimension"] + DIMENSION_FIELDS)

    by_iteration = summarize_by_iteration(records)
    iteration_csv = output_dir / "by_iteration.csv"
    iteration_md = output_dir / "by_iteration.md"
    _write_csv(iteration_csv, by_iteration, ["iteration"] + ITERATION_FIELDS)
    _write_markdown(iteration_md, by_iteration, ["iteration"] + ITERATION_FIELDS)

    by_capability = summarize_by_capability(records)
    capability_csv = output_dir / "by_capability.csv"
    capability_md = output_dir / "by_capability.md"
    _write_csv(capability_csv, by_capability, ["capability"] + COVERAGE_FIELDS)
    _write_markdown(capability_md, by_capability, ["capability"] + COVERAGE_FIELDS)

    by_operator = summarize_by_domain_operator(records)
    operator_csv = output_dir / "by_domain_operator.csv"
    operator_md = output_dir / "by_domain_operator.md"
    _write_csv(operator_csv, by_operator, ["domain_operator"] + COVERAGE_FIELDS)
    _write_markdown(operator_md, by_operator, ["domain_operator"] + COVERAGE_FIELDS)
    badcase_tables = write_badcase_reports(records_path, output_dir)

    return {
        "by_dimension_csv": str(dimension_csv),
        "by_dimension_md": str(dimension_md),
        "by_iteration_csv": str(iteration_csv),
        "by_iteration_md": str(iteration_md),
        "by_capability_csv": str(capability_csv),
        "by_capability_md": str(capability_md),
        "by_domain_operator_csv": str(operator_csv),
        "by_domain_operator_md": str(operator_md),
        **badcase_tables,
    }


def write_matrix_tables(matrix_report_path: Path) -> dict[str, str]:
    summary = _read_json(matrix_report_path)
    output_dir = matrix_report_path.parent
    comparison = summary.get("comparison", [])
    comparison_csv = output_dir / "comparison.csv"
    comparison_md = output_dir / "comparison.md"
    _write_csv(comparison_csv, comparison, COMPARISON_FIELDS)
    _write_markdown(comparison_md, comparison, COMPARISON_FIELDS)

    dimension_rows: list[dict[str, Any]] = []
    all_badcases: list[dict[str, Any]] = []
    for run in summary.get("runs", []):
        records_path = Path(run.get("records_path", ""))
        if not records_path.is_absolute():
            records_path = matrix_report_path.parents[2] / records_path
        if not records_path.exists():
            continue
        records = _read_jsonl(records_path)
        for badcase in extract_badcases(records):
            all_badcases.append(
                {"name": run["name"], "experiment_name": run["experiment_name"], **badcase}
            )
        for row in summarize_by_dimension(records):
            row = {"name": run["name"], "experiment_name": run["experiment_name"], **row}
            dimension_rows.append(row)

    dimension_csv = output_dir / "by_dimension.csv"
    dimension_md = output_dir / "by_dimension.md"
    dimension_fields = ["name", "experiment_name", "dimension"] + DIMENSION_FIELDS
    _write_csv(dimension_csv, dimension_rows, dimension_fields)
    _write_markdown(dimension_md, dimension_rows, dimension_fields)

    iteration_rows: list[dict[str, Any]] = []
    capability_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    for run in summary.get("runs", []):
        records_path = Path(run.get("records_path", ""))
        if not records_path.is_absolute():
            records_path = matrix_report_path.parents[2] / records_path
        if not records_path.exists():
            continue
        records = _read_jsonl(records_path)
        for row in summarize_by_iteration(records):
            iteration_rows.append(
                {"name": run["name"], "experiment_name": run["experiment_name"], **row}
            )
        for row in summarize_by_capability(records):
            capability_rows.append(
                {"name": run["name"], "experiment_name": run["experiment_name"], **row}
            )
        for row in summarize_by_domain_operator(records):
            operator_rows.append(
                {"name": run["name"], "experiment_name": run["experiment_name"], **row}
            )

    iteration_csv = output_dir / "by_iteration.csv"
    iteration_md = output_dir / "by_iteration.md"
    iteration_fields = ["name", "experiment_name", "iteration"] + ITERATION_FIELDS
    _write_csv(iteration_csv, iteration_rows, iteration_fields)
    _write_markdown(iteration_md, iteration_rows, iteration_fields)

    capability_csv = output_dir / "by_capability.csv"
    capability_md = output_dir / "by_capability.md"
    capability_fields = ["name", "experiment_name", "capability"] + COVERAGE_FIELDS
    _write_csv(capability_csv, capability_rows, capability_fields)
    _write_markdown(capability_md, capability_rows, capability_fields)

    operator_csv = output_dir / "by_domain_operator.csv"
    operator_md = output_dir / "by_domain_operator.md"
    operator_fields = ["name", "experiment_name", "domain_operator"] + COVERAGE_FIELDS
    _write_csv(operator_csv, operator_rows, operator_fields)
    _write_markdown(operator_md, operator_rows, operator_fields)

    badcase_jsonl = output_dir / "badcases.jsonl"
    with badcase_jsonl.open("w", encoding="utf-8") as f:
        for row in all_badcases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    badcase_summary = []
    for run in summary.get("runs", []):
        rows = [row for row in all_badcases if row.get("name") == run["name"]]
        for item in summarize_badcases(rows):
            badcase_summary.append(
                {
                    "name": run["name"],
                    "experiment_name": run["experiment_name"],
                    **item,
                }
            )
    badcase_summary_csv = output_dir / "badcase_summary.csv"
    badcase_summary_md = output_dir / "badcase_summary.md"
    badcase_fields = [
        "name",
        "experiment_name",
        "capability",
        "count",
        "avg_score",
        "top_failure_modes",
        "top_dimensions",
        "top_domain_operators",
    ]
    _write_csv(badcase_summary_csv, badcase_summary, badcase_fields)
    _write_markdown(badcase_summary_md, badcase_summary, badcase_fields)

    return {
        "comparison_csv": str(comparison_csv),
        "comparison_md": str(comparison_md),
        "by_dimension_csv": str(dimension_csv),
        "by_dimension_md": str(dimension_md),
        "by_iteration_csv": str(iteration_csv),
        "by_iteration_md": str(iteration_md),
        "by_capability_csv": str(capability_csv),
        "by_capability_md": str(capability_md),
        "by_domain_operator_csv": str(operator_csv),
        "by_domain_operator_md": str(operator_md),
        "badcases_jsonl": str(badcase_jsonl),
        "badcase_summary_csv": str(badcase_summary_csv),
        "badcase_summary_md": str(badcase_summary_md),
    }


def summarize_by_dimension(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        intervention = record.get("intervention") or {}
        dimension = intervention.get("dimension") or "no_intervention"
        grouped[dimension].append(record)

    rows = []
    for dimension, group in sorted(grouped.items()):
        cf_scores = [
            rec.get("counterfactual_score", {}).get("total")
            for rec in group
            if rec.get("counterfactual_score")
        ]
        cf_metrics = [
            rec.get("metadata", {}).get("counterfactual_hard_metrics", {}) for rec in group
        ]
        cf_metrics = [item for item in cf_metrics if item]
        row = {
            "dimension": dimension,
            "count": len(group),
            **_score_metric_row(cf_scores, cf_metrics),
            "counterfactual_commonsense_pass_rate": _bool_rate(
                cf_metrics,
                "commonsense_pass",
            ),
            "counterfactual_hard_pass_rate": _bool_rate(cf_metrics, "hard_pass"),
        }
        rows.append(row)
    return rows


def summarize_by_iteration(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        iteration = int(record.get("metadata", {}).get("iteration") or 1)
        grouped[iteration].append(record)

    rows = []
    for iteration, group in sorted(grouped.items()):
        cf_records = [rec for rec in group if rec.get("counterfactual_score")]
        cf_scores = [rec["counterfactual_score"].get("total") for rec in cf_records]
        cf_metrics = [
            rec.get("metadata", {}).get("counterfactual_hard_metrics", {}) for rec in cf_records
        ]
        capabilities = {cap for rec in cf_records for cap in _record_capabilities(rec)}
        operators = {_record_domain_operator(rec) for rec in cf_records if rec.get("intervention")}
        dimensions = {
            (rec.get("intervention") or {}).get("dimension")
            for rec in cf_records
            if rec.get("intervention")
        }
        wm_labels = [_wm_label(rec) for rec in cf_records]
        prediction_rows = [
            rec.get("metadata", {}).get("wm_prediction_metrics", {})
            for rec in cf_records
            if rec.get("metadata", {}).get("wm_prediction_metrics")
        ]
        rows.append(
            {
                "iteration": iteration,
                "count": len(group),
                "counterfactual_count": len(cf_records),
                **_score_metric_row(cf_scores, [item for item in cf_metrics if item]),
                "capability_coverage": len({item for item in capabilities if item}),
                "domain_operator_coverage": len({item for item in operators if item}),
                "dimension_coverage": len({item for item in dimensions if item}),
                "high_value_wm_challenge_rate": _label_rate(
                    wm_labels,
                    "high_value_challenge",
                ),
                "invalid_or_unfair_wm_challenge_rate": _label_rate(
                    wm_labels,
                    "invalid_or_unfair",
                ),
                **_prediction_metric_row(prediction_rows),
            }
        )
    return rows


def summarize_by_capability(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for cap in _record_capabilities(record) or ["unknown"]:
            grouped[str(cap)].append(record)
    return _summarize_grouped_records(grouped, "capability")


def summarize_by_domain_operator(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        operator = _record_domain_operator(record)
        grouped[str(operator)].append(record)
    return _summarize_grouped_records(grouped, "domain_operator")


def _summarize_grouped_records(
    grouped: dict[str, list[dict[str, Any]]],
    group_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for name, group in sorted(grouped.items()):
        cf_scores = [
            rec.get("counterfactual_score", {}).get("total")
            for rec in group
            if rec.get("counterfactual_score")
        ]
        cf_metrics = [
            rec.get("metadata", {}).get("counterfactual_hard_metrics", {}) for rec in group
        ]
        rows.append(
            {
                group_key: name,
                "count": len(group),
                **_score_metric_row(cf_scores, [item for item in cf_metrics if item]),
            }
        )
    return rows


def _score_metric_row(
    cf_scores: list[Any],
    cf_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "counterfactual_recovery_avg": _avg_values(cf_scores),
        "counterfactual_low_score_rate": _low_score_rate(cf_scores),
        "counterfactual_final_pass_rate": _bool_rate(cf_metrics, "final_pass"),
        "counterfactual_unavailable_usage_rate": _bool_rate(
            [item for item in cf_metrics if int(item.get("unavailable_id_count") or 0) > 0],
            "uses_unavailable_identifier",
        ),
        "counterfactual_llm_error_rate": _bool_rate(cf_metrics, "llm_error"),
    }


def _prediction_metric_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "wm_score_bin_accuracy": 0.0,
            "wm_failure_mode_hit_rate": 0.0,
            "wm_failure_mode_precision": 0.0,
            "wm_failure_mode_recall": 0.0,
            "wm_prediction_avg_confidence": 0.0,
        }
    return {
        "wm_score_bin_accuracy": _bool_rate(rows, "score_bin_hit"),
        "wm_failure_mode_hit_rate": _bool_rate(rows, "failure_mode_hit"),
        "wm_failure_mode_precision": round(
            sum(float(row.get("failure_mode_precision") or 0.0) for row in rows) / len(rows),
            4,
        ),
        "wm_failure_mode_recall": round(
            sum(float(row.get("failure_mode_recall") or 0.0) for row in rows) / len(rows),
            4,
        ),
        "wm_prediction_avg_confidence": round(
            sum(float(row.get("confidence") or 0.0) for row in rows) / len(rows),
            4,
        ),
    }


def _record_capabilities(record: dict[str, Any]) -> list[str]:
    intervention = record.get("intervention") or {}
    caps = intervention.get("capability_targets") or []
    if caps:
        return [str(cap) for cap in caps]
    dimension = intervention.get("dimension")
    return infer_capability_targets(str(dimension)) if dimension else []


def _record_domain_operator(record: dict[str, Any]) -> str:
    intervention = record.get("intervention") or {}
    operator = intervention.get("domain_operator")
    if operator:
        return str(operator)
    dimension = intervention.get("dimension")
    return infer_domain_operator(str(dimension)) if dimension else "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|")


def _bool_rate(values: list[dict[str, Any]], key: str) -> float:
    present = [bool(item[key]) for item in values if key in item and item[key] is not None]
    if not present:
        return 0.0
    return round(sum(1 for item in present if item) / len(present), 4)


def _avg_values(values: list[Any]) -> float:
    nums = [float(item) for item in values if isinstance(item, (int, float))]
    if not nums:
        return 0.0
    return round(sum(nums) / len(nums), 4)


def _low_score_rate(values: list[Any], threshold: float = 2.0) -> float:
    nums = [float(item) for item in values if isinstance(item, (int, float))]
    if not nums:
        return 0.0
    return round(sum(1 for item in nums if item < threshold) / len(nums), 4)


def _wm_label(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if metadata.get("wm_label"):
        return str(metadata["wm_label"])
    score = (record.get("counterfactual_score") or {}).get("total")
    validity = float(metadata.get("intervention_validity") or 0.0)
    hard = metadata.get("counterfactual_hard_metrics", {})
    if validity < 0.7:
        return "invalid_or_unfair"
    diagnosticity = 0.0
    if isinstance(score, (int, float)) and score < 3.0:
        diagnosticity += 0.45
    if hard.get("uses_unavailable_identifier"):
        diagnosticity += 0.35
    if hard.get("parse_success"):
        diagnosticity += 0.20
    if diagnosticity >= 0.7:
        return "high_value_challenge"
    if diagnosticity >= 0.4:
        return "medium_value_challenge"
    return "low_value_or_too_easy"


def _label_rate(labels: list[str], target: str) -> float:
    if not labels:
        return 0.0
    return round(sum(1 for label in labels if label == target) / len(labels), 4)
