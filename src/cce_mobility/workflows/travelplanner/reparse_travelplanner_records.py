from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.benchmarks.travelplanner import TravelPlannerAdapter  # noqa: E402
from cce_mobility.benchmarks.travelplanner_eval import (  # noqa: E402
    EVALUATOR_PROTOCOL_VERSION,
)
from cce_mobility.core.config import BenchmarkConfig  # noqa: E402
from cce_mobility.core.metrics import aggregate_records  # noqa: E402
from cce_mobility.core.schema import EvaluationRecord  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute TravelPlanner hard/parse metrics from existing responses without "
            "calling an LLM. Original records and reports are left unchanged."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--records")
    source.add_argument(
        "--matrix-report",
        help="Reparse every records_path in a completed matrix report.",
    )
    parser.add_argument("--output-records", default="")
    parser.add_argument("--output-report", default="")
    parser.add_argument("--output-matrix-report", default="")
    args = parser.parse_args()

    if args.matrix_report:
        if args.output_records or args.output_report:
            parser.error("--output-records/--output-report can only be used with --records")
        matrix_path = _resolve(args.matrix_report)
        output_matrix_path = (
            _resolve(args.output_matrix_report)
            if args.output_matrix_report
            else matrix_path.with_name("matrix_report_format_reparsed.json")
        )
        summary = reparse_matrix_report(matrix_path, output_matrix_path)
        print(json.dumps(summary["format_reparse"], ensure_ascii=False, indent=2))
        print(f"matrix report: {output_matrix_path}")
        return 0

    assert args.records is not None
    records_path = _resolve(args.records)
    output_records = (
        _resolve(args.output_records)
        if args.output_records
        else records_path.with_name("records_format_reparsed.jsonl")
    )
    output_report = (
        _resolve(args.output_report)
        if args.output_report
        else records_path.with_name("report_format_reparsed.json")
    )

    report = reparse_records_file(
        records_path,
        output_records=output_records,
        output_report=output_report,
    )
    print(json.dumps(report["format_reparse"], ensure_ascii=False, indent=2))
    print(f"records: {output_records}")
    print(f"report: {output_report}")
    return 0


def reparse_records_file(
    records_path: Path,
    *,
    output_records: Path,
    output_report: Path,
) -> dict[str, Any]:
    records = _read_records(records_path)
    old_metrics = aggregate_records(records)
    reparsed, changed = reparse_records(records)
    new_metrics = aggregate_records(reparsed)

    output_records.parent.mkdir(parents=True, exist_ok=True)
    with output_records.open("w", encoding="utf-8") as f:
        for record in reparsed:
            f.write(record.model_dump_json() + "\n")

    original_report_path = records_path.with_name("report.json")
    report = _read_json(original_report_path) if original_report_path.exists() else {}
    preserved_metrics = dict(report.get("metrics") or {})
    preserved_metrics.update(new_metrics)
    preserved_metrics["format_reparse_changed_factual_count"] = changed["factual"]
    preserved_metrics["format_reparse_changed_counterfactual_count"] = changed["counterfactual"]
    report.update(
        {
            "metrics": preserved_metrics,
            "records_path": str(output_records),
            "format_reparse": {
                "evaluator_protocol_version": EVALUATOR_PROTOCOL_VERSION,
                "source_records": str(records_path),
                "llm_calls": 0,
                "original_factual_parse_success_rate": old_metrics.get(
                    "factual_parse_success_rate", 0.0
                ),
                "reparsed_factual_parse_success_rate": new_metrics.get(
                    "factual_parse_success_rate", 0.0
                ),
                "original_counterfactual_parse_success_rate": old_metrics.get(
                    "counterfactual_parse_success_rate", 0.0
                ),
                "reparsed_counterfactual_parse_success_rate": new_metrics.get(
                    "counterfactual_parse_success_rate", 0.0
                ),
                "original_counterfactual_unavailable_usage_rate": old_metrics.get(
                    "counterfactual_unavailable_usage_rate", 0.0
                ),
                "reparsed_counterfactual_unavailable_usage_rate": new_metrics.get(
                    "counterfactual_unavailable_usage_rate", 0.0
                ),
                "original_factual_final_pass_rate": old_metrics.get("factual_final_pass_rate", 0.0),
                "reparsed_factual_final_pass_rate": new_metrics.get("factual_final_pass_rate", 0.0),
                "original_counterfactual_final_pass_rate": old_metrics.get(
                    "counterfactual_final_pass_rate", 0.0
                ),
                "reparsed_counterfactual_final_pass_rate": new_metrics.get(
                    "counterfactual_final_pass_rate", 0.0
                ),
                "changed_factual_count": changed["factual"],
                "changed_counterfactual_count": changed["counterfactual"],
            },
        }
    )
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def reparse_matrix_report(
    matrix_path: Path,
    output_matrix_path: Path,
) -> dict[str, Any]:
    matrix = _read_json(matrix_path)
    updated_runs: list[dict[str, Any]] = []
    changed_factual = 0
    changed_counterfactual = 0

    for run in matrix.get("runs", []):
        records_path = _resolve_matrix_records_path(
            str(run.get("records_path") or ""),
            matrix_path,
        )
        if not records_path.exists():
            raise FileNotFoundError(
                f"Matrix run {run.get('name', '<unnamed>')} records not found: {records_path}"
            )
        report = reparse_records_file(
            records_path,
            output_records=records_path.with_name("records_format_reparsed.jsonl"),
            output_report=records_path.with_name("report_format_reparsed.json"),
        )
        reparse_summary = report["format_reparse"]
        changed_factual += int(reparse_summary["changed_factual_count"])
        changed_counterfactual += int(reparse_summary["changed_counterfactual_count"])
        updated = dict(run)
        updated["metrics"] = report["metrics"]
        updated["records_path"] = report["records_path"]
        updated["format_reparse"] = reparse_summary
        updated_runs.append(updated)

    matrix["runs"] = updated_runs
    matrix["comparison"] = _updated_comparison(
        matrix.get("comparison", []),
        updated_runs,
    )
    matrix["format_reparse"] = {
        "evaluator_protocol_version": EVALUATOR_PROTOCOL_VERSION,
        "source_matrix_report": str(matrix_path),
        "llm_calls": 0,
        "run_count": len(updated_runs),
        "changed_factual_count": changed_factual,
        "changed_counterfactual_count": changed_counterfactual,
    }
    output_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    output_matrix_path.write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return matrix


def _updated_comparison(
    existing: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {str(row.get("name") or ""): dict(row) for row in existing if row.get("name")}
    rows = []
    for run in runs:
        name = str(run.get("name") or "")
        row = by_name.get(name, {})
        row.update(
            {
                "name": name,
                "experiment_name": run.get("experiment_name", ""),
                **(run.get("metrics") or {}),
            }
        )
        rows.append(row)
    return rows


def _resolve_matrix_records_path(value: str, matrix_path: Path) -> Path:
    if not value:
        return Path("__missing_records_path__")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return matrix_path.parent / path


def reparse_records(
    records: list[EvaluationRecord],
) -> tuple[list[EvaluationRecord], dict[str, int]]:
    adapter = TravelPlannerAdapter(BenchmarkConfig(name="travelplanner"))
    changed = {"factual": 0, "counterfactual": 0}
    seen_factual: set[str] = set()

    for record in records:
        metadata = dict(record.metadata)
        if record.factual is not None:
            new_factual = adapter.evaluate_trajectory(record.sample, record.factual, None)
            old_factual = metadata.get("factual_hard_metrics") or {}
            sample_id = record.sample.sample_id
            if sample_id not in seen_factual and bool(new_factual.get("parse_success")) != bool(
                old_factual.get("parse_success")
            ):
                changed["factual"] += 1
            seen_factual.add(sample_id)
            metadata["factual_hard_metrics"] = new_factual

        if record.counterfactual is not None and record.intervention is not None:
            new_counterfactual = adapter.evaluate_trajectory(
                record.sample,
                record.counterfactual,
                record.intervention,
            )
            old_counterfactual = metadata.get("counterfactual_hard_metrics") or {}
            if bool(new_counterfactual.get("parse_success")) != bool(
                old_counterfactual.get("parse_success")
            ):
                changed["counterfactual"] += 1
            metadata["counterfactual_hard_metrics"] = new_counterfactual

        record.metadata = metadata
    return records, changed


def _read_records(path: Path) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(EvaluationRecord.model_validate(json.loads(line)))
    return records


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
