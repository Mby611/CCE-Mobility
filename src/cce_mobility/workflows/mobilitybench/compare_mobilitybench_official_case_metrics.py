from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

from cce_mobility.workflows.shared.analyze_paired_ablation import _bootstrap_mean_ci

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.integrations.mobilitybench_official_audit import (  # noqa: E402
    official_raw_dirty_reasons,
)

DEFAULT_METRICS = ["tool", "instruction", "planning", "answer", "efficiency"]
OFFICIAL_SUBDIMENSIONS = {
    "tool": {"TS_P", "TS_R", "SC"},
    "instruction": {"ID", "IE"},
    "planning": {"DEC_P", "DEC_R"},
    "answer": {"DR", "FPR"},
    "efficiency": {"IT", "OT"},
}
LOWER_IS_BETTER = {
    ("efficiency", "IT"),
    ("efficiency", "OT"),
}
SKIP_COLUMNS = {
    "case_id",
    "Case ID",
    "metric_name",
    "score",
    "error",
    "source_file",
    "task_scenario",
    "intent_family",
    "note",
    "slots_found",
    "pred_steps",
    "gold_steps",
}
QUERY_COLUMNS = [
    "query",
    "question",
    "user_query",
    "instruction",
    "input",
    "human",
    "text",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare official MobilityBench per-case metric CSVs between two "
            "evaluation directories."
        )
    )
    parser.add_argument("--baseline-eval-dir", required=True)
    parser.add_argument("--target-eval-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="aihub-qwen35b")
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--baseline-raw", default="")
    parser.add_argument("--target-raw", default="")
    parser.add_argument(
        "--common-clean-only",
        action="store_true",
        help="Compare only case IDs with clean official raw traces in both arms.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--focus-intent", action="append", default=[])
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--sort",
        choices=["worse", "improved", "absolute"],
        default="worse",
        help="How to sort the case summary in Markdown/JSON.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()
    if args.common_clean_only != bool(args.baseline_raw and args.target_raw):
        parser.error("--common-clean-only requires both --baseline-raw and --target-raw")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")

    metrics = args.metric or DEFAULT_METRICS
    dataset = _load_dataset(Path(args.dataset), strict=args.common_clean_only)
    clean_audit = None
    common_clean_ids = None
    if args.common_clean_only:
        clean_audit, common_clean_ids = _common_clean_audit(
            Path(args.baseline_raw), Path(args.target_raw), set(dataset)
        )
    case_deltas = _compare_eval_dirs(
        baseline_eval_dir=Path(args.baseline_eval_dir),
        target_eval_dir=Path(args.target_eval_dir),
        model=args.model,
        metrics=metrics,
        dataset=dataset,
        common_clean_ids=common_clean_ids,
    )
    if args.focus_intent:
        focus = set(args.focus_intent)
        case_deltas = [row for row in case_deltas if row.get("intent_family", "") in focus]

    sorted_cases = _sort_cases(case_deltas, args.sort)
    summary = {
        "baseline_eval_dir": args.baseline_eval_dir,
        "target_eval_dir": args.target_eval_dir,
        "dataset": args.dataset,
        "model": args.model,
        "metrics": metrics,
        "focus_intent": args.focus_intent,
        "sort": args.sort,
        "case_count": len(sorted_cases),
        "aggregate": _aggregate_cases(sorted_cases),
        "top_cases": sorted_cases[: args.top_n],
        "all_cases": sorted_cases,
        "metric_note": (
            "raw_delta is target - baseline. signed_delta is positive when "
            "the target improves the metric; for efficiency IT/OT, lower is better."
        ),
    }
    if clean_audit is not None:
        summary["common_clean_audit"] = clean_audit
        summary["by_metric"] = _aggregate_by_metric(
            sorted_cases, bootstrap_samples=args.bootstrap_samples
        )
        summary["bootstrap_samples"] = args.bootstrap_samples
        summary["metric_note"] += (
            " Compare paired per-metric rows only: signed_delta_sum combines "
            "incommensurable score and token units. CSV error rows are excluded."
        )

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        _write_text(Path(args.output_json), text + "\n")
    if args.output_md:
        _write_text(Path(args.output_md), _render_markdown(summary, top_n=args.top_n))
    if args.output_csv:
        _write_csv(Path(args.output_csv), sorted_cases)
    return 0


def _load_dataset(path: Path, *, strict: bool = False) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for index, row in enumerate(reader):
            case_id = str(row.get("case_id") or row.get("Case ID") or index)
            if strict and case_id in rows:
                raise ValueError(f"duplicate dataset case_id: {case_id}")
            rows[case_id] = row
    return rows


def _load_raw_quality(path: Path, dataset_ids: set[str]) -> set[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected raw_results list: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("case_id") is None:
            raise ValueError(f"raw_results row missing case_id: {path}")
        case_id = str(row["case_id"])
        if case_id in by_id:
            raise ValueError(f"duplicate raw case_id {case_id}: {path}")
        by_id[case_id] = row
    if set(by_id) != dataset_ids:
        raise ValueError(
            f"raw/dataset case IDs differ in {path}: "
            f"missing={len(dataset_ids - set(by_id))} extra={len(set(by_id) - dataset_ids)}"
        )
    return {case_id for case_id, row in by_id.items() if official_raw_dirty_reasons(row)}


def _common_clean_audit(
    baseline_raw: Path, target_raw: Path, dataset_ids: set[str]
) -> tuple[dict[str, Any], set[str]]:
    baseline_dirty = _load_raw_quality(baseline_raw, dataset_ids)
    target_dirty = _load_raw_quality(target_raw, dataset_ids)
    common_clean = dataset_ids - (baseline_dirty | target_dirty)
    if not common_clean:
        raise ValueError("no common clean case IDs")
    return {
        "baseline_raw": str(baseline_raw),
        "target_raw": str(target_raw),
        "dataset_count": len(dataset_ids),
        "baseline_dirty_count": len(baseline_dirty),
        "target_dirty_count": len(target_dirty),
        "dirty_overlap_count": len(baseline_dirty & target_dirty),
        "common_clean_count": len(common_clean),
    }, common_clean


def _compare_eval_dirs(
    *,
    baseline_eval_dir: Path,
    target_eval_dir: Path,
    model: str,
    metrics: list[str],
    dataset: dict[str, dict[str, str]],
    common_clean_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    per_case: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        baseline_rows = _load_metric_csv(
            baseline_eval_dir / model / f"{metric}_evaluation.csv",
            strict=common_clean_ids is not None,
        )
        target_rows = _load_metric_csv(
            target_eval_dir / model / f"{metric}_evaluation.csv",
            strict=common_clean_ids is not None,
        )
        if common_clean_ids is not None:
            for path, rows in ((baseline_eval_dir, baseline_rows), (target_eval_dir, target_rows)):
                unknown = set(rows) - set(dataset)
                if unknown:
                    raise ValueError(f"evaluation has {len(unknown)} unknown case IDs: {path}")
        common_case_ids = sorted(set(baseline_rows) & set(target_rows))
        if common_clean_ids is not None:
            common_case_ids = sorted(set(common_case_ids) & common_clean_ids)
        for case_id in common_case_ids:
            base = baseline_rows[case_id]
            target = target_rows[case_id]
            if common_clean_ids is not None and (_row_has_error(base) or _row_has_error(target)):
                continue
            for sub_dimension in _numeric_columns(base, target):
                if common_clean_ids is not None and sub_dimension not in OFFICIAL_SUBDIMENSIONS.get(
                    metric, set()
                ):
                    continue
                base_value = _as_float(base.get(sub_dimension))
                target_value = _as_float(target.get(sub_dimension))
                if base_value is None or target_value is None:
                    continue
                raw_delta = round(target_value - base_value, 6)
                signed_delta = _signed_delta(metric, sub_dimension, raw_delta)
                item = _case_item(per_case, case_id, dataset.get(case_id, {}))
                item["metric_deltas"].append(
                    {
                        "metric": metric,
                        "sub_dimension": sub_dimension,
                        "baseline": _round(base_value),
                        "target": _round(target_value),
                        "raw_delta": raw_delta,
                        "signed_delta": signed_delta,
                        "preferred_direction": _preferred_direction(metric, sub_dimension),
                        "status": _status(signed_delta),
                    }
                )

    for item in per_case.values():
        deltas = item["metric_deltas"]
        item["signed_delta_sum"] = round(sum(d["signed_delta"] for d in deltas), 6)
        item["absolute_delta_sum"] = round(sum(abs(d["signed_delta"]) for d in deltas), 6)
        item["improved_metric_count"] = sum(d["signed_delta"] > 0 for d in deltas)
        item["worse_metric_count"] = sum(d["signed_delta"] < 0 for d in deltas)
        item["same_metric_count"] = sum(d["signed_delta"] == 0 for d in deltas)
        item["top_negative_deltas"] = _top_deltas(deltas, reverse=False)
        item["top_positive_deltas"] = _top_deltas(deltas, reverse=True)
    return list(per_case.values())


def _load_metric_csv(path: Path, *, strict: bool = False) -> dict[str, dict[str, str]]:
    if not path.exists():
        if strict:
            raise FileNotFoundError(path)
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            case_id = str(row.get("case_id") or row.get("Case ID") or "")
            if case_id:
                if strict and case_id in rows:
                    raise ValueError(f"duplicate evaluation case_id {case_id}: {path}")
                rows[case_id] = row
            elif strict:
                raise ValueError(f"evaluation row missing case_id: {path}")
    return rows


def _row_has_error(row: dict[str, str]) -> bool:
    return str(row.get("error") or "").strip().lower() not in {"", "none", "null"}


def _numeric_columns(*rows: dict[str, str]) -> list[str]:
    columns = sorted(set().union(*(row.keys() for row in rows)))
    return [
        column
        for column in columns
        if column not in SKIP_COLUMNS
        and any(_as_float(row.get(column)) is not None for row in rows)
    ]


def _case_item(
    per_case: dict[str, dict[str, Any]],
    case_id: str,
    dataset_row: dict[str, str],
) -> dict[str, Any]:
    if case_id not in per_case:
        per_case[case_id] = {
            "case_id": case_id,
            "intent_family": dataset_row.get("intent_family", ""),
            "task_scenario": dataset_row.get("task_scenario", ""),
            "query_preview": _query_preview(dataset_row),
            "metric_deltas": [],
        }
    return per_case[case_id]


def _query_preview(row: dict[str, str], limit: int = 120) -> str:
    text = ""
    for column in QUERY_COLUMNS:
        if row.get(column):
            text = str(row[column])
            break
    if not text:
        for value in row.values():
            if isinstance(value, str) and value.strip():
                text = value
                break
    text = " ".join(str(text).split())
    return text[:limit]


def _signed_delta(metric: str, sub_dimension: str, raw_delta: float) -> float:
    return round(-raw_delta if (metric, sub_dimension) in LOWER_IS_BETTER else raw_delta, 6)


def _preferred_direction(metric: str, sub_dimension: str) -> str:
    return "lower" if (metric, sub_dimension) in LOWER_IS_BETTER else "higher"


def _status(signed_delta: float) -> str:
    if signed_delta > 0:
        return "improved"
    if signed_delta < 0:
        return "worse"
    return "same"


def _sort_cases(rows: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
    if sort == "improved":
        return sorted(rows, key=lambda row: row["signed_delta_sum"], reverse=True)
    if sort == "absolute":
        return sorted(rows, key=lambda row: row["absolute_delta_sum"], reverse=True)
    return sorted(rows, key=lambda row: row["signed_delta_sum"])


def _aggregate_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "improved_cases": 0,
            "worse_cases": 0,
            "same_cases": 0,
            "mean_signed_delta_sum": 0.0,
        }
    return {
        "improved_cases": sum(row["signed_delta_sum"] > 0 for row in rows),
        "worse_cases": sum(row["signed_delta_sum"] < 0 for row in rows),
        "same_cases": sum(row["signed_delta_sum"] == 0 for row in rows),
        "mean_signed_delta_sum": round(sum(row["signed_delta_sum"] for row in rows) / len(rows), 6),
        "mean_absolute_delta_sum": round(
            sum(row["absolute_delta_sum"] for row in rows) / len(rows), 6
        ),
    }


def _aggregate_by_metric(
    rows: list[dict[str, Any]], *, bootstrap_samples: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for delta in row["metric_deltas"]:
            groups.setdefault((delta["metric"], delta["sub_dimension"]), []).append(delta)
    result = []
    for (metric, sub_dimension), deltas in sorted(groups.items()):
        raw_deltas = [delta["raw_delta"] for delta in deltas]
        ci_low, ci_high = _bootstrap_mean_ci(raw_deltas, samples=bootstrap_samples)
        count = len(deltas)
        result.append(
            {
                "metric": metric,
                "sub_dimension": sub_dimension,
                "paired_count": count,
                "baseline_mean": round(sum(delta["baseline"] for delta in deltas) / count, 6),
                "target_mean": round(sum(delta["target"] for delta in deltas) / count, 6),
                "mean_delta": round(sum(raw_deltas) / count, 6),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "preferred_direction": _preferred_direction(metric, sub_dimension),
                "improved": sum(delta["signed_delta"] > 0 for delta in deltas),
                "worse": sum(delta["signed_delta"] < 0 for delta in deltas),
                "same": sum(delta["signed_delta"] == 0 for delta in deltas),
            }
        )
    return result


def _top_deltas(
    deltas: list[dict[str, Any]],
    *,
    reverse: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    filtered = [
        delta
        for delta in deltas
        if (delta["signed_delta"] > 0 if reverse else delta["signed_delta"] < 0)
    ]
    return sorted(filtered, key=lambda row: row["signed_delta"], reverse=reverse)[:limit]


def _render_markdown(summary: dict[str, Any], *, top_n: int) -> str:
    if "common_clean_audit" in summary:
        audit = summary["common_clean_audit"]
        lines = [
            "# MobilityBench Official Common-Clean Paired Sensitivity",
            "",
            summary["metric_note"],
            "",
            f"Dataset: {audit['dataset_count']}; baseline dirty: {audit['baseline_dirty_count']}; "
            f"target dirty: {audit['target_dirty_count']}; dirty overlap: {audit['dirty_overlap_count']}; "
            f"common clean: {audit['common_clean_count']}.",
            "",
            "Each metric uses case IDs with non-error numeric values in both evaluator CSVs. "
            "The per-metric paired count can be smaller than the common-clean count.",
            "",
            "| metric | sub_dimension | paired | baseline | target | delta | 95% bootstrap CI | preferred | improved | worse | same |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ]
        for row in summary["by_metric"]:
            lines.append(
                "| {metric} | {sub_dimension} | {paired_count} | {baseline_mean} | "
                "{target_mean} | {mean_delta} | [{bootstrap_ci_low}, {bootstrap_ci_high}] | "
                "{preferred_direction} | {improved} | {worse} | {same} |".format(**row)
            )
        return "\n".join(lines).rstrip() + "\n"
    lines = [
        "# MobilityBench Official Case-Level Metric Comparison",
        "",
        summary["metric_note"],
        "",
        "## Aggregate",
        "",
        "| case_count | improved_cases | worse_cases | same_cases | mean_signed_delta_sum | mean_absolute_delta_sum |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    aggregate = summary["aggregate"]
    lines.append(
        "| {case_count} | {improved_cases} | {worse_cases} | {same_cases} | "
        "{mean_signed_delta_sum} | {mean_absolute_delta_sum} |".format(
            case_count=summary["case_count"],
            improved_cases=aggregate["improved_cases"],
            worse_cases=aggregate["worse_cases"],
            same_cases=aggregate["same_cases"],
            mean_signed_delta_sum=aggregate["mean_signed_delta_sum"],
            mean_absolute_delta_sum=aggregate["mean_absolute_delta_sum"],
        )
    )
    lines.extend(["", f"## Top {top_n} Cases", ""])
    lines.extend(_render_case_table(summary["top_cases"]))
    return "\n".join(lines).rstrip() + "\n"


def _render_case_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| case_id | intent_family | task_scenario | signed_sum | worse | improved | negative_deltas | positive_deltas | query_preview |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {intent_family} | {task_scenario} | {signed_delta_sum} | "
            "{worse_metric_count} | {improved_metric_count} | {neg} | {pos} | {query} |".format(
                case_id=_escape_md(row["case_id"]),
                intent_family=_escape_md(row["intent_family"]),
                task_scenario=_escape_md(row["task_scenario"]),
                signed_delta_sum=row["signed_delta_sum"],
                worse_metric_count=row["worse_metric_count"],
                improved_metric_count=row["improved_metric_count"],
                neg=_escape_md(_compact_deltas(row["top_negative_deltas"])),
                pos=_escape_md(_compact_deltas(row["top_positive_deltas"])),
                query=_escape_md(row["query_preview"]),
            )
        )
    return lines


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "intent_family",
                "task_scenario",
                "query_preview",
                "signed_delta_sum",
                "absolute_delta_sum",
                "improved_metric_count",
                "worse_metric_count",
                "same_metric_count",
                "top_negative_deltas",
                "top_positive_deltas",
            ],
        )
        writer.writeheader()
        for row in rows:
            copy = dict(row)
            copy["top_negative_deltas"] = _compact_deltas(row["top_negative_deltas"])
            copy["top_positive_deltas"] = _compact_deltas(row["top_positive_deltas"])
            copy.pop("metric_deltas", None)
            writer.writerow(copy)


def _compact_deltas(deltas: list[dict[str, Any]]) -> str:
    return "; ".join(f"{d['metric']}.{d['sub_dimension']}={d['raw_delta']}" for d in deltas)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _round(value: float | None) -> float | None:
    return round(value, 6) if isinstance(value, float) else value


if __name__ == "__main__":
    raise SystemExit(main())
