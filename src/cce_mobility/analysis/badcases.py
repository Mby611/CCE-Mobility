from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cce_mobility.core.capabilities import infer_capability_targets, infer_domain_operator


BADCASE_SCORE_THRESHOLD = 2.0


def write_badcase_reports(records_path: Path, output_dir: Path) -> dict[str, str]:
    records = _read_jsonl(records_path)
    badcases = extract_badcases(records)

    badcase_jsonl = output_dir / "badcases.jsonl"
    with badcase_jsonl.open("w", encoding="utf-8") as f:
        for row in badcases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_badcases(badcases)
    summary_csv = output_dir / "badcase_summary.csv"
    summary_md = output_dir / "badcase_summary.md"
    fields = [
        "capability",
        "count",
        "avg_score",
        "top_failure_modes",
        "top_dimensions",
        "top_domain_operators",
    ]
    _write_csv(summary_csv, summary, fields)
    _write_markdown(summary_md, summary, fields)
    return {
        "badcases_jsonl": str(badcase_jsonl),
        "badcase_summary_csv": str(summary_csv),
        "badcase_summary_md": str(summary_md),
    }


def extract_badcases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        score = (record.get("counterfactual_score") or {}).get("total")
        hard = record.get("metadata", {}).get("counterfactual_hard_metrics", {})
        if not _is_badcase(score, hard):
            continue
        intervention = record.get("intervention") or {}
        dimension = intervention.get("dimension")
        capabilities = intervention.get("capability_targets") or (
            infer_capability_targets(str(dimension)) if dimension else []
        )
        domain_operator = intervention.get("domain_operator") or (
            infer_domain_operator(str(dimension)) if dimension else "unknown"
        )
        prediction = record.get("metadata", {}).get("wm_prediction_metrics", {})
        rows.append(
            {
                "sample_id": (record.get("sample") or {}).get("sample_id"),
                "iteration": record.get("metadata", {}).get("iteration"),
                "dimension": dimension,
                "domain_operator": domain_operator,
                "capability_targets": capabilities,
                "score": score,
                "strategy": (record.get("counterfactual_score") or {}).get("strategy"),
                "failure_modes": prediction.get("actual_failure_modes") or [],
                "predicted_failure_modes": prediction.get("predicted_failure_modes") or [],
                "prediction_hit": prediction.get("failure_mode_hit"),
                "uses_unavailable_identifier": hard.get("uses_unavailable_identifier"),
                "llm_error": hard.get("llm_error"),
                "intervention_id": intervention.get("intervention_id"),
            }
        )
    return rows


def summarize_badcases(badcases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in badcases:
        caps = row.get("capability_targets") or ["unknown"]
        for cap in caps:
            grouped[str(cap)].append(row)
    summary = []
    for cap, rows in sorted(grouped.items()):
        scores = [float(row["score"]) for row in rows if isinstance(row.get("score"), (int, float))]
        failure_modes = Counter(
            mode for row in rows for mode in (row.get("failure_modes") or ["unknown_failure"])
        )
        dimensions = Counter(str(row.get("dimension") or "unknown") for row in rows)
        operators = Counter(str(row.get("domain_operator") or "unknown") for row in rows)
        summary.append(
            {
                "capability": cap,
                "count": len(rows),
                "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "top_failure_modes": _format_counter(failure_modes),
                "top_dimensions": _format_counter(dimensions),
                "top_domain_operators": _format_counter(operators),
            }
        )
    return summary


def _is_badcase(score: Any, hard: dict[str, Any]) -> bool:
    low_score = isinstance(score, (int, float)) and float(score) < BADCASE_SCORE_THRESHOLD
    unavailable = bool(hard.get("uses_unavailable_identifier"))
    llm_error = bool(hard.get("llm_error"))
    return low_score or unavailable or llm_error


def _format_counter(counter: Counter[str], limit: int = 4) -> str:
    return "; ".join(f"{key}:{value}" for key, value in counter.most_common(limit))


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
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(field, "")).replace("|", "\\|") for field in fields)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
