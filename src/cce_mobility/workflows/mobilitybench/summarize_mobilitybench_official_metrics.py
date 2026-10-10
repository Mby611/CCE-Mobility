from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize official MobilityBench factual metrics together with "
            "optional CCE counterfactual recovery metrics."
        )
    )
    parser.add_argument(
        "--official-eval",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Official evaluation_results.json, e.g. base=.../evaluation_results.json.",
    )
    parser.add_argument(
        "--cce-report",
        default="",
        help="Optional CCE matrix_report.json or single report.json.",
    )
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    parser.add_argument("--output-md", default="", help="Optional output Markdown path.")
    args = parser.parse_args()

    official = {}
    for item in args.official_eval:
        name, path = _split_named_path(item)
        official[name] = _summarize_official_eval(Path(path))

    cce = _summarize_cce_report(Path(args.cce_report)) if args.cce_report else {}
    summary = {
        "official_factual_metrics": official,
        "cce_counterfactual_metrics": cce,
        "metric_boundary": (
            "Official MobilityBench metrics are computed on factual official "
            "raw_results. CCE metrics are computed on post-hoc counterfactual "
            "recovery records and should be reported as dynamic recovery metrics."
        ),
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_render_markdown(summary), encoding="utf-8")
    return 0


def _split_named_path(item: str) -> tuple[str, str]:
    if "=" not in item:
        path = Path(item)
        return path.parent.name or path.stem, item
    name, path = item.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"invalid NAME=PATH argument: {item}")
    return name, path


def _summarize_official_eval(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    for metric_name, models in data.items():
        if not isinstance(models, dict):
            continue
        for model_name, payload in models.items():
            summary = payload.get("summary") if isinstance(payload, dict) else None
            if not isinstance(summary, dict):
                continue
            row = rows.setdefault(
                model_name,
                {
                    "model": model_name,
                    "metrics": {},
                },
            )
            row["metrics"][metric_name] = {
                "average_score": _round(summary.get("average_score")),
                "total_cases": summary.get("total_cases"),
                "successful_cases": summary.get("successful_cases"),
                "failed_cases": summary.get("failed_cases"),
                "sub_scores": {
                    key: _round(value) for key, value in (summary.get("sub_scores") or {}).items()
                },
            }
    return {
        "path": str(path),
        "models": rows,
    }


def _summarize_cce_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "runs" in data:
        return {
            "path": str(path),
            "matrix_name": data.get("matrix_name", ""),
            "runs": {
                run.get("name", run.get("experiment_name", "")): _cce_metric_subset(
                    run.get("metrics", {})
                )
                for run in data.get("runs", [])
            },
        }
    return {
        "path": str(path),
        "experiment_name": data.get("experiment_name", ""),
        "metrics": _cce_metric_subset(data.get("metrics", {})),
    }


def _cce_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "count",
        "counterfactual_recovery_avg",
        "counterfactual_low_score_rate",
        "counterfactual_final_pass_rate",
        "counterfactual_hard_pass_rate",
        "counterfactual_llm_error_rate",
        "agent_owned_retrieval_enabled",
        "agent_owned_retrieval_selected_case_count",
        "agent_owned_retrieval_no_skill_rate",
        "agent_owned_retrieval_unique_selected_skill_ids",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MobilityBench Official And CCE Metric Summary",
        "",
        summary["metric_boundary"],
        "",
        "## Official Factual Metrics",
        "",
    ]
    for run_name, run in summary["official_factual_metrics"].items():
        lines.append(f"### {run_name}")
        for model_name, model in run.get("models", {}).items():
            lines.append(f"- model: `{model_name}`")
            for metric_name, metric in model.get("metrics", {}).items():
                lines.append(
                    f"- {metric_name}: avg={metric.get('average_score')}, "
                    f"cases={metric.get('total_cases')}, "
                    f"failed={metric.get('failed_cases')}"
                )
                sub_scores = metric.get("sub_scores") or {}
                if sub_scores:
                    rendered = ", ".join(f"{k}={v}" for k, v in sub_scores.items())
                    lines.append(f"  - sub_scores: {rendered}")
        lines.append("")

    cce = summary.get("cce_counterfactual_metrics") or {}
    if cce:
        lines.extend(["## CCE Counterfactual Recovery Metrics", ""])
        for run_name, metrics in (cce.get("runs") or {}).items():
            rendered = ", ".join(f"{k}={v}" for k, v in metrics.items())
            lines.append(f"- {run_name}: {rendered}")
        if cce.get("metrics"):
            rendered = ", ".join(f"{k}={v}" for k, v in cce["metrics"].items())
            lines.append(f"- {cce.get('experiment_name', 'run')}: {rendered}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
