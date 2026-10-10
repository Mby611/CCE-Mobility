from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only paired-delta analysis for frozen ablation matrix reports. "
            "Prints markdown to stdout and does not modify run outputs."
        ),
    )
    parser.add_argument("matrix_report", help="Path to runs/matrices/.../matrix_report.json")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline run name. Defaults to the first run containing no_evolution.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Primary target run name. Defaults to the first run containing ours.",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=2.0,
        help="Threshold used for low-score tail metrics.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of markdown.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of deterministic bootstrap samples for mean-delta confidence intervals.",
    )
    parser.add_argument(
        "--exclude-llm-errors",
        action="store_true",
        help="Exclude a pair if either counterfactual agent call has an LLM error.",
    )
    args = parser.parse_args()

    report_path = Path(args.matrix_report)
    report = _read_json(report_path)
    runs = report.get("runs", [])
    if not runs:
        raise SystemExit(f"No runs found in {report_path}")

    run_records = {
        run["name"]: _load_counterfactual_records(_resolve_records_path(run, report_path))
        for run in runs
    }
    baseline_name = args.baseline or _infer_run_name(runs, contains="no_evolution", fallback=0)
    target_name = args.target or _infer_run_name(runs, contains="ours", fallback=len(runs) - 1)
    if baseline_name not in run_records:
        raise SystemExit(f"Baseline run not found: {baseline_name}")
    if target_name not in run_records:
        raise SystemExit(f"Target run not found: {target_name}")

    comparisons: list[dict[str, Any]] = []
    for run in runs:
        name = run["name"]
        if name == baseline_name:
            continue
        comparisons.append(
            _paired_comparison(
                left_name=baseline_name,
                right_name=name,
                left=run_records[baseline_name],
                right=run_records[name],
                low_score_threshold=args.low_score_threshold,
                bootstrap_samples=args.bootstrap_samples,
                exclude_llm_errors=args.exclude_llm_errors,
            )
        )
    for run in runs:
        name = run["name"]
        if name in {baseline_name, target_name}:
            continue
        comparisons.append(
            _paired_comparison(
                left_name=name,
                right_name=target_name,
                left=run_records[name],
                right=run_records[target_name],
                low_score_threshold=args.low_score_threshold,
                bootstrap_samples=args.bootstrap_samples,
                exclude_llm_errors=args.exclude_llm_errors,
            )
        )

    summary = {
        "matrix_report": str(report_path),
        "baseline": baseline_name,
        "target": target_name,
        "low_score_threshold": args.low_score_threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "exclude_llm_errors": args.exclude_llm_errors,
        "comparisons": comparisons,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(_to_markdown(summary))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_records_path(run: dict[str, Any], matrix_report_path: Path) -> Path:
    path = Path(run.get("records_path", ""))
    if path.is_absolute():
        return path
    candidate = matrix_report_path.parents[2] / path
    if candidate.exists():
        return candidate
    return PROJECT_ROOT / path


def _load_counterfactual_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("counterfactual") or not record.get("counterfactual_score"):
                continue
            sample = record.get("sample") or {}
            intervention = record.get("intervention") or {}
            key = (str(sample.get("sample_id")), str(intervention.get("intervention_id")))
            records[key] = record
    return records


def _infer_run_name(runs: list[dict[str, Any]], *, contains: str, fallback: int) -> str:
    needle = contains.lower()
    for run in runs:
        if needle in str(run.get("name", "")).lower():
            return str(run["name"])
    return str(runs[fallback]["name"])


def _paired_comparison(
    *,
    left_name: str,
    right_name: str,
    left: dict[tuple[str, str], dict[str, Any]],
    right: dict[tuple[str, str], dict[str, Any]],
    low_score_threshold: float,
    bootstrap_samples: int,
    exclude_llm_errors: bool = False,
) -> dict[str, Any]:
    common_keys = sorted(set(left) & set(right))
    error_left = {key for key in common_keys if _counterfactual_llm_error(left[key])}
    error_right = {key for key in common_keys if _counterfactual_llm_error(right[key])}
    error_keys = error_left | error_right
    selected_keys = [key for key in common_keys if not exclude_llm_errors or key not in error_keys]
    pairs = [(_row(left[key]), _row(right[key])) for key in selected_keys]
    overall = _summarize_pairs(
        pairs,
        low_score_threshold=low_score_threshold,
        bootstrap_samples=bootstrap_samples,
    )
    result = {
        "left": left_name,
        "right": right_name,
        "overall": overall,
        "by_dimension": _summarize_grouped(
            pairs,
            key_fn=lambda pair: pair[0]["dimension"],
            low_score_threshold=low_score_threshold,
            bootstrap_samples=bootstrap_samples,
        ),
        "by_strategy": _summarize_grouped(
            pairs,
            key_fn=lambda pair: pair[1]["strategy"],
            low_score_threshold=low_score_threshold,
            bootstrap_samples=bootstrap_samples,
        ),
        "by_strategy_transition": _summarize_grouped(
            pairs,
            key_fn=lambda pair: f"{pair[0]['strategy']}->{pair[1]['strategy']}",
            low_score_threshold=low_score_threshold,
            bootstrap_samples=bootstrap_samples,
        ),
        "by_capability": _summarize_capabilities(
            pairs,
            low_score_threshold=low_score_threshold,
            bootstrap_samples=bootstrap_samples,
        ),
    }
    if exclude_llm_errors:
        result["llm_error_filter"] = {
            "common_before_filter": len(common_keys),
            "left_error_cases": len(error_left),
            "right_error_cases": len(error_right),
            "excluded_union_cases": len(error_keys),
            "paired_after_filter": len(selected_keys),
        }
    return result


def _counterfactual_llm_error(record: dict[str, Any]) -> bool:
    hard = (record.get("metadata") or {}).get("counterfactual_hard_metrics") or {}
    raw = (record.get("counterfactual") or {}).get("raw") or {}
    return bool(hard.get("llm_error") or raw.get("llm_error"))


def _row(record: dict[str, Any]) -> dict[str, Any]:
    score = record.get("counterfactual_score") or {}
    intervention = record.get("intervention") or {}
    hard = record.get("metadata", {}).get("counterfactual_hard_metrics", {}) or {}
    capabilities = intervention.get("capability_targets") or ["unknown"]
    return {
        "score": float(score.get("total") or 0.0),
        "strategy": str(score.get("strategy") or "unknown"),
        "dimension": str(intervention.get("dimension") or "unknown"),
        "capabilities": [str(item) for item in capabilities],
        "uses_unavailable": bool(hard.get("uses_unavailable_identifier")),
        "has_unavailable": int(hard.get("unavailable_id_count") or 0) > 0,
        "final_pass": bool(hard.get("final_pass")),
    }


def _summarize_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    low_score_threshold: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not pairs:
        return {
            "common_cases": 0,
            "left_avg": 0.0,
            "right_avg": 0.0,
            "mean_delta": 0.0,
            "bootstrap_ci_low": 0.0,
            "bootstrap_ci_high": 0.0,
            "improved": 0,
            "worse": 0,
            "same": 0,
            "changed_cases": 0,
            "sign_test_p_value": 1.0,
            "left_low_score_rate": 0.0,
            "right_low_score_rate": 0.0,
            "low_score_rate_delta": 0.0,
            "left_unavailable_usage_rate": 0.0,
            "right_unavailable_usage_rate": 0.0,
            "unavailable_usage_rate_delta": 0.0,
            "left_unavailable_usage_rate_all_cases": 0.0,
            "right_unavailable_usage_rate_all_cases": 0.0,
            "unavailable_usage_rate_all_cases_delta": 0.0,
            "left_final_pass_rate": 0.0,
            "right_final_pass_rate": 0.0,
            "final_pass_rate_delta": 0.0,
        }
    left_scores = [left["score"] for left, _ in pairs]
    right_scores = [right["score"] for _, right in pairs]
    deltas = [right - left for left, right in zip(left_scores, right_scores, strict=True)]
    improved = sum(1 for delta in deltas if delta > 0)
    worse = sum(1 for delta in deltas if delta < 0)
    same = sum(1 for delta in deltas if delta == 0)
    changed = improved + worse
    ci_low, ci_high = _bootstrap_mean_ci(deltas, samples=bootstrap_samples)
    left_low = _rate(score < low_score_threshold for score in left_scores)
    right_low = _rate(score < low_score_threshold for score in right_scores)
    left_unavailable_all = _rate(left["uses_unavailable"] for left, _ in pairs)
    right_unavailable_all = _rate(right["uses_unavailable"] for _, right in pairs)
    applicable_pairs = [
        (left, right)
        for left, right in pairs
        if left["has_unavailable"] or right["has_unavailable"]
    ]
    left_unavailable = _rate(left["uses_unavailable"] for left, _ in applicable_pairs)
    right_unavailable = _rate(right["uses_unavailable"] for _, right in applicable_pairs)
    left_final = _rate(left["final_pass"] for left, _ in pairs)
    right_final = _rate(right["final_pass"] for _, right in pairs)
    return {
        "common_cases": len(pairs),
        "left_avg": _avg(left_scores),
        "right_avg": _avg(right_scores),
        "mean_delta": _avg(deltas),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "improved": improved,
        "worse": worse,
        "same": same,
        "changed_cases": changed,
        "sign_test_p_value": _sign_test_p_value(improved, worse),
        "left_low_score_rate": left_low,
        "right_low_score_rate": right_low,
        "low_score_rate_delta": round(right_low - left_low, 4),
        "left_unavailable_usage_rate": left_unavailable,
        "right_unavailable_usage_rate": right_unavailable,
        "unavailable_usage_rate_delta": round(right_unavailable - left_unavailable, 4),
        "left_unavailable_usage_rate_all_cases": left_unavailable_all,
        "right_unavailable_usage_rate_all_cases": right_unavailable_all,
        "unavailable_usage_rate_all_cases_delta": round(
            right_unavailable_all - left_unavailable_all,
            4,
        ),
        "left_final_pass_rate": left_final,
        "right_final_pass_rate": right_final,
        "final_pass_rate_delta": round(right_final - left_final, 4),
    }


def _summarize_grouped(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    key_fn,
    low_score_threshold: float,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(key_fn(pair))].append(pair)
    return [
        {
            "group": key,
            **_summarize_pairs(
                rows,
                low_score_threshold=low_score_threshold,
                bootstrap_samples=bootstrap_samples,
            ),
        }
        for key, rows in sorted(grouped.items())
    ]


def _summarize_capabilities(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    low_score_threshold: float,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for left, right in pairs:
        for capability in sorted(set(left["capabilities"]) | set(right["capabilities"])):
            grouped[capability].append((left, right))
    return [
        {
            "group": key,
            **_summarize_pairs(
                rows,
                low_score_threshold=low_score_threshold,
                bootstrap_samples=bootstrap_samples,
            ),
        }
        for key, rows in sorted(grouped.items())
    ]


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _rate(values) -> float:
    items = list(values)
    return round(sum(1 for item in items if item) / len(items), 4) if items else 0.0


def _sign_test_p_value(improved: int, worse: int) -> float:
    n = improved + worse
    if n == 0:
        return 1.0
    tail = min(improved, worse)
    probability = _binomial_half_cdf(n, tail)
    return round(min(1.0, 2 * probability), 6)


def _binomial_half_cdf(n: int, tail: int) -> float:
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)
        for k in range(tail + 1)
    ]
    max_log = max(log_terms)
    return math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)


def _bootstrap_mean_ci(values: list[float], *, samples: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if samples <= 0:
        mean = _avg(values)
        return mean, mean
    rng = random.Random(13)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low_index = min(len(means) - 1, max(0, int(0.025 * len(means))))
    high_index = min(len(means) - 1, max(0, int(0.975 * len(means)) - 1))
    return round(means[low_index], 4), round(means[high_index], 4)


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Paired Ablation Analysis",
        "",
        f"- Matrix report: `{summary['matrix_report']}`",
        f"- Baseline: `{summary['baseline']}`",
        f"- Primary target: `{summary['target']}`",
        f"- Low-score threshold: `{summary['low_score_threshold']}`",
        f"- Bootstrap samples: `{summary['bootstrap_samples']}`",
        f"- Exclude pairs with counterfactual LLM errors: `{summary['exclude_llm_errors']}`",
        "",
    ]
    for comparison in summary["comparisons"]:
        lines.extend(_comparison_markdown(comparison))
    return "\n".join(lines)


def _comparison_markdown(comparison: dict[str, Any]) -> list[str]:
    left = comparison["left"]
    right = comparison["right"]
    lines = [
        f"## {right} vs {left}",
        "",
    ]
    if "llm_error_filter" in comparison:
        audit = comparison["llm_error_filter"]
        lines.extend(
            [
                f"LLM error filter: {audit['common_before_filter']} common before, "
                f"{audit['left_error_cases']} left error, {audit['right_error_cases']} right error, "
                f"{audit['excluded_union_cases']} excluded in union, "
                f"{audit['paired_after_filter']} paired after.",
                "",
            ]
        )
    lines.extend(
        [
            _table(
                [
                    "common_cases",
                    "left_avg",
                    "right_avg",
                    "mean_delta",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "improved",
                    "worse",
                    "same",
                    "changed_cases",
                    "sign_test_p_value",
                    "left_low_score_rate",
                    "right_low_score_rate",
                    "low_score_rate_delta",
                    "left_unavailable_usage_rate",
                    "right_unavailable_usage_rate",
                    "unavailable_usage_rate_delta",
                    "left_final_pass_rate",
                    "right_final_pass_rate",
                    "final_pass_rate_delta",
                ],
                [comparison["overall"]],
            ),
            "",
            "### By Dimension",
            "",
            _table(
                [
                    "group",
                    "common_cases",
                    "mean_delta",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "improved",
                    "worse",
                    "same",
                    "changed_cases",
                    "sign_test_p_value",
                    "low_score_rate_delta",
                    "unavailable_usage_rate_delta",
                    "final_pass_rate_delta",
                ],
                comparison["by_dimension"],
            ),
            "",
            "### By Target Strategy",
            "",
            _table(
                [
                    "group",
                    "common_cases",
                    "mean_delta",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "improved",
                    "worse",
                    "same",
                    "changed_cases",
                    "sign_test_p_value",
                    "low_score_rate_delta",
                    "unavailable_usage_rate_delta",
                    "final_pass_rate_delta",
                ],
                comparison["by_strategy"],
            ),
            "",
            "### By Strategy Transition",
            "",
            _table(
                [
                    "group",
                    "common_cases",
                    "mean_delta",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "improved",
                    "worse",
                    "same",
                    "changed_cases",
                    "sign_test_p_value",
                    "low_score_rate_delta",
                    "unavailable_usage_rate_delta",
                    "final_pass_rate_delta",
                ],
                comparison["by_strategy_transition"],
            ),
            "",
            "### By Capability",
            "",
            _table(
                [
                    "group",
                    "common_cases",
                    "mean_delta",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "improved",
                    "worse",
                    "same",
                    "changed_cases",
                    "sign_test_p_value",
                    "low_score_rate_delta",
                    "unavailable_usage_rate_delta",
                    "final_pass_rate_delta",
                ],
                comparison["by_capability"],
            ),
            "",
        ]
    )
    return lines


def _table(fields: list[str], rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    raise SystemExit(main())
