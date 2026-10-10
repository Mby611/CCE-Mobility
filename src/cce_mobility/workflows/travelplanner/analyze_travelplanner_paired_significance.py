from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cce_mobility.workflows.shared.analyze_paired_ablation import (
    _counterfactual_llm_error,
    _load_counterfactual_records,
    _paired_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PairedJob:
    name: str
    records_path: Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one primary TravelPlanner method against multiple baselines "
            "using shared counterfactual cases, bootstrap confidence intervals, "
            "and sign tests."
        )
    )
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="Job in NAME=PATH_TO_RECORDS_JSONL form. Can be repeated.",
    )
    parser.add_argument(
        "--primary",
        required=True,
        help="Primary method name from --job, e.g. cce_mobility.",
    )
    parser.add_argument(
        "--compare-to",
        action="append",
        default=[],
        help="Comparator method name from --job. Defaults to every job except --primary.",
    )
    parser.add_argument("--low-score-threshold", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--exclude-llm-errors",
        action="store_true",
        help="Exclude the union of paired cases with a counterfactual Agent LLM error.",
    )
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")
    parser.add_argument("--output-md", default="", help="Optional markdown output path.")
    args = parser.parse_args()

    jobs = [_parse_job(raw_job) for raw_job in args.job]
    by_name = {job.name: job for job in jobs}
    if args.primary not in by_name:
        raise SystemExit(f"Primary job not found: {args.primary}")
    compare_to = args.compare_to or [job.name for job in jobs if job.name != args.primary]
    missing = [name for name in compare_to if name not in by_name]
    if missing:
        raise SystemExit(f"Comparator job(s) not found: {', '.join(missing)}")

    records = {
        name: _load_counterfactual_records(job.records_path) for name, job in by_name.items()
    }
    comparisons = []
    for comparator in compare_to:
        comparison = _paired_comparison(
            left_name=comparator,
            right_name=args.primary,
            left=records[comparator],
            right=records[args.primary],
            low_score_threshold=args.low_score_threshold,
            bootstrap_samples=args.bootstrap_samples,
            exclude_llm_errors=args.exclude_llm_errors,
        )
        comparison["overall"].update(
            _sample_cluster_bootstrap(
                left=records[comparator],
                right=records[args.primary],
                samples=args.bootstrap_samples,
                exclude_llm_errors=args.exclude_llm_errors,
            )
        )
        comparison["evidence_label"] = _evidence_label(comparison["overall"])
        comparisons.append(comparison)

    report = {
        "primary": args.primary,
        "compare_to": compare_to,
        "low_score_threshold": args.low_score_threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "exclude_llm_errors": args.exclude_llm_errors,
        "jobs": {name: str(job.records_path) for name, job in by_name.items()},
        "comparisons": comparisons,
        "interpretation_boundary": (
            "If a bootstrap confidence interval crosses zero, phrase the result "
            "as a point-estimate difference or trade-off rather than a statistically "
            "conclusive win. Prefer the sample-clustered interval for paper claims."
        ),
    }

    markdown = to_markdown(report)
    if args.output_json:
        output_json = _resolve_path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.output_md:
        output_md = _resolve_path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def _parse_job(raw_job: str) -> PairedJob:
    if "=" not in raw_job:
        raise SystemExit(f"Invalid --job value, expected NAME=PATH: {raw_job}")
    name, raw_path = raw_job.split("=", 1)
    return PairedJob(name=name.strip(), records_path=_resolve_path(raw_path.strip()))


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _evidence_label(overall: dict[str, Any]) -> str:
    ci_low = float(overall.get("bootstrap_ci_low") or 0.0)
    ci_high = float(overall.get("bootstrap_ci_high") or 0.0)
    if ci_low > 0:
        return "positive_ci"
    if ci_high < 0:
        return "negative_ci"
    return "inconclusive_ci"


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TravelPlanner Paired Significance Analysis",
        "",
        f"- primary: `{report['primary']}`",
        f"- low_score_threshold: `{report['low_score_threshold']}`",
        f"- bootstrap_samples: `{report['bootstrap_samples']}`",
        f"- exclude_llm_errors: `{report['exclude_llm_errors']}`",
        f"- interpretation_boundary: {report['interpretation_boundary']}",
        "",
        "## Overall",
        "",
        (
            "| Comparator | Primary | Common Cases | Comparator Avg | Primary Avg | "
            "Mean Delta | Case Bootstrap 95% CI | Sample Clusters | "
            "Sample-Clustered 95% CI | Improved | Worse | Same | "
            "Sign-Test p | Low-Score Delta | Unavailable Delta | Final-Pass Delta | Evidence |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for comparison in report["comparisons"]:
        overall = comparison["overall"]
        lines.append(
            "| {left} | {right} | {cases} | {left_avg} | {right_avg} | {delta} | "
            "[{ci_low}, {ci_high}] | {clusters} | [{cluster_low}, {cluster_high}] | "
            "{improved} | {worse} | {same} | {p} | "
            "{low_delta} | {unavailable_delta} | {final_delta} | {label} |".format(
                left=comparison["left"],
                right=comparison["right"],
                cases=overall["common_cases"],
                left_avg=_fmt(overall["left_avg"]),
                right_avg=_fmt(overall["right_avg"]),
                delta=_fmt(overall["mean_delta"]),
                ci_low=_fmt(overall["bootstrap_ci_low"]),
                ci_high=_fmt(overall["bootstrap_ci_high"]),
                clusters=overall["sample_cluster_count"],
                cluster_low=_fmt(overall["sample_cluster_bootstrap_ci_low"]),
                cluster_high=_fmt(overall["sample_cluster_bootstrap_ci_high"]),
                improved=overall["improved"],
                worse=overall["worse"],
                same=overall["same"],
                p=_fmt(overall["sign_test_p_value"]),
                low_delta=_fmt(overall["low_score_rate_delta"]),
                unavailable_delta=_fmt(overall["unavailable_usage_rate_delta"]),
                final_delta=_fmt(overall["final_pass_rate_delta"]),
                label=comparison["evidence_label"],
            )
        )
    lines.append("")

    for comparison in report["comparisons"]:
        lines.extend(_detail_markdown(comparison))
    return "\n".join(lines) + "\n"


def _sample_cluster_bootstrap(
    *,
    left: dict[tuple[str, str], dict[str, Any]],
    right: dict[tuple[str, str], dict[str, Any]],
    samples: int,
    exclude_llm_errors: bool,
) -> dict[str, Any]:
    common_keys = sorted(set(left) & set(right))
    if exclude_llm_errors:
        common_keys = [
            key
            for key in common_keys
            if not _counterfactual_llm_error(left[key])
            and not _counterfactual_llm_error(right[key])
        ]
    by_sample: dict[str, list[float]] = {}
    for key in common_keys:
        left_score = float((left[key].get("counterfactual_score") or {}).get("total") or 0.0)
        right_score = float((right[key].get("counterfactual_score") or {}).get("total") or 0.0)
        by_sample.setdefault(key[0], []).append(right_score - left_score)

    cluster_ids = sorted(by_sample)
    if not cluster_ids:
        return {
            "sample_cluster_count": 0,
            "sample_cluster_bootstrap_ci_low": 0.0,
            "sample_cluster_bootstrap_ci_high": 0.0,
        }
    all_deltas = [delta for cluster_id in cluster_ids for delta in by_sample[cluster_id]]
    if samples <= 0:
        mean = sum(all_deltas) / len(all_deltas)
        return {
            "sample_cluster_count": len(cluster_ids),
            "sample_cluster_bootstrap_ci_low": round(mean, 4),
            "sample_cluster_bootstrap_ci_high": round(mean, 4),
        }

    rng = random.Random(29)
    means = []
    cluster_count = len(cluster_ids)
    for _ in range(samples):
        sampled = [cluster_ids[rng.randrange(cluster_count)] for _ in range(cluster_count)]
        deltas = [delta for cluster_id in sampled for delta in by_sample[cluster_id]]
        means.append(sum(deltas) / len(deltas))
    means.sort()
    low_index = min(len(means) - 1, max(0, int(0.025 * len(means))))
    high_index = min(len(means) - 1, max(0, int(0.975 * len(means)) - 1))
    return {
        "sample_cluster_count": cluster_count,
        "sample_cluster_bootstrap_ci_low": round(means[low_index], 4),
        "sample_cluster_bootstrap_ci_high": round(means[high_index], 4),
    }


def _detail_markdown(comparison: dict[str, Any]) -> list[str]:
    title = f"{comparison['right']} vs {comparison['left']}"
    lines = [
        f"## {title}",
        "",
    ]
    audit = comparison.get("llm_error_filter")
    if audit:
        lines.extend(
            [
                (
                    "- LLM-error filter: common before `{common_before_filter}`, "
                    "left errors `{left_error_cases}`, right errors `{right_error_cases}`, "
                    "excluded union `{excluded_union_cases}`, paired after `{paired_after_filter}`."
                ).format(**audit),
                "",
            ]
        )
    lines.extend(
        [
            "### By Dimension",
            "",
            _group_table(comparison["by_dimension"]),
            "",
            "### By Capability",
            "",
            _group_table(comparison["by_capability"]),
            "",
        ]
    )
    return lines


def _group_table(rows: list[dict[str, Any]]) -> str:
    fields = [
        "group",
        "common_cases",
        "mean_delta",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "improved",
        "worse",
        "same",
        "low_score_rate_delta",
        "unavailable_usage_rate_delta",
        "final_pass_rate_delta",
    ]
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
