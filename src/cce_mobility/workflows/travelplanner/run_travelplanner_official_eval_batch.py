from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]


METRIC_PATTERNS = {
    "delivery_rate": re.compile(r"^Delivery Rate:\s*([0-9.]+)%", re.MULTILINE),
    "commonsense_micro_pass_rate": re.compile(
        r"^Commonsense Constraint Micro Pass Rate:\s*([0-9.]+)%", re.MULTILINE
    ),
    "commonsense_macro_pass_rate": re.compile(
        r"^Commonsense Constraint Macro Pass Rate:\s*([0-9.]+)%", re.MULTILINE
    ),
    "hard_micro_pass_rate": re.compile(
        r"^Hard Constraint Micro Pass Rate:\s*([0-9.]+)%", re.MULTILINE
    ),
    "hard_macro_pass_rate": re.compile(
        r"^Hard Constraint Macro Pass Rate:\s*([0-9.]+)%", re.MULTILINE
    ),
    "final_pass_rate": re.compile(r"^Final Pass Rate:\s*([0-9.]+)%", re.MULTILINE),
}


@dataclass(frozen=True)
class OfficialEvalJob:
    name: str
    records_path: Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch export CCE TravelPlanner factual records to the official TravelPlanner "
            "evaluation JSONL format, run the official evaluator, and summarize metrics."
        )
    )
    parser.add_argument(
        "--matrix-glob",
        action="append",
        default=[],
        help=(
            "Glob for matrix_report.json files. Relative paths are resolved from the "
            "project root. Can be repeated."
        ),
    )
    parser.add_argument(
        "--matrix-report",
        action="append",
        default=[],
        help="Explicit matrix_report.json path. Can be repeated.",
    )
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="Explicit job in NAME=PATH_TO_RECORDS_JSONL form. Can be repeated.",
    )
    parser.add_argument(
        "--include-run-regex",
        action="append",
        default=[],
        help="Only include matrix runs whose name matches at least one regex.",
    )
    parser.add_argument(
        "--exclude-run-regex",
        action="append",
        default=[],
        help="Exclude matrix runs whose name matches any regex.",
    )
    parser.add_argument("--set-type", default="test", choices=["train", "validation", "test"])
    parser.add_argument(
        "--repo-path",
        default="third_party/TravelPlanner",
        help="Path to the official TravelPlanner repository.",
    )
    parser.add_argument(
        "--hf-dir",
        default="",
        help="Directory containing TravelPlanner_hf CSV files. Passed as TRAVELPLANNER_HF_DIR.",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/official_tp",
        help="Directory for per-run official evaluation logs.",
    )
    parser.add_argument(
        "--summary-output",
        default="runs/official_tp_eval_summary.json",
        help="Path to write the batch summary JSON.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Export only available samples instead of padding missing split entries. "
            "Use this for val20/val60 diagnostic gates."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered jobs and write a dry-run summary without running official eval.",
    )
    args = parser.parse_args()

    jobs = _collect_jobs(args)
    if not jobs:
        raise SystemExit("No official evaluation jobs found.")

    logs_dir = _resolve_path(args.logs_dir)
    summary_output = _resolve_path(args.summary_output)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    print(f"Found {len(jobs)} official TravelPlanner eval job(s).")
    for job in jobs:
        log_path = logs_dir / f"{_safe_name(job.name)}.log"
        output_path = job.records_path.parent / f"official_eval_{args.set_type}.jsonl"
        print(f"- {job.name}: {job.records_path}")
        if args.dry_run:
            summaries.append(
                {
                    "name": job.name,
                    "records_path": str(job.records_path),
                    "official_output": str(output_path),
                    "log_path": str(log_path),
                    "allow_partial": args.allow_partial,
                    "dry_run": True,
                }
            )
            continue

        proc = _run_job(
            job=job,
            output_path=output_path,
            log_path=log_path,
            set_type=args.set_type,
            repo_path=Path(args.repo_path),
            hf_dir=args.hf_dir,
            allow_partial=args.allow_partial,
        )
        metrics = _parse_official_metrics(log_path.read_text(encoding="utf-8", errors="replace"))
        summaries.append(
            {
                "name": job.name,
                "records_path": str(job.records_path),
                "official_output": str(output_path),
                "log_path": str(log_path),
                "returncode": proc.returncode,
                "metrics": metrics,
            }
        )

    report = {
        "set_type": args.set_type,
        "repo_path": args.repo_path,
        "hf_dir": args.hf_dir,
        "allow_partial": args.allow_partial,
        "job_count": len(summaries),
        "runs": summaries,
    }
    summary_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote summary: {summary_output}")
    _print_table(summaries)
    return 0


def _collect_jobs(args: argparse.Namespace) -> list[OfficialEvalJob]:
    include_patterns = [re.compile(pattern) for pattern in args.include_run_regex]
    exclude_patterns = [re.compile(pattern) for pattern in args.exclude_run_regex]
    jobs: dict[Path, OfficialEvalJob] = {}

    for raw_job in args.job:
        name, records_path = _parse_explicit_job(raw_job)
        if not _include_name(name, include_patterns, exclude_patterns):
            continue
        if records_path.exists():
            jobs[records_path.resolve()] = OfficialEvalJob(
                name=name, records_path=records_path.resolve()
            )

    for report_path in _iter_matrix_reports(args):
        for job in _jobs_from_matrix_report(report_path):
            if not _include_name(job.name, include_patterns, exclude_patterns):
                continue
            if job.records_path.exists():
                jobs[job.records_path.resolve()] = OfficialEvalJob(
                    name=job.name,
                    records_path=job.records_path.resolve(),
                )

    return list(jobs.values())


def _iter_matrix_reports(args: argparse.Namespace) -> list[Path]:
    reports: set[Path] = set()
    for raw_path in args.matrix_report:
        path = _resolve_path(raw_path)
        if path.exists():
            reports.add(path.resolve())
    for raw_glob in args.matrix_glob or []:
        raw_path = Path(raw_glob)
        if raw_path.is_absolute():
            matches = raw_path.parent.glob(raw_path.name)
        else:
            matches = PROJECT_ROOT.glob(raw_glob)
        reports.update(path.resolve() for path in matches if path.name == "matrix_report.json")
    if not args.matrix_glob and not args.matrix_report and not args.job:
        reports.update((PROJECT_ROOT / "runs" / "matrices").glob("*test1000*/matrix_report.json"))
    return sorted(reports)


def _jobs_from_matrix_report(report_path: Path) -> list[OfficialEvalJob]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    jobs = []
    for run in report.get("runs", []):
        records_value = run.get("records_path")
        if not records_value:
            continue
        records_path = Path(records_value)
        if not records_path.is_absolute():
            records_path = (PROJECT_ROOT / records_path).resolve()
        name = str(run.get("name") or records_path.parent.name)
        jobs.append(OfficialEvalJob(name=name, records_path=records_path))
    return jobs


def _parse_explicit_job(raw_job: str) -> tuple[str, Path]:
    if "=" not in raw_job:
        raise SystemExit(f"Invalid --job value, expected NAME=PATH: {raw_job}")
    name, raw_path = raw_job.split("=", 1)
    return name.strip(), _resolve_path(raw_path.strip())


def _include_name(
    name: str, include_patterns: list[re.Pattern[str]], exclude_patterns: list[re.Pattern[str]]
) -> bool:
    if include_patterns and not any(pattern.search(name) for pattern in include_patterns):
        return False
    return not any(pattern.search(name) for pattern in exclude_patterns)


def _run_job(
    *,
    job: OfficialEvalJob,
    output_path: Path,
    log_path: Path,
    set_type: str,
    repo_path: Path,
    hf_dir: str,
    allow_partial: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if hf_dir:
        env["TRAVELPLANNER_HF_DIR"] = hf_dir
    command = [
        sys.executable,
        "-m",
        "cce_mobility.workflows.travelplanner.export_travelplanner_official_eval",
        "--records",
        str(job.records_path),
        "--output",
        str(output_path),
        "--set-type",
        set_type,
        "--repo-path",
        str(repo_path),
        "--run-official",
    ]
    if allow_partial:
        command.append("--allow-partial")
    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc


def _parse_official_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def _print_table(summaries: list[dict[str, Any]]) -> None:
    if not summaries or all(not item.get("metrics") for item in summaries):
        return
    columns = [
        "name",
        "delivery_rate",
        "commonsense_micro_pass_rate",
        "commonsense_macro_pass_rate",
        "hard_micro_pass_rate",
        "hard_macro_pass_rate",
        "final_pass_rate",
    ]
    print("\n| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for item in summaries:
        metrics = item.get("metrics") or {}
        values = [str(item.get("name", ""))]
        for column in columns[1:]:
            value = metrics.get(column)
            values.append("" if value is None else f"{value:.4f}")
        print("| " + " | ".join(values) + " |")


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in name)


def _resolve_path(path: str) -> Path:
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw)


if __name__ == "__main__":
    raise SystemExit(main())
