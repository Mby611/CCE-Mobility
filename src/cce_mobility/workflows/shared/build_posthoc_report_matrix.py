from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.analysis.tables import write_matrix_tables  # noqa: E402
from cce_mobility.runners.matrix import _comparison_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble an existing set of posthoc report.json files into a matrix_report.json "
            "without rerunning the experiments."
        ),
    )
    parser.add_argument("--matrix-name", required=True)
    parser.add_argument("--output-dir", default="runs/matrices")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=REPORT_JSON",
        help="Run name and path to its report.json. Repeat for each run.",
    )
    args = parser.parse_args()

    matrix_output = _resolve_project_path(Path(args.output_dir)) / args.matrix_name
    matrix_output.mkdir(parents=True, exist_ok=True)

    reports = [_load_named_report(spec) for spec in args.run]
    summary: dict[str, Any] = {
        "matrix_name": args.matrix_name,
        "runs": reports,
        "comparison": _comparison_table(reports),
    }
    report_path = matrix_output / "matrix_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["tables"] = write_matrix_tables(report_path)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_named_report(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise SystemExit(f"--run must use NAME=REPORT_JSON, got: {spec}")
    name, raw_path = spec.split("=", 1)
    if not name.strip():
        raise SystemExit(f"Empty run name in --run: {spec}")
    report_path = _resolve_project_path(Path(raw_path))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records_path = Path(str(report.get("records_path", "")))
    if not records_path.is_absolute():
        records_path = (report_path.parent / records_path.name).resolve()
        if not records_path.exists():
            records_path = (PROJECT_ROOT / str(report.get("records_path", ""))).resolve()
    return {
        "name": name.strip(),
        "experiment_name": report.get("experiment_name", ""),
        "benchmark": report.get("benchmark", ""),
        "method": report.get("method", ""),
        "total_samples": report.get("total_samples", ""),
        "metrics": report.get("metrics", {}),
        "records_path": str(records_path),
    }


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
