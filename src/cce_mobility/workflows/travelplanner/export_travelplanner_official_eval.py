from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.benchmarks.travelplanner_eval import parse_plan  # noqa: E402


SPLIT_SIZES = {
    "train": 45,
    "validation": 180,
    "test": 1000,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export factual TravelPlanner plans from CCE records to the official "
            "TravelPlanner evaluation JSONL format."
        )
    )
    parser.add_argument("--records", required=True, help="CCE records.jsonl path.")
    parser.add_argument("--output", required=True, help="Official-format JSONL output path.")
    parser.add_argument("--set-type", default="validation", choices=sorted(SPLIT_SIZES))
    parser.add_argument(
        "--repo-path",
        default="third_party/TravelPlanner",
        help="TravelPlanner repository path.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Export only available samples instead of padding missing split entries with null plans.",
    )
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Run TravelPlanner/evaluation/eval.py after exporting.",
    )
    args = parser.parse_args()

    records_path = Path(args.records)
    output_path = Path(args.output)
    records = _load_jsonl(records_path)
    plans = _first_factual_plan_by_index(records)
    split_size = SPLIT_SIZES[args.set_type]
    max_index = max(plans) if plans else -1
    export_size = max_index + 1 if args.allow_partial else split_size

    if export_size > split_size:
        raise ValueError(
            f"Records contain index {max_index}, which exceeds {args.set_type} split size {split_size}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample_idx in range(export_size):
            item = plans.get(sample_idx)
            if item is None:
                row = {
                    "idx": sample_idx + 1,
                    "query": "",
                    "plan": None,
                    "metadata": {"missing_from_records": True},
                }
            else:
                row = item
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "records": str(records_path),
        "output": str(output_path),
        "set_type": args.set_type,
        "expected_split_size": split_size,
        "exported_rows": export_size,
        "available_factual_plans": len(plans),
        "missing_rows": export_size - len(plans),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.run_official:
        eval_script = Path(args.repo_path) / "evaluation" / "eval.py"
        if not eval_script.exists():
            raise FileNotFoundError(f"TravelPlanner official evaluator not found: {eval_script}")
        proc = subprocess.run(
            [
                sys.executable,
                str(eval_script.name),
                "--set_type",
                args.set_type,
                "--evaluation_file_path",
                str(output_path.resolve()),
            ],
            cwd=str(eval_script.parent),
            check=False,
            text=True,
        )
        return proc.returncode
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first_factual_plan_by_index(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    plans: dict[int, dict[str, Any]] = {}
    for record in records:
        sample = record.get("sample") or {}
        sample_idx = _sample_index(sample)
        if sample_idx is None or sample_idx in plans:
            continue
        factual = record.get("factual") or {}
        final_response = str(factual.get("final_response") or "").strip()
        plans[sample_idx] = {
            "idx": sample_idx + 1,
            "query": str(sample.get("query") or ""),
            "plan": _official_plan(final_response),
            "metadata": {
                "sample_id": sample.get("sample_id"),
                "source": "cce_factual_final_response",
                "parse_success": bool(parse_plan(final_response).parse_success),
            },
        }
    return dict(sorted(plans.items()))


def _sample_index(sample: dict[str, Any]) -> int | None:
    metadata = sample.get("metadata") or {}
    if "sample_idx" in metadata:
        try:
            return int(metadata["sample_idx"])
        except (TypeError, ValueError):
            return None
    sample_id = str(sample.get("sample_id") or "")
    if sample_id.startswith("tp_"):
        try:
            return int(sample_id.removeprefix("tp_"))
        except ValueError:
            return None
    return None


def _official_plan(plan_text: str) -> list[dict[str, Any]] | None:
    parsed = parse_plan(plan_text)
    if not parsed.parse_success:
        return None
    plan = []
    for day in parsed.days:
        plan.append(
            {
                "day": day.day_number,
                "current_city": day.current_city
                if not day.from_city
                else f"from {day.from_city} to {day.to_city}",
                "transportation": day.transportation or "-",
                "breakfast": day.breakfast or "-",
                "attraction": _normalize_attractions(day.attraction),
                "lunch": day.lunch or "-",
                "dinner": day.dinner or "-",
                "accommodation": day.accommodation or "-",
            }
        )
    return plan


def _normalize_attractions(value: str) -> str:
    if not value:
        return "-"
    stripped = value.strip()
    if stripped == "-":
        return "-"
    return stripped if stripped.endswith(";") else f"{stripped};"


if __name__ == "__main__":
    raise SystemExit(main())
