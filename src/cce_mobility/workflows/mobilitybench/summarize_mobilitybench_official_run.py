from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.integrations.mobilitybench_official_audit import (  # noqa: E402
    official_raw_dirty_reasons,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize official MobilityBench raw_results.json for CCE integration."
    )
    parser.add_argument("run_dir", help="Official MobilityBench run output directory.")
    parser.add_argument("--model", default="", help="Model subdirectory to inspect.")
    parser.add_argument("--output", default="", help="Optional JSON summary output path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    model_dirs = _model_dirs(run_dir, args.model)
    if not model_dirs:
        raise FileNotFoundError(f"no model raw_results.json found under: {run_dir}")

    summary = {
        "run_dir": str(run_dir),
        "models": {},
    }
    for model_dir in model_dirs:
        raw_path = model_dir / "raw_results.json"
        rows = json.loads(raw_path.read_text(encoding="utf-8"))
        summary["models"][model_dir.name] = _summarize_rows(rows, raw_path)

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


def _model_dirs(run_dir: Path, model: str) -> list[Path]:
    if model:
        return [run_dir / model] if (run_dir / model / "raw_results.json").exists() else []
    return sorted(path for path in run_dir.iterdir() if (path / "raw_results.json").exists())


def _summarize_rows(rows: list[dict[str, Any]], raw_path: Path) -> dict[str, Any]:
    errors = [row for row in rows if row.get("error")]
    planner_present = sum(bool(row.get("planner_output")) for row in rows)
    worker_counts = [_safe_len(row.get("worker_output")) for row in rows]
    reporter_present = sum(bool(str(row.get("reporter_output") or "").strip()) for row in rows)
    first_success = next((row for row in rows if not row.get("error")), None)
    error_types = Counter(str(row.get("error") or "")[:200] for row in errors)
    dirty = []
    dirty_reasons = {}
    for row in rows:
        reasons = _dirty_reasons(row)
        if reasons:
            case_id = str(row.get("case_id") or "")
            dirty.append(case_id)
            dirty_reasons[case_id] = reasons
    return {
        "raw_results_path": str(raw_path),
        "count": len(rows),
        "success_count": len(rows) - len(errors),
        "error_count": len(errors),
        "dirty_case_count": len(dirty),
        "dirty_case_ids": dirty,
        "dirty_case_reasons": dirty_reasons,
        "planner_output_present": planner_present,
        "worker_output_avg_len": round(sum(worker_counts) / len(worker_counts), 4)
        if worker_counts
        else 0.0,
        "reporter_output_present": reporter_present,
        "error_examples": [
            {"error": error, "count": count} for error, count in error_types.most_common(5)
        ],
        "first_success_preview": _preview_row(first_success) if first_success else None,
    }


def _safe_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _dirty_reasons(row: dict[str, Any]) -> list[str]:
    return official_raw_dirty_reasons(row)


def _preview_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "model_name": row.get("model_name"),
        "planner_keys": sorted((row.get("planner_output") or {}).keys()),
        "worker_output_len": _safe_len(row.get("worker_output")),
        "reporter_output_preview": str(row.get("reporter_output") or "")[:500],
        "token_usage": row.get("token_usage") or {},
    }


if __name__ == "__main__":
    raise SystemExit(main())
