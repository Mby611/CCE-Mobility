from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official MobilityBench evaluator on an arbitrary run directory."
    )
    parser.add_argument("--mobilitybench-root", required=True, help="MobilityBench repo root.")
    parser.add_argument("--run-dir", required=True, help="Official mbench run output directory.")
    parser.add_argument("--ground-truth", required=True, help="Dataset CSV/XLSX used by the run.")
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory.")
    parser.add_argument(
        "--metrics",
        default="tool,instruction,planning,answer,efficiency",
        help="Comma-separated official metrics.",
    )
    args = parser.parse_args()

    mb_root = Path(args.mobilitybench_root).resolve()
    sys.path.insert(0, str(mb_root / "src"))

    from mobility_bench.config.settings import Settings  # noqa: E402
    from mobility_bench.evaluation.runner import EvaluationRunner  # noqa: E402

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    settings = Settings.load(mb_root / "configs" / "evaluation" / "default.yaml")
    runner = EvaluationRunner(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir),
        ground_truth_path=Path(args.ground_truth),
        settings=settings,
    )
    results = {}
    for metric in metrics:
        try:
            results[metric] = runner.evaluate_metric(metric)
        except Exception as exc:
            results[metric] = {"error": str(exc)}
    summary = runner.aggregate_results(results)
    runner.save_results(results, summary)

    serializable = {
        "mobilitybench_root": str(mb_root),
        "run_dir": args.run_dir,
        "ground_truth": args.ground_truth,
        "output_dir": args.output_dir,
        "metrics": metrics,
        "summary": summary,
        "result_keys": list(results.keys()),
    }
    print(json.dumps(serializable, ensure_ascii=False, indent=2, default=str))
    return (
        1 if any(isinstance(value, dict) and "error" in value for value in results.values()) else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
