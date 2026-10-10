from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.analysis.tables import write_matrix_tables  # noqa: E402
from cce_mobility.core.config import ExperimentConfig  # noqa: E402
from cce_mobility.core.env import load_dotenv  # noqa: E402
from cce_mobility.runners.matrix import (  # noqa: E402
    ExperimentMatrixConfig,
    _comparison_table,
    _deep_merge,
    _load_base_config,
)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_mobilitybench_official_posthoc_matrix.py <matrix-config.json>")

    matrix_path = Path(sys.argv[1])
    summary = run_official_posthoc_matrix(matrix_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_official_posthoc_matrix(path: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    matrix = ExperimentMatrixConfig.from_json(path)
    base = _load_base_config(matrix.base_config, path.parent)
    output_dir = Path(matrix.output_dir) / matrix.matrix_name
    config_dir = output_dir / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for run_spec in matrix.runs:
        data = _deep_merge(copy.deepcopy(base), run_spec.overrides)
        data["experiment_name"] = data.get("experiment_name") or run_spec.name
        cfg = ExperimentConfig.model_validate(data)
        config_path = config_dir / f"{run_spec.name}.json"
        config_path.write_text(
            json.dumps(cfg.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "cce_mobility.workflows.mobilitybench.run_mobilitybench_official_posthoc",
                str(config_path),
            ],
            check=True,
        )
        report_path = Path(cfg.output_dir) / cfg.experiment_name / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports.append(
            {
                "name": run_spec.name,
                "experiment_name": report.get("experiment_name", cfg.experiment_name),
                "benchmark": report.get("benchmark", cfg.benchmark.name),
                "method": report.get("method", cfg.method),
                "total_samples": report.get("total_samples", ""),
                "metrics": report.get("metrics", {}),
                "records_path": str(Path(report.get("records_path", "")).resolve()),
            }
        )

    summary = {
        "matrix_name": matrix.matrix_name,
        "runs": reports,
        "comparison": _comparison_table(reports),
    }
    report_path = output_dir / "matrix_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["tables"] = write_matrix_tables(report_path)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
