from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.core.env import load_dotenv
from cce_mobility.core.config import ExperimentConfig
from cce_mobility.runners.evaluation import EvaluationRunner


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT / "configs" / "smoke" / "travelplanner_offline.json"
    )
    cfg = ExperimentConfig.from_yaml(config_path)
    report = EvaluationRunner(cfg).run()
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
