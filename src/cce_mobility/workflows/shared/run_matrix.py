from __future__ import annotations

# ruff: noqa: E402

import argparse
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.core.env import load_dotenv
from cce_mobility.runners.matrix import run_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CCE experiment matrix.")
    parser.add_argument("matrix_config")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Skip complete clean runs and resume partial or retryable runs.",
    )
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    summary = run_matrix(
        Path(args.matrix_config),
        resume_existing=args.resume_existing,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
