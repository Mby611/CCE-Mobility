from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.benchmarks.travelplanner import TravelPlannerAdapter  # noqa: E402
from cce_mobility.core.config import BenchmarkConfig  # noqa: E402
from cce_mobility.core.schema import InterventionCandidate  # noqa: E402


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "travelplanner"
DEFAULT_REPO_PATH = PROJECT_ROOT / "third_party" / "TravelPlanner"


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize CCE tasks for official SkillOpt.")
    parser.add_argument("--train-interventions", required=True)
    parser.add_argument("--val-interventions", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/skillopt/travelplanner_dataset",
    )
    parser.add_argument("--repo-path", default=str(DEFAULT_REPO_PATH))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    repo_path = Path(args.repo_path).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"TravelPlanner repository not found: {repo_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"TravelPlanner data root not found: {data_root}")
    counts = {}
    for split, intervention_path in (
        ("train", Path(args.train_interventions)),
        ("val", Path(args.val_interventions)),
    ):
        items = _build_items(
            split,
            intervention_path,
            repo_path=repo_path,
            data_root=data_root,
        )
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "items.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        counts[split] = len(items)

    test_dir = output_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "items.json").write_text("[]\n", encoding="utf-8")
    counts["test"] = 0
    summary = {
        "output_dir": str(output_dir),
        "counts": counts,
        "train_interventions": args.train_interventions,
        "val_interventions": args.val_interventions,
        "repo_path": str(repo_path),
        "data_root": str(data_root),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_items(
    split: str,
    intervention_path: Path,
    *,
    repo_path: Path = DEFAULT_REPO_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> list[dict]:
    travelplanner_split = "validation" if split == "val" else split
    benchmark = TravelPlannerAdapter(
        BenchmarkConfig(
            name="travelplanner",
            repo_path=str(repo_path),
            data_path=str(data_root / f"{travelplanner_split}_ref_info.jsonl"),
            split=travelplanner_split,
            extra={"planner_mode": "adaptive_rule"},
        )
    )
    samples = {sample.sample_id: sample for sample in benchmark.load_samples()}
    items = []
    with intervention_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = InterventionCandidate.model_validate(json.loads(line))
            sample = samples.get(candidate.spec.sample_id)
            if sample is None or candidate.recovery_spec is None:
                continue
            items.append(
                {
                    "id": f"{sample.sample_id}::{candidate.spec.intervention_id}",
                    "sample": sample.model_dump(mode="json"),
                    "intervention": candidate.spec.model_dump(mode="json"),
                    "recovery_spec": candidate.recovery_spec.model_dump(mode="json"),
                    "task_type": candidate.spec.domain_operator or candidate.spec.dimension,
                }
            )
    if split == "val":
        random.Random(42).shuffle(items)
    return items


if __name__ == "__main__":
    raise SystemExit(main())
