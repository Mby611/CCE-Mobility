from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.core.schema import EvaluationRecord, InterventionCandidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize official MobilityBench trace recovery tasks for Microsoft SkillOpt."
        )
    )
    parser.add_argument("--train-factual-records", required=True)
    parser.add_argument("--train-interventions", required=True)
    parser.add_argument("--val-factual-records", required=True)
    parser.add_argument("--val-interventions", required=True)
    parser.add_argument(
        "--output-dir",
        default="runs/local_data/mb_skillopt_dataset",
    )
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_items, train_summary = _build_items(
        split="train",
        factual_path=Path(args.train_factual_records),
        intervention_path=Path(args.train_interventions),
        limit=args.train_limit,
    )
    val_items, val_summary = _build_items(
        split="val",
        factual_path=Path(args.val_factual_records),
        intervention_path=Path(args.val_interventions),
        limit=args.val_limit,
    )

    for split, items in (("train", train_items), ("val", val_items)):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "items.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    test_dir = output_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "items.json").write_text("[]\n", encoding="utf-8")

    summary = {
        "output_dir": str(output_dir),
        "counts": {
            "train": len(train_items),
            "val": len(val_items),
            "test": 0,
        },
        "train": train_summary,
        "val": val_summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_items(
    *,
    split: str,
    factual_path: Path,
    intervention_path: Path,
    limit: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    factual_records = _load_factual_records(factual_path)
    factual_by_sample = {
        record.sample.sample_id: record for record in factual_records if record.factual is not None
    }
    candidates = _load_intervention_candidates(intervention_path)

    items: list[dict[str, Any]] = []
    skipped_missing_factual = 0
    skipped_missing_recovery_spec = 0
    for candidate in candidates:
        sample_id = candidate.spec.sample_id
        record = factual_by_sample.get(sample_id)
        if record is None or record.factual is None:
            skipped_missing_factual += 1
            continue
        if candidate.recovery_spec is None:
            skipped_missing_recovery_spec += 1
            continue
        items.append(
            {
                "id": f"{sample_id}::{candidate.spec.intervention_id}",
                "sample": record.sample.model_dump(mode="json"),
                "factual": record.factual.model_dump(mode="json"),
                "intervention": candidate.spec.model_dump(mode="json"),
                "recovery_spec": candidate.recovery_spec.model_dump(mode="json"),
                "task_type": candidate.spec.domain_operator
                or candidate.spec.dimension
                or "mobility_recovery",
                "source_split": split,
                "metadata": {
                    "factual_records_path": str(factual_path),
                    "intervention_path": str(intervention_path),
                    "frozen_generation_experiment": candidate.spec.metadata.get(
                        "frozen_generation_experiment"
                    ),
                    "llm_wm_selected_ontology_id": candidate.spec.metadata.get(
                        "llm_wm_selected_ontology_id"
                    ),
                    "llm_wm_selected_spec_id": candidate.spec.metadata.get(
                        "llm_wm_selected_spec_id"
                    ),
                },
            }
        )
    if limit > 0:
        items = items[:limit]
    summary = {
        "split": split,
        "factual_records_path": str(factual_path),
        "intervention_path": str(intervention_path),
        "loaded_factual_records": len(factual_records),
        "loaded_candidates": len(candidates),
        "materialized_items": len(items),
        "skipped_missing_factual": skipped_missing_factual,
        "skipped_missing_recovery_spec": skipped_missing_recovery_spec,
        "limit": limit,
    }
    return items, summary


def _load_factual_records(path: Path) -> list[EvaluationRecord]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(EvaluationRecord.model_validate(json.loads(line)))
    return rows


def _load_intervention_candidates(path: Path) -> list[InterventionCandidate]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(InterventionCandidate.model_validate(json.loads(line)))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
