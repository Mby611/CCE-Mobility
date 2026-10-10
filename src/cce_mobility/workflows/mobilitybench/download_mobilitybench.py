from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download MobilityBench from HuggingFace and create reproducible CSV splits."
    )
    parser.add_argument(
        "--repo-id",
        default="GD-ML/MobilityBench",
        help="HuggingFace dataset repo id.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/local_data/mobilitybench_csv_files",
        help="Directory for query.csv and derived split CSV files.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="",
        help="Optional HF_ENDPOINT, for example https://hf-mirror.com.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only create derived splits from an existing query.csv in output-dir.",
    )
    parser.add_argument(
        "--source-csv",
        default="",
        help="Optional existing source CSV to split instead of output-dir/query.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument(
        "--test-size",
        default="remaining",
        help="Number of test rows, or 'remaining'/'auto' to use all rows left after train/validation.",
    )
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: dict[str, dict[str, Any]] = {}
    if args.source_csv:
        source_split = Path(args.source_csv).stem
        source_path = Path(args.source_csv)
        if not source_path.exists():
            raise FileNotFoundError(f"--source-csv not found: {source_path}")
    elif args.skip_download:
        source_split = "query"
        source_path = output_dir / "query.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"--skip-download requires existing file: {source_path}")
    else:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError:
            print(
                "Missing dependency: datasets. Run with a temporary uv dependency:\n"
                "  uv run --with datasets python -m "
                "cce_mobility.workflows.mobilitybench.download_mobilitybench\n"
                "or use an existing query.csv:\n"
                "  uv run python -m cce_mobility.workflows.mobilitybench.download_mobilitybench "
                "--skip-download",
                file=sys.stderr,
            )
            return 2

        dataset = load_dataset(args.repo_id)
        split_names = list(dataset.keys())

        for split_name, split_data in dataset.items():
            path = output_dir / f"{split_name}.csv"
            split_data.to_csv(str(path))
            exported[split_name] = {"path": str(path), "rows": len(split_data)}
            print(f"saved {split_name}: {path} ({len(split_data)} rows)")

        source_split = "query" if "query" in dataset else split_names[0]
        source_path = output_dir / f"{source_split}.csv"
    derived = _write_derived_splits(
        source_path=source_path,
        output_dir=output_dir,
        seed=args.seed,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )

    manifest = {
        "repo_id": args.repo_id,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "seed": args.seed,
        "source_split": source_split,
        "exported_splits": exported,
        "derived_splits": derived,
    }
    manifest_path = output_dir / "cce_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _write_derived_splits(
    *,
    source_path: Path,
    output_dir: Path,
    seed: int,
    train_size: int,
    validation_size: int,
    test_size: int | str,
) -> dict[str, dict[str, Any]]:
    with source_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "case_id" not in fieldnames:
        fieldnames = ["case_id", *fieldnames]

    parsed_test_size = _parse_test_size(
        test_size,
        total_rows=len(rows),
        train_size=train_size,
        validation_size=validation_size,
    )
    total_requested = train_size + validation_size + parsed_test_size
    if total_requested > len(rows):
        raise ValueError(
            f"requested derived splits need {total_requested} rows, but {source_path} has {len(rows)}"
        )

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    split_indices = {
        "train": indices[:train_size],
        "validation": indices[train_size : train_size + validation_size],
        "test": indices[
            train_size + validation_size : train_size + validation_size + parsed_test_size
        ],
    }
    manifest: dict[str, dict[str, Any]] = {}
    for split_name, idxs in split_indices.items():
        path = output_dir / f"{split_name}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx in idxs:
                row = dict(rows[idx])
                row.setdefault("case_id", str(idx))
                if not row.get("case_id"):
                    row["case_id"] = str(idx)
                writer.writerow(row)
        manifest[split_name] = {
            "path": str(path),
            "rows": len(idxs),
            "source_indices_head": idxs[:10],
        }
        print(f"derived {split_name}: {path} ({len(idxs)} rows)")
    return manifest


def _parse_test_size(
    value: int | str,
    *,
    total_rows: int,
    train_size: int,
    validation_size: int,
) -> int:
    if isinstance(value, int):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"remaining", "auto", "rest"}:
        remaining = total_rows - train_size - validation_size
        if remaining <= 0:
            raise ValueError(
                f"no rows left for test split: total={total_rows}, "
                f"train={train_size}, validation={validation_size}"
            )
        return remaining
    parsed = int(normalized)
    if parsed <= 0:
        raise ValueError(f"test-size must be positive or remaining, got: {value}")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
