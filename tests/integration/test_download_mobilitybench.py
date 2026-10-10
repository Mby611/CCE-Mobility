from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/cce_mobility/workflows/mobilitybench"
    / "download_mobilitybench.py"
)
SPEC = importlib.util.spec_from_file_location("download_mobilitybench", SCRIPT_PATH)
assert SPEC and SPEC.loader
download_mobilitybench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_mobilitybench)
_write_derived_splits = download_mobilitybench._write_derived_splits


def test_write_derived_splits_adds_case_id(tmp_path):
    source = tmp_path / "query.csv"
    with source.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "context"])
        writer.writeheader()
        for idx in range(6):
            writer.writerow({"query": f"q{idx}", "context": "{}"})

    manifest = _write_derived_splits(
        source_path=source,
        output_dir=tmp_path,
        seed=1,
        train_size=3,
        validation_size=2,
        test_size=1,
    )

    assert manifest["train"]["rows"] == 3
    with (tmp_path / "train.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    assert "case_id" in row
    assert row["case_id"]


def test_write_derived_splits_supports_remaining_test_size(tmp_path):
    source = tmp_path / "query.csv"
    with source.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "context"])
        writer.writeheader()
        for idx in range(8):
            writer.writerow({"query": f"q{idx}", "context": "{}"})

    manifest = _write_derived_splits(
        source_path=source,
        output_dir=tmp_path,
        seed=1,
        train_size=3,
        validation_size=2,
        test_size="remaining",
    )

    assert manifest["test"]["rows"] == 3
