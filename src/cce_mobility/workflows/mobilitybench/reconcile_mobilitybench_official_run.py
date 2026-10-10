from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.integrations.mobilitybench_official_audit import (  # noqa: E402
    official_raw_dirty_reasons,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and merge MobilityBench transport/parser repairs without score-based retry."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--raw-results", type=Path, required=True)
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--output-csv", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--base-raw", type=Path, required=True)
    merge.add_argument("--repair-raw", type=Path, required=True)
    merge.add_argument("--dataset", type=Path, required=True)
    merge.add_argument("--output-raw", type=Path, required=True)
    merge.add_argument("--next-repair-csv", type=Path)
    canonicalize = subparsers.add_parser("canonicalize")
    canonicalize.add_argument("--source-raw", type=Path, required=True)
    canonicalize.add_argument("--output-raw", type=Path, required=True)
    canonicalize.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare_repair(
            raw_results=args.raw_results,
            dataset=args.dataset,
            output_csv=args.output_csv,
        )
    elif args.command == "merge":
        result = merge_clean_replacements(
            base_raw=args.base_raw,
            repair_raw=args.repair_raw,
            dataset=args.dataset,
            output_raw=args.output_raw,
            next_repair_csv=args.next_repair_csv,
        )
    else:
        result = canonicalize_clean_raw(
            source_raw=args.source_raw,
            output_raw=args.output_raw,
            expected_count=args.expected_count,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command in {"prepare", "canonicalize"}:
        return 0
    return 0 if int(result["remaining_bad_count"]) == 0 else 3


def canonicalize_clean_raw(
    *, source_raw: Path, output_raw: Path, expected_count: int
) -> dict[str, Any]:
    source_raw = source_raw.expanduser().resolve()
    output_raw = output_raw.expanduser().resolve()
    if output_raw.exists():
        raise FileExistsError(f"Refusing to overwrite canonical raw results: {output_raw}")
    rows = _read_raw(source_raw)
    dirty = {
        str(row.get("case_id") or ""): _dirty_reasons(row) for row in rows if _dirty_reasons(row)
    }
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} rows before canonicalization, found {len(rows)}"
        )
    if dirty:
        raise ValueError(f"Refusing to canonicalize {len(dirty)} dirty rows: {list(dirty)[:10]}")
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_raw, output_raw)
    result = {
        "operation": "canonicalize",
        "count": len(rows),
        "dirty_case_count": 0,
        "source_raw": _artifact(source_raw),
        "canonical_raw": _artifact(output_raw),
        "byte_identical": _sha256(source_raw) == _sha256(output_raw),
    }
    _write_manifest(output_raw.parent.parent / "canonical_source.json", result)
    return result


def prepare_repair(*, raw_results: Path, dataset: Path, output_csv: Path) -> dict[str, Any]:
    raw_results = raw_results.expanduser().resolve()
    dataset = dataset.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    rows = _read_raw(raw_results)
    bad = {
        str(row.get("case_id") or ""): _dirty_reasons(row) for row in rows if _dirty_reasons(row)
    }
    if bad:
        _write_selected_csv(dataset, output_csv, list(bad))
    result = {
        "operation": "prepare",
        "base_count": len(rows),
        "remaining_bad_count": len(bad),
        "remaining_bad_ids": list(bad),
        "remaining_bad_reasons": bad,
        "raw_results": _artifact(raw_results),
        "dataset": _artifact(dataset),
        "repair_csv": _artifact(output_csv) if bad else None,
    }
    _write_manifest(output_csv.with_suffix(".manifest.json"), result)
    return result


def merge_clean_replacements(
    *,
    base_raw: Path,
    repair_raw: Path,
    dataset: Path,
    output_raw: Path,
    next_repair_csv: Path | None,
) -> dict[str, Any]:
    base_raw = base_raw.expanduser().resolve()
    repair_raw = repair_raw.expanduser().resolve()
    dataset = dataset.expanduser().resolve()
    output_raw = output_raw.expanduser().resolve()
    if output_raw.exists():
        raise FileExistsError(f"Refusing to overwrite merged raw results: {output_raw}")
    base_rows = _read_raw(base_raw)
    repair_rows = _read_raw(repair_raw)
    repair_by_id = {str(row.get("case_id") or ""): row for row in repair_rows}
    if len(repair_by_id) != len(repair_rows):
        raise ValueError(f"Duplicate case_id in repair raw: {repair_raw}")
    clean_ids = {case_id for case_id, row in repair_by_id.items() if not _dirty_reasons(row)}
    merged = [
        repair_by_id.get(str(row.get("case_id") or ""), row)
        if str(row.get("case_id") or "") in clean_ids
        else row
        for row in base_rows
    ]
    remaining = {
        str(row.get("case_id") or ""): _dirty_reasons(row) for row in merged if _dirty_reasons(row)
    }
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    output_raw.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if remaining:
        if next_repair_csv is None:
            raise ValueError("Dirty cases remain; --next-repair-csv is required")
        _write_selected_csv(dataset, next_repair_csv.expanduser().resolve(), list(remaining))
    result = {
        "operation": "merge",
        "base_count": len(base_rows),
        "repair_count": len(repair_rows),
        "clean_repair_count": len(clean_ids),
        "dirty_repair_count": len(repair_rows) - len(clean_ids),
        "remaining_bad_count": len(remaining),
        "remaining_bad_ids": list(remaining),
        "remaining_bad_reasons": remaining,
        "base_raw": _artifact(base_raw),
        "repair_raw": _artifact(repair_raw),
        "merged_raw": _artifact(output_raw),
        "next_repair_csv": (
            _artifact(next_repair_csv.expanduser().resolve())
            if remaining and next_repair_csv
            else None
        ),
    }
    _write_manifest(output_raw.parent / "merge_manifest.json", result)
    return result


def _dirty_reasons(row: dict[str, Any]) -> list[str]:
    return official_raw_dirty_reasons(row)


def _write_selected_csv(dataset: Path, output: Path, ids: list[str]) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite repair CSV: {output}")
    with dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    by_id = {str(row.get("case_id") or row.get("Case ID") or ""): row for row in rows}
    missing = [case_id for case_id in ids if case_id not in by_id]
    if missing:
        raise ValueError(f"Dirty ids missing from source dataset: {missing[:10]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_id[case_id] for case_id in ids)


def _read_raw(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected raw_results JSON array: {path}")
    ids = [str(row.get("case_id") or "") for row in rows]
    if any(not case_id for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Missing or duplicate case_id in {path}")
    return rows


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
