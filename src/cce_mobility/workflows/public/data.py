from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .artifacts import artifact, read_json, sha256, write_locked

UPSTREAM = {
    "travelplanner": ("OSU-NLP-Group/TravelPlanner", "osunlp/TravelPlanner"),
    "mobilitybench": ("AMAP-ML/MobilityBench", "GD-ML/MobilityBench"),
}


def prepare_data(args) -> dict[str, Any]:
    target = args.output.resolve()
    manifest_path = target / "split_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        request = manifest["request"]
        expected = _request(args)
        if request != expected:
            raise ValueError("Data preparation options changed; choose a new --output")
        for item in manifest["files"].values():
            if sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Prepared data mutated: {item['path']}")
        return manifest

    if args.source:
        source = args.source.resolve()
        revision = {"kind": "local", "requested": args.revision, "resolved": None}
    else:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install data extras: pip install -e '.[data]'") from exc
        repo = UPSTREAM[args.benchmark][1]
        resolved = HfApi().dataset_info(repo, revision=args.revision).sha
        patterns = (
            ["*_ref_info.jsonl", "*.csv"]
            if args.benchmark == "travelplanner"
            else ["datasets/**", "sandbox/**"]
        )
        source = Path(
            snapshot_download(repo, repo_type="dataset", revision=resolved, allow_patterns=patterns)
        )
        revision = {
            "kind": "huggingface",
            "repository": repo,
            "requested": args.revision,
            "resolved": resolved,
        }

    target.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    if args.benchmark == "travelplanner":
        seen: dict[str, str] = {}
        for split in ("train", "validation", "test"):
            path = source / f"{split}_ref_info.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Official reference data missing: {path}")
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not rows:
                raise ValueError(f"Empty split: {path}")
            csv_source = source / f"{split}.csv"
            queries = []
            if csv_source.is_file():
                with csv_source.open(encoding="utf-8-sig", newline="") as handle:
                    queries = list(csv.DictReader(handle))
                if len(queries) != len(rows):
                    raise ValueError(f"Query/reference row counts differ for {split}")
            identities = []
            for index, row in enumerate(rows):
                if not isinstance(row, dict) or not row:
                    raise ValueError(f"Malformed {split} reference row {index}: expected an object")
                query = row.get("query") or (queries[index].get("query") if queries else None)
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(
                        f"Malformed {split} row {index}: require query CSV aligned with nonempty reference JSONL"
                    )
                identity = hashlib.sha256(query.strip().encode()).hexdigest()
                if identity in seen and seen[identity] != split:
                    raise ValueError(f"Query occurs in both {seen[identity]} and {split}")
                seen[identity] = split
                identities.append(identity)
            _copy_checked(path, target / path.name)
            files[path.name] = artifact(target / path.name)
            if csv_source.is_file():
                _copy_checked(csv_source, target / csv_source.name)
                files[csv_source.name] = artifact(target / csv_source.name)
            splits[split] = {
                "count": len(rows),
                "sample_ids": [f"tp_{i}" for i in range(len(rows))],
                "query_hashes": identities,
            }
    else:
        path = source if source.is_file() else source / "query.csv"
        if not path.exists():
            candidates = sorted((source / "datasets").glob("*.csv"))
            full = [p for p in candidates if "all_data" in p.name]
            if len(full) != 1:
                raise ValueError(
                    "Specify --source with the official full query CSV; ambiguous download layout"
                )
            path = full[0]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
        if not rows or "query" not in columns:
            raise ValueError("MobilityBench CSV must contain nonempty query rows")
        if any(not str(row.get("query") or "").strip() for row in rows):
            raise ValueError("MobilityBench query rows must not be empty")
        if "case_id" not in columns:
            columns.insert(0, "case_id")
            for i, row in enumerate(rows):
                row["case_id"] = str(i)
        ids = [str(row["case_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("MobilityBench case_id must be unique")
        requested = args.train_size + args.validation_size + args.test_size
        if requested > len(rows) or min(args.train_size, args.validation_size, args.test_size) < 1:
            raise ValueError("Split sizes must be positive and fit the official source")
        random.Random(args.seed).shuffle(rows)
        offset = 0
        for split, size in (
            ("train", args.train_size),
            ("validation", args.validation_size),
            ("test", args.test_size),
        ):
            selected = rows[offset : offset + size]
            offset += size
            dest = target / f"{split}.csv"
            if dest.exists():
                raise FileExistsError(f"Incomplete preparation; refusing to overwrite {dest}")
            with dest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(selected)
            files[dest.name] = artifact(dest)
            splits[split] = {"count": size, "case_ids": [row["case_id"] for row in selected]}
        files["source"] = artifact(path)
        sandbox = args.sandbox_source or (source / "sandbox" if source.is_dir() else None)
        if sandbox and Path(sandbox).is_dir():
            if not any(item.is_file() for item in Path(sandbox).rglob("*")):
                raise ValueError("Official sandbox directory is empty")
            for item in sorted(Path(sandbox).rglob("*")):
                if item.is_file():
                    dest = target / "sandbox" / item.relative_to(sandbox)
                    _copy_checked(item, dest)
                    files[dest.relative_to(target).as_posix()] = artifact(dest)
        else:
            raise FileNotFoundError(
                "MB continuation requires official replay files: supply --sandbox-source or download sandbox/**"
            )
    manifest = {
        "schema": "cce-data-manifest",
        "benchmark": args.benchmark,
        "request": _request(args),
        "upstream": revision,
        "splits": splits,
        "files": files,
        "split_overlap": False,
    }
    write_locked(manifest_path, manifest)
    return manifest


def _request(args) -> dict[str, Any]:
    return {
        "benchmark": args.benchmark,
        "source": str(args.source.resolve()) if args.source else None,
        "revision": args.revision,
        "seed": args.seed,
        "train_size": args.train_size,
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "sandbox_source": str(args.sandbox_source.resolve()) if args.sandbox_source else None,
    }


def _copy_checked(source: Path, target: Path) -> None:
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError(f"Existing file differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def pin_checkout(benchmark: str, path: Path, revision: str) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"Clone https://github.com/{UPSTREAM[benchmark][0]} into {path}")

    def git(*command):
        return subprocess.check_output(["git", "-C", str(path), *command], text=True).strip()

    commit = git("rev-parse", "HEAD")
    if revision and git("rev-parse", revision) != commit:
        raise ValueError(
            "Upstream HEAD differs from --code-revision; explicitly checkout that revision first"
        )
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"Upstream tracked files are modified: {path}")
    return {"repository": UPSTREAM[benchmark][0], "revision": commit}
