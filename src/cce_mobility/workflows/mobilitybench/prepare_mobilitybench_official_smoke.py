from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import yaml


DEFAULT_OUTPUT_DIR = "outputs/mobilitybench_official"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a tiny dataset and config for official MobilityBench smoke runs."
    )
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start row offset in the source CSV. Use this to build heldout slices.",
    )
    parser.add_argument("--model-alias", default="local-model")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    if not args.base_url.startswith(("https://", "http://")):
        parser.error("--base-url must be an OpenAI-compatible HTTP(S) endpoint")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env):
        parser.error("--api-key-env must be an environment variable name")

    source_csv = Path(args.source_csv)
    if not source_csv.exists():
        raise FileNotFoundError(f"source CSV not found: {source_csv}")
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"official_smoke_{args.limit}.csv"
    config_path = output_dir / "official_model_config.yaml"
    manifest_path = output_dir / "official_smoke_manifest.json"

    with source_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)[args.offset : args.offset + args.limit]
    if len(rows) < args.limit:
        raise ValueError(
            f"Requested {args.limit} rows from offset {args.offset}, "
            f"but only found {len(rows)} rows in {source_csv}."
        )
    if "query" not in columns or any(not row["query"].strip() for row in rows):
        raise ValueError("Source CSV must contain nonempty queries")
    if "case_id" not in columns:
        columns.insert(0, "case_id")
        for index, row in enumerate(rows, start=args.offset):
            row["case_id"] = str(index)
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "models": {
            args.model_alias: {
                "provider": "openai",
                "base_url": args.base_url,
                "api_key": f"${{{args.api_key_env}}}",
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "roles": {
                    "planner": args.model_name,
                    "worker": args.model_name,
                    "reporter": args.model_name,
                },
            }
        },
        "agent": {
            "default_framework": "plan_and_execute",
            "sandbox_data_dir": "data/sandbox",
            "plan_and_execute": {
                "max_plan_iterations": 10,
                "max_worker_retries": 3,
            },
            "react": {
                "max_iterations": 10,
            },
        },
        "evaluation": {
            "metrics": ["tool", "instruction", "planning", "answer", "efficiency"],
            "tool": {
                "check_min": {
                    "query_poi": ["keywords"],
                    "driving_route": ["origin", "destination"],
                    "walking_route": ["origin", "destination"],
                    "bicycling_route": ["origin", "destination"],
                    "bus_route": ["origin", "destination"],
                    "weather_query": ["city"],
                    "reverse_geocoding": ["longitude", "latitude"],
                    "search_around_poi": ["location", "keywords"],
                    "traffic_status": ["name", "city"],
                },
                "check_max": {
                    "query_poi": ["keywords", "city"],
                    "driving_route": ["origin", "destination", "waypoints", "strategy"],
                    "bicycling_route": ["origin", "destination"],
                    "walking_route": ["origin", "destination"],
                    "bus_route": ["origin", "destination", "strategy"],
                    "weather_query": ["city", "need_forecast"],
                    "reverse_geocoding": ["longitude", "latitude", "radius"],
                    "search_around_poi": ["location", "keywords", "radius"],
                    "traffic_status": ["name", "city"],
                },
            },
            "instruction": {"similarity_threshold": 0.7, "model_path": "./embedding-model"},
            "planning": {"alpha": 0.8, "beta": 0.2},
            "answer": {"similarity_threshold": 0.8, "distance_tolerance": 0.05},
        },
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    manifest = {
        "source_sha256": _sha256(source_csv),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_file": dataset_path.name,
        "config_file": config_path.name,
        "limit": args.limit,
        "offset": args.offset,
        "model_alias": args.model_alias,
        "model_name": args.model_name,
        "api_key_env": args.api_key_env,
        "case_ids": [row["case_id"] for row in rows],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
