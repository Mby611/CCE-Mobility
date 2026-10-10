from __future__ import annotations

from pathlib import Path
import argparse
import json

from cce_mobility.core.config import ExperimentConfig
from cce_mobility.runners.evaluation import EvaluationRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="CCE-Mobility experiment CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run-config", help="Run an evaluation config")
    run_p.add_argument("config", type=Path)

    inspect_p = sub.add_parser("inspect-config", help="Validate and print a config")
    inspect_p.add_argument("config", type=Path)

    data_p = sub.add_parser("prepare-data", help="Prepare official data and lock split hashes")
    data_p.add_argument("--benchmark", choices=("travelplanner", "mobilitybench"), required=True)
    data_p.add_argument("--output", type=Path, required=True)
    data_p.add_argument(
        "--source", type=Path, help="Existing official dataset directory or MB query CSV"
    )
    data_p.add_argument("--sandbox-source", type=Path)
    data_p.add_argument(
        "--revision",
        default="main",
        help="HF revision, resolved to an immutable commit on first download",
    )
    data_p.add_argument("--seed", type=int, default=42)
    data_p.add_argument("--train-size", type=int, default=5000)
    data_p.add_argument("--validation-size", type=int, default=5000)
    data_p.add_argument("--test-size", type=int, default=5000)

    prep_p = sub.add_parser("prepare-config", help="Generate one local experiment configuration")
    prep_p.add_argument("--benchmark", choices=("travelplanner", "mobilitybench"), required=True)
    prep_p.add_argument("--models", type=Path, required=True)
    prep_p.add_argument("--data-dir", type=Path, required=True)
    prep_p.add_argument("--repo-path", type=Path, required=True)
    prep_p.add_argument(
        "--code-revision",
        default="",
        help="Expected upstream commit; default locks the checked-out HEAD",
    )
    prep_p.add_argument("--mb-python", type=Path)
    prep_p.add_argument("--output", type=Path, required=True)
    prep_p.add_argument("--output-root", type=Path, required=True)
    prep_p.add_argument("--name", default="cce_experiment")
    prep_p.add_argument(
        "--variant", choices=("full", "agent-only", "without-memory-curation"), default="full"
    )
    prep_p.add_argument("--train-tasks", type=int, default=None)
    prep_p.add_argument("--cycles", type=int, default=3)
    prep_p.add_argument("--max-phase-updates", type=int, default=42)
    prep_p.add_argument("--seed", type=int, default=42)
    prep_p.add_argument("--validation-tasks", type=int, default=None)
    prep_p.add_argument("--test-tasks", type=int, default=None)
    prep_p.add_argument(
        "--frozen-validation",
        type=Path,
        help="Shared validation challenge JSONL with hash manifest",
    )
    prep_p.add_argument(
        "--frozen-test", type=Path, help="Shared test challenge JSONL with hash manifest"
    )
    prep_p.add_argument(
        "--base-agent-guidance",
        type=Path,
        help="Frozen external SkillOpt artifact for residual-memory learning",
    )

    check_p = sub.add_parser(
        "check-models", help="Paid connectivity probe; no automatic model aliases"
    )
    check_p.add_argument(
        "config", type=Path, help="Local model file or generated experiment configuration"
    )
    for stage in ("train", "validate", "test", "evaluate-official"):
        stage_p = sub.add_parser(
            stage, help=f"Run the {stage} stage with one generated experiment configuration"
        )
        stage_p.add_argument("config", type=Path)

    args = parser.parse_args()
    if args.command == "prepare-data":
        from cce_mobility.workflows.public.data import prepare_data

        print(json.dumps(prepare_data(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-config":
        from cce_mobility.workflows.public.configuration import prepare_config

        tp = args.benchmark == "travelplanner"
        args.train_tasks = args.train_tasks if args.train_tasks is not None else (45 if tp else 20)
        args.validation_tasks = (
            args.validation_tasks if args.validation_tasks is not None else (180 if tp else 100)
        )
        args.test_tasks = args.test_tasks if args.test_tasks is not None else (1000 if tp else 500)
        print(json.dumps(prepare_config(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "check-models":
        from cce_mobility.workflows.public.pipeline import check_models

        result = check_models(args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["all_ok"] else 1
    if args.command in {"train", "validate", "test", "evaluate-official"}:
        from cce_mobility.workflows.public.pipeline import run_stage

        print(json.dumps(run_stage(args.config, args.command), ensure_ascii=False, indent=2))
        return 0
    cfg = ExperimentConfig.from_yaml(args.config)

    if args.command == "inspect-config":
        print(cfg.model_dump_json(indent=2))
        return 0

    report = EvaluationRunner(cfg).run()
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
