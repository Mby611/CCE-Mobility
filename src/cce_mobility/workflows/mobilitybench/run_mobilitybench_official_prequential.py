from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.core.config import ExperimentConfig  # noqa: E402
from cce_mobility.core.llm import ChatClient  # noqa: E402
from cce_mobility.core.metrics import add_experiment_coverage_metrics, aggregate_records  # noqa: E402
from cce_mobility.core.schema import EvaluationRecord, ExperimentReport  # noqa: E402
from cce_mobility.interventions.analyzer import HeuristicTrajectoryAnalyzer  # noqa: E402
from cce_mobility.interventions.frontier import FrontierSelector  # noqa: E402
from cce_mobility.interventions.generator import FrozenInterventionGenerator  # noqa: E402
from cce_mobility.interventions.predictor import HeuristicWMPredictiveCritic  # noqa: E402
from cce_mobility.interventions.spec_builder import RuleSpecBuilder  # noqa: E402
from cce_mobility.interventions.validator import HeuristicInterventionValidator  # noqa: E402
from cce_mobility.judges.recovery import HeuristicRecoveryJudge  # noqa: E402
from cce_mobility.judges.spec_llm import SpecGroundedLLMJudge  # noqa: E402
from cce_mobility.methods.factory import create_method  # noqa: E402
from cce_mobility.skills.retrieval import sanitize_agent_visible_text  # noqa: E402
from cce_mobility.workflows.mobilitybench.convert_mobilitybench_official_to_cce import (  # noqa: E402
    _load_cases,
    _task_sample,
    _trajectory_from_official,
)
from cce_mobility.workflows.mobilitybench.run_mobilitybench_official_posthoc import (  # noqa: E402
    RECOVERY_SYSTEM_PROMPT,  # noqa: F401
    _agent_guidance,
    _agent_skill_registry,
    _configure_method,
    _official_factual_hard_metrics,
    _posthoc_hard_metrics,
    _prediction_metrics,
    _run_recovery_agent,
    _world_model_guidance,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run official MobilityBench one case at a time while updating a "
            "Reflexion/Voyager-style memory after each scored case."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["reflexion_like", "voyager_like"],
        help="Online/prequential method to update after each scored case.",
    )
    parser.add_argument("--dataset", required=True, help="Official MobilityBench CSV.")
    parser.add_argument("--config", required=True, help="Official MobilityBench AIHub config.")
    parser.add_argument("--output-dir", required=True, help="Aggregate official run directory.")
    parser.add_argument(
        "--mobilitybench-root", required=True, help="Official MobilityBench repo root."
    )
    parser.add_argument(
        "--posthoc-base-config",
        required=True,
        help="CCE posthoc config that provides model, judge, and intervention settings.",
    )
    parser.add_argument(
        "--frozen-intervention-path",
        default="",
        help="Frozen intervention set. Defaults to extra.frozen_intervention_path in posthoc config.",
    )
    parser.add_argument("--model", default="aihub-qwen35b", help="Official MB model id.")
    parser.add_argument("--framework", default="plan_and_execute", help="Official MB framework.")
    parser.add_argument(
        "--prompt-roles",
        default="worker,reporter",
        help="Official MB roles that receive the current method guidance prompt.",
    )
    parser.add_argument("--offset", type=int, default=0, help="Start offset in dataset rows.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of cases.")
    parser.add_argument(
        "--uv-cache-dir",
        default="",
        help="uv cache dir used by official MB. Defaults to CCE_ROOT/.uv-cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write no official/CCE outputs.",
    )
    parser.add_argument(
        "--save-memory-every",
        type=int,
        default=50,
        help="Save online method memory every N cases. Use 0 to save only final memory.",
    )
    args = parser.parse_args()

    dataset_path = _resolve_path(args.dataset)
    official_config_path = _resolve_path(args.config)
    output_dir = _resolve_path(args.output_dir)
    mobilitybench_root = _resolve_path(args.mobilitybench_root)
    posthoc_config_path = _resolve_path(args.posthoc_base_config)
    config = ExperimentConfig.from_yaml(posthoc_config_path)
    frozen_path = _resolve_path(
        args.frozen_intervention_path or str(config.extra.get("frozen_intervention_path") or "")
    )
    if not frozen_path.exists():
        raise FileNotFoundError(f"frozen intervention set not found: {frozen_path}")

    rows, fieldnames = _read_dataset_rows(dataset_path, offset=args.offset, limit=args.limit)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": args.method,
                    "dataset": str(dataset_path),
                    "case_count": len(rows),
                    "official_config": str(official_config_path),
                    "output_dir": str(output_dir),
                    "mobilitybench_root": str(mobilitybench_root),
                    "posthoc_base_config": str(posthoc_config_path),
                    "frozen_intervention_path": str(frozen_path),
                    "prequential_update_rule": "score case t, then update memory for case t+1",
                    "save_memory_every": args.save_memory_every,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_results_dir = output_dir / args.model
    raw_results_path = raw_results_dir / "raw_results.json"
    records_path = output_dir / "records.jsonl"
    factual_records_path = output_dir / "cce_factual_records.jsonl"
    records_path.write_text("", encoding="utf-8")
    factual_records_path.write_text("", encoding="utf-8")

    method = create_method(args.method)
    _configure_method(method, config.extra)
    method.before_iteration([])

    agent_client = ChatClient(config.agent)
    judge_client = ChatClient(config.judge)
    analyzer = HeuristicTrajectoryAnalyzer()
    generator = FrozenInterventionGenerator(str(frozen_path))
    spec_builder = RuleSpecBuilder()
    validator = HeuristicInterventionValidator(config.intervention.validity_threshold)
    selector = FrontierSelector(
        low=config.intervention.frontier_low,
        high=config.intervention.frontier_high,
    )
    wm_predictor = HeuristicWMPredictiveCritic()
    judge = (
        HeuristicRecoveryJudge()
        if config.judge.provider in {"local", "heuristic", "placeholder"}
        else SpecGroundedLLMJudge(judge_client)
    )
    cases = _load_cases(dataset_path)

    raw_rows: list[dict[str, Any]] = []
    output_records: list[EvaluationRecord] = []
    case_dataset_dir = output_dir / "_case_datasets"
    case_run_dir = output_dir / "_case_runs"
    case_log_dir = output_dir / "_case_logs"
    prompt_dir = output_dir / "_prequential_prompts"
    for path in [case_dataset_dir, case_run_dir, case_log_dir, prompt_dir]:
        path.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        case_id = _case_id(row, index + args.offset - 1)
        safe_id = _safe(case_id)
        print(
            f"[official_prequential:{args.method}] case {index}/{len(rows)} {case_id}",
            flush=True,
        )
        guidance = sanitize_agent_visible_text(str(_agent_guidance(method) or "")).strip()
        prompt_path = prompt_dir / f"{index:04d}_{safe_id}.md"
        prompt_path.write_text(guidance + ("\n" if guidance else ""), encoding="utf-8")
        one_case_dataset = case_dataset_dir / f"{index:04d}_{safe_id}.csv"
        _write_single_case_dataset(one_case_dataset, fieldnames, row)
        one_case_output = case_run_dir / f"{index:04d}_{safe_id}"
        one_case_log = case_log_dir / f"{index:04d}_{safe_id}.log"

        _run_official_case(
            mobilitybench_root=mobilitybench_root,
            official_config_path=official_config_path,
            dataset_path=one_case_dataset,
            output_dir=one_case_output,
            model=args.model,
            framework=args.framework,
            prompt_path=prompt_path if guidance else None,
            prompt_roles=args.prompt_roles,
            log_path=one_case_log,
            uv_cache_dir=args.uv_cache_dir or str(PROJECT_ROOT / ".uv-cache"),
        )

        raw = _read_case_raw_result(one_case_output, args.model, case_id)
        raw_rows.append(raw)
        _write_aggregate_raw_results(raw_results_path, raw_rows)

        sample = _task_sample(cases.get(case_id, row), dataset_path)
        factual = _trajectory_from_official(raw, sample.sample_id, model=args.model)
        factual_record = EvaluationRecord(
            sample=sample,
            factual=factual,
            metadata={
                "source": "official_mobilitybench_prequential_raw_results",
                "official_case_id": case_id,
                "official_model_name": raw.get("model_name"),
                "official_error": raw.get("error"),
                "official_execution_time": raw.get("execution_time"),
                "official_token_usage": raw.get("token_usage") or {},
                "prequential_index": index,
                "prequential_method": args.method,
            },
        )
        _append_jsonl(factual_records_path, factual_record)

        sample_records = _score_posthoc_case(
            base_record=factual_record,
            config=config,
            method=method,
            analyzer=analyzer,
            generator=generator,
            spec_builder=spec_builder,
            validator=validator,
            selector=selector,
            wm_predictor=wm_predictor,
            judge=judge,
            agent_client=agent_client,
            index=index,
        )
        if not sample_records:
            continue
        for record in sample_records:
            output_records.append(record)
            _append_jsonl(records_path, record)
        method.observe(sample_records)
        if args.save_memory_every > 0 and index % args.save_memory_every == 0:
            _save_method_memory(method, output_dir / f"skill_memory_after_case_{index:04d}.jsonl")

    _save_method_memory(method, output_dir / "skill_memory_final.jsonl")
    metrics = aggregate_records(output_records)
    add_experiment_coverage_metrics(
        metrics,
        sample_count=len(rows),
        iterations=1,
        selected_budget=config.intervention.selected_budget,
    )
    metrics.update(
        {
            "agent_model": config.agent.model,
            "agent_provider": config.agent.provider,
            "world_model": config.world_model.model,
            "world_model_provider": config.world_model.provider,
            "judge_model": config.judge.model,
            "judge_provider": config.judge.provider,
            "official_prequential_method": args.method,
            "official_prequential_online_update_after_sample": True,
            "official_prequential_raw_results_path": str(raw_results_path),
            "official_prequential_factual_records_path": str(factual_records_path),
            "official_prequential_prompt_roles": args.prompt_roles,
            "official_prequential_case_count": len(raw_rows),
            "posthoc_official_aligned": True,
        }
    )
    method_metrics = getattr(method, "report_metrics", None)
    if callable(method_metrics):
        metrics.update(method_metrics())

    report = ExperimentReport(
        experiment_name=output_dir.name,
        benchmark=config.benchmark.name,
        method=args.method,
        total_samples=len(rows),
        metrics=metrics,
        records_path=str(records_path),
    )
    (output_dir / "report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(report.model_dump_json(indent=2), flush=True)
    return 0


def _score_posthoc_case(
    *,
    base_record: EvaluationRecord,
    config: ExperimentConfig,
    method: Any,
    analyzer: HeuristicTrajectoryAnalyzer,
    generator: FrozenInterventionGenerator,
    spec_builder: RuleSpecBuilder,
    validator: HeuristicInterventionValidator,
    selector: FrontierSelector,
    wm_predictor: HeuristicWMPredictiveCritic,
    judge: Any,
    agent_client: ChatClient,
    index: int,
) -> list[EvaluationRecord]:
    sample = base_record.sample
    factual = base_record.factual
    if factual is None:
        return []

    factual_score = judge.score(EvaluationRecord(sample=sample, factual=factual))
    points = analyzer.find_points(sample, factual)
    candidates = generator.generate(sample, factual, points)
    prepared = []
    for candidate in candidates[: config.intervention.candidate_budget]:
        candidate.recovery_spec = candidate.recovery_spec or spec_builder.build(candidate, sample)
        candidate = validator.validate(candidate, sample)
        if validator.is_valid(candidate):
            prepared.append(candidate)
    selected = selector.select(prepared, budget=config.intervention.selected_budget)
    print(f"  selected {len(selected)} environment update(s)", flush=True)

    if not selected:
        record = base_record.model_copy(deep=True)
        record.factual_score = factual_score
        record.metadata.update(
            {
                "prequential_index": index,
                "posthoc_iteration": 1,
                "iteration": 1,
            }
        )
        return [record]

    records = []
    for candidate in selected:
        wm_prediction = wm_predictor.predict(
            sample=sample,
            factual=factual,
            candidate=candidate,
            world_model_guidance=_world_model_guidance(method),
        )
        cf, agent_retrieval = _run_recovery_agent(
            agent_client=agent_client,
            sample_query=sample.query,
            factual=factual,
            intervention=candidate.spec,
            agent_guidance=_agent_guidance(method),
            agent_skill_registry=_agent_skill_registry(method),
            agent_owned_retrieval=False,
        )
        record = EvaluationRecord(
            sample=sample,
            factual=factual,
            counterfactual=cf,
            intervention=candidate.spec,
            recovery_spec=candidate.recovery_spec,
            wm_prediction=wm_prediction,
            factual_score=factual_score,
            metadata={
                **base_record.metadata,
                "intervention_validity": candidate.validity_score,
                "intervention_utility": candidate.utility,
                "validation_notes": candidate.validation_notes,
                "factual_hard_metrics": _official_factual_hard_metrics(factual),
                "counterfactual_hard_metrics": _posthoc_hard_metrics(cf, candidate.spec),
                "agent_skill_retrieval": agent_retrieval,
                "prequential_index": index,
                "posthoc_iteration": 1,
                "iteration": 1,
            },
        )
        record.counterfactual_score = judge.score(record)
        record.metadata["wm_prediction_metrics"] = _prediction_metrics(record)
        records.append(record)
    return records


def _run_official_case(
    *,
    mobilitybench_root: Path,
    official_config_path: Path,
    dataset_path: Path,
    output_dir: Path,
    model: str,
    framework: str,
    prompt_path: Path | None,
    prompt_roles: str,
    log_path: Path,
    uv_cache_dir: str,
) -> None:
    env = _official_env(
        mobilitybench_root=mobilitybench_root,
        dataset_path=dataset_path,
        output_dir=output_dir,
        model=model,
        framework=framework,
        official_config_path=official_config_path,
        prompt_path=prompt_path,
        prompt_roles=prompt_roles,
        uv_cache_dir=uv_cache_dir,
    )
    cmd = [
        "bash",
        str(PROJECT_ROOT / "scripts" / "run_mobilitybench_official_smoke.sh"),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        print(
            f"  official MB command exited with code {completed.returncode}; "
            f"keeping raw fallback for repair audit",
            flush=True,
        )


def _official_env(
    *,
    mobilitybench_root: Path,
    dataset_path: Path,
    output_dir: Path,
    model: str,
    framework: str,
    official_config_path: Path,
    prompt_path: Path | None,
    prompt_roles: str,
    uv_cache_dir: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CCE_ROOT": str(PROJECT_ROOT),
            "DATASET": str(dataset_path),
            "CONFIG": str(official_config_path),
            "OUTPUT_DIR": str(output_dir),
            "MODEL": model,
            "FRAMEWORK": framework,
            "MB_ROOT": str(mobilitybench_root),
            "UV_CACHE_DIR": uv_cache_dir,
            "CCE_MOBILITYBENCH_SKILL_PROMPT_ROLES": prompt_roles,
        }
    )
    if prompt_path is not None:
        env["CCE_MOBILITYBENCH_SKILL_PROMPT_PATH"] = str(prompt_path)
    else:
        env.pop("CCE_MOBILITYBENCH_SKILL_PROMPT_PATH", None)
    return env


def _read_case_raw_result(output_dir: Path, model: str, case_id: str) -> dict[str, Any]:
    raw_path = output_dir / model / "raw_results.json"
    if not raw_path.exists():
        candidates = list(output_dir.glob("*/raw_results.json"))
        raw_path = candidates[0] if candidates else raw_path
    if not raw_path.exists():
        return _fallback_raw_result(case_id, model, "missing raw_results.json")
    try:
        rows = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _fallback_raw_result(case_id, model, f"raw_results parse error: {exc}")
    if not rows:
        return _fallback_raw_result(case_id, model, "empty raw_results.json")
    raw = rows[0]
    raw["case_id"] = str(raw.get("case_id") or case_id)
    raw["model_name"] = str(raw.get("model_name") or model)
    return raw


def _fallback_raw_result(case_id: str, model: str, error: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "model_name": model,
        "planner_output": {},
        "worker_output": [],
        "reporter_output": "",
        "token_usage": {},
        "execution_time": 0.0,
        "error": error,
    }


def _write_aggregate_raw_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_dataset_rows(
    path: Path,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows, fieldnames


def _write_single_case_dataset(
    path: Path,
    fieldnames: list[str],
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def _append_jsonl(path: Path, record: EvaluationRecord) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def _save_method_memory(method: Any, path: Path) -> None:
    save_fn = getattr(method, "save_registry", None)
    if callable(save_fn):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_fn(path)


def _case_id(row: dict[str, Any], fallback_idx: int) -> str:
    return str(row.get("case_id") or row.get("Case ID") or fallback_idx)


def _safe(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "case"


def _resolve_path(value: str) -> Path:
    if not value:
        return Path("")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
