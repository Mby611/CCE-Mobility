from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any
from cce_mobility.core.config import ModelConfig
from cce_mobility.core.llm import ChatClient
from cce_mobility.core.metrics import aggregate_records
from cce_mobility.core.schema import EvaluationRecord, ExperimentReport
from cce_mobility.integrations.mobilitybench_continuation import (
    CONTINUATION_PROTOCOL,
    MobilityBenchContinuationCase,
    cce_continuation_runtime_provenance,
    continuation_agent_history_text,
    file_sha256,
    intervention_target_terms,
    load_prompt_bundle,
    official_mobilitybench_provenance,
    run_official_continuation,
    stable_hash,
    validate_continuation_case,
)
from cce_mobility.judges.spec_llm import JUDGE_PROMPT, SpecGroundedLLMJudge
from cce_mobility.skills.registry import SkillRegistry
from cce_mobility.skills.retrieval import (
    AGENT_RETRIEVAL_SYSTEM_PROMPT,
    active_agent_skills,
    build_agent_retrieval_prompt,
    heuristic_retrieval_decision,
    parse_retrieval_decision,
    render_agent_skill_index,
    render_selected_agent_skill_blocks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_SCHEMA = "mobilitybench-continuation-eval-v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official MobilityBench state-reconstruction continuation evaluation."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--smoke-per-dimension",
        type=int,
        default=0,
        help="Run a deterministic small subset in a separate smoke output directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = _read_json(args.config)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.smoke_per_dimension:
        config["smoke_per_dimension"] = args.smoke_per_dimension
        if not args.output_dir:
            config["output_dir"] = str(Path(config["output_dir"]) / "protocol_smoke")
    report = asyncio.run(run(config, config_path=args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


async def run(config: dict[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    _validate_config(config)
    _load_local_secret_into_process(config["agent"]["api_key_env"], "LLM_API_KEY")
    output_dir = _project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = stable_hash(_portable_config(config))
    _guard_config(output_dir, config, config_hash, config_path)
    cases_path = _project_path(config["continuation_cases_path"])
    cases = _load_cases(cases_path)
    cases = _select_smoke_cases(cases, int(config.get("smoke_per_dimension") or 0))
    expected_count = int(config.get("expected_count") or len(cases))
    if not config.get("smoke_per_dimension") and len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} continuation cases, got {len(cases)}")
    prompt_bundle_path = _project_path(config["official_prompt_bundle_path"])
    prompt_bundle_payload = _read_json(prompt_bundle_path)
    base_prompts = load_prompt_bundle(prompt_bundle_path)
    current_official_provenance = official_mobilitybench_provenance()
    frozen_official_provenance = prompt_bundle_payload.get("official_mobilitybench_provenance")
    if not _same_official_code(current_official_provenance, frozen_official_provenance):
        raise ValueError(
            f"Official MobilityBench code differs from the frozen prompt bundle: current={current_official_provenance}, frozen={frozen_official_provenance}"
        )
    current_cce_provenance = cce_continuation_runtime_provenance(PROJECT_ROOT)
    frozen_cce_provenance = prompt_bundle_payload.get("cce_continuation_runtime_provenance")
    if not _same_cce_runtime(current_cce_provenance, frozen_cce_provenance):
        raise ValueError(
            f"CCE continuation runtime differs from the frozen prompt bundle: current={current_cce_provenance}, frozen={frozen_cce_provenance}"
        )
    runtime_contract = _runtime_contract()
    if runtime_contract != prompt_bundle_payload.get("environment_contract"):
        raise ValueError(
            f"Official runtime environment differs from the frozen prompt bundle: current={runtime_contract}, frozen={prompt_bundle_payload.get('environment_contract')}"
        )
    official_config_path = _project_path(config["official_config_path"])
    agent_config = ModelConfig.model_validate(config["agent"])
    judge_config = ModelConfig.model_validate(config["judge"])
    retrieval_client = ChatClient(agent_config)
    judge_client = ChatClient(judge_config)
    judge = SpecGroundedLLMJudge(judge_client)
    registry = _load_agent_registry(config.get("agent_memory_path"))
    records_path = output_dir / "records.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    counts = Counter()
    max_case_attempts = max(1, int(config.get("max_case_attempts", 2)))
    loaded = _load_records_by_pair_key(records_path)
    attempt_counts, last_attempt_records = _load_attempt_state(attempts_path)
    for key, record in last_attempt_records.items():
        loaded.setdefault(key, record)
    existing = {
        key: record
        for key, record in loaded.items()
        if not _retry_reason(record) or attempt_counts[key] >= max_case_attempts
    }
    retryable_dirty = sorted(set(loaded) - set(existing))
    exhausted_dirty = sorted((key for key, record in existing.items() if _retry_reason(record)))
    if retryable_dirty:
        counts["resumed_dirty_records_retried"] = len(retryable_dirty)
        _write_canonical_records(records_path, cases, existing)
    if exhausted_dirty:
        counts["resumed_exhausted_dirty_records"] = len(exhausted_dirty)
    for index, case in enumerate(cases, start=1):
        if case.pair_key in existing:
            counts["resumed_records"] += 1
            continue
        print(
            f"[{config['experiment_name']}] {index}/{len(cases)} {case.task_id} {case.intervention.dimension}",
            flush=True,
        )
        final_record: EvaluationRecord | None = None
        first_attempt = int(attempt_counts[case.pair_key]) + 1
        for attempt in range(first_attempt, max_case_attempts + 1):
            retrieval_transport_before = retrieval_client.transport_metrics()
            selected_guidance, retrieval = _retrieve_guidance(
                client=retrieval_client,
                registry=registry,
                case=case,
                retrieval_config=config.get("retrieval") or {},
            )
            retrieval_transport = _transport_delta(
                retrieval_transport_before, retrieval_client.transport_metrics()
            )
            counterfactual = await run_official_continuation(
                case=case,
                model_alias=config["official_model_alias"],
                official_config_path=official_config_path,
                base_prompts=base_prompts,
                selected_guidance=selected_guidance,
            )
            hard_metrics = _hard_metrics(counterfactual, case)
            record = EvaluationRecord(
                sample=case.sample,
                factual=case.factual,
                counterfactual=counterfactual,
                intervention=case.intervention,
                recovery_spec=case.recovery_spec,
                metadata={
                    "protocol": CONTINUATION_PROTOCOL,
                    "continuation_mode": "state_reconstruction_continuation",
                    "bitwise_graph_resume": False,
                    **_intervention_evaluation_metadata(case),
                    "pair_key": case.pair_key,
                    "case_hash": case.case_hash,
                    "checkpoint_position": case.checkpoint_position,
                    "factual_source_hash": case.factual_source_hash,
                    "intervention_source_hash": case.intervention_source_hash,
                    "recovery_contract_hash": case.recovery_contract_hash,
                    "continuation_cases_sha256": file_sha256(cases_path),
                    "official_prompt_bundle_sha256": stable_hash(base_prompts),
                    "official_mobilitybench_code_sha256": current_official_provenance[
                        "python_source_sha256"
                    ],
                    "official_mobilitybench_git_commit": current_official_provenance["git_commit"],
                    "cce_runtime_code_sha256": current_cce_provenance["runtime_source_sha256"],
                    "cce_runtime_git_commit": current_cce_provenance["git_commit"],
                    "official_runtime_contract_sha256": stable_hash(runtime_contract),
                    "official_config_sha256": file_sha256(official_config_path),
                    "official_tool_schema_sha256": counterfactual.raw.get(
                        "official_tool_schema_sha256", ""
                    ),
                    "official_model_config_sha256": counterfactual.raw.get(
                        "official_model_config_sha256", ""
                    ),
                    "agent_model_config_sha256": stable_hash(config["agent"]),
                    "judge_model_config_sha256": stable_hash(config["judge"]),
                    "judge_prompt_sha256": stable_hash(JUDGE_PROMPT),
                    "retrieval_config_sha256": stable_hash(config.get("retrieval") or {}),
                    "eval_config_sha256": config_hash,
                    "runtime_prompt_hashes": counterfactual.raw.get("prompt_hashes", {}),
                    "agent_skill_retrieval": retrieval,
                    "counterfactual_hard_metrics": hard_metrics,
                    "factual_hard_metrics": _factual_hard_metrics(case),
                    "attempt": attempt,
                    "training_series": config.get("training_series", "unknown"),
                    "method_version": config.get("method_version", "unknown"),
                    "completed_cycle": int(config.get("completed_cycle", 0)),
                    "raw_iteration": int(config.get("raw_iteration", 0)),
                    "agent_update_count": int(config.get("agent_update_count", 0)),
                    "training_rollout_count": int(config.get("training_rollout_count", 0)),
                    "agent_memory_path": config.get("agent_memory_path") or "",
                    "agent_memory_sha256": file_sha256(_project_path(config["agent_memory_path"]))
                    if config.get("agent_memory_path")
                    else stable_hash(""),
                    "world_memory_path": config.get("world_memory_path") or "",
                    "world_memory_sha256": file_sha256(_project_path(config["world_memory_path"]))
                    if config.get("world_memory_path")
                    else stable_hash(""),
                    "world_ontology_path": config.get("world_ontology_path") or "",
                    "world_ontology_sha256": file_sha256(
                        _project_path(config["world_ontology_path"])
                    )
                    if config.get("world_ontology_path")
                    else stable_hash(""),
                    "artifact_commit": str(config.get("artifact_commit") or "unknown"),
                },
            )
            judge_transport_before = judge_client.transport_metrics()
            record.counterfactual_score = judge.score(record)
            judge_transport = _transport_delta(
                judge_transport_before, judge_client.transport_metrics()
            )
            judge_error = _judge_error(record)
            record.metadata["judge_error"] = judge_error
            record.metadata["judge_error_message"] = (
                record.counterfactual_score.rationale if judge_error else ""
            )
            retry_reason = _retry_reason(record)
            attempt_row = {
                "pair_key": case.pair_key,
                "attempt": attempt,
                "retry_reason": retry_reason,
                "retrieval_transport": retrieval_transport,
                "judge_transport": judge_transport,
                "record": record.model_dump(mode="json"),
            }
            _append_jsonl(attempts_path, attempt_row)
            attempt_counts[case.pair_key] = attempt
            final_record = record
            if not retry_reason:
                break
            counts[f"retry_{retry_reason}"] += 1
        assert final_record is not None
        existing[case.pair_key] = final_record
        _write_canonical_records(records_path, cases, existing)
    ordered_records = [existing[case.pair_key] for case in cases if case.pair_key in existing]
    report = _build_report(
        config=config,
        config_hash=config_hash,
        cases_path=cases_path,
        records=ordered_records,
        expected_count=len(cases) if config.get("smoke_per_dimension") else expected_count,
        run_counts=counts,
        attempts_path=attempts_path,
        max_case_attempts=max_case_attempts,
        retrieval_client=retrieval_client,
        judge_client=judge_client,
    )
    _write_json(output_dir / "report.json", report)
    return report


def _same_official_code(current: dict[str, Any], frozen: Any) -> bool:
    if not isinstance(frozen, dict):
        return False
    return bool(current.get("python_source_sha256")) and (
        current.get("python_source_sha256") == frozen.get("python_source_sha256")
        and (not current.get("runtime_dirty"))
    )


def _same_cce_runtime(
    current: dict[str, Any], frozen: Any, *, allow_qoder_only_transition: bool = False
) -> bool:
    if not isinstance(frozen, dict):
        return False
    if current.get("runtime_source_sha256") == frozen.get("runtime_source_sha256"):
        return bool(current.get("runtime_source_sha256")) and (not current.get("runtime_dirty"))
    from cce_mobility.integrations.mobilitybench_continuation import (
        verify_scoped_provenance_transition,
    )

    try:
        verify_scoped_provenance_transition(
            current["project_root"],
            frozen,
            scope="runtime",
            allow_qoder_only_transition=allow_qoder_only_transition,
        )
    except (KeyError, ValueError):
        return False
    return True


def _retrieve_guidance(
    *,
    client: ChatClient,
    registry: SkillRegistry,
    case: MobilityBenchContinuationCase,
    retrieval_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    enabled = bool(retrieval_config.get("enabled", True))
    max_skills = max(1, int(retrieval_config.get("max_skills", 2)))
    skills = active_agent_skills(registry, limit=int(retrieval_config.get("catalog_limit", 8)))
    if not enabled or not skills:
        return (
            "",
            {
                "enabled": enabled,
                "source": "disabled" if not enabled else "empty_registry",
                "selected_skill_ids": [],
                "ignored_skill_ids": [skill.skill_id for skill in skills],
                "no_skill_needed": True,
                "available_skill_count": len(skills),
                "retrieval_prompt_sha256": stable_hash(""),
            },
        )
    environment = continuation_agent_history_text(case)
    prompt = build_agent_retrieval_prompt(
        query=case.sample.query,
        environment_observation=environment,
        skill_index=render_agent_skill_index(skills),
        max_skills=max_skills,
    )
    available_ids = {skill.skill_id for skill in skills}
    response = client.chat(
        system_prompt=AGENT_RETRIEVAL_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.0,
        max_tokens=1024,
    )
    strict = not bool(retrieval_config.get("allow_heuristic_fallback", False))
    retrieval_error = ""
    if response.ok:
        decision = parse_retrieval_decision(
            response.text, available_skill_ids=available_ids, max_skills=max_skills
        )
        if decision.parse_error:
            retrieval_error = f"retrieval_parse_error: {decision.parse_error}"
            if strict:
                decision.selected_skill_ids = []
                decision.source = "llm_parse_error_strict"
            else:
                fallback = heuristic_retrieval_decision(
                    query=case.sample.query,
                    environment_observation=environment,
                    skills=skills,
                    max_skills=max_skills,
                )
                fallback.source = "llm_parse_error_heuristic"
                fallback.raw_response = response.text
                fallback.parse_error = decision.parse_error
                decision = fallback
    else:
        retrieval_error = f"retrieval_api_error: {response.error}"
        if strict:
            decision = parse_retrieval_decision(
                "", available_skill_ids=available_ids, max_skills=max_skills
            )
            decision.source = "llm_error_strict"
            decision.raw_response = response.error
        else:
            decision = heuristic_retrieval_decision(
                query=case.sample.query,
                environment_observation=environment,
                skills=skills,
                max_skills=max_skills,
            )
            decision.source = "llm_error_heuristic"
            decision.raw_response = response.error
    guidance = render_selected_agent_skill_blocks(registry, decision.selected_skill_ids)
    metadata = {
        "enabled": True,
        **decision.model_dump(),
        "available_skill_count": len(skills),
        "rendered_skill_ids": sorted(available_ids),
        "retrieval_prompt_sha256": stable_hash(prompt),
        "observable_state_sha256": stable_hash(environment),
        "selected_guidance_sha256": stable_hash(guidance),
        "token_usage": _response_usage(response.raw),
        "error": retrieval_error,
        "strict_no_fallback": strict,
    }
    return (guidance, metadata)


def _hard_metrics(counterfactual, case: MobilityBenchContinuationCase) -> dict[str, Any]:
    text = counterfactual.final_response.strip()
    targets = intervention_target_terms(case.intervention)
    used_targets = [target for target in targets if _recommends_unavailable_target(text, target)]
    llm_error = bool(counterfactual.raw.get("llm_error"))
    tool_calls = counterfactual.raw.get("tool_call_log") or []
    overlay_calls = [call for call in tool_calls if call.get("overlay_applied")]
    parse_success = bool(text) and (not llm_error)
    return {
        "benchmark": "mobilitybench",
        "run_mode": "counterfactual",
        "adapter_scope": CONTINUATION_PROTOCOL,
        "parse_success": parse_success,
        "commonsense_pass": parse_success,
        "hard_pass": parse_success and (not used_targets),
        "final_pass": parse_success and (not used_targets),
        "uses_unavailable_identifier": bool(used_targets),
        "unavailable_id_count": len(used_targets),
        "unavailable_identifiers": used_targets,
        "estimated_cost": 0.0,
        "llm_error": llm_error,
        "llm_error_message": counterfactual.raw.get("llm_error_message", ""),
        "official_continuation_tool_call_count": len(tool_calls),
        "persistent_overlay_call_count": len(overlay_calls),
        "termination_reason": counterfactual.raw.get("termination_reason", "unknown"),
        "official_planner_present": bool(counterfactual.raw.get("official_planner_present")),
        "official_worker_tool_call_present": bool(tool_calls),
        "official_reporter_present": bool(text),
    }


def _intervention_evaluation_metadata(case: MobilityBenchContinuationCase) -> dict[str, Any]:
    validity = case.source_metadata.get("intervention_validity")
    if isinstance(validity, bool) or not isinstance(validity, (int, float)):
        raise ValueError(f"Continuation case lacks numeric intervention_validity: {case.pair_key}")
    notes = case.source_metadata.get("validation_notes") or []
    if not isinstance(notes, list) or not all((isinstance(note, str) for note in notes)):
        raise ValueError(f"Continuation case has invalid validation_notes: {case.pair_key}")
    return {"intervention_validity": float(validity), "intervention_validation_notes": list(notes)}


def _recommends_unavailable_target(text: str, target: str) -> bool:
    """Detect actionable reuse, while allowing explicit acknowledgement of unavailability."""
    if not target or target.lower() not in text.lower():
        return False
    normalized = text.lower()
    needle = target.lower()
    negative_markers = (
        "closed",
        "unavailable",
        "blocked",
        "cannot use",
        "do not use",
        "avoid",
        "not recommend",
        "关闭",
        "不可用",
        "无法使用",
        "禁止通行",
        "已封闭",
        "避开",
        "不建议",
        "不要使用",
    )
    positive_markers = (
        "recommend",
        "go to",
        "navigate to",
        "use ",
        "choose",
        "推荐",
        "前往",
        "导航至",
        "使用",
        "选择",
        "按照",
    )
    start = 0
    while True:
        index = normalized.find(needle, start)
        if index < 0:
            break
        window = normalized[max(0, index - 100) : min(len(normalized), index + len(needle) + 100)]
        if any((marker in window for marker in negative_markers)):
            start = index + len(needle)
            continue
        if any((marker in window for marker in positive_markers)):
            return True
        start = index + len(needle)
    return False


def _factual_hard_metrics(case: MobilityBenchContinuationCase) -> dict[str, Any]:
    text = case.factual.final_response.strip()
    error = str(case.factual.raw.get("official_error") or "")
    return {
        "benchmark": "mobilitybench",
        "run_mode": "factual",
        "adapter_scope": "official_mbench_raw_results",
        "parse_success": bool(text) and (not error),
        "commonsense_pass": bool(text) and (not error),
        "hard_pass": bool(text) and (not error),
        "final_pass": bool(text) and (not error),
        "uses_unavailable_identifier": False,
        "unavailable_id_count": 0,
        "estimated_cost": 0.0,
        "llm_error": bool(error),
        "llm_error_message": error,
    }


def _judge_error(record: EvaluationRecord) -> bool:
    score = record.counterfactual_score
    if score is None:
        return True
    return score.rationale.startswith(("LLM judge failed:", "LLM judge JSON parse failed;"))


def _retry_reason(record: EvaluationRecord) -> str:
    retrieval = record.metadata.get("agent_skill_retrieval", {})
    if retrieval.get("error"):
        return "agent_retrieval_api_or_parse_error"
    hard = record.metadata.get("counterfactual_hard_metrics", {})
    if hard.get("llm_error"):
        return "agent_api_or_execution_error"
    if not hard.get("parse_success"):
        return "agent_empty_or_parse_error"
    if not hard.get("official_planner_present"):
        return "official_planner_missing"
    if not hard.get("official_worker_tool_call_present"):
        return "official_worker_tool_call_missing"
    if not hard.get("official_reporter_present"):
        return "official_reporter_missing"
    if record.metadata.get("judge_error"):
        return "judge_api_or_parse_error"
    return ""


def _build_report(
    *,
    config: dict[str, Any],
    config_hash: str,
    cases_path: Path,
    records: list[EvaluationRecord],
    expected_count: int,
    run_counts: Counter,
    attempts_path: Path,
    max_case_attempts: int,
    retrieval_client: ChatClient,
    judge_client: ChatClient,
) -> dict[str, Any]:
    metrics = aggregate_records(records)
    errors = Counter()
    for record in records:
        retry_reason = _retry_reason(record)
        if retry_reason:
            errors[retry_reason] += 1
    pair_keys = [str(record.metadata.get("pair_key") or "") for record in records]
    duplicate_count = len(pair_keys) - len(set(pair_keys))
    completed = len(records)
    unavailable_count = sum(
        (
            bool(
                record.metadata.get("counterfactual_hard_metrics", {}).get(
                    "uses_unavailable_identifier"
                )
            )
            for record in records
        )
    )
    final_official_usage = _aggregate_official_usage(records)
    final_retrieval_usage = _aggregate_retrieval_usage(records)
    final_judge_usage = _aggregate_judge_usage(records)
    attempt_records = _load_all_attempt_records(attempts_path)
    official_usage = _aggregate_official_usage(attempt_records)
    retrieval_usage = _aggregate_retrieval_usage(attempt_records)
    judge_usage = _aggregate_judge_usage(attempt_records)
    retrieval_transport = _aggregate_attempt_transport(attempts_path, field="retrieval_transport")
    judge_transport = _aggregate_attempt_transport(attempts_path, field="judge_transport")
    attempt_counts, _ = _load_attempt_state(attempts_path)
    exhausted_error_count = sum(
        (
            bool(_retry_reason(record))
            and int(attempt_counts[str(record.metadata.get("pair_key") or "")]) >= max_case_attempts
            for record in records
        )
    )
    retryable_error_count = sum(
        (
            bool(_retry_reason(record))
            and int(attempt_counts[str(record.metadata.get("pair_key") or "")]) < max_case_attempts
            for record in records
        )
    )
    metrics.update(
        {
            "expected_count": expected_count,
            "completed_count": completed,
            "missing_count": max(0, expected_count - completed),
            "duplicate_pair_key_count": duplicate_count,
            "agent_error_count": errors["agent_api_or_execution_error"]
            + errors["agent_empty_or_parse_error"]
            + errors["agent_retrieval_api_or_parse_error"]
            + errors["official_planner_missing"]
            + errors["official_worker_tool_call_missing"]
            + errors["official_reporter_missing"],
            "judge_error_count": errors["judge_api_or_parse_error"],
            "parse_error_count": errors["agent_empty_or_parse_error"],
            "coverage_rate": round(completed / expected_count, 6) if expected_count else 0.0,
            "counterfactual_unavailable_usage_rate": round(unavailable_count / completed, 6)
            if completed
            else 0.0,
            "clean": completed == expected_count
            and duplicate_count == 0
            and (not sum(errors.values())),
            "continuation_protocol": CONTINUATION_PROTOCOL,
            "continuation_mode": "state_reconstruction_continuation",
            "bitwise_graph_resume": False,
            "continuation_cases_path": str(cases_path),
            "continuation_cases_sha256": file_sha256(cases_path),
            "eval_config_sha256": config_hash,
            "official_model_alias": config["official_model_alias"],
            "agent_model": config["agent"]["model"],
            "judge_model": config["judge"]["model"],
            "training_series": config.get("training_series", "unknown"),
            "method_version": config.get("method_version", "unknown"),
            "completed_cycle": int(config.get("completed_cycle", 0)),
            "raw_iteration": int(config.get("raw_iteration", 0)),
            "agent_update_count": int(config.get("agent_update_count", 0)),
            "training_rollout_count": int(config.get("training_rollout_count", 0)),
            "retry_counts": dict(run_counts),
            "max_case_attempts": max_case_attempts,
            "total_attempt_count": sum(attempt_counts.values()),
            "retry_budget_exhausted_case_count": exhausted_error_count,
            "retryable_error_count": retryable_error_count,
            "official_agent_token_usage": official_usage,
            "retrieval_token_usage": retrieval_usage,
            "judge_call_count": int(judge_usage.get("call_count") or 0),
            "judge_token_usage": judge_usage,
            "official_agent_token_usage_final_records": final_official_usage,
            "retrieval_token_usage_final_records": final_retrieval_usage,
            "judge_token_usage_final_records": final_judge_usage,
            "retrieval_transport": retrieval_transport,
            "judge_transport": judge_transport,
            "retrieval_transport_current_invocation": retrieval_client.transport_metrics(),
            "judge_transport_current_invocation": judge_client.transport_metrics(),
        }
    )
    return ExperimentReport(
        experiment_name=config["experiment_name"],
        benchmark="mobilitybench",
        method="formal_state_reconstruction_continuation",
        total_samples=len(records),
        metrics=metrics,
        records_path=str(_project_path(config["output_dir"]) / "records.jsonl"),
    ).model_dump(mode="json")


def _load_agent_registry(path: str | None) -> SkillRegistry:
    if not path:
        return SkillRegistry()
    resolved = _project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Agent memory not found: {resolved}")
    return SkillRegistry.load_jsonl(resolved)


def _load_cases(path: Path) -> list[MobilityBenchContinuationCase]:
    cases: list[MobilityBenchContinuationCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                case = MobilityBenchContinuationCase.model_validate_json(line)
                validate_continuation_case(case)
                cases.append(case)
    pair_keys = [case.pair_key for case in cases]
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError("Duplicate pair_key in continuation case artifact")
    return cases


def _select_smoke_cases(
    cases: list[MobilityBenchContinuationCase], per_dimension: int
) -> list[MobilityBenchContinuationCase]:
    if per_dimension <= 0:
        return cases
    grouped: dict[str, list[MobilityBenchContinuationCase]] = {}
    for case in cases:
        grouped.setdefault(case.intervention.dimension, []).append(case)
    return [case for dimension in sorted(grouped) for case in grouped[dimension][:per_dimension]]


def _load_records_by_pair_key(path: Path) -> dict[str, EvaluationRecord]:
    records: dict[str, EvaluationRecord] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = EvaluationRecord.model_validate_json(line)
            pair_key = str(record.metadata.get("pair_key") or "")
            if not pair_key:
                raise ValueError(f"Existing record lacks pair_key: {path}")
            if pair_key in records:
                raise ValueError(f"Duplicate pair_key in existing records: {pair_key}")
            records[pair_key] = record
    return records


def _load_attempt_state(path: Path) -> tuple[Counter[str], dict[str, EvaluationRecord]]:
    counts: Counter[str] = Counter()
    last_records: dict[str, EvaluationRecord] = {}
    if not path.exists():
        return (counts, last_records)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_key = str(row.get("pair_key") or "")
            if not pair_key:
                raise ValueError(f"Attempt lacks pair_key at {path}:{line_number}")
            counts[pair_key] += 1
            raw_record = row.get("record")
            if isinstance(raw_record, dict):
                last_records[pair_key] = EvaluationRecord.model_validate(raw_record)
    return (counts, last_records)


def _load_all_attempt_records(path: Path) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            raw_record = row.get("record")
            if not isinstance(raw_record, dict):
                raise ValueError(f"Attempt lacks record at {path}:{line_number}")
            records.append(EvaluationRecord.model_validate(raw_record))
    return records


def _transport_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    numeric = (
        "call_count",
        "request_count",
        "retry_count",
        "error_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "credits",
        "duration_ms",
        "duration_api_ms",
    )
    result = {
        key: after[key]
        for key in (
            "backend",
            "provider",
            "configured_model",
            "underlying_model_disclosed",
            "temperature_controlled",
        )
        if key in after
    }
    for key in numeric:
        if key in before or key in after:
            result[key] = (after.get(key) or 0) - (before.get(key) or 0)
    if "reported_model_usage_keys" in after:
        result["reported_model_usage_keys"] = sorted(
            set(after.get("reported_model_usage_keys") or [])
            - set(before.get("reported_model_usage_keys") or [])
        )
    return result


def _aggregate_attempt_transport(path: Path, *, field: str) -> dict[str, Any]:
    totals = Counter()
    models: set[str] = set()
    identities: dict[str, Any] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line).get(field) or {}
            if not isinstance(item, dict):
                continue
            for key in (
                "call_count",
                "request_count",
                "retry_count",
                "error_count",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "credits",
                "duration_ms",
                "duration_api_ms",
            ):
                if key in item:
                    totals[key] += item.get(key) or 0
            models.update((str(value) for value in item.get("reported_model_usage_keys") or []))
            for key in ("backend", "provider", "configured_model"):
                if item.get(key):
                    identities[key] = item[key]
    return {**identities, **dict(totals), "reported_model_usage_keys": sorted(models)}


def _write_canonical_records(
    path: Path, cases: list[MobilityBenchContinuationCase], records: dict[str, EvaluationRecord]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            record = records.get(case.pair_key)
            if record is not None:
                handle.write(record.model_dump_json() + "\n")
    temporary.replace(path)


def _aggregate_official_usage(records: list[EvaluationRecord]) -> dict[str, int]:
    totals = Counter()
    for record in records:
        usage = (
            (record.counterfactual.raw if record.counterfactual else {})
            .get("official_result", {})
            .get("token_usage", {})
        )
        for role in ("planner", "worker", "reporter"):
            role_usage = usage.get(role) or {}
            totals[f"{role}_prompt_tokens"] += int(
                role_usage.get("prompt_tokens", role_usage.get("input_tokens", 0)) or 0
            )
            totals[f"{role}_completion_tokens"] += int(
                role_usage.get("completion_tokens", role_usage.get("output_tokens", 0)) or 0
            )
    totals["official_graph_run_count"] = len(records)
    return dict(totals)


def _aggregate_retrieval_usage(records: list[EvaluationRecord]) -> dict[str, int]:
    totals = Counter()
    for record in records:
        retrieval = record.metadata.get("agent_skill_retrieval", {})
        usage = retrieval.get("token_usage") or {}
        if retrieval.get("enabled") and int(retrieval.get("available_skill_count") or 0) > 0:
            totals["call_count"] += 1
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
    return dict(totals)


def _aggregate_judge_usage(records: list[EvaluationRecord]) -> dict[str, int]:
    totals = Counter()
    for record in records:
        score = record.counterfactual_score
        usage = (score.raw if score else {}).get("token_usage") or {}
        totals["call_count"] += int(score is not None)
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
    return dict(totals)


def _guard_config(
    output_dir: Path, config: dict[str, Any], config_hash: str, config_path: Path | None
) -> None:
    guard_path = output_dir / "protocol_config.json"
    payload = {
        "schema_version": CONFIG_SCHEMA,
        "config_sha256": config_hash,
        "config_source": str(config_path.resolve()) if config_path else "programmatic",
        "config": config,
    }
    if guard_path.exists():
        previous = _read_json(guard_path)
        if previous.get("config_sha256") != config_hash:
            raise ValueError(
                f"Refusing to resume {output_dir}: protocol config hash changed ({previous.get('config_sha256')} != {config_hash})"
            )
        return
    _write_json(guard_path, payload)


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "experiment_name",
        "output_dir",
        "continuation_cases_path",
        "official_config_path",
        "official_prompt_bundle_path",
        "official_model_alias",
        "agent",
        "judge",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Continuation config missing keys: {missing}")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"Expected schema_version={CONFIG_SCHEMA}")


def _portable_config(config: dict[str, Any]) -> dict[str, Any]:
    forbidden = [
        key
        for key in _flatten_keys(config)
        if "api_key" in key and (not key.endswith("api_key_env"))
    ]
    if forbidden:
        raise ValueError(f"Literal API key fields are forbidden: {forbidden}")
    return config


def _runtime_contract() -> dict[str, str]:
    return {
        "disable_thinking": os.getenv("CCE_MOBILITYBENCH_DISABLE_THINKING", "1"),
        "force_tool_choice_required": os.getenv(
            "CCE_MOBILITYBENCH_FORCE_TOOL_CHOICE_REQUIRED", "1"
        ),
        "normalize_tool_args": os.getenv("CCE_MOBILITYBENCH_NORMALIZE_TOOL_ARGS", "1"),
        "plan_json_fallback": os.getenv("CCE_MOBILITYBENCH_PLAN_JSON_FALLBACK", "0"),
        "synthetic_plan_fallback": os.getenv("CCE_MOBILITYBENCH_SYNTHETIC_PLAN_FALLBACK", "0"),
    }


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_flatten_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            keys.extend(_flatten_keys(child, f"{prefix}[{index}]"))
    return keys


def _load_local_secret_into_process(source_name: str, target_name: str) -> None:
    if os.getenv(target_name):
        return
    try:
        from cce_mobility import local_secrets
    except ImportError:
        return
    value = getattr(local_secrets, source_name, "")
    if value:
        os.environ[target_name] = value


def _response_usage(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    usage = raw.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    random.seed(42)
    raise SystemExit(main())
