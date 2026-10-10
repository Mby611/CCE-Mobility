from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from cce_mobility.core.config import ExperimentConfig
from cce_mobility.core.llm import ChatClient
from cce_mobility.core.metrics import add_experiment_coverage_metrics, aggregate_records
from cce_mobility.core.schema import (
    AgentTrajectory,
    EvaluationRecord,
    ExperimentReport,
    InterventionSpec,
    RunMode,
    ToolEvent,
)
from cce_mobility.interventions.analyzer import HeuristicTrajectoryAnalyzer
from cce_mobility.interventions.frontier import FrontierSelector
from cce_mobility.interventions.generator import (
    FrozenInterventionGenerator,
    GridInterventionGenerator,
    LLMInterventionGenerator,
    OntologyAwareLLMInterventionGenerator,
)
from cce_mobility.interventions.predictor import HeuristicWMPredictiveCritic
from cce_mobility.interventions.spec_builder import RuleSpecBuilder
from cce_mobility.interventions.validator import HeuristicInterventionValidator
from cce_mobility.judges.recovery import HeuristicRecoveryJudge
from cce_mobility.judges.spec_llm import SpecGroundedLLMJudge
from cce_mobility.methods.factory import create_method
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
RECOVERY_SYSTEM_PROMPT = "You are a mobility assistant operating in a dynamic environment.\n\nYou can only rely on the current user request, the official factual trajectory summary,\nand the current environment update. Do not mention internal evaluation or experiment\ndetails. Detect changed task conditions, avoid using invalidated information, and\ngive a concise recovery response to the user."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CCE post-hoc counterfactual recovery eval over official MobilityBench traces."
    )
    parser.add_argument("config", help="Experiment JSON/YAML config.")
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(Path(args.config))
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    factual_records = _load_factual_records(Path(config.benchmark.data_path))
    if config.benchmark.limit:
        factual_records = factual_records[: config.benchmark.limit]
    agent_client = ChatClient(config.agent)
    world_model_client = ChatClient(config.world_model)
    judge_client = ChatClient(config.judge)
    method = create_method(config.method)
    _configure_method(method, config.extra)
    _load_method_memory(method, config.extra, config.experiment_name)
    _maybe_freeze_method(method, config.extra, config.experiment_name)
    analyzer = HeuristicTrajectoryAnalyzer()
    grid_generator = GridInterventionGenerator(
        dimensions=config.intervention.dimensions or None,
        severity_grid=config.intervention.severity_grid,
    )
    if config.intervention.policy == "frozen":
        frozen_path = config.extra.get("frozen_intervention_path", "")
        generator = FrozenInterventionGenerator(frozen_path)
    elif config.intervention.policy in {
        "ontology",
        "ontology_discovery",
        "llm_ontology",
        "ontology_frontier",
    }:
        generator = OntologyAwareLLMInterventionGenerator(
            world_model_client,
            dimensions=config.intervention.dimensions or None,
            severity_grid=config.intervention.severity_grid,
            fallback=grid_generator,
        )
    elif config.intervention.policy in {"llm", "llm_frontier", "world_model"}:
        generator = LLMInterventionGenerator(
            world_model_client,
            dimensions=config.intervention.dimensions or None,
            severity_grid=config.intervention.severity_grid,
            fallback=grid_generator,
        )
    else:
        generator = grid_generator
    seed_ontology_fn = getattr(method, "seed_intervention_ontology", None)
    if callable(seed_ontology_fn):
        seed_ontology_fn(config.intervention.dimensions or None)
    spec_builder = RuleSpecBuilder()
    validator = HeuristicInterventionValidator(config.intervention.validity_threshold)
    selector = FrontierSelector(
        low=config.intervention.frontier_low, high=config.intervention.frontier_high
    )
    wm_predictor = HeuristicWMPredictiveCritic()
    judge = (
        HeuristicRecoveryJudge()
        if config.judge.provider in {"local", "heuristic", "placeholder"}
        else SpecGroundedLLMJudge(judge_client)
    )
    output_records: list[EvaluationRecord] = []
    actual_iterations = 0
    online_update_after_sample = bool(config.extra.get("online_update_after_sample", False))
    for iteration in range(1, config.iterations + 1):
        actual_iterations = iteration
        iteration_path = output_dir / f"iteration_{iteration}.jsonl"
        iteration_path.write_text("", encoding="utf-8")
        method.before_iteration(output_records)
        _apply_method_guidance(method, generator)
        iter_records: list[EvaluationRecord] = []
        print(
            f"[{config.experiment_name}] iteration {iteration}/{config.iterations}: {len(factual_records)} official traces"
        )
        for idx, base_record in enumerate(factual_records, start=1):
            sample_records: list[EvaluationRecord] = []
            sample = base_record.sample
            factual = base_record.factual
            if factual is None:
                continue
            print(
                f"[{config.experiment_name}] sample {idx}/{len(factual_records)} {sample.sample_id}"
            )
            factual_score = judge.score(EvaluationRecord(sample=sample, factual=factual))
            points = analyzer.find_points(sample, factual)
            candidates = generator.generate(sample, factual, points)
            prepared = []
            for cand in candidates[: config.intervention.candidate_budget]:
                cand.recovery_spec = cand.recovery_spec or spec_builder.build(cand, sample)
                cand = validator.validate(cand, sample)
                if validator.is_valid(cand):
                    prepared.append(cand)
            selected = selector.select(prepared, budget=config.intervention.selected_budget)
            print(f"  selected {len(selected)} counterfactual(s)")
            if not selected:
                record = base_record.model_copy(deep=True)
                record.factual_score = factual_score
                record.metadata["posthoc_iteration"] = iteration
                record.metadata["iteration"] = iteration
                iter_records.append(record)
                sample_records.append(record)
                output_records.append(record)
                _append_jsonl(records_path, record)
                _append_jsonl(iteration_path, record)
                if online_update_after_sample:
                    method.observe(sample_records)
                continue
            for cand in selected:
                wm_prediction = wm_predictor.predict(
                    sample=sample,
                    factual=factual,
                    candidate=cand,
                    world_model_guidance=_world_model_guidance(method),
                )
                cf, agent_retrieval = _run_recovery_agent(
                    agent_client=agent_client,
                    sample_query=sample.query,
                    factual=factual,
                    intervention=cand.spec,
                    agent_guidance=_agent_guidance(method),
                    agent_skill_registry=_agent_skill_registry(method),
                    agent_owned_retrieval=bool(config.extra.get("agent_owned_retrieval", False)),
                    agent_retrieval_max_skills=int(
                        config.extra.get("agent_retrieval_max_skills", 3)
                    ),
                )
                counterfactual_metrics = _posthoc_hard_metrics(cf, cand.spec)
                record = EvaluationRecord(
                    sample=sample,
                    factual=factual,
                    counterfactual=cf,
                    intervention=cand.spec,
                    recovery_spec=cand.recovery_spec,
                    wm_prediction=wm_prediction,
                    factual_score=factual_score,
                    metadata={
                        **base_record.metadata,
                        "intervention_validity": cand.validity_score,
                        "intervention_utility": cand.utility,
                        "validation_notes": cand.validation_notes,
                        "factual_hard_metrics": _official_factual_hard_metrics(factual),
                        "counterfactual_hard_metrics": counterfactual_metrics,
                        "agent_skill_retrieval": agent_retrieval,
                        "posthoc_iteration": iteration,
                        "iteration": iteration,
                    },
                )
                record.counterfactual_score = judge.score(record)
                record.metadata["wm_prediction_metrics"] = _prediction_metrics(record)
                iter_records.append(record)
                sample_records.append(record)
                output_records.append(record)
                _append_jsonl(records_path, record)
                _append_jsonl(iteration_path, record)
            if online_update_after_sample and sample_records:
                method.observe(sample_records)
        if not online_update_after_sample:
            method.observe(iter_records)
        _save_method_memory_checkpoint(method, output_dir, iteration, config.extra)
        should_stop_fn = getattr(method, "should_stop", None)
        if callable(should_stop_fn) and should_stop_fn():
            print(
                f"[{config.experiment_name}] method requested early stop after iteration {iteration}"
            )
            break
    metrics = aggregate_records(output_records)
    add_experiment_coverage_metrics(
        metrics,
        sample_count=len(factual_records),
        iterations=actual_iterations,
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
            "official_factual_records_path": config.benchmark.data_path,
            "posthoc_official_aligned": True,
            "posthoc_method_lifecycle_enabled": True,
        }
    )
    method_metrics_fn = getattr(method, "report_metrics", None)
    if callable(method_metrics_fn):
        metrics.update(method_metrics_fn())
    metrics.update(_agent_retrieval_metrics(output_records))
    _save_method_memory_final(method, output_dir, config.extra)
    report = ExperimentReport(
        experiment_name=config.experiment_name,
        benchmark=config.benchmark.name,
        method=config.method,
        total_samples=len(factual_records),
        metrics=metrics,
        records_path=str(records_path),
    )
    (output_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 0


def _load_factual_records(path: Path) -> list[EvaluationRecord]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(EvaluationRecord.model_validate(json.loads(line)))
    return rows


def _run_recovery_agent(
    *,
    agent_client: ChatClient,
    sample_query: str,
    factual: AgentTrajectory,
    intervention: InterventionSpec,
    agent_guidance: str = "",
    agent_skill_registry: Any | None = None,
    agent_owned_retrieval: bool = False,
    agent_retrieval_max_skills: int = 3,
) -> tuple[AgentTrajectory, dict[str, Any]]:
    update = _agent_visible_update(intervention)
    trajectory_summary = _trajectory_summary(factual)
    selected_guidance = ""
    retrieval_metadata: dict[str, Any] = {}
    if agent_owned_retrieval:
        selected_guidance, retrieval_metadata = _retrieve_posthoc_agent_skills(
            agent_client=agent_client,
            registry=agent_skill_registry,
            query=sample_query,
            environment_observation=f"Official factual trajectory summary:\n{trajectory_summary}\n\nCurrent environment update:\n{update}",
            max_skills=agent_retrieval_max_skills,
        )
    else:
        selected_guidance = _agent_visible_text(agent_guidance) if agent_guidance else ""
        retrieval_metadata = {
            "enabled": False,
            "source": "disabled",
            "selected_skill_ids": [],
            "ignored_skill_ids": [],
            "no_skill_needed": True,
        }
    parts = [
        f"User request:\n{sample_query}",
        f"Official factual trajectory summary:\n{trajectory_summary}",
        f"Current environment update:\n{update}",
    ]
    if selected_guidance:
        heading = (
            "Selected recovery guidance" if agent_owned_retrieval else "Recovery memory catalog"
        )
        parts.append(f"{heading}:\n{selected_guidance}")
    parts.append(
        "Continue assisting the user under the current environment. Explain the changed condition and provide a feasible recovery or clarification."
    )
    user_prompt = "\n\n".join(parts)
    if agent_client.is_placeholder:
        text = f"I noticed the current environment changed: {update}. I will avoid the invalidated option and suggest a feasible alternative or ask for clarification."
        error = ""
    else:
        response = agent_client.chat(
            system_prompt=RECOVERY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=2048,
        )
        text = response.text if response.ok else f"[LLM_ERROR] {response.error}"
        error = "" if response.ok else response.error
    events = [
        *factual.events,
        ToolEvent(
            step=len(factual.events),
            tool_name="official_posthoc_environment_update",
            observation=f"[ENVIRONMENT_UPDATE] {update}",
            action_text=text,
            success=not bool(error),
            metadata={
                "agent_visible": True,
                "posthoc_recovery": True,
                "intervention_id": intervention.intervention_id,
                "intervention_dimension": intervention.dimension,
            },
        ),
    ]
    return (
        AgentTrajectory(
            sample_id=factual.sample_id,
            run_mode=RunMode.COUNTERFACTUAL,
            events=events,
            final_response=text,
            raw={
                **factual.raw,
                "adapter": "official_mobilitybench_posthoc_recovery",
                "adapter_scope": "official_trace_posthoc_environment_update",
                "llm_error": bool(error),
                "llm_error_message": error,
                "intervention_dimension": intervention.dimension,
                "agent_skill_retrieval": retrieval_metadata,
            },
        ),
        retrieval_metadata,
    )


def _retrieve_posthoc_agent_skills(
    *,
    agent_client: ChatClient,
    registry: Any | None,
    query: str,
    environment_observation: str,
    max_skills: int,
) -> tuple[str, dict[str, Any]]:
    if registry is None:
        return (
            "",
            {
                "enabled": True,
                "source": "no_registry",
                "selected_skill_ids": [],
                "ignored_skill_ids": [],
                "no_skill_needed": True,
                "available_skill_count": 0,
                "rendered_skill_ids": [],
            },
        )
    max_skills = max(1, max_skills)
    skills = active_agent_skills(registry, limit=8)
    if not skills:
        return (
            "",
            {
                "enabled": True,
                "source": "empty_registry",
                "selected_skill_ids": [],
                "ignored_skill_ids": [],
                "no_skill_needed": True,
                "available_skill_count": 0,
                "rendered_skill_ids": [],
            },
        )
    skill_index = render_agent_skill_index(skills)
    available_ids = {skill.skill_id for skill in skills}
    prompt = build_agent_retrieval_prompt(
        query=query,
        environment_observation=environment_observation,
        skill_index=skill_index,
        max_skills=max_skills,
    )
    if agent_client.is_placeholder:
        decision = heuristic_retrieval_decision(
            query=query,
            environment_observation=environment_observation,
            skills=skills,
            max_skills=max_skills,
        )
    else:
        response = agent_client.chat(
            system_prompt=AGENT_RETRIEVAL_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=1024,
        )
        if response.ok:
            decision = parse_retrieval_decision(
                response.text, available_skill_ids=available_ids, max_skills=max_skills
            )
            if decision.parse_error:
                fallback = heuristic_retrieval_decision(
                    query=query,
                    environment_observation=environment_observation,
                    skills=skills,
                    max_skills=max_skills,
                )
                decision.selected_skill_ids = fallback.selected_skill_ids
                decision.ignored_skill_ids = fallback.ignored_skill_ids
                decision.selection_rationale = fallback.selection_rationale
                decision.supporting_observable_evidence = fallback.supporting_observable_evidence
                decision.trigger_match_reason = fallback.trigger_match_reason
                decision.ignored_skill_reasons = fallback.ignored_skill_reasons
                decision.no_skill_needed = fallback.no_skill_needed
                decision.source = "llm_parse_error_heuristic"
        else:
            decision = heuristic_retrieval_decision(
                query=query,
                environment_observation=environment_observation,
                skills=skills,
                max_skills=max_skills,
            )
            decision.source = "llm_error_heuristic"
            decision.raw_response = response.error
    selected_guidance = render_selected_agent_skill_blocks(registry, decision.selected_skill_ids)
    metadata = decision.model_dump()
    metadata.update(
        {
            "enabled": True,
            "available_skill_count": len(skills),
            "rendered_skill_ids": decision.selected_skill_ids,
        }
    )
    return (selected_guidance, metadata)


def _trajectory_summary(trajectory: AgentTrajectory, limit: int = 4000) -> str:
    parts = []
    for event in trajectory.events:
        if event.metadata.get("official_role") == "planner":
            parts.append(
                f"Planner:\n- intent: {event.parameters.get('intent', '')}\n- steps: {event.parameters.get('steps', [])}\n- thinking: {event.action_text[:700]}"
            )
        elif event.metadata.get("official_role") == "worker":
            parts.append(
                f"Worker/tool:\n- tool: {event.tool_name}\n- args: {event.parameters}\n- observation: {event.observation[:900]}"
            )
        elif event.metadata.get("official_role") == "reporter":
            parts.append(f"Reporter final response:\n{event.action_text[:900]}")
    return "\n\n".join(parts)[:limit]


def _agent_visible_update(intervention: InterventionSpec) -> str:
    text = intervention.condition_text or intervention.diagnostic_goal or intervention.dimension
    return (
        text.replace("counterfactual", "current")
        .replace("Counterfactual", "Current")
        .replace("intervention", "environment update")
        .replace("Intervention", "Environment update")
    )


def _official_factual_hard_metrics(factual: AgentTrajectory) -> dict[str, Any]:
    llm_error = bool(factual.raw.get("official_error"))
    return {
        "benchmark": "mobilitybench",
        "run_mode": "factual",
        "adapter_scope": "official_mbench_raw_results",
        "official_score": None,
        "parse_success": bool(factual.final_response) and (not llm_error),
        "commonsense_pass": bool(factual.final_response) and (not llm_error),
        "hard_pass": bool(factual.final_response) and (not llm_error),
        "final_pass": bool(factual.final_response) and (not llm_error),
        "uses_unavailable_identifier": False,
        "unavailable_id_count": 0,
        "estimated_cost": 0.0,
        "llm_error": llm_error,
        "llm_error_message": str(factual.raw.get("official_error") or ""),
    }


def _posthoc_hard_metrics(cf: AgentTrajectory, intervention: InterventionSpec) -> dict[str, Any]:
    text = cf.final_response.strip()
    lower = text.lower()
    llm_error = bool(cf.raw.get("llm_error"))
    parse_success = bool(text) and (not llm_error)
    detects = any(
        (
            token in lower
            for token in [
                "changed",
                "closed",
                "blocked",
                "unavailable",
                "delay",
                "empty",
                "conflict",
                "不可用",
                "关闭",
                "封闭",
                "拥堵",
                "延误",
                "变化",
                "无法",
                "没有结果",
            ]
        )
    )
    recovers = any(
        (
            token in lower
            for token in [
                "alternative",
                "replan",
                "instead",
                "verify",
                "clarification",
                "替代",
                "重新规划",
                "建议",
                "确认",
                "换",
            ]
        )
    )
    negative = intervention.is_negative
    hard_pass = parse_success and (not negative and (detects or recovers) or negative)
    return {
        "benchmark": "mobilitybench",
        "run_mode": "counterfactual",
        "adapter_scope": "official_trace_posthoc_environment_update",
        "official_score": None,
        "parse_success": parse_success,
        "commonsense_pass": parse_success,
        "hard_pass": hard_pass,
        "final_pass": hard_pass,
        "uses_unavailable_identifier": False,
        "unavailable_id_count": 0,
        "estimated_cost": 0.0,
        "llm_error": llm_error,
        "llm_error_message": cf.raw.get("llm_error_message", ""),
        "detected_environment_update": detects,
        "showed_recovery": recovers,
        "intervention_dimension": intervention.dimension,
    }


def _append_jsonl(path: Path, record: EvaluationRecord) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def _configure_method(method: Any, extra: dict[str, Any]) -> None:
    configure_fn = getattr(method, "configure", None)
    if callable(configure_fn):
        configure_fn(extra)


def _load_method_memory(method: Any, extra: dict[str, Any], experiment_name: str) -> None:
    path = extra.get("load_skill_memory_path") or extra.get("load_registry_path")
    if path:
        load_fn = getattr(method, "load_registry", None)
        if callable(load_fn):
            load_fn(path)
            print(f"[{experiment_name}] loaded skill/spec memory from {path}")


def _maybe_freeze_method(method: Any, extra: dict[str, Any], experiment_name: str) -> None:
    if not bool(extra.get("freeze_method", False)):
        return
    freeze_fn = getattr(method, "freeze", None)
    if callable(freeze_fn):
        freeze_fn()
        print(f"[{experiment_name}] method memory frozen")


def _apply_method_guidance(method: Any, generator: Any) -> None:
    wm_guidance_fn = getattr(method, "world_model_guidance", None)
    if callable(wm_guidance_fn) and hasattr(generator, "set_guidance"):
        generator.set_guidance(wm_guidance_fn())
    ontology_registry_fn = getattr(method, "intervention_ontology_registry", None)
    if callable(ontology_registry_fn) and hasattr(generator, "set_ontology_registry"):
        generator.set_ontology_registry(ontology_registry_fn())


def _agent_guidance(method: Any) -> str:
    guidance_fn = getattr(method, "agent_guidance", None)
    return guidance_fn() if callable(guidance_fn) else ""


def _agent_skill_registry(method: Any) -> Any | None:
    registry_fn = getattr(method, "agent_skill_registry", None)
    return registry_fn() if callable(registry_fn) else None


def _world_model_guidance(method: Any) -> str:
    guidance_fn = getattr(method, "world_model_guidance", None)
    return guidance_fn() if callable(guidance_fn) else ""


def _agent_retrieval_metrics(records: list[EvaluationRecord]) -> dict[str, Any]:
    from cce_mobility.core.metrics import _aggregate_agent_retrieval_metrics

    return _aggregate_agent_retrieval_metrics(records)


def _save_method_memory_checkpoint(
    method: Any, output_dir: Path, iteration: int, extra: dict[str, Any]
) -> None:
    if not bool(extra.get("save_skill_memory", True)):
        return
    save_fn = getattr(method, "save_registry", None)
    if callable(save_fn):
        save_fn(output_dir / f"skill_memory_iteration_{iteration}.jsonl")
    extra_save_fn = getattr(method, "save_extra_artifacts", None)
    if callable(extra_save_fn):
        extra_save_fn(output_dir, iteration=iteration)


def _save_method_memory_final(method: Any, output_dir: Path, extra: dict[str, Any]) -> None:
    if not bool(extra.get("save_skill_memory", True)):
        return
    save_fn = getattr(method, "save_registry", None)
    if callable(save_fn):
        save_fn(output_dir / "skill_memory_final.jsonl")
    extra_save_fn = getattr(method, "save_extra_artifacts", None)
    if callable(extra_save_fn):
        extra_save_fn(output_dir, iteration=None)


def _agent_visible_text(text: str) -> str:
    return (
        text.replace("counterfactual", "current environment")
        .replace("Counterfactual", "Current environment")
        .replace("intervention", "environment update")
        .replace("Intervention", "Environment update")
        .replace("evaluator", "system")
        .replace("Evaluator", "System")
    )


def _prediction_metrics(record: EvaluationRecord) -> dict[str, Any]:
    if record.wm_prediction is None or record.counterfactual_score is None:
        return {}
    score = record.counterfactual_score.total
    actual_bin = "low" if score < 2.0 else "medium" if score < 3.0 else "high"
    predicted = record.wm_prediction.predicted_recovery_score_bin
    actual_failures = _actual_failure_modes(record)
    predicted_failures = set(record.wm_prediction.predicted_failure_modes)
    overlap = predicted_failures.intersection(actual_failures)
    return {
        "predicted_score_bin": predicted,
        "actual_score_bin": actual_bin,
        "score_bin_hit": predicted == actual_bin,
        "predicted_failure_modes": sorted(predicted_failures),
        "actual_failure_modes": sorted(actual_failures),
        "failure_mode_hit": bool(overlap),
        "failure_mode_precision": round(len(overlap) / len(predicted_failures), 4)
        if predicted_failures
        else 0.0,
        "failure_mode_recall": round(len(overlap) / len(actual_failures), 4)
        if actual_failures
        else 0.0,
        "confidence": record.wm_prediction.confidence,
    }


def _actual_failure_modes(record: EvaluationRecord) -> set[str]:
    score = record.counterfactual_score
    if score is None or score.total >= 3.0:
        return set()
    failures = set()
    if score.detection < 0.5:
        failures.add("missed_change_detection")
    if score.replanning < 0.5:
        failures.add("no_replanning")
    if score.verification < 0.5:
        failures.add("constraint_not_rechecked")
    if score.communication < 0.5:
        failures.add("poor_user_communication")
    if score.safety < 0.5:
        failures.add("used_invalid_resource")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
