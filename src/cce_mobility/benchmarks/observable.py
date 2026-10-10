from __future__ import annotations
import ast
import csv
import json
from pathlib import Path
from typing import Any
from cce_mobility.core.config import BenchmarkConfig
from cce_mobility.core.llm import ChatClient
from cce_mobility.core.schema import (
    AgentTrajectory,
    BenchmarkName,
    InterventionSpec,
    RunMode,
    TaskSample,
    ToolEvent,
)
from cce_mobility.skills.registry import SkillRegistry
from cce_mobility.skills.retrieval import (
    AGENT_RETRIEVAL_SYSTEM_PROMPT,
    active_agent_skills,
    build_agent_retrieval_prompt,
    heuristic_retrieval_decision,
    normalize_filter_values,
    parse_retrieval_decision,
    render_agent_skill_index,
    render_selected_agent_skill_blocks,
    sanitize_agent_visible_text,
)

OBSERVABLE_SYSTEM_PROMPT = "You are a careful travel and mobility assistant. Answer using only the current observable task state, environment observations, and tool evidence provided in the prompt. If the current state invalidates a previous option, verify constraints, choose a feasible alternative when possible, and explain the recovery clearly to the user."


class ObservableStateAdapter:
    """Lightweight adapter for benchmarks before full tool replay is wired.

    This adapter intentionally does not claim official benchmark scoring. It
    gives every benchmark the same CCE-facing surface: normalized samples,
    agent-visible current state, optional environment updates, and simple hard
    metrics for smoke tests.
    """

    name: str

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.chat_client: ChatClient | None = None
        self.agent_guidance = ""
        self.agent_skill_registry: SkillRegistry | None = None
        self.runtime_options: dict[str, Any] = dict(config.extra or {})

    def set_chat_client(self, chat_client: ChatClient) -> None:
        self.chat_client = chat_client

    def set_agent_guidance(self, guidance: str) -> None:
        self.agent_guidance = sanitize_agent_visible_text(guidance)

    def set_agent_skill_registry(self, registry: SkillRegistry) -> None:
        self.agent_skill_registry = registry

    def set_runtime_options(self, options: dict[str, Any]) -> None:
        self.runtime_options.update(options or {})

    def run_agent(
        self, sample: TaskSample, *, agent_id: str, intervention: InterventionSpec | None = None
    ) -> AgentTrajectory:
        run_mode = RunMode.COUNTERFACTUAL if intervention else RunMode.FACTUAL
        observation = self._observable_state(sample)
        if intervention is not None:
            observation = "\n".join(
                [observation, f"[ENVIRONMENT_UPDATE] {_agent_visible_condition(intervention)}"]
            )
        final, skill_retrieval = self._generate_response(sample, observation, intervention)
        llm_error = _extract_llm_error(final)
        return AgentTrajectory(
            sample_id=sample.sample_id,
            run_mode=run_mode,
            events=[
                ToolEvent(
                    step=0,
                    tool_name=f"{self.name}_observable_state",
                    observation=observation,
                    action_text=final,
                    metadata={"agent_visible": True},
                )
            ],
            final_response=final,
            raw={
                "adapter": f"{self.name}_observable_light",
                "adapter_scope": "lightweight_observable_state",
                "agent_id": agent_id,
                "llm_error": bool(llm_error),
                "llm_error_message": llm_error,
                "intervention_dimension": intervention.dimension if intervention else None,
                "skill_retrieval": skill_retrieval,
            },
        )

    def evaluate_trajectory(
        self,
        sample: TaskSample,
        trajectory: AgentTrajectory,
        intervention: InterventionSpec | None = None,
    ) -> dict[str, Any]:
        text = trajectory.final_response.strip()
        lower = text.lower()
        llm_error = bool(trajectory.raw.get("llm_error"))
        parse_success = bool(text) and (not llm_error)
        detects_update = _detects_environment_update(lower, intervention)
        recovers = _shows_recovery(lower)
        negative = bool(intervention and intervention.is_negative)
        if intervention is None:
            hard_pass = parse_success
        elif negative:
            hard_pass = parse_success and (not _overreacts(lower))
        else:
            hard_pass = parse_success and (detects_update or recovers)
        return {
            "benchmark": self.name,
            "run_mode": trajectory.run_mode.value,
            "adapter_scope": "lightweight_observable_state",
            "official_score": None,
            "parse_success": parse_success,
            "commonsense_pass": parse_success,
            "hard_pass": hard_pass,
            "final_pass": hard_pass,
            "uses_unavailable_identifier": False,
            "unavailable_id_count": 0,
            "estimated_cost": 0.0,
            "llm_error": llm_error,
            "llm_error_message": trajectory.raw.get("llm_error_message", ""),
            "detected_environment_update": detects_update,
            "showed_recovery": recovers,
            "intervention_dimension": intervention.dimension if intervention else None,
        }

    def _observable_state(self, sample: TaskSample) -> str:
        raise NotImplementedError

    def _generate_response(
        self, sample: TaskSample, observation: str, intervention: InterventionSpec | None
    ) -> tuple[str, dict[str, Any]]:
        if self.chat_client is None or self.chat_client.is_placeholder:
            return (_placeholder_mobility_response(sample, observation, intervention), {})
        selected_guidance, retrieval_metadata = self._retrieve_agent_skills(
            query=sample.query, environment_observation=observation
        )
        guidance = ""
        retrieval_enabled = bool(retrieval_metadata.get("enabled"))
        base_guidance = sanitize_agent_visible_text(
            str(self.runtime_options.get("base_agent_guidance") or "")
        ).strip()
        if base_guidance:
            guidance += f"\n\nBaseline planning guidance from prior training:\n{base_guidance}\n\nUse this guidance only when it is supported by the current observable task state and tool results."
        if selected_guidance:
            guidance += f"\n\nRelevant prior recovery guidance selected from the skill catalog:\n{selected_guidance}\n\nUse selected guidance only where it is supported by the current observable task state. Ignore unrelated guidance."
        elif self.agent_guidance and (not retrieval_enabled) and (not guidance):
            guidance = f"\n\nAgent recovery skill memory catalog for self-retrieval:\n{self.agent_guidance}"
        user_prompt = "\n\n".join(
            [
                f"User request:\n{sample.query}",
                f"Current observable task state:\n{observation}",
                guidance,
                "Respond to the user. Use normal assistant language; do not mention benchmark or evaluator internals.",
            ]
        )
        response = self.chat_client.chat(
            system_prompt=OBSERVABLE_SYSTEM_PROMPT, user_prompt=user_prompt
        )
        if not response.ok:
            return (f"[LLM_ERROR] {response.error}", retrieval_metadata)
        return (sanitize_agent_visible_text(response.text.strip()), retrieval_metadata)

    def _retrieve_agent_skills(
        self, *, query: str, environment_observation: str
    ) -> tuple[str, dict[str, Any]]:
        if self.agent_skill_registry is None:
            return ("", {})
        if not bool(self.runtime_options.get("agent_owned_retrieval", True)):
            return (
                "",
                {
                    "enabled": False,
                    "source": "disabled",
                    "selected_skill_ids": [],
                    "ignored_skill_ids": [],
                    "rendered_skill_ids": [],
                    "no_skill_needed": True,
                },
            )
        max_skills = max(1, int(self.runtime_options.get("agent_retrieval_max_skills", 3)))
        index_limit = max(
            max_skills, int(self.runtime_options.get("agent_retrieval_index_limit", 8))
        )
        exclude_dimensions = normalize_filter_values(
            self.runtime_options.get("agent_retrieval_exclude_dimensions")
        )
        exclude_domain_operators = normalize_filter_values(
            self.runtime_options.get("agent_retrieval_exclude_domain_operators")
        )
        unfiltered_skill_count = len(
            active_agent_skills(
                self.agent_skill_registry,
                limit=max(index_limit, len(self.agent_skill_registry.all())),
            )
        )
        skills = active_agent_skills(
            self.agent_skill_registry,
            limit=index_limit,
            exclude_dimensions=exclude_dimensions,
            exclude_domain_operators=exclude_domain_operators,
        )
        if not skills:
            return (
                "",
                {
                    "available_skill_count": 0,
                    "unfiltered_available_skill_count": unfiltered_skill_count,
                    "enabled": True,
                    "source": "filtered_empty",
                    "selected_skill_ids": [],
                    "ignored_skill_ids": [],
                    "rendered_skill_ids": [],
                    "no_skill_needed": True,
                    "max_skills": max_skills,
                    "excluded_dimensions": sorted(exclude_dimensions),
                    "excluded_domain_operators": sorted(exclude_domain_operators),
                },
            )
        available_ids = {skill.skill_id for skill in skills}
        skill_index = render_agent_skill_index(skills)
        prompt = build_agent_retrieval_prompt(
            query=query,
            environment_observation=environment_observation,
            skill_index=skill_index,
            max_skills=max_skills,
        )
        if self.chat_client is None or self.chat_client.is_placeholder:
            decision = heuristic_retrieval_decision(
                query=query,
                environment_observation=environment_observation,
                skills=skills,
                max_skills=max_skills,
            )
        else:
            response = self.chat_client.chat(
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
                    decision.supporting_observable_evidence = (
                        fallback.supporting_observable_evidence
                    )
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
        selected_guidance = render_selected_agent_skill_blocks(
            self.agent_skill_registry, decision.selected_skill_ids
        )
        metadata = decision.model_dump()
        metadata.update(
            {
                "available_skill_count": len(skills),
                "unfiltered_available_skill_count": unfiltered_skill_count,
                "enabled": True,
                "max_skills": max_skills,
                "excluded_dimensions": sorted(exclude_dimensions),
                "excluded_domain_operators": sorted(exclude_domain_operators),
                "rendered_skill_ids": decision.selected_skill_ids,
            }
        )
        return (selected_guidance, metadata)


class TravelBenchAdapter(ObservableStateAdapter):
    """TravelBench data adapter with CCE observable-state execution."""

    name = "travelbench"

    def load_samples(self, limit: int | None = None) -> list[TaskSample]:
        data_path = _resolve_data_path(
            self.config,
            default_parts=("datas", f"{self.config.extra.get('variant', 'single_turn')}.jsonl"),
        )
        rows = _load_jsonl(data_path)
        n = _limit_count(len(rows), limit, self.config.limit)
        samples: list[TaskSample] = []
        for idx, row in enumerate(rows[:n]):
            trace_id = str(row.get("trace_id") or idx)
            query = str(row.get("query") or row.get("intent") or "")
            samples.append(
                TaskSample(
                    sample_id=f"tb_{trace_id}",
                    benchmark=BenchmarkName.TRAVEL_BENCH,
                    query=query,
                    context={
                        "observable_state": str(row.get("context") or ""),
                        "time": str(row.get("time") or ""),
                        "intent": str(row.get("intent") or ""),
                        "primary_intent": str(row.get("primary_intent") or ""),
                        "raw": row,
                    },
                    ground_truth={
                        "intent": row.get("intent"),
                        "missing_tool": row.get("missing_tool"),
                        "missing_info": row.get("missing_info"),
                        "no_actionable": row.get("no_actionable"),
                    },
                    metadata={
                        "sample_idx": idx,
                        "trace_id": trace_id,
                        "source_path": str(data_path),
                        "split": self.config.split,
                    },
                )
            )
        return samples

    def _observable_state(self, sample: TaskSample) -> str:
        parts = [
            f"Time: {sample.context.get('time') or 'unknown'}",
            f"Primary intent: {sample.context.get('primary_intent') or 'unknown'}",
            f"Task intent: {sample.context.get('intent') or 'unknown'}",
            "Current environment observation:",
            str(sample.context.get("observable_state") or "(none)")[:2400],
        ]
        return "\n".join(parts)


class MobilityBenchAdapter(ObservableStateAdapter):
    """MobilityBench CSV adapter with CCE observable-state execution."""

    name = "mobilitybench"

    def load_samples(self, limit: int | None = None) -> list[TaskSample]:
        data_path = _resolve_mobilitybench_data_path(self.config)
        rows = _load_csv(data_path)
        n = _limit_count(len(rows), limit, self.config.limit)
        samples: list[TaskSample] = []
        for idx, row in enumerate(rows[:n]):
            query = str(row.get("query") or "")
            split = _safe_split_name(self.config.split or data_path.stem)
            source_case_id = str(row.get("case_id") or row.get("Case ID") or idx)
            sample_id = f"mb_{split}_{source_case_id}"
            tool_list = _parse_literal_or_text(row.get("tool_list") or row.get("tools_list"))
            samples.append(
                TaskSample(
                    sample_id=sample_id,
                    benchmark=BenchmarkName.MOBILITY_BENCH,
                    query=query,
                    context={
                        "observable_state": _clean_text(row.get("context") or ""),
                        "city": str(row.get("city") or row.get("user_city") or ""),
                        "task_scenario": str(row.get("task_scenario") or ""),
                        "intent_family": str(row.get("intent_family") or ""),
                        "tool_list": tool_list,
                        "route_answer": _first_non_empty(
                            row,
                            "route_ans",
                            "route_info",
                            "drive_ans_list",
                            "bike_ans_list",
                            "walk_ans_list",
                            "bus_ans_list",
                        ),
                        "poi_answer": _first_non_empty(
                            row, "poi_result", "near_poi_ans", "query_poi", "location_ans", "ans"
                        ),
                        "weather": _first_non_empty(row, "weather_description", "weather"),
                        "start_poi": _first_non_empty(row, "query_start_poi", "std_start"),
                        "transit_poi": _first_non_empty(row, "query_transit_poi", "std_transit"),
                        "end_poi": _first_non_empty(row, "query_end_poi", "std_end"),
                        "raw_fields": {
                            "road": _clean_text(row.get("road") or ""),
                            "duration": _clean_text(row.get("duration") or ""),
                            "time": _clean_text(row.get("time") or ""),
                            "slot_city": _clean_text(row.get("slot_city") or ""),
                        },
                        "raw": row,
                    },
                    ground_truth={
                        "tool_list": tool_list,
                        "route_ans": row.get("route_ans"),
                        "ans": row.get("ans"),
                        "route_info": row.get("route_info"),
                        "poi_result": row.get("poi_result"),
                    },
                    metadata={
                        "sample_idx": idx,
                        "source_case_id": source_case_id,
                        "source_path": str(data_path),
                        "split": self.config.split,
                        "task_scenario": row.get("task_scenario", ""),
                        "intent_family": row.get("intent_family", ""),
                    },
                )
            )
        return samples

    def _observable_state(self, sample: TaskSample) -> str:
        route_answer = str(sample.context.get("route_answer") or "")
        poi_answer = str(sample.context.get("poi_answer") or "")
        raw_fields = sample.context.get("raw_fields") or {}
        parts = [
            f"City: {sample.context.get('city') or 'unknown'}",
            f"Task scenario: {sample.context.get('task_scenario') or 'unknown'}",
            f"Intent family: {sample.context.get('intent_family') or 'unknown'}",
            f"Start POI: {_compact_text(sample.context.get('start_poi'), limit=500)}",
            f"Transit POI: {_compact_text(sample.context.get('transit_poi'), limit=500)}",
            f"End POI: {_compact_text(sample.context.get('end_poi'), limit=500)}",
            f"Tool evidence: {_compact_text(sample.context.get('tool_list'), limit=900)}",
            f"Route evidence: {(route_answer[:1600] if route_answer else '(none)')}",
            f"POI evidence: {(poi_answer[:900] if poi_answer else '(none)')}",
            f"Weather evidence: {_compact_text(sample.context.get('weather'), limit=500)}",
            f"Other mobility fields: {_compact_text(raw_fields, limit=700)}",
            "Current environment observation:",
            str(sample.context.get("observable_state") or "(none)")[:1200],
        ]
        return "\n".join(parts)


def _resolve_data_path(config: BenchmarkConfig, *, default_parts: tuple[str, ...]) -> Path:
    if config.data_path:
        path = Path(config.data_path)
    elif config.repo_path:
        path = Path(config.repo_path).joinpath(*default_parts)
    else:
        raise FileNotFoundError(
            f"{config.name} data_path is empty. Set benchmark.data_path or benchmark.repo_path."
        )
    if not path.exists():
        raise FileNotFoundError(f"{config.name} data_path not found: {path}")
    return path


def _resolve_mobilitybench_data_path(config: BenchmarkConfig) -> Path:
    if config.data_path:
        path = Path(config.data_path)
        if path.is_dir():
            split_path = path / f"{config.split}.csv"
            path = split_path if split_path.exists() else path / "query.csv"
    elif config.repo_path:
        csv_dir = Path(config.repo_path) / "MobilityBench_csv_files"
        split_path = csv_dir / f"{config.split}.csv"
        path = split_path if split_path.exists() else csv_dir / "query.csv"
    else:
        raise FileNotFoundError(
            "mobilitybench data_path is empty. Set benchmark.data_path or benchmark.repo_path."
        )
    if not path.exists():
        raise FileNotFoundError(f"mobilitybench data_path not found: {path}")
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _limit_count(total: int, explicit: int | None, config_limit: int | None) -> int:
    n = total
    if explicit is not None:
        n = min(n, explicit)
    elif config_limit is not None:
        n = min(n, config_limit)
    return n


def _parse_literal_or_text(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _first_non_empty(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _clean_text(row.get(key))
        if value:
            return value
    return ""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() in {"nan", "none", "null"} else text


def _safe_split_name(split: str) -> str:
    safe = "".join((ch if ch.isalnum() else "_" for ch in str(split).lower())).strip("_")
    return safe or "query"


def _placeholder_mobility_response(
    sample: TaskSample, observation: str, intervention: InterventionSpec | None
) -> str:
    if intervention is None:
        return "I reviewed the current task state and available evidence. I would answer the user's travel or mobility request by checking route, POI, timing, and preference constraints before giving the next action."
    update = _agent_visible_condition(intervention)
    if intervention.is_negative:
        return f"I checked the latest environment update: {update}. It does not change the user's actionable route or POI constraints, so I will keep the task focused and avoid unnecessary replanning."
    return f"I noticed the latest environment update: {update}. I will verify affected route, POI, time, and user constraints, avoid unavailable or contradicted options, and provide a feasible alternative before confirming with the user."


def _agent_visible_condition(intervention: InterventionSpec) -> str:
    text = intervention.condition_text.strip()
    if not text:
        text = f"{intervention.dimension.replace('_', ' ')} at severity {intervention.severity:g}"
    return sanitize_agent_visible_text(text)


def _sanitize_agent_visible_text(text: str) -> str:
    replacements = {
        "counterfactual notice": "latest environment observation",
        "counterfactual change": "current environment update",
        "counterfactual failures": "planning and recovery failures",
        "counterfactual failure": "planning and recovery failure",
        "counterfactual noise": "environment noise",
        "current counterfactual": "current observable task state",
        "counterfactual": "environment update",
        "intervention notice": "latest environment observation",
        "intervention": "environment update",
        "modified environment state": "current environment state",
        "modified data": "current data",
        "changed world": "current environment state",
        "changed travel state": "current travel state",
    }
    sanitized = text
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)
        sanitized = sanitized.replace(old.capitalize(), new.capitalize())
        sanitized = sanitized.replace(old.upper(), new.upper())
    return sanitized


def _extract_llm_error(text: str) -> str:
    prefix = "[LLM_ERROR]"
    if not text.startswith(prefix):
        return ""
    return text.removeprefix(prefix).strip()


def _detects_environment_update(text: str, intervention: InterventionSpec | None) -> bool:
    if intervention is None:
        return False
    update_terms = {
        "closed",
        "cancel",
        "unavailable",
        "blocked",
        "traffic",
        "congestion",
        "delay",
        "weather",
        "empty",
        "conflict",
        "change",
        "update",
        "关闭",
        "停运",
        "取消",
        "不可用",
        "不可达",
        "封闭",
        "拥堵",
        "堵车",
        "延误",
        "变更",
        "更新",
        "冲突",
    }
    dimension_terms = {
        token for token in intervention.dimension.replace("-", "_").split("_") if len(token) > 2
    }
    return bool((update_terms | dimension_terms) & set(text.replace("/", " ").split()))


def _shows_recovery(text: str) -> bool:
    return any(
        (
            term in text
            for term in (
                "alternative",
                "instead",
                "replan",
                "adjust",
                "replace",
                "switch",
                "verify",
                "confirm",
                "avoid",
                "clarification",
                "替代",
                "改走",
                "绕行",
                "重新规划",
                "重规划",
                "调整",
                "更换",
                "避开",
                "确认",
                "核实",
                "重新查询",
            )
        )
    )


def _overreacts(text: str) -> bool:
    return any((term in text for term in ("replan", "switch", "replace", "reroute")))


def _compact_text(value: Any, *, limit: int) -> str:
    if value in (None, "", []):
        return "(none)"
    return json.dumps(value, ensure_ascii=False, default=str)[:limit]
