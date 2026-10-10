from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from cce_mobility.core.schema import (
    AgentTrajectory,
    EvaluationRecord,
    InterventionCandidate,
    InterventionSpec,
    RunMode,
    TaskSample,
    ToolEvent,
)


CONTINUATION_PROTOCOL = "mobilitybench-state-reconstruction-continuation-v1"
FORMAL_EXPERIMENT_SCRIPTS = (
    "src/cce_mobility/workflows/mobilitybench/convert_mobilitybench_official_to_cce.py",
    "src/cce_mobility/workflows/mobilitybench/generate_mobilitybench_official_frozen_intervention_set.py",
    "src/cce_mobility/workflows/mobilitybench/prepare_mobilitybench_continuation_protocol.py",
    "src/cce_mobility/workflows/mobilitybench/reconcile_mobilitybench_official_run.py",
    "src/cce_mobility/workflows/mobilitybench/run_mobilitybench_formal_coevolution_curve.py",
    "src/cce_mobility/workflows/mobilitybench/run_mobilitybench_formal_coevolution_train.py",
    "src/cce_mobility/workflows/mobilitybench/run_mobilitybench_official_continuation.py",
    "scripts/run_mobilitybench_official_smoke.sh",
    "src/cce_mobility/workflows/mobilitybench/summarize_mobilitybench_official_run.py",
)

# These paths belong to the separate TP/SkillOpt experiment lines. They are
# deliberately outside the MB execution contract. Keep the exact helper in
# the legacy exclusion set so bundles made before it was introduced remain
# byte-compatible; the broader prefixes are used only when filtering status
# and validating a transition.
NON_MB_CONTINUATION_SOURCES = frozenset(
    {
        "src/cce_mobility/integrations/skillopt_qwen_compat.py",
    }
)
NON_MB_CONTINUATION_SOURCE_PREFIXES = (
    "src/cce_mobility/analysis",
    "src/cce_mobility/benchmarks/travelplanner",
    "src/cce_mobility/integrations/skillopt_",
    "src/cce_mobility/interventions/travelplanner",
    "src/cce_mobility/methods/cce_coevolution.py",
    "src/cce_mobility/methods/skill_evolution.py",
)


def _is_non_mb_continuation_source(relative: str) -> bool:
    relative = relative.strip().strip('"')
    return relative in NON_MB_CONTINUATION_SOURCES or any(
        relative.startswith(prefix) for prefix in NON_MB_CONTINUATION_SOURCE_PREFIXES
    )


SUPPORTED_TOOL_INTERVENTION_DIMENSIONS = {
    "road_blocked",
    "congestion_spike",
    "eta_shift",
    "poi_closed",
    "pickup_point_unreachable",
    "parking_unavailable",
    "charging_unavailable",
    "restriction_policy_changed",
    "weather_degraded",
    "tool_empty_result",
    "tool_conflicting_info",
}
PRIVILEGED_AGENT_KEYS = {
    "capability_targets",
    "diagnostic_goal",
    "expected_recovery_actions",
    "intervention_id",
    "intervention_type",
    "recovery_contract",
    "recovery_spec",
    "rubric",
    "validity_constraints",
    "wm_prediction",
}
PRIVILEGED_AGENT_TEXT_MARKERS = {
    "capability target",
    "capability_targets",
    "diagnostic goal",
    "expected recovery actions",
    "expected recovery",
    "expected_recovery",
    "failure mode",
    "grounded recovery contract",
    "intervention id",
    "intervention_id",
    "judge output",
    "judge score",
    "oracle recovery",
    "recovery contract",
    "recovery_spec",
    "reference action",
    "reference recovery",
    "rubric",
    "test challenge",
    "validity constraint",
    "validity_constraints",
    "wm prediction",
    "world model prediction",
    "world-model prediction",
    "世界模型预测",
    "失败模式",
    "测试挑战",
    "诊断目标",
    "参考恢复动作",
    "预期恢复",
    "恢复契约",
    "评分标准",
    "裁判输出",
}
AGENT_VISIBLE_CONTEXT_KEYS = {"observable_state"}


class ContinuationToolEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_event_index: int
    source_step: int
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    action_text: str = ""
    observation: str = ""
    success: bool = True


class MobilityBenchContinuationCase(BaseModel):
    """Frozen input for an auditable reconstruction of an official MB continuation.

    ``agent_visible`` is the only payload passed to the acting planner/worker/reporter.
    Evaluator-only intervention and recovery fields remain outside that mapping.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mobilitybench-continuation-case-v1"] = (
        "mobilitybench-continuation-case-v1"
    )
    protocol: Literal["mobilitybench-state-reconstruction-continuation-v1"] = CONTINUATION_PROTOCOL
    pair_key: str
    task_id: str
    intervention_id: str
    factual_source_hash: str
    intervention_source_hash: str
    recovery_contract_hash: str
    checkpoint_position: int
    trigger_source_event_index: int
    trigger_tool_name: str
    trigger_tool_parameters: dict[str, Any] = Field(default_factory=dict)
    trigger_resolution: str
    factual_prefix: list[ContinuationToolEvent] = Field(default_factory=list)
    agent_visible: dict[str, Any]
    sample: TaskSample
    factual: AgentTrajectory
    intervention: InterventionSpec
    recovery_spec: dict[str, Any]
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    case_hash: str


def build_continuation_case(
    base_record: EvaluationRecord,
    candidate: InterventionCandidate,
) -> MobilityBenchContinuationCase:
    """Bind one intervention to a concrete official worker/tool position.

    Historical MB frozen sets often store ``trigger_step=0`` as a logical marker.
    This function deliberately resolves it against the actual worker events instead
    of treating that marker as an executable checkpoint.
    """

    factual = base_record.factual
    if factual is None:
        raise ValueError(f"Factual trajectory missing for sample {base_record.sample.sample_id}")
    if candidate.recovery_spec is None:
        raise ValueError(f"Grounded recovery contract missing for {candidate.spec.intervention_id}")
    if candidate.spec.sample_id != base_record.sample.sample_id:
        raise ValueError(
            "Intervention sample does not match factual record: "
            f"{candidate.spec.sample_id!r} != {base_record.sample.sample_id!r}"
        )
    dimension = candidate.spec.dimension.strip().lower()
    if dimension not in SUPPORTED_TOOL_INTERVENTION_DIMENSIONS:
        raise ValueError(
            "Formal MobilityBench continuation requires a tool-observable intervention "
            f"dimension; unsupported dimension={candidate.spec.dimension!r}"
        )

    worker_events = [
        (index, event)
        for index, event in enumerate(factual.events)
        if event.metadata.get("official_role") == "worker" and event.tool_name
    ]
    if not worker_events:
        raise ValueError(
            f"No official worker/tool event found for sample {base_record.sample.sample_id}"
        )
    source_index, trigger, resolution = resolve_trigger_event(candidate.spec, worker_events)
    prefix = [
        ContinuationToolEvent(
            source_event_index=index,
            source_step=event.step,
            tool_name=event.tool_name,
            parameters=event.parameters,
            action_text=event.action_text,
            observation=event.observation,
            success=event.success,
        )
        for index, event in worker_events
        if index < source_index
    ]

    observable_update = render_observable_update(candidate.spec)
    changed_observation = overlay_observation(
        trigger.observation,
        candidate.spec,
        tool_name=trigger.tool_name,
    )
    agent_visible = {
        "task_id": base_record.sample.sample_id,
        "query": base_record.sample.query,
        "context": _agent_visible_context(base_record.sample.context),
        "prior_tool_history": [event.model_dump(mode="json") for event in prefix],
        "trigger_tool_call": {
            "tool_name": trigger.tool_name,
            "parameters": trigger.parameters,
            "observation": changed_observation,
            "observable_update": observable_update,
        },
        "continuation_instruction": (
            "Continue the same task from the latest observable tool result. Preserve "
            "still-valid information, revise invalidated choices and their dependencies, "
            "and use the available tools when further evidence is needed."
        ),
    }
    assert_agent_payload_is_clean(agent_visible)

    factual_hash = stable_hash(base_record.model_dump(mode="json"))
    intervention_hash = stable_hash(candidate.spec.model_dump(mode="json"))
    recovery_hash = stable_hash(candidate.recovery_spec.model_dump(mode="json"))
    pair_key = f"{base_record.sample.sample_id}::{candidate.spec.intervention_id}::{source_index}"
    payload = {
        "schema_version": "mobilitybench-continuation-case-v1",
        "protocol": CONTINUATION_PROTOCOL,
        "pair_key": pair_key,
        "task_id": base_record.sample.sample_id,
        "intervention_id": candidate.spec.intervention_id,
        "factual_source_hash": factual_hash,
        "intervention_source_hash": intervention_hash,
        "recovery_contract_hash": recovery_hash,
        "checkpoint_position": source_index,
        "trigger_source_event_index": source_index,
        "trigger_tool_name": trigger.tool_name,
        "trigger_tool_parameters": trigger.parameters,
        "trigger_resolution": resolution,
        "factual_prefix": [event.model_dump(mode="json") for event in prefix],
        "agent_visible": agent_visible,
        "sample": base_record.sample.model_dump(mode="json"),
        "factual": factual.model_dump(mode="json"),
        "intervention": candidate.spec.model_dump(mode="json"),
        "recovery_spec": candidate.recovery_spec.model_dump(mode="json"),
        "source_metadata": {
            "factual_adapter": factual.raw.get("adapter", "unknown"),
            "official_run_id": factual.raw.get("official_run_id", "unknown"),
            "historical_trigger_step": candidate.spec.trigger_step,
            "intervention_validity": candidate.validity_score,
            "validation_notes": candidate.validation_notes,
        },
    }
    return MobilityBenchContinuationCase(**payload, case_hash=stable_hash(payload))


def validate_continuation_case(case: MobilityBenchContinuationCase) -> None:
    assert_agent_payload_is_clean(case.agent_visible)
    if case.protocol != CONTINUATION_PROTOCOL:
        raise ValueError(f"Unsupported continuation protocol: {case.protocol}")
    if case.trigger_tool_name != case.agent_visible["trigger_tool_call"]["tool_name"]:
        raise ValueError("Trigger tool differs between audit and agent-visible payload")
    payload = case.model_dump(mode="json", exclude={"case_hash"})
    if stable_hash(payload) != case.case_hash:
        raise ValueError(f"Continuation case hash mismatch for {case.pair_key}")


def resolve_trigger_event(
    spec: InterventionSpec,
    worker_events: list[tuple[int, ToolEvent]],
) -> tuple[int, ToolEvent, str]:
    """Resolve a logical intervention to a real factual tool event."""

    allowed_tools = tools_for_dimension(spec.dimension)
    target_terms = intervention_target_terms(spec)

    ranked: list[tuple[int, int, ToolEvent, str]] = []
    for index, event in worker_events:
        haystack = _canonical_json(
            {
                "tool_name": event.tool_name,
                "parameters": event.parameters,
                "observation": event.observation,
                "action_text": event.action_text,
            }
        ).lower()
        target_hits = sum(term.lower() in haystack for term in target_terms)
        tool_match = int(not allowed_tools or event.tool_name in allowed_tools)
        historical_match = int(spec.trigger_step is not None and event.step == spec.trigger_step)
        # A target string in an unrelated tool result is not an executable checkpoint.
        # Keep the candidate only when its tool type can actually expose the change.
        score = 100 * target_hits + 10 * tool_match + historical_match if tool_match else -1
        reason = (
            "target_and_tool"
            if target_hits and tool_match
            else "dimension_tool_match"
            if tool_match
            else "fallback"
        )
        ranked.append((score, -index, event, reason))

    ranked.sort(reverse=True, key=lambda row: (row[0], row[1]))
    score, neg_index, event, reason = ranked[0]
    source_index = -neg_index
    if score <= 0:
        reason = "first_worker_event_fallback"
    return source_index, event, reason


def tools_for_dimension(dimension: str) -> set[str]:
    dimension = dimension.strip().lower()
    if dimension in {
        "road_blocked",
        "congestion_spike",
        "eta_shift",
        "restriction_policy_changed",
    }:
        return {
            "driving_route",
            "walking_route",
            "bicycling_route",
            "bus_route",
            "traffic_status",
        }
    if dimension in {
        "poi_closed",
        "pickup_point_unreachable",
        "parking_unavailable",
        "charging_unavailable",
    }:
        return {"query_poi", "search_around_poi", "reverse_geocoding"}
    if dimension == "weather_degraded":
        return {"weather_query"}
    if dimension in {"tool_conflicting_info", "tool_empty_result"}:
        return set()
    return set()


def render_observable_update(spec: InterventionSpec) -> str:
    text = spec.condition_text or spec.trigger_condition or spec.dimension.replace("_", " ")
    replacements = {
        "counterfactual": "current",
        "Counterfactual": "Current",
        "intervention": "environment update",
        "Intervention": "Environment update",
        "evaluator": "system",
        "Evaluator": "System",
        "judge": "review",
        "Judge": "Review",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def overlay_observation(
    original: Any,
    spec: InterventionSpec,
    *,
    tool_name: str,
) -> str:
    """Apply a persistent, observable world change to one tool result."""

    original_text = _to_text(original)
    update = render_observable_update(spec)
    dimension = spec.dimension.strip().lower()
    targets = intervention_target_terms(spec)

    if dimension == "negative_irrelevant":
        return original_text
    if dimension == "tool_empty_result":
        return _canonical_json(
            {
                "status": "success",
                "data": [],
                "message": update or f"{tool_name} returned no current result.",
            }
        )

    parsed = _parse_json_value(original_text)
    if dimension in {
        "poi_closed",
        "pickup_point_unreachable",
        "parking_unavailable",
        "charging_unavailable",
    }:
        transformed, changed = _mark_matching_resources_unavailable(parsed, targets, update)
        if changed:
            return _canonical_json(transformed)
    if dimension in {"road_blocked", "congestion_spike", "eta_shift", "restriction_policy_changed"}:
        transformed = _annotate_mapping(parsed, update, status="changed")
        return _canonical_json(transformed)
    if dimension == "weather_degraded":
        transformed = _annotate_mapping(parsed, update, status="degraded")
        return _canonical_json(transformed)
    if dimension == "tool_conflicting_info":
        transformed = _annotate_mapping(parsed, update, status="conflicting")
        return _canonical_json(transformed)

    return _canonical_json(
        {
            "status": "changed",
            "observable_update": update,
            "original_result": parsed,
        }
    )


def overlay_tool_result(result: Any, spec: InterventionSpec, *, tool_name: str) -> Any:
    """Overlay helper that preserves mapping/list return types where possible."""

    overlaid = overlay_observation(result, spec, tool_name=tool_name)
    parsed = _parse_json_value(overlaid)
    if isinstance(result, str):
        return overlaid
    return parsed


def intervention_applies_to_tool(
    spec: InterventionSpec,
    tool_name: str,
    *,
    trigger_tool_name: str = "",
) -> bool:
    target_tool = str(spec.params.get("tool_name") or spec.target_entity.get("tool_name") or "")
    if target_tool:
        return target_tool == tool_name
    tools = tools_for_dimension(spec.dimension)
    if tools:
        return tool_name in tools
    # Generic tool-output interventions are scoped to the concrete tool that
    # exposed the frozen trigger, rather than contaminating every later tool.
    return not trigger_tool_name or trigger_tool_name == tool_name


def assert_agent_payload_is_clean(payload: dict[str, Any]) -> None:
    violations: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if str(key) in PRIVILEGED_AGENT_KEYS:
                    violations.append(child_path)
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            lowered = value.lower()
            for marker in PRIVILEGED_AGENT_TEXT_MARKERS:
                if marker.lower() in lowered:
                    violations.append(f"{path}:text:{marker}")

    walk(payload, "")
    if violations:
        raise ValueError("Privileged evaluator fields leaked to agent: " + ", ".join(violations))


def continuation_agent_history_text(case: MobilityBenchContinuationCase) -> str:
    """Render the exact observable state used by Agent-owned retrieval."""

    visible = case.agent_visible
    lines = [f"User request: {visible['query']}"]
    if visible.get("context"):
        lines.append(f"Context: {_to_text(visible['context'])}")
    for event in visible.get("prior_tool_history", []):
        lines.append(
            f"Tool {event.get('tool_name', '')}({_canonical_json(event.get('parameters') or {})}) "
            f"returned: {event.get('observation', '')}"
        )
    trigger = visible["trigger_tool_call"]
    lines.append(
        f"Latest tool {trigger['tool_name']}({_canonical_json(trigger.get('parameters') or {})}) "
        f"returned: {trigger.get('observation', '')}"
    )
    return "\n".join(lines)


def build_official_messages(case: MobilityBenchContinuationCase) -> list[Any]:
    """Build LangChain messages lazily inside the official MB runtime."""

    try:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    except ImportError as exc:  # pragma: no cover - exercised in the MB environment
        raise RuntimeError(
            "Official MobilityBench dependencies are unavailable. Run with the "
            "MobilityBench virtual environment and include this repository's src on PYTHONPATH."
        ) from exc

    visible = case.agent_visible
    messages: list[Any] = []
    if visible.get("context"):
        messages.append(
            HumanMessage(content=f"Context information:\n{_to_text(visible['context'])}")
        )
    messages.append(HumanMessage(content=f"User requirement:\n{visible['query']}"))
    events = [*visible.get("prior_tool_history", []), visible["trigger_tool_call"]]
    for index, event in enumerate(events):
        tool_call_id = f"cce_reconstructed_{index}"
        name = str(event.get("tool_name") or "unknown_tool")
        args = dict(event.get("parameters") or {})
        messages.append(
            AIMessage(
                content=str(event.get("action_text") or ""),
                tool_calls=[{"id": tool_call_id, "name": name, "args": args, "type": "tool_call"}],
            )
        )
        messages.append(
            ToolMessage(
                content=str(event.get("observation") or ""),
                tool_call_id=tool_call_id,
                name=name,
            )
        )
    messages.append(HumanMessage(content=visible["continuation_instruction"]))
    return messages


@contextmanager
def official_runtime_patches(
    *,
    case: MobilityBenchContinuationCase,
    base_prompts: dict[str, str],
    selected_guidance: str = "",
) -> Iterator[dict[str, Any]]:
    """Patch official prompts and sandbox tools for one sequential continuation."""

    from mobility_bench.agent import prompts as prompt_module
    from mobility_bench.agent.frameworks.plan_and_execute import nodes
    from mobility_bench.tools.registry import ToolRegistry

    original_prompt_module = prompt_module.get_system_prompt
    original_nodes_prompt = nodes.get_system_prompt
    tools = ToolRegistry.get_all()
    originals: list[tuple[Any, Any]] = []
    call_log: list[dict[str, Any]] = []

    def fixed_prompt(framework: str, role: str, template_vars=None):  # type: ignore[no-untyped-def]
        if framework != "plan_and_execute" or role not in base_prompts:
            return original_prompt_module(framework, role, template_vars)
        return _runtime_prompt(base_prompts[role], selected_guidance, role)

    prompt_module.get_system_prompt = fixed_prompt
    nodes.get_system_prompt = fixed_prompt
    try:
        for tool in tools:
            name = str(getattr(tool, "name", getattr(tool, "__name__", "unknown_tool")))
            if hasattr(tool, "func") and callable(tool.func):
                original = tool.func
                wrapped = _tool_wrapper(
                    name,
                    original,
                    case.intervention,
                    call_log,
                    trigger_tool_name=case.trigger_tool_name,
                )
                try:
                    tool.func = wrapped
                except Exception:
                    object.__setattr__(tool, "func", wrapped)
                originals.append((tool, original))
        yield {
            "tool_calls": call_log,
            "prompt_hashes": _prompt_hashes(base_prompts, selected_guidance),
        }
    finally:
        prompt_module.get_system_prompt = original_prompt_module
        nodes.get_system_prompt = original_nodes_prompt
        for tool, original in originals:
            try:
                tool.func = original
            except Exception:
                object.__setattr__(tool, "func", original)


async def run_official_continuation(
    *,
    case: MobilityBenchContinuationCase,
    model_alias: str,
    official_config_path: str | Path,
    base_prompts: dict[str, str],
    selected_guidance: str = "",
) -> AgentTrajectory:
    """Run the official planner/worker/reporter graph from reconstructed state."""

    validate_continuation_case(case)
    try:
        from mobility_bench.agent.frameworks import FrameworkFactory
        from mobility_bench.config.settings import Settings
        from mobility_bench.tools.registry import ToolRegistry
    except ImportError as exc:  # pragma: no cover - exercised on the server
        raise RuntimeError("Official MobilityBench package is not importable") from exc

    official_provenance = official_mobilitybench_provenance()
    settings = Settings.load(Path(official_config_path))
    model_config = settings.models.get(model_alias)
    if model_config is None:
        raise ValueError(f"Official model alias not found in config: {model_alias}")
    ToolRegistry.clear()
    ToolRegistry.load_default_tools(mode="sandbox")
    tool_schema = _official_tool_schema(ToolRegistry.get_all())
    framework = FrameworkFactory.create("plan_and_execute", settings, model_config)
    graph = framework.build_graph()
    initial_state = {
        "query": case.agent_visible["query"],
        "context": _to_text(case.agent_visible.get("context") or ""),
        "messages": build_official_messages(case),
        "poi_map": {},
        "current_plan": None,
        # Non-zero prevents the graph from appending the original query/context again.
        "plan_iterations": 1,
    }
    with official_runtime_patches(
        case=case,
        base_prompts=base_prompts,
        selected_guidance=selected_guidance,
    ) as audit:
        try:
            final_state = await graph.ainvoke(initial_state)
            extracted = framework.extract_result(final_state)
            error = ""
        except Exception as exc:  # pragma: no cover - runtime transport failures
            final_state = {}
            extracted = {}
            error = f"{type(exc).__name__}: {exc}"

    final_response = str(extracted.get("plan_result") or "")
    continuation_events = _events_from_official_result(
        extracted,
        start_step=len(case.factual_prefix) + 1,
        tool_call_log=audit["tool_calls"],
    )
    return AgentTrajectory(
        sample_id=case.task_id,
        run_mode=RunMode.COUNTERFACTUAL,
        events=continuation_events,
        final_response=final_response,
        raw={
            "adapter": "official_mobilitybench_state_reconstruction_continuation",
            "adapter_scope": CONTINUATION_PROTOCOL,
            "continuation_mode": "state_reconstruction_continuation",
            "bitwise_graph_resume": False,
            "pair_key": case.pair_key,
            "case_hash": case.case_hash,
            "prompt_hashes": audit["prompt_hashes"],
            "official_model_config_sha256": stable_hash(
                model_config.model_dump(mode="json")
                if hasattr(model_config, "model_dump")
                else str(model_config)
            ),
            "official_tool_schema_sha256": stable_hash(tool_schema),
            "official_tool_schema": tool_schema,
            "official_mobilitybench_provenance": official_provenance,
            "official_result": extracted,
            "official_planner_present": bool(extracted.get("planner_training_data")),
            "official_worker_present": bool(extracted.get("worker_training_data")),
            "official_reporter_present": bool(final_response),
            "termination_reason": _termination_reason(final_state, error),
            "llm_error": bool(error or not final_response),
            "llm_error_message": error or ("empty_reporter_output" if not final_response else ""),
            "tool_call_log": audit["tool_calls"],
        },
    )


def load_prompt_bundle(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, dict):
        raise ValueError(f"Invalid official prompt bundle: {path}")
    required = {"planner", "worker", "reporter"}
    if required - set(prompts):
        raise ValueError(f"Official prompt bundle missing roles: {sorted(required - set(prompts))}")
    return {role: str(prompts[role]) for role in required}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_mobilitybench_provenance() -> dict[str, Any]:
    """Fingerprint the official package code used by the reconstructed runner."""

    try:
        import mobility_bench
    except ImportError as exc:  # pragma: no cover - requires official MB runtime
        raise RuntimeError("Official MobilityBench package is not importable") from exc
    package_root = Path(mobility_bench.__file__).resolve().parent
    repo_root = _git_toplevel(package_root)
    tracked_status = (
        _git_output(repo_root, ["status", "--porcelain", "--untracked-files=no"])
        if repo_root
        else ""
    )
    runtime_status = ""
    if repo_root:
        runtime_scope = str(package_root.relative_to(repo_root))
        runtime_status = _git_output(
            repo_root,
            [
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                runtime_scope,
            ],
        )
    return {
        "package_root": str(package_root),
        "repo_root": str(repo_root) if repo_root else "unknown",
        "git_commit": _git_output(repo_root, ["rev-parse", "HEAD"]) if repo_root else "unknown",
        "tracked_dirty": bool(tracked_status) if repo_root else None,
        "tracked_status": tracked_status.splitlines(),
        "runtime_dirty": bool(runtime_status) if repo_root else None,
        "runtime_status": runtime_status.splitlines(),
        "python_source_sha256": _python_tree_sha256(package_root),
    }


def cce_continuation_runtime_provenance(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fingerprint CCE code that can affect continuation execution or scoring."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    source_roots = [root / "src" / "cce_mobility", root / "patches" / "mobilitybench"]
    files = [
        path
        for source_root in source_roots
        if source_root.exists()
        for path in source_root.rglob("*.py")
    ]
    files = [
        path
        for path in files
        if str(path.relative_to(root)) not in NON_MB_CONTINUATION_SOURCES
        and not path.name.startswith("local_secrets")
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CCE continuation runtime files missing: {missing}")
    tracked_status = _git_output(root, ["status", "--porcelain", "--untracked-files=no"])
    runtime_scope = [str(path.relative_to(root)) for path in source_roots if path.exists()]
    runtime_status = _filter_non_mb_status(
        _git_output(
            root,
            ["status", "--porcelain", "--untracked-files=all", "--", *runtime_scope],
        )
    )
    return {
        "project_root": str(root),
        "git_commit": _git_output(root, ["rev-parse", "HEAD"]) or "unknown",
        "tracked_dirty": bool(tracked_status),
        "tracked_status": tracked_status.splitlines(),
        "runtime_dirty": bool(runtime_status),
        "runtime_status": runtime_status.splitlines(),
        "runtime_source_sha256": _paths_sha256(files, root=root),
        "runtime_file_count": len(files),
    }


def formal_experiment_code_provenance(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fingerprint every local code path used to create the formal curve."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    source_roots = [root / "src" / "cce_mobility", root / "patches" / "mobilitybench"]
    files = [
        path
        for source_root in source_roots
        if source_root.exists()
        for path in source_root.rglob("*.py")
        if not path.name.startswith("local_secrets")
    ]
    files.extend(root / relative for relative in FORMAL_EXPERIMENT_SCRIPTS)
    missing = sorted(str(path) for path in files if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Formal experiment code files missing: {missing}")
    relative_scope = [
        "src/cce_mobility",
        "patches/mobilitybench",
        *FORMAL_EXPERIMENT_SCRIPTS,
    ]
    status = _filter_non_mb_status(
        _git_output(
            root,
            ["status", "--porcelain", "--untracked-files=all", "--", *relative_scope],
        )
    )
    return {
        "project_root": str(root),
        "git_commit": _git_output(root, ["rev-parse", "HEAD"]) or "unknown",
        "relevant_dirty": bool(status),
        "relevant_status": status.splitlines(),
        "formal_source_sha256": _paths_sha256(files, root=root),
        "formal_file_count": len(set(files)),
    }


def _git_toplevel(path: Path) -> Path | None:
    value = _git_output(path, ["rev-parse", "--show-toplevel"])
    return Path(value).resolve() if value else None


def _git_output(path: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _filter_non_mb_status(status: str) -> str:
    """Keep MB-owned status lines and ignore unrelated experiment edits."""

    retained: list[str] = []
    for line in status.splitlines():
        payload = line[3:].strip() if len(line) >= 3 else line.strip()
        paths = [part.strip() for part in payload.split(" -> ")]
        if paths and all(_is_non_mb_continuation_source(path) for path in paths):
            continue
        retained.append(line)
    return "\n".join(retained)


def _python_tree_sha256(package_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _paths_sha256(paths: list[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_scoped_provenance_transition(
    project_root: str | Path,
    frozen: dict[str, Any],
    *,
    scope: Literal["runtime", "formal"],
    allow_qoder_only_transition: bool = False,
) -> dict[str, Any]:
    """Allow legacy checkpoints only across audited, execution-equivalent changes."""

    root = Path(project_root).expanduser().resolve()
    current = (
        cce_continuation_runtime_provenance(root)
        if scope == "runtime"
        else formal_experiment_code_provenance(root)
    )
    hash_key = "runtime_source_sha256" if scope == "runtime" else "formal_source_sha256"
    dirty_key = "runtime_dirty" if scope == "runtime" else "relevant_dirty"
    if not isinstance(frozen, dict) or not frozen.get(hash_key):
        raise ValueError(f"Frozen {scope} provenance lacks {hash_key}")
    if current.get(dirty_key):
        raise ValueError(f"Current {scope} source has uncommitted changes")
    if current[hash_key] == frozen[hash_key]:
        return {
            "scope": scope,
            "frozen_commit": frozen.get("git_commit"),
            "current_commit": current["git_commit"],
            "frozen_source_sha256": frozen[hash_key],
            "current_source_sha256": current[hash_key],
            "equivalence": "byte_identical",
            "policy_only_files": [],
            "backend_only_files": [],
            "ignored_files": [],
        }

    frozen_commit = str(frozen.get("git_commit") or "")
    if frozen_commit in {"", "unknown"}:
        raise ValueError("Cannot verify a legacy source without its frozen commit")
    source_roots = (root / "src/cce_mobility", root / "patches/mobilitybench")
    paths = [path for source in source_roots if source.exists() for path in source.rglob("*.py")]
    if scope == "formal":
        paths.extend(root / path for path in FORMAL_EXPERIMENT_SCRIPTS)
    current_sources = {str(path.relative_to(root)): path.read_bytes() for path in set(paths)}
    frozen_sources: dict[str, bytes] = {}
    ignored_missing: list[str] = []
    for relative, content in current_sources.items():
        historical = subprocess.run(
            ["git", "-C", str(root), "show", f"{frozen_commit}:{relative}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if historical.returncode == 0:
            frozen_sources[relative] = historical.stdout
            continue
        if _is_non_mb_continuation_source(relative):
            ignored_missing.append(relative)
            continue
        ignored = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", "--", relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError(f"Untracked source not present at frozen commit: {relative}")
        frozen_sources[relative] = content
    if _source_bytes_digest(frozen_sources) != frozen[hash_key]:
        raise ValueError(f"Frozen {scope} hash cannot be reconstructed from its commit")

    policy_functions = {
        "src/cce_mobility/integrations/mobilitybench_continuation.py": {
            "cce_continuation_runtime_provenance",
            "formal_experiment_code_provenance",
            "_filter_non_mb_status",
            "_is_non_mb_continuation_source",
            "verify_scoped_provenance_transition",
            "_source_bytes_digest",
            "_policy_ast",
            "_openai_compatible_llm_ast",
        },
        "src/cce_mobility/workflows/mobilitybench/run_mobilitybench_official_continuation.py": {
            "_same_cce_runtime",
            "_same_official_code",
        },
        # The standalone MB continuation runner never invokes the TP matrix runner.
        "src/cce_mobility/runners/matrix.py": {"run_matrix"},
        "src/cce_mobility/workflows/mobilitybench/run_mobilitybench_formal_coevolution_train.py": {
            "prepare",
            "run_cycle",
            "_run_evaluation",
            "_report_has_only_retryable_execution_errors",
            "_repair_exhausted_evaluation",
            "_merge_error_only_repair",
            "_records_by_pair_key",
            "_continuation_retry_reason",
            "_write_jsonl_payloads",
            "_assert_clean_report",
            "_select_training_evidence",
            "_paired_training_evidence",
            "_training_record_key",
        },
        "src/cce_mobility/workflows/mobilitybench/run_mobilitybench_formal_coevolution_curve.py": {
            "prepare",
            "_reuse_protocol_lock",
            "_protocol_audit",
            "_require_matching_formal_code",
            "_validate_training_manifest",
        },
    }
    changed: list[str] = []
    backend_only: list[str] = []
    ignored_files: list[str] = list(ignored_missing)
    for relative, content in current_sources.items():
        if relative in ignored_missing:
            continue
        before = frozen_sources[relative]
        if content == before:
            continue
        if _is_non_mb_continuation_source(relative):
            ignored_files.append(relative)
            continue
        if (
            allow_qoder_only_transition
            and relative == "src/cce_mobility/core/llm.py"
            and _openai_compatible_llm_ast(before) == _openai_compatible_llm_ast(content)
        ):
            backend_only.append(relative)
            continue
        if relative not in policy_functions or _policy_ast(
            before, policy_functions[relative], relative=relative
        ) != _policy_ast(content, policy_functions[relative], relative=relative):
            raise ValueError(f"Execution-relevant {scope} source changed: {relative}")
        changed.append(relative)
    return {
        "scope": scope,
        "frozen_commit": frozen_commit,
        "current_commit": current["git_commit"],
        "frozen_source_sha256": frozen[hash_key],
        "current_source_sha256": current[hash_key],
        "equivalence": (
            "backend_scoped_ast_verified" if backend_only else "policy_only_ast_verified"
        ),
        "policy_only_files": sorted(changed),
        "backend_only_files": sorted(backend_only),
        "ignored_files": sorted(set(ignored_files)),
    }


def _source_bytes_digest(sources: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative, content in sorted(sources.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _policy_ast(content: bytes, ignored_functions: set[str], *, relative: str) -> str:
    tree = ast.parse(content)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in ignored_functions
        )
        and not (
            isinstance(node, ast.Import)
            and len(node.names) == 1
            and node.names[0].name in {"ast", "shutil"}
            and node.names[0].asname is None
        )
        and not (
            relative == "src/cce_mobility/runners/matrix.py"
            and isinstance(node, ast.ImportFrom)
            and node.module == "cce_mobility.core.schema"
            and [(name.name, name.asname) for name in node.names] == [("ExperimentReport", None)]
        )
        and not (
            relative == "src/cce_mobility/integrations/mobilitybench_continuation.py"
            and isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name)
                and target.id
                in {
                    "NON_MB_CONTINUATION_SOURCES",
                    "NON_MB_CONTINUATION_SOURCE_PREFIXES",
                }
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        )
    ]
    if (
        relative
        == "src/cce_mobility/workflows/mobilitybench/run_mobilitybench_official_continuation.py"
    ):
        # Normalize only the explicit AIHub provenance guard, not the run logic.
        ai_hub_only = ast.parse(
            'all(config[role]["provider"] == "aihub" for role in ("agent", "judge"))',
            mode="eval",
        ).body
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_same_cce_runtime"
                and len(node.args) == 2
                and len(node.keywords) == 1
                and node.keywords[0].arg == "allow_qoder_only_transition"
                and ast.dump(node.keywords[0].value) == ast.dump(ai_hub_only)
            ):
                node.keywords = []
    return ast.dump(tree, include_attributes=False)


def _openai_compatible_llm_ast(content: bytes) -> str:
    """Ignore only the Qoder body and the guarded Qoder metrics return."""

    tree = ast.parse(content)
    clients = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChatClient"
    ]
    if len(clients) != 1:
        raise ValueError("Cannot isolate ChatClient backend branches")
    methods = {node.name: node for node in clients[0].body if isinstance(node, ast.FunctionDef)}
    qoder = methods.get("_chat_qoder_cli")
    metrics = methods.get("transport_metrics")
    if qoder is None or metrics is None or len(metrics.body) < 3:
        raise ValueError("Cannot isolate Qoder-only ChatClient changes")
    guard, qoder_return = metrics.body[-2:]
    if not (
        isinstance(guard, ast.If)
        and isinstance(guard.test, ast.UnaryOp)
        and isinstance(guard.test.op, ast.Not)
        and isinstance(guard.test.operand, ast.Attribute)
        and isinstance(guard.test.operand.value, ast.Name)
        and guard.test.operand.value.id == "self"
        and guard.test.operand.attr == "is_qoder_cli"
        and guard.body
        and isinstance(guard.body[-1], ast.Return)
        and not guard.orelse
        and isinstance(qoder_return, ast.Return)
    ):
        raise ValueError("Qoder metrics are not guarded from the OpenAI-compatible path")
    qoder.body = [ast.Pass()]
    qoder_return.value = ast.Constant(value=None)
    return ast.dump(tree, include_attributes=False)


def _tool_wrapper(
    name: str,
    original: Callable[..., Any],
    spec: InterventionSpec,
    call_log: list[dict[str, Any]],
    *,
    trigger_tool_name: str = "",
) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(original):

        async def async_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = await original(*args, **kwargs)
            overlaid = (
                overlay_tool_result(result, spec, tool_name=name)
                if intervention_applies_to_tool(
                    spec,
                    name,
                    trigger_tool_name=trigger_tool_name,
                )
                else result
            )
            call_log.append(
                {
                    "tool_name": name,
                    "parameters": kwargs,
                    "overlay_applied": overlaid != result,
                    "result": _to_text(overlaid),
                }
            )
            return overlaid

        return async_wrapper

    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        result = original(*args, **kwargs)
        overlaid = (
            overlay_tool_result(result, spec, tool_name=name)
            if intervention_applies_to_tool(
                spec,
                name,
                trigger_tool_name=trigger_tool_name,
            )
            else result
        )
        call_log.append(
            {
                "tool_name": name,
                "parameters": kwargs,
                "overlay_applied": overlaid != result,
                "result": _to_text(overlaid),
            }
        )
        return overlaid

    return wrapper


def _events_from_official_result(
    result: dict[str, Any],
    *,
    start_step: int,
    tool_call_log: list[dict[str, Any]],
) -> list[ToolEvent]:
    events: list[ToolEvent] = []
    step = start_step
    planner = result.get("planner_training_data") or {}
    if planner:
        events.append(
            ToolEvent(
                step=step,
                tool_name="official_plan_and_execute_planner",
                parameters={"intent": planner.get("intent"), "steps": planner.get("steps")},
                action_text=str(planner.get("thinking") or ""),
                metadata={"agent_visible": False, "official_role": "planner"},
            )
        )
        step += 1
    for call in tool_call_log:
        events.append(
            ToolEvent(
                step=step,
                tool_name=str(call.get("tool_name") or ""),
                parameters=dict(call.get("parameters") or {}),
                observation=str(call.get("result") or ""),
                success=True,
                metadata={
                    "agent_visible": True,
                    "official_role": "worker",
                    "persistent_world_overlay_applied": bool(call.get("overlay_applied")),
                },
            )
        )
        step += 1
    reporter = result.get("reporter_training_data") or {}
    final_answer = str(result.get("plan_result") or reporter.get("final_answer") or "")
    events.append(
        ToolEvent(
            step=step,
            tool_name="official_plan_and_execute_reporter",
            action_text=final_answer,
            success=bool(final_answer),
            metadata={"agent_visible": True, "official_role": "reporter"},
        )
    )
    return events


def _termination_reason(final_state: dict[str, Any], error: str) -> str:
    if error:
        return "official_graph_exception"
    if final_state.get("plan_result"):
        return "official_reporter_completed"
    return "official_reporter_empty"


def _prompt_hashes(base_prompts: dict[str, str], selected_guidance: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for role, prompt in base_prompts.items():
        runtime = _runtime_prompt(prompt, selected_guidance, role)
        hashes[role] = stable_hash(runtime)
    hashes["base_bundle"] = stable_hash(base_prompts)
    hashes["selected_guidance"] = stable_hash(selected_guidance)
    return hashes


def _runtime_prompt(base_prompt: str, selected_guidance: str, role: str) -> str:
    if not selected_guidance or role not in {"worker", "reporter"}:
        return base_prompt
    return (
        f"{base_prompt.rstrip()}\n\nReusable recovery guidance from prior mobility-assistant "
        "experience:\nUse it only when current observable evidence supports it. "
        "Ignore unrelated guidance.\n"
        f"{selected_guidance.strip()}"
    )


def _official_tool_schema(tools: list[Any]) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for tool in tools:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None and hasattr(args_schema, "model_json_schema"):
            try:
                parameters = args_schema.model_json_schema()
            except Exception:
                parameters = str(args_schema)
        else:
            parameters = "unknown"
        schema.append(
            {
                "name": str(getattr(tool, "name", getattr(tool, "__name__", "unknown"))),
                "description": str(getattr(tool, "description", "")),
                "parameters": parameters,
            }
        )
    return sorted(schema, key=lambda item: item["name"])


def intervention_target_terms(spec: InterventionSpec) -> list[str]:
    values: list[Any] = []
    for key in ("target_ids", "target_id", "target_name", "poi_name", "road_name"):
        value = spec.params.get(key)
        if value is not None:
            values.extend(value if isinstance(value, list) else [value])
        value = spec.target_entity.get(key)
        if value is not None:
            values.extend(value if isinstance(value, list) else [value])
    if spec.target_selector:
        values.append(spec.target_selector)
    return [str(value).strip() for value in values if str(value).strip()]


def _mark_matching_resources_unavailable(
    value: Any,
    targets: list[str],
    update: str,
) -> tuple[Any, bool]:
    changed = False
    if isinstance(value, dict):
        rendered = _canonical_json(value).lower()
        matches = not targets or any(target.lower() in rendered for target in targets)
        out: dict[str, Any] = {}
        for key, child in value.items():
            new_child, child_changed = _mark_matching_resources_unavailable(child, targets, update)
            out[key] = new_child
            changed = changed or child_changed
        if matches and any(key in value for key in ("name", "id", "poi_id", "address", "title")):
            out["availability"] = "unavailable"
            out["observable_update"] = update
            changed = True
        return out, changed
    if isinstance(value, list):
        out_list = []
        for child in value:
            new_child, child_changed = _mark_matching_resources_unavailable(child, targets, update)
            out_list.append(new_child)
            changed = changed or child_changed
        return out_list, changed
    return value, False


def _annotate_mapping(value: Any, update: str, *, status: str) -> Any:
    if isinstance(value, dict):
        return {**value, "environment_status": status, "observable_update": update}
    return {"data": value, "environment_status": status, "observable_update": update}


def _agent_visible_context(context: dict[str, Any]) -> dict[str, Any]:
    # The converter also stores evaluator metadata and gold answers in TaskSample.context.
    # Only the original official ``context`` field, mapped to observable_state, is an
    # execution input. A whitelist prevents new dataset columns from silently leaking.
    return {
        key: context[key]
        for key in AGENT_VISIBLE_CONTEXT_KEYS
        if key in context and context[key] not in (None, "", {}, [])
    }


def _parse_json_value(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _to_text(value: Any) -> str:
    return value if isinstance(value, str) else _canonical_json(value)
