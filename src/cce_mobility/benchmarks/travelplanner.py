from __future__ import annotations
import ast
import csv
import json
from pathlib import Path
from typing import Any
from cce_mobility.core.config import BenchmarkConfig
from cce_mobility.core.llm import ChatClient
from cce_mobility.core.recovery_context import (
    render_factual_recovery_context,
    validate_factual_recovery_context,
)
from cce_mobility.core.schema import (
    AgentTrajectory,
    BenchmarkName,
    FactualRecoveryContext,
    InterventionSpec,
    RunMode,
    TaskSample,
    ToolEvent,
)
from cce_mobility.benchmarks.travelplanner_eval import evaluate_plan, extract_unavailable_ids
from cce_mobility.benchmarks.travelplanner_rule_wm import TravelPlannerRuleWorldModel
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

TRAVELPLANNER_SYSTEM_PROMPT = "You are a proficient travel planner. Generate detailed travel plans based on the provided information, including flight numbers, restaurant names, and accommodation names. All information should be derived from the provided data. Adhere to the given format."
TRAVELPLANNER_PLANNER_PROMPT = "You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers, restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data.\n\n***** Example *****\nQuery: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?\nTravel Plan:\nDay 1:\nCurrent City: from Ithaca to Charlotte\nTransportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46\nBreakfast: Nagaland's Kitchen, Charlotte\nAttraction: The Charlotte Museum of History, Charlotte\nLunch: Cafe Maple Street, Charlotte\nDinner: Bombay Vada Pav, Charlotte\nAccommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte\n\nDay 2:\nCurrent City: Charlotte\nTransportation: -\nBreakfast: Olive Tree Cafe, Charlotte\nAttraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.\nLunch: Birbal Ji Dhaba, Charlotte\nDinner: Pind Balluchi, Charlotte\nAccommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte\n\nDay 3:\nCurrent City: from Charlotte to Ithaca\nTransportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26\nBreakfast: Subway, Charlotte\nAttraction: Books Monument, Charlotte.\nLunch: Olive Tree Cafe, Charlotte\nDinner: Kylin Skybar, Charlotte\nAccommodation: -\n\n***** Example Ends *****\n\nGiven information:\n{info_text}\n\nQuery: {query}\n\nTravel Plan:"


class TravelPlannerAdapter:
    """Lightweight TravelPlanner adapter for CCE experiments.

    The first implementation intentionally supports a transparent sole-planning
    path. It treats ref_info as the environment state and records how an
    intervention would rewrite that state. A later adapter can call the official
    ReAct tools directly.
    """

    name = "travelplanner"

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.repo_path = Path(config.repo_path) if config.repo_path else None
        self.chat_client: ChatClient | None = None
        self.planner_mode = str(config.extra.get("planner_mode", "adaptive_rule"))
        self.agent_guidance = ""
        self.agent_skill_registry: SkillRegistry | None = None
        self.runtime_options: dict[str, Any] = dict(config.extra or {})

    def set_chat_client(self, chat_client: ChatClient) -> None:
        self.chat_client = chat_client

    def set_agent_guidance(self, guidance: str) -> None:
        self.agent_guidance = guidance

    def set_agent_skill_registry(self, registry: SkillRegistry) -> None:
        self.agent_skill_registry = registry

    def set_runtime_options(self, options: dict[str, Any]) -> None:
        self.runtime_options.update(options or {})

    def load_samples(self, limit: int | None = None) -> list[TaskSample]:
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"TravelPlanner data_path not found: {data_path}")
        ref_rows = _load_jsonl(data_path)
        query_rows = self._load_queries(data_path)
        n = min(len(ref_rows), len(query_rows))
        selected_indices = _selected_sample_indices(self.config.extra, n)
        if selected_indices is None:
            selected_indices = list(range(n))
        effective_limit = limit if limit is not None else self.config.limit
        if effective_limit is not None:
            selected_indices = selected_indices[:effective_limit]
        samples: list[TaskSample] = []
        for idx in selected_indices:
            query = _build_query_text(query_rows[idx])
            samples.append(
                TaskSample(
                    sample_id=f"tp_{idx}",
                    benchmark=BenchmarkName.TRAVEL_PLANNER,
                    query=query,
                    context={"ref_info": ref_rows[idx], "query_data": query_rows[idx]},
                    metadata={"sample_idx": idx},
                )
            )
        return samples

    def run_agent(
        self,
        sample: TaskSample,
        *,
        agent_id: str,
        intervention: InterventionSpec | None = None,
        recovery_context: FactualRecoveryContext | None = None,
    ) -> AgentTrajectory:
        """Run sole-planning with an optional environment-state rewrite."""
        if recovery_context is not None:
            if intervention is None:
                raise ValueError("Recovery context is only valid for an intervened rollout")
            validate_factual_recovery_context(sample, recovery_context)
        run_mode = RunMode.COUNTERFACTUAL if intervention else RunMode.FACTUAL
        ref_info, effective_ref_info, intervention_records = self._effective_ref_info(
            sample, intervention
        )
        unavailable_ids = extract_unavailable_ids(effective_ref_info)
        env_summary = _summarize_ref_info(ref_info)
        if intervention:
            cf_summary = _summarize_ref_info(effective_ref_info)
            env_summary = f"{cf_summary}\n[ENVIRONMENT_UPDATE] {sanitize_agent_visible_text(intervention.condition_text)}"
        planner_ref_info = self._select_planner_ref_info(
            original_ref_info=ref_info,
            effective_ref_info=effective_ref_info,
            intervention=intervention,
        )
        final, skill_retrieval = self._generate_plan(
            sample.query,
            planner_ref_info,
            sample.context.get("query_data", {}),
            environment_observation=env_summary,
            recovery_context=recovery_context,
        )
        llm_error = _extract_llm_error(final)
        return AgentTrajectory(
            sample_id=sample.sample_id,
            run_mode=run_mode,
            events=[
                ToolEvent(step=0, tool_name="Planner", observation=env_summary, action_text=final)
            ],
            final_response=final,
            raw={
                "adapter": "travelplanner_sole_planning",
                "agent_id": agent_id,
                "planner_mode": self.planner_mode,
                "intervention_records": intervention_records,
                "unavailable_ids": unavailable_ids,
                "effective_ref_info_summary": _summarize_ref_info(effective_ref_info),
                "planner_ref_info_summary": _summarize_ref_info(planner_ref_info),
                "skill_retrieval": skill_retrieval,
                "recovery_context_used": recovery_context is not None,
                "factual_prefix_hash": recovery_context.prefix_hash if recovery_context else "",
                "llm_error": bool(llm_error),
                "llm_error_message": llm_error,
            },
        )

    def evaluate_trajectory(
        self,
        sample: TaskSample,
        trajectory: AgentTrajectory,
        intervention: InterventionSpec | None = None,
    ) -> dict[str, Any]:
        """Return deterministic TravelPlanner parsing and constraint metrics."""
        _, effective_ref_info, intervention_records = self._effective_ref_info(sample, intervention)
        unavailable_ids = trajectory.raw.get("unavailable_ids") or extract_unavailable_ids(
            effective_ref_info
        )
        metrics = evaluate_plan(
            trajectory.final_response,
            effective_ref_info,
            sample.context.get("query_data", {}),
            unavailable_ids=unavailable_ids,
        )
        metrics.update(
            {
                "benchmark": self.name,
                "run_mode": trajectory.run_mode.value,
                "intervention_dimension": intervention.dimension if intervention else None,
                "intervention_record_count": len(intervention_records),
                "llm_error": bool(trajectory.raw.get("llm_error")),
                "llm_error_message": trajectory.raw.get("llm_error_message", ""),
            }
        )
        return metrics

    def _generate_plan(
        self,
        query: str,
        ref_info: dict[str, Any],
        query_data: dict[str, Any],
        *,
        environment_observation: str,
        recovery_context: FactualRecoveryContext | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if self.chat_client is None or self.chat_client.is_placeholder:
            return (_generate_rule_plan(query, ref_info, query_data), {})
        info_text = _ref_info_to_text(ref_info)
        recovery_text = (
            render_factual_recovery_context(recovery_context) if recovery_context else ""
        )
        retrieval_observation = environment_observation
        if recovery_text:
            retrieval_observation = f"{recovery_text}\n\n{environment_observation}"
        selected_guidance, retrieval_metadata = self._retrieve_agent_skills(
            query=query, environment_observation=retrieval_observation
        )
        retrieval_enabled = bool(retrieval_metadata.get("enabled"))
        guidance_parts: list[str] = []
        base_guidance = sanitize_agent_visible_text(
            str(self.runtime_options.get("base_agent_guidance") or "")
        ).strip()
        if base_guidance:
            guidance_parts.append(
                f"Baseline planning guidance from prior training:\n{base_guidance}\n\nUse this guidance only when it is supported by the current observable task state and provided records."
            )
        if selected_guidance:
            guidance_parts.append(selected_guidance)
        if not guidance_parts and (not retrieval_enabled) and self.agent_guidance:
            guidance_parts.append(f"Optional planning guidance:\n{self.agent_guidance}")
        guidance = "\n\n".join(guidance_parts)
        if guidance:
            guidance = "\n\n" + guidance
        if selected_guidance:
            guidance += "\n\nUse the selected guidance only where it is supported by the current observable task state. Preserve unaffected parts of the plan, verify constraints, and explain material changes to the user."
        user_prompt = TRAVELPLANNER_PLANNER_PROMPT.format(info_text=info_text, query=query)
        if recovery_text:
            continuation_prompt = f"{recovery_text}\n\nCurrent observable environment after the earlier plan:\n{environment_observation}\n\nReturn the complete revised plan."
            plan_marker = "\nTravel Plan:"
            prompt_prefix, marker, prompt_suffix = user_prompt.rpartition(plan_marker)
            if marker:
                user_prompt = f"{prompt_prefix}\n\n{continuation_prompt}{marker}{prompt_suffix}"
            else:
                user_prompt += f"\n\n{continuation_prompt}"
        response = self.chat_client.chat(
            system_prompt=TRAVELPLANNER_SYSTEM_PROMPT, user_prompt=user_prompt + guidance
        )
        if not response.ok:
            return (f"[LLM_ERROR] {response.error}", retrieval_metadata)
        return (response.text.strip(), retrieval_metadata)

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
        skill_index = render_agent_skill_index(skills)
        available_ids = {skill.skill_id for skill in skills}
        retrieval_prompt = build_agent_retrieval_prompt(
            query=query,
            environment_observation=environment_observation,
            skill_index=skill_index,
            max_skills=max_skills,
        )
        response = self.chat_client.chat(
            system_prompt=AGENT_RETRIEVAL_SYSTEM_PROMPT,
            user_prompt=retrieval_prompt,
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
            decision.raw_response = response.error
            decision.source = "llm_error_heuristic"
        selected_blocks = render_selected_agent_skill_blocks(
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
        return (selected_blocks, metadata)

    def _load_queries(self, ref_info_path: Path) -> list[dict[str, Any]]:
        split = self.config.split or "validation"
        candidates = [
            ref_info_path.with_name(f"{split}_queries.jsonl"),
            ref_info_path.with_name(f"{split}.jsonl"),
            ref_info_path.with_name(f"{split}.csv"),
        ]
        for path in candidates:
            if path.exists() and path.suffix == ".jsonl":
                return _load_jsonl(path)
            if path.exists() and path.suffix == ".csv":
                with path.open("r", encoding="utf-8") as f:
                    return list(csv.DictReader(f))
        return [
            {"query": f"Plan a trip for sample {i}"} for i in range(_count_jsonl(ref_info_path))
        ]

    def _effective_ref_info(
        self, sample: TaskSample, intervention: InterventionSpec | None
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        ref_info = sample.context.get("ref_info", {})
        if not intervention:
            return (ref_info, ref_info, [])
        seed = 42 + int(sample.metadata.get("sample_idx", 0))
        wm = TravelPlannerRuleWorldModel(
            dimension=intervention.dimension,
            severity=intervention.severity,
            seed=seed,
            params=intervention.params,
        )
        effective_ref_info = wm.intervene_ref_info(ref_info)
        return (ref_info, effective_ref_info, [record.__dict__ for record in wm.records])

    def _select_planner_ref_info(
        self,
        *,
        original_ref_info: dict[str, Any],
        effective_ref_info: dict[str, Any],
        intervention: InterventionSpec | None,
    ) -> dict[str, Any]:
        if intervention and self.planner_mode in {"static_replay", "factual_replay"}:
            return original_ref_info
        return effective_ref_info


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum((1 for line in f if line.strip()))


def _selected_sample_indices(extra: dict[str, Any], total: int) -> list[int] | None:
    raw_indices = extra.get("sample_indices")
    raw_ids = extra.get("sample_ids")
    if raw_indices in (None, "") and raw_ids in (None, ""):
        return None
    selected: list[int] = []
    for raw in _as_sample_selector_list(raw_indices):
        selected.append(_parse_sample_index(raw))
    for raw in _as_sample_selector_list(raw_ids):
        selected.append(_parse_sample_index(raw))
    deduped: list[int] = []
    seen: set[int] = set()
    for idx in selected:
        if idx < 0 or idx >= total:
            raise ValueError(f"TravelPlanner sample index out of range: {idx} not in [0, {total})")
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def _as_sample_selector_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_sample_index(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.startswith("tp_"):
        text = text.removeprefix("tp_")
    if not text.isdigit():
        raise ValueError(f"Unsupported TravelPlanner sample id/index: {value!r}")
    return int(text)


def _build_query_text(row: dict[str, Any]) -> str:
    if row.get("query"):
        return str(row["query"])
    parts = []
    if row.get("org"):
        parts.append(f"from {row['org']}")
    if row.get("dest"):
        parts.append(f"to {row['dest']}")
    if row.get("days"):
        parts.append(f"for {row['days']} days")
    if row.get("people_number"):
        parts.append(f"for {row['people_number']} people")
    if row.get("budget"):
        parts.append(f"with a budget of ${row['budget']}")
    return "Please plan a trip " + ", ".join(parts) if parts else "Please plan a trip."


def _extract_llm_error(text: str) -> str:
    prefix = "[LLM_ERROR]"
    if not text.startswith(prefix):
        return ""
    return text.removeprefix(prefix).strip()


def _generate_rule_plan(query: str, ref_info: dict[str, Any], query_data: dict[str, Any]) -> str:
    """Small deterministic planner for smoke tests and verifier calibration."""
    org = str(query_data.get("org") or _infer_origin(ref_info) or "Origin")
    days = int(_to_float(query_data.get("days"), 3) or 3)
    cities = _infer_visit_cities(ref_info, str(query_data.get("dest") or ""))
    if not cities:
        cities = [str(query_data.get("dest") or "Destination")]
    dates = _parse_dates(query_data.get("date"))
    restaurant_cursor: dict[str, int] = {}
    attraction_cursor: dict[str, int] = {}
    lines = ["Travel Plan:"]
    previous_city = org
    for day_idx in range(days):
        day_number = day_idx + 1
        city = cities[min(day_idx * len(cities) // max(days, 1), len(cities) - 1)]
        is_last = day_number == days
        if (
            day_idx > 0
            and cities[min((day_idx - 1) * len(cities) // max(days, 1), len(cities) - 1)] == city
        ):
            current_city = city if not is_last else f"from {city} to {org}"
        else:
            current_city = f"from {previous_city} to {city}"
        if is_last:
            current_city = f"from {city} to {org}"
        if day_idx == 0:
            transport = _select_transport(
                ref_info, org, city, dates[0] if dates else None, query_data
            )
        elif is_last:
            transport = _select_transport(
                ref_info, city, org, dates[-1] if dates else None, query_data
            )
        elif current_city.startswith("from"):
            transport = _select_transport(
                ref_info, previous_city, city, dates[day_idx] if dates else None, query_data
            )
        else:
            transport = "-"
        breakfast = _select_restaurant(ref_info, city, restaurant_cursor, query_data)
        lunch = _select_restaurant(ref_info, city, restaurant_cursor, query_data)
        dinner = _select_restaurant(ref_info, city, restaurant_cursor, query_data)
        attraction = _select_attraction(ref_info, city, attraction_cursor)
        accommodation = "-" if is_last else _select_accommodation(ref_info, city, query_data)
        lines.extend(
            [
                f"Day {day_number}:",
                f"Current City: {current_city}",
                f"Transportation: {transport}",
                f"Breakfast: {_with_city(breakfast, city)}",
                f"Attraction: {_with_city(attraction, city)}",
                f"Lunch: {_with_city(lunch, city)}",
                f"Dinner: {_with_city(dinner, city)}",
                f"Accommodation: {(_with_city(accommodation, city) if accommodation != '-' else '-')}",
                "",
            ]
        )
        previous_city = city
    if not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _summarize_ref_info(ref_info: dict[str, Any], max_keys: int = 8) -> str:
    lines = []
    for key, value in list(ref_info.items())[:max_keys]:
        if isinstance(value, list):
            lines.append(f"- {key}: {len(value)} item(s)")
        else:
            text = str(value)
            lines.append(f"- {key}: {text[:120]}")
    return "\n".join(lines)


def _infer_visit_cities(ref_info: dict[str, Any], requested_dest: str) -> list[str]:
    cities = []
    for prefix in ["Attractions in ", "Restaurants in ", "Accommodations in "]:
        for key in ref_info:
            if key.startswith(prefix):
                city = key.removeprefix(prefix).strip()
                if city and city not in cities:
                    cities.append(city)
    if requested_dest and requested_dest in cities:
        cities.remove(requested_dest)
        cities.insert(0, requested_dest)
    return cities


def _infer_origin(ref_info: dict[str, Any]) -> str:
    for key in ref_info:
        if key.startswith("Flight from "):
            rest = key.removeprefix("Flight from ")
            return rest.split(" to ")[0].strip()
        if key.startswith("Self-driving from "):
            rest = key.removeprefix("Self-driving from ")
            return rest.split(" to ")[0].strip()
    return ""


def _parse_dates(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _select_transport(
    ref_info: dict[str, Any], origin: str, dest: str, date: str | None, query_data: dict[str, Any]
) -> str:
    local_constraint = _parse_local_constraint(query_data.get("local_constraint"))
    transport_pref = str(local_constraint.get("transportation") or "").casefold()
    if "no flight" not in transport_pref and "self driving" not in transport_pref:
        flight = _find_available_flight(ref_info, origin, dest, date)
        if flight:
            return f"Flight Number: {flight.get('Flight Number')}, from {origin} to {dest}, Departure Time: {flight.get('DepTime', 'Unknown')}, Arrival Time: {flight.get('ArrTime', 'Unknown')}"
    if "no self-driving" not in transport_pref:
        route = _find_route(ref_info, "Self-driving", origin, dest)
        if route:
            return route
    route = _find_route(ref_info, "Taxi", origin, dest)
    return route or "-"


def _find_available_flight(
    ref_info: dict[str, Any], origin: str, dest: str, date: str | None
) -> dict[str, Any] | None:
    candidates = []
    for key, value in ref_info.items():
        if not key.startswith(f"Flight from {origin} to {dest}"):
            continue
        if date and date not in key:
            continue
        if not isinstance(value, list):
            continue
        candidates.extend(
            (item for item in value if isinstance(item, dict) and (not item.get("Status")))
        )
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: _to_float(item.get("Price"), 0.0))[0]


def _find_route(ref_info: dict[str, Any], mode: str, origin: str, dest: str) -> str:
    value = ref_info.get(f"{mode} from {origin} to {dest}")
    return value if isinstance(value, str) else ""


def _select_restaurant(
    ref_info: dict[str, Any], city: str, cursor: dict[str, int], query_data: dict[str, Any]
) -> str:
    restaurants = _available_items(ref_info.get(f"Restaurants in {city}", []), name_key="Name")
    local_constraint = _parse_local_constraint(query_data.get("local_constraint"))
    cuisines = _as_list(local_constraint.get("cuisine"))
    if cuisines:
        cuisine_filtered = [
            item
            for item in restaurants
            if any(
                (
                    cuisine.casefold() in str(item.get("Cuisines", "")).casefold()
                    for cuisine in cuisines
                )
            )
        ]
        if cuisine_filtered:
            restaurants = cuisine_filtered
    restaurants = sorted(
        restaurants,
        key=lambda item: (
            _to_float(item.get("Average Cost"), 0.0),
            -_to_float(item.get("Aggregate Rating"), 0.0),
        ),
    )
    idx = cursor.get(city, 0)
    cursor[city] = idx + 1
    if idx >= len(restaurants):
        return "-"
    return str(restaurants[idx].get("Name", "-"))


def _select_attraction(ref_info: dict[str, Any], city: str, cursor: dict[str, int]) -> str:
    attractions = _available_items(ref_info.get(f"Attractions in {city}", []), name_key="Name")
    idx = cursor.get(city, 0)
    cursor[city] = idx + 1
    if idx >= len(attractions):
        return "-"
    return str(attractions[idx].get("Name", "-"))


def _select_accommodation(ref_info: dict[str, Any], city: str, query_data: dict[str, Any]) -> str:
    hotels = _available_items(ref_info.get(f"Accommodations in {city}", []), name_key="NAME")
    local_constraint = _parse_local_constraint(query_data.get("local_constraint"))
    room_type = str(local_constraint.get("room type") or "").casefold()
    house_rules = _as_list(local_constraint.get("house rule"))
    if room_type:
        hotels = [
            hotel
            for hotel in hotels
            if _room_type_matches(room_type, str(hotel.get("room type", "")).casefold())
        ] or hotels
    if house_rules:
        filtered = [
            hotel
            for hotel in hotels
            if not _violates_room_rules(house_rules, str(hotel.get("house_rules", "")).casefold())
        ]
        hotels = filtered or hotels
    if not hotels:
        return "-"
    chosen = sorted(hotels, key=lambda item: _to_float(item.get("price"), 0.0))[0]
    return str(chosen.get("NAME", "-"))


def _available_items(value: Any, name_key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if isinstance(item, dict) and item.get(name_key) and (not item.get("Status"))
    ]


def _with_city(name: str, city: str) -> str:
    return "-" if not name or name == "-" else f"{name}, {city}"


def _parse_local_constraint(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return [str(value)]


def _room_type_matches(required: str, actual: str) -> bool:
    if required in actual:
        return True
    if required == "entire room" and "entire home/apt" in actual:
        return True
    return required == "shared room" and "shared" in actual


def _violates_room_rules(required_rules: list[str], actual_rules: str) -> bool:
    required = [rule.casefold().removeprefix("no ").strip() for rule in required_rules]
    return any(
        [
            "smoking" in required and "no smoking" in actual_rules,
            "pets" in required and "no pets" in actual_rules,
            "parties" in required and "no parties" in actual_rules,
            "children under 10" in required and "no children under 10" in actual_rules,
            "visitors" in required and "no visitors" in actual_rules,
        ]
    )


def _to_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        if not cleaned:
            return default
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def _ref_info_to_text(ref_info: dict[str, Any]) -> str:
    parts = []
    for key, value in ref_info.items():
        if isinstance(value, list):
            lines = []
            for item in value[:12]:
                if isinstance(item, dict):
                    compact = ", ".join((f"{k}: {v}" for k, v in list(item.items())[:8]))
                    lines.append(f"  {compact}")
                else:
                    lines.append(f"  {item}")
            parts.append(f"[{key}]\n" + "\n".join(lines))
        else:
            parts.append(f"[{key}]\n{value}")
    return "\n\n".join(parts)
