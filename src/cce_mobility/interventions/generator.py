from __future__ import annotations

import hashlib
import json
import re
from itertools import product
from pathlib import Path
from typing import Any

from cce_mobility.core.llm import ChatClient
from cce_mobility.core.capabilities import (
    benchmark_domain,
    expected_recovery_actions,
    infer_capability_targets,
    infer_domain_operator,
)
from cce_mobility.core.schema import (
    AgentTrajectory,
    InterventionCandidate,
    InterventionSpec,
    TaskSample,
    VulnerabilityPoint,
)
from cce_mobility.interventions.library import default_dimensions
from cce_mobility.interventions.ontology import (
    InterventionOntologyRegistry,
    registry_from_seed_dimensions,
)


class GridInterventionGenerator:
    """Generate a transparent candidate pool from dimensions and severity grid."""

    def __init__(
        self, dimensions: list[str] | None = None, severity_grid: list[float] | None = None
    ):
        self.dimensions = dimensions
        self.severity_grid = severity_grid or [0.5, 1.0, 1.5]

    def generate(
        self,
        sample: TaskSample,
        trajectory: AgentTrajectory | None,
        points: list[VulnerabilityPoint],
    ) -> list[InterventionCandidate]:
        dims = self.dimensions or default_dimensions(sample.benchmark)
        point = points[0] if points else None
        candidates: list[InterventionCandidate] = []
        for dimension, severity in product(dims, self.severity_grid):
            trigger_step = point.step if point else None
            intervention_type = _dimension_to_type(dimension)
            condition = _condition_text(dimension, severity)
            iid = _stable_id(sample.sample_id, dimension, severity, trigger_step)
            capability_targets = infer_capability_targets(dimension)
            candidates.append(
                InterventionCandidate(
                    spec=InterventionSpec(
                        intervention_id=iid,
                        sample_id=sample.sample_id,
                        benchmark=sample.benchmark,
                        dimension=dimension,
                        domain=benchmark_domain(sample.benchmark),
                        domain_operator=infer_domain_operator(dimension),
                        capability_targets=capability_targets,
                        intervention_type=intervention_type,
                        severity=severity,
                        trigger_step=trigger_step,
                        trigger_condition=point.reason if point else "",
                        diagnostic_goal=_diagnostic_goal(dimension, capability_targets),
                        expected_recovery_actions=expected_recovery_actions(capability_targets),
                        validity_constraints=_validity_constraints(dimension),
                        params={"severity": severity},
                        condition_text=condition,
                        is_negative=(dimension == "negative_irrelevant"),
                        source="grid",
                        metadata={"vulnerability_reason": point.reason if point else ""},
                    )
                )
            )
        return candidates


WM_PROPOSAL_PROMPT = """You are a world model proposing counterfactual interventions for a mobility/travel agent evaluation.

Goal:
Generate local, realistic, recoverable interventions that are likely to expose whether the agent can detect a changed travel world and recover.

Allowed dimensions:
{allowed_dimensions}

Allowed capability targets:
{allowed_capability_targets}

Allowed domain operators:
{allowed_domain_operators}

Generic capability targets and domain operators may appear in prior guidance.
Use them first to decide what capability gap to test, then ground the decision
into one executable benchmark_dimension from the allowed list.

Rules:
- Only use dimensions from the allowed list.
- Only use capability_target values from the allowed capability targets list.
- Only use domain_operator values from the allowed domain operators list.
- Each intervention must be local, verifiable, and recoverable.
- Do not make the task impossible.
- Do not reveal the correct answer.
- Prefer interventions tied to the agent trajectory and visible travel resources.
- Include one negative_irrelevant candidate only when useful as an overreaction control.
- If prior guidance contains a matching spec, include selected_spec_id.
- Make capability_target and domain_operator consistent with benchmark_dimension.

User query:
{query}

Trajectory summary:
{trajectory}

Vulnerability points:
{points}

Prior world-model guidance:
{guidance}

Return strict JSON only:
{{
  "candidates": [
    {{
      "capability_target": "change_detection",
      "domain_operator": "selected_transport_unavailable",
      "benchmark_dimension": "flight_cancelled",
      "selected_spec_id": "optional spec id from prior guidance",
      "severity": 1.0,
      "rationale": "why this is informative",
      "mapping_rationale": "why the generic operator maps to this benchmark dimension",
      "condition_text": "short natural-language counterfactual",
      "target_ids": ["optional exact flight number, hotel name, restaurant name, or attraction name"]
    }}
  ]
}}"""


ONTOLOGY_WM_PROPOSAL_PROMPT = """You are a world model proposing counterfactual interventions for a mobility/travel agent evaluation.

Goal:
Generate local, realistic, recoverable interventions that are likely to expose
whether the agent can detect a changed travel world and recover.

Discovered intervention ontology:
{ontology_registry}

Allowed executable benchmark dimensions:
{allowed_dimensions}

Allowed capability targets:
{allowed_capability_targets}

Allowed domain operators:
{allowed_domain_operators}

Rules:
- First reason in generic ontology terms: capability_target, domain_operator,
  affected_entity_type, and surface_dimension.
- You may propose a novel surface_dimension when the trajectory suggests one.
- benchmark_dimension must still be one value from the allowed executable list,
  so the current benchmark adapter can execute the intervention.
- Only use capability_target values from the allowed capability targets list.
- Prefer domain_operator values from the discovered ontology or allowed domain
  operators list.
- Each intervention must be local, verifiable, and recoverable.
- Do not make the task impossible.
- Do not reveal the correct answer.
- Prefer interventions tied to the agent trajectory and visible travel resources.
- Include one negative_irrelevant candidate only when useful as an overreaction control.

User query:
{query}

Trajectory summary:
{trajectory}

Vulnerability points:
{points}

Prior world-model guidance:
{guidance}

Return strict JSON only:
{{
  "candidates": [
    {{
      "capability_target": "change_detection",
      "domain_operator": "selected_poi_unavailable",
      "affected_entity_type": "poi",
      "surface_dimension": "selected_destination_temporarily_closed",
      "benchmark_dimension": "poi_closed",
      "selected_ontology_id": "optional id from discovered ontology",
      "selected_spec_id": "optional spec id from prior guidance",
      "severity": 1.0,
      "rationale": "why this is informative",
      "mapping_rationale": "why this ontology operator maps to the executable benchmark dimension",
      "condition_text": "short natural-language environment update",
      "target_ids": ["optional exact target resource id or name"]
    }}
  ]
}}"""


class LLMInterventionGenerator:
    """Use a world model to propose bounded, executable interventions.

    The LLM proposes dimensions and rationales. Execution remains rule-backed by
    benchmark adapters, so every proposed intervention must map to an allowed
    dimension.
    """

    def __init__(
        self,
        client: ChatClient,
        *,
        dimensions: list[str] | None = None,
        severity_grid: list[float] | None = None,
        fallback: GridInterventionGenerator | None = None,
    ):
        self.client = client
        self.dimensions = dimensions
        self.severity_grid = severity_grid or [1.0]
        self.fallback = fallback or GridInterventionGenerator(dimensions, self.severity_grid)
        self.guidance = ""

    def set_guidance(self, guidance: str) -> None:
        self.guidance = guidance

    def generate(
        self,
        sample: TaskSample,
        trajectory: AgentTrajectory | None,
        points: list[VulnerabilityPoint],
    ) -> list[InterventionCandidate]:
        allowed = self.dimensions or default_dimensions(sample.benchmark)
        if self.client.is_placeholder:
            return self.fallback.generate(sample, trajectory, points)

        response = self.client.chat(
            system_prompt="You are a strict world model. Output only valid JSON.",
            user_prompt=WM_PROPOSAL_PROMPT.format(
                allowed_dimensions=", ".join(allowed),
                allowed_capability_targets=", ".join(_allowed_capability_targets(allowed)),
                allowed_domain_operators=", ".join(_allowed_domain_operators(allowed)),
                query=sample.query,
                trajectory=_trajectory_summary(trajectory),
                points=_points_summary(points),
                guidance=self.guidance or "(none)",
            ),
            temperature=0.3,
            max_tokens=2048,
        )
        if not response.ok:
            candidates = self.fallback.generate(sample, trajectory, points)
            for cand in candidates:
                cand.validation_notes.append(f"LLM WM failed; fallback grid used: {response.error}")
                cand.spec.metadata["llm_wm_error"] = response.error
            return candidates

        parsed = _parse_json(response.text)
        proposals = parsed.get("candidates", []) if isinstance(parsed, dict) else []
        candidates = _proposal_candidates(
            proposals,
            sample=sample,
            points=points,
            allowed_dimensions=allowed,
            raw_response=response.text,
        )
        if candidates:
            return candidates

        fallback = self.fallback.generate(sample, trajectory, points)
        for cand in fallback:
            cand.validation_notes.append("LLM WM produced no valid candidates; fallback grid used.")
            cand.spec.metadata["llm_wm_response"] = response.text
        return fallback


class OntologyAwareLLMInterventionGenerator(LLMInterventionGenerator):
    """Ontology-aware WM generator.

    This v2 generator lets the world model reason over generic intervention
    ontology entries while still mapping each proposal to an executable
    benchmark dimension. That keeps early experiments comparable and avoids
    random unexecutable open-set labels.
    """

    def __init__(
        self,
        client: ChatClient,
        *,
        dimensions: list[str] | None = None,
        severity_grid: list[float] | None = None,
        fallback: GridInterventionGenerator | None = None,
        ontology_registry: InterventionOntologyRegistry | None = None,
    ):
        super().__init__(
            client,
            dimensions=dimensions,
            severity_grid=severity_grid,
            fallback=fallback,
        )
        self.ontology_registry = ontology_registry or InterventionOntologyRegistry()

    def set_ontology_registry(self, registry: InterventionOntologyRegistry) -> None:
        self.ontology_registry = registry

    def generate(
        self,
        sample: TaskSample,
        trajectory: AgentTrajectory | None,
        points: list[VulnerabilityPoint],
    ) -> list[InterventionCandidate]:
        allowed = self.dimensions or default_dimensions(sample.benchmark)
        ontology_registry = self.ontology_registry
        if len(ontology_registry) == 0:
            ontology_registry = registry_from_seed_dimensions(allowed)

        if self.client.is_placeholder:
            return self.fallback.generate(sample, trajectory, points)

        response = self.client.chat(
            system_prompt="You are a strict ontology-aware world model. Output only valid JSON.",
            user_prompt=ONTOLOGY_WM_PROPOSAL_PROMPT.format(
                ontology_registry=ontology_registry.to_prompt(limit=12) or "(none)",
                allowed_dimensions=", ".join(allowed),
                allowed_capability_targets=", ".join(_allowed_capability_targets(allowed)),
                allowed_domain_operators=", ".join(_allowed_domain_operators(allowed)),
                query=sample.query,
                trajectory=_trajectory_summary(trajectory),
                points=_points_summary(points),
                guidance=self.guidance or "(none)",
            ),
            temperature=0.25,
            max_tokens=2048,
        )
        if not response.ok:
            candidates = self.fallback.generate(sample, trajectory, points)
            for cand in candidates:
                cand.validation_notes.append(
                    f"Ontology-aware LLM WM failed; fallback grid used: {response.error}"
                )
                cand.spec.metadata["llm_wm_error"] = response.error
                cand.spec.metadata["ontology_aware_generation"] = True
            return candidates

        parsed = _parse_json(response.text)
        proposals = parsed.get("candidates", []) if isinstance(parsed, dict) else []
        candidates = _proposal_candidates(
            proposals,
            sample=sample,
            points=points,
            allowed_dimensions=allowed,
            raw_response=response.text,
        )
        for cand in candidates:
            cand.spec.metadata["ontology_aware_generation"] = True
        if candidates:
            return candidates

        fallback = self.fallback.generate(sample, trajectory, points)
        for cand in fallback:
            cand.validation_notes.append(
                "Ontology-aware LLM WM produced no valid candidates; fallback grid used."
            )
            cand.spec.metadata["llm_wm_response"] = response.text
            cand.spec.metadata["ontology_aware_generation"] = True
        return fallback


class FrozenInterventionGenerator:
    """Replay a fixed intervention set for fair baseline comparisons."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.by_sample: dict[str, list[InterventionCandidate]] = {}
        self._load()

    def generate(
        self,
        sample: TaskSample,
        trajectory: AgentTrajectory | None,
        points: list[VulnerabilityPoint],
    ) -> list[InterventionCandidate]:
        return [
            candidate.model_copy(deep=True)
            for candidate in self.by_sample.get(sample.sample_id, [])
        ]

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Frozen intervention set not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                candidate = InterventionCandidate.model_validate(json.loads(line))
                self.by_sample.setdefault(candidate.spec.sample_id, []).append(candidate)


def _dimension_to_type(dimension: str) -> str:
    if dimension == "negative_irrelevant":
        return "negative"
    if "price" in dimension or "budget" in dimension:
        return "cost"
    if "closed" in dimension or "full" in dimension or "unavailable" in dimension:
        return "availability"
    if "route" in dimension or "road" in dimension or "congestion" in dimension:
        return "route"
    if "weather" in dimension:
        return "weather"
    if "tool" in dimension:
        return "tool_reliability"
    return "state_change"


def _condition_text(dimension: str, severity: float) -> str:
    mobility_conditions = {
        "road_blocked": "A road segment on the currently suggested route is blocked and cannot be used.",
        "road_closed": "A key road on the current route is temporarily closed.",
        "congestion": "Traffic on the current route is heavier than expected and may affect arrival time.",
        "congestion_spike": "A sudden congestion spike appears on the current route, causing material delay risk.",
        "eta_shift": "The estimated arrival time for the current route has increased beyond the user's original expectation.",
        "poi_closed": "The destination or selected POI is currently closed or unavailable.",
        "pickup_point_unreachable": "The current pickup point is unreachable and the user needs a feasible nearby alternative.",
        "parking_unavailable": "Parking near the destination is unavailable.",
        "parking_full": "The selected parking option near the destination is full.",
        "charging_unavailable": "The charging option needed for this trip is unavailable.",
        "charging_pile_unavailable": "The selected charging pile is unavailable or occupied.",
        "restriction_policy_changed": "A local traffic restriction now affects the planned route or vehicle choice.",
        "weather_degraded": "Weather has degraded and may affect the current travel mode or route safety.",
        "user_preference_change": "The user has updated a travel preference that affects route or mode choice.",
        "tool_empty_result": "The latest tool query returns no usable result for a needed route or POI lookup.",
        "tool_conflicting_info": "Two tool observations provide conflicting route or POI information that must be verified.",
        "negative_irrelevant": "A minor unrelated environment update appears but does not affect the user's current route or constraints.",
    }
    if dimension in mobility_conditions:
        return mobility_conditions[dimension]
    return f"{dimension} intervention at severity {severity:g}"


def _stable_id(sample_id: str, dimension: str, severity: float, trigger_step: int | None) -> str:
    raw = f"{sample_id}:{dimension}:{severity}:{trigger_step}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"int_{digest}"


def _proposal_candidates(
    proposals: Any,
    *,
    sample: TaskSample,
    points: list[VulnerabilityPoint],
    allowed_dimensions: list[str],
    raw_response: str,
) -> list[InterventionCandidate]:
    if not isinstance(proposals, list):
        return []
    point = points[0] if points else None
    candidates: list[InterventionCandidate] = []
    seen: set[tuple[str, float, int | None]] = set()
    allowed = set(allowed_dimensions)

    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        dimension = _proposal_dimension(proposal)
        if dimension not in allowed:
            continue
        severity = _coerce_severity(proposal.get("severity", 1.0))
        trigger_step = point.step if point else None
        key = (dimension, severity, trigger_step)
        if key in seen:
            continue
        seen.add(key)
        condition_text = str(proposal.get("condition_text") or _condition_text(dimension, severity))
        rationale = str(proposal.get("rationale", ""))
        raw_capability = _proposal_capability(proposal)
        raw_operator = _proposal_operator(proposal)
        proposed_capability = _canonical_capability(raw_capability)
        proposed_operator = _canonical_domain_operator(raw_operator, dimension)
        selected_spec_id = str(proposal.get("selected_spec_id") or "").strip()
        selected_ontology_id = str(proposal.get("selected_ontology_id") or "").strip()
        mapping_rationale = str(proposal.get("mapping_rationale") or "").strip()
        target_ids = _extract_target_ids(proposal, condition_text, rationale)
        iid = _stable_id(sample.sample_id, f"llm:{dimension}:{rationale}", severity, trigger_step)
        capability_targets = infer_capability_targets(dimension)
        inferred_operator = infer_domain_operator(dimension)
        mapping_valid = _mapping_is_consistent(
            proposed_capability=proposed_capability,
            proposed_operator=proposed_operator,
            inferred_capabilities=capability_targets,
            inferred_operator=inferred_operator,
        )
        candidates.append(
            InterventionCandidate(
                spec=InterventionSpec(
                    intervention_id=iid,
                    sample_id=sample.sample_id,
                    benchmark=sample.benchmark,
                    dimension=dimension,
                    domain=benchmark_domain(sample.benchmark),
                    domain_operator=inferred_operator,
                    capability_targets=capability_targets,
                    intervention_type=_dimension_to_type(dimension),
                    severity=severity,
                    trigger_step=trigger_step,
                    trigger_condition=point.reason if point else "",
                    target_selector=_target_selector_for_dimension(dimension),
                    target_entity={"ids": target_ids} if target_ids else {},
                    diagnostic_goal=_diagnostic_goal(dimension, capability_targets),
                    expected_recovery_actions=expected_recovery_actions(capability_targets),
                    validity_constraints=_validity_constraints(dimension),
                    params={"severity": severity, "target_ids": target_ids},
                    condition_text=condition_text,
                    is_negative=(dimension == "negative_irrelevant"),
                    source="llm_world_model",
                    metadata={
                        "vulnerability_reason": point.reason if point else "",
                        "llm_wm_rationale": rationale,
                        "llm_wm_response": raw_response,
                        "llm_wm_target_ids": target_ids,
                        "llm_wm_capability_target": proposed_capability,
                        "llm_wm_domain_operator": proposed_operator,
                        "llm_wm_raw_capability_target": raw_capability,
                        "llm_wm_raw_domain_operator": raw_operator,
                        "llm_wm_benchmark_dimension": dimension,
                        "llm_wm_surface_dimension": str(
                            proposal.get("surface_dimension") or ""
                        ).strip(),
                        "llm_wm_affected_entity_type": str(
                            proposal.get("affected_entity_type") or ""
                        ).strip(),
                        "llm_wm_selected_spec_id": selected_spec_id,
                        "llm_wm_selected_ontology_id": selected_ontology_id,
                        "llm_wm_mapping_rationale": mapping_rationale,
                        "llm_wm_mapping_valid": mapping_valid,
                    },
                )
            )
        )
    return candidates


def _proposal_dimension(proposal: dict[str, Any]) -> str:
    for key in ("benchmark_dimension", "dimension"):
        value = str(proposal.get(key) or "").strip()
        if value:
            return value
    return ""


def _proposal_capability(proposal: dict[str, Any]) -> str:
    value = proposal.get("capability_target")
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value or "").strip()


def _proposal_operator(proposal: dict[str, Any]) -> str:
    return str(proposal.get("domain_operator") or "").strip()


def _allowed_capability_targets(dimensions: list[str]) -> list[str]:
    return sorted({cap for dimension in dimensions for cap in infer_capability_targets(dimension)})


def _allowed_domain_operators(dimensions: list[str]) -> list[str]:
    return sorted({infer_domain_operator(dimension) for dimension in dimensions})


def _canonical_capability(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "constraint_satisfaction": "constraint_verification",
        "constraint_following": "constraint_verification",
        "constraint_adherence": "constraint_verification",
        "dynamic_replanning": "cascading_replan",
        "replanning": "cascading_replan",
        "noise_resilience": "overreaction_control",
        "noise_robustness": "overreaction_control",
    }
    return aliases.get(normalized, normalized)


def _canonical_domain_operator(value: str, dimension: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return ""
    broad_resource_aliases = {
        "selected_resource_unavailable",
        "resource_unavailable",
        "selected_resource_closed",
        "resource_closed",
    }
    if normalized in broad_resource_aliases:
        return infer_domain_operator(dimension)
    aliases = {
        "selected_accommodation_unavailable": "selected_lodging_unavailable",
        "selected_hotel_unavailable": "selected_lodging_unavailable",
        "selected_lodging_full": "selected_lodging_unavailable",
        "selected_restaurant_unavailable": "selected_poi_unavailable",
        "selected_restaurant_closed": "selected_poi_unavailable",
        "selected_dining_unavailable": "selected_poi_unavailable",
        "selected_attraction_unavailable": "selected_poi_unavailable",
        "selected_attraction_closed": "selected_poi_unavailable",
        "irrelevant_noise": "irrelevant_environment_noise",
        "irrelevant_info": "irrelevant_environment_noise",
        "irrelevant_info_change": "irrelevant_environment_noise",
        "irrelevant_info_noise": "irrelevant_environment_noise",
        "irrelevant_information": "irrelevant_environment_noise",
        "irrelevant_change": "irrelevant_environment_noise",
        "irrelevant_context_change": "irrelevant_environment_noise",
        "noise_introduction": "irrelevant_environment_noise",
        "noise_intervention": "irrelevant_environment_noise",
        "negative_irrelevant": "irrelevant_environment_noise",
    }
    return aliases.get(normalized, normalized)


def _mapping_is_consistent(
    *,
    proposed_capability: str,
    proposed_operator: str,
    inferred_capabilities: list[str],
    inferred_operator: str,
) -> bool:
    capability_ok = not proposed_capability or proposed_capability in inferred_capabilities
    operator_ok = not proposed_operator or proposed_operator == inferred_operator
    return capability_ok and operator_ok


def _coerce_severity(value: Any) -> float:
    try:
        severity = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.25, min(3.0, severity))


def _extract_target_ids(proposal: dict[str, Any], condition_text: str, rationale: str) -> list[str]:
    explicit = proposal.get("target_ids", [])
    if isinstance(explicit, str):
        targets = [explicit]
    elif isinstance(explicit, list):
        targets = [str(item) for item in explicit if item]
    else:
        targets = []

    text = f"{condition_text}\n{rationale}"
    targets.extend(re.findall(r"\bF\d{5,}\b", text))
    for pattern in [
        r"Restaurant\s+(.+?)\s+in\s+",
        r"Accommodation\s+(.+?)\s+in\s+",
        r"Hotel\s+(.+?)\s+in\s+",
        r"Attraction\s+(.+?)\s+in\s+",
        r"'([^']+)'",
        r'"([^"]+)"',
    ]:
        targets.extend(match.strip() for match in re.findall(pattern, text) if match.strip())
    return sorted({target for target in targets if target})


def _diagnostic_goal(dimension: str, capability_targets: list[str]) -> str:
    caps = ", ".join(capability_targets)
    return f"Test {caps} under the {dimension} domain operator."


def _validity_constraints(dimension: str) -> list[str]:
    constraints = [
        "The change must be local and verifiable from benchmark evidence.",
        "The task must remain recoverable unless this is an explicit negative control.",
        "The intervention must not reveal the correct recovery answer.",
    ]
    if dimension == "negative_irrelevant":
        constraints.append(
            "The notice must not affect selected resources or hard user constraints."
        )
    else:
        constraints.append(
            "At least one feasible alternative should remain available when possible."
        )
    return constraints


def _target_selector_for_dimension(dimension: str) -> str:
    if dimension == "flight_cancelled":
        return "Select a flight identifier used in the factual trajectory."
    if dimension in {"hotel_full", "hotel_price_surge"}:
        return "Select an accommodation used in the factual trajectory."
    if dimension == "restaurant_closed":
        return "Select a restaurant used in the factual trajectory."
    if dimension == "attraction_closed":
        return "Select an attraction used in the factual trajectory."
    if dimension == "negative_irrelevant":
        return "Select no target resource; generate irrelevant but plausible noise."
    return "Select a task-relevant environment object from the factual trajectory."


def _trajectory_summary(trajectory: AgentTrajectory | None, limit: int = 1800) -> str:
    if trajectory is None:
        return "(none)"
    parts = []
    for event in trajectory.events[:4]:
        parts.append(
            "\n".join(
                [
                    f"step={event.step}",
                    f"tool={event.tool_name}",
                    f"observation={event.observation[:500]}",
                    f"action={event.action_text[:500]}",
                ]
            )
        )
    if trajectory.final_response:
        parts.append(f"final_response={trajectory.final_response[:700]}")
    text = "\n\n".join(parts)
    return text[:limit] if text else "(empty trajectory)"


def _points_summary(points: list[VulnerabilityPoint]) -> str:
    if not points:
        return "(none)"
    lines = []
    for point in points[:5]:
        lines.append(
            f"- step={point.step}; target={point.target}; score={point.score:.2f}; "
            f"reason={point.reason}; evidence={point.evidence[:240]}"
        )
    return "\n".join(lines)


def _parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for idx, ch in enumerate(text[start:], start):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end < 0:
        return None
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
