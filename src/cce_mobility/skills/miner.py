from __future__ import annotations

from collections import defaultdict
from typing import Any

from cce_mobility.core.capabilities import (
    expected_recovery_actions,
    infer_capability_targets,
    infer_domain_operator,
    infer_failure_types,
)
from cce_mobility.core.schema import EvaluationRecord
from cce_mobility.skills.registry import SkillRegistry
from cce_mobility.skills.schema import Skill, SkillStatus, SkillType


FAILURE_STRATEGIES = {"ignore", "hallucinate", "passive_notify"}


def mine_skills_from_records(records: list[EvaluationRecord]) -> SkillRegistry:
    """Mine reusable skills directly from evaluation records."""
    registry = SkillRegistry()
    for skill in _mine_agent_skills_from_records(records):
        registry.add_or_merge(skill)
    for skill in _mine_wm_skills_from_records(records):
        registry.add_or_merge(skill)
    return registry


def mine_agent_skills_from_rows(rows: list[dict[str, Any]]) -> list[Skill]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dimension = str(row.get("dimension") or "general")
        primary_capability = _primary_capability(row)
        domain_operator = str(row.get("domain_operator") or infer_domain_operator(dimension))
        signal = str(row.get("training_signal") or "improve_recovery_quality")
        grouped[(primary_capability, domain_operator, signal)].append(row)
    return [
        _agent_skill_from_group(capability, domain_operator, signal, items)
        for (capability, domain_operator, signal), items in sorted(grouped.items())
    ]


def mine_wm_skills_from_rows(rows: list[dict[str, Any]]) -> list[Skill]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dimension = str(row.get("dimension") or "general")
        primary_capability = _primary_capability(row)
        domain_operator = str(row.get("domain_operator") or infer_domain_operator(dimension))
        label = str(row.get("wm_label") or "low_value_or_too_easy")
        grouped[(primary_capability, domain_operator, label)].append(row)
    return [
        _wm_skill_from_group(capability, domain_operator, label, items)
        for (capability, domain_operator, label), items in sorted(grouped.items())
    ]


def _mine_agent_skills_from_records(records: list[EvaluationRecord]) -> list[Skill]:
    rows = []
    for record in records:
        if not _is_agent_failure(record):
            continue
        score = record.counterfactual_score
        intervention = record.intervention
        assert score is not None and intervention is not None
        hard = record.metadata.get("counterfactual_hard_metrics", {})
        rows.append(
            {
                "sample_id": record.sample.sample_id,
                "dimension": intervention.dimension,
                "domain_operator": intervention.domain_operator
                or infer_domain_operator(intervention.dimension),
                "capability_targets": intervention.capability_targets
                or infer_capability_targets(intervention.dimension),
                "failure_types": infer_failure_types(
                    dimension=intervention.dimension,
                    judge_strategy=score.strategy.value,
                    hard_metrics=hard,
                    score_total=score.total,
                ),
                "training_signal": _agent_training_signal(record),
                "judge_total": score.total,
                "judge_strategy": score.strategy.value,
                "uses_unavailable_identifier": hard.get("uses_unavailable_identifier", False),
            }
        )
    return mine_agent_skills_from_rows(rows)


def _mine_wm_skills_from_records(records: list[EvaluationRecord]) -> list[Skill]:
    rows = []
    for record in records:
        if not record.intervention or not record.counterfactual_score:
            continue
        hard = record.metadata.get("counterfactual_hard_metrics", {})
        diagnosticity = _diagnosticity(record.counterfactual_score.total, hard)
        rows.append(
            {
                "sample_id": record.sample.sample_id,
                "dimension": record.intervention.dimension,
                "domain_operator": record.intervention.domain_operator
                or infer_domain_operator(record.intervention.dimension),
                "capability_targets": record.intervention.capability_targets
                or infer_capability_targets(record.intervention.dimension),
                "failure_types": infer_failure_types(
                    dimension=record.intervention.dimension,
                    judge_strategy=record.counterfactual_score.strategy.value,
                    hard_metrics=hard,
                    score_total=record.counterfactual_score.total,
                ),
                "wm_label": _wm_label(record, diagnosticity),
                "validity_score": record.metadata.get("intervention_validity", 0.0),
                "diagnosticity": diagnosticity,
                "uses_unavailable_identifier": hard.get("uses_unavailable_identifier", False),
            }
        )
    return mine_wm_skills_from_rows(rows)


def _is_agent_failure(record: EvaluationRecord) -> bool:
    if not record.intervention or not record.counterfactual_score:
        return False
    score = record.counterfactual_score
    hard = record.metadata.get("counterfactual_hard_metrics", {})
    return (
        score.total < 3.0
        or score.strategy.value in FAILURE_STRATEGIES
        or bool(hard.get("uses_unavailable_identifier"))
    )


def _agent_training_signal(record: EvaluationRecord) -> str:
    score = record.counterfactual_score
    hard = record.metadata.get("counterfactual_hard_metrics", {})
    if hard.get("uses_unavailable_identifier"):
        return "avoid_unavailable_resource"
    if score and score.strategy.value == "ignore":
        return "detect_and_replan"
    if score and score.strategy.value == "hallucinate":
        return "ground_response_in_modified_environment"
    return "improve_recovery_quality"


def _agent_skill_from_group(
    capability: str,
    domain_operator: str,
    signal: str,
    rows: list[dict[str, Any]],
) -> Skill:
    dimensions = _unique(row.get("dimension") for row in rows)
    capability_targets = _unique(
        cap for row in rows for cap in _as_list(row.get("capability_targets"))
    ) or [capability]
    failure_types = _unique(
        failure for row in rows for failure in _as_list(row.get("failure_types"))
    )
    domain_operators = _unique(row.get("domain_operator") for row in rows) or [domain_operator]
    template = _agent_template(
        primary_dimension=dimensions[0] if dimensions else domain_operator,
        capability=capability,
        domain_operator=domain_operator,
        signal=signal,
        capability_targets=capability_targets,
    )
    evidence_count = len(rows)
    avg_score = _avg(row.get("judge_total") for row in rows)
    status = SkillStatus.ACTIVE if evidence_count >= 1 else SkillStatus.CANDIDATE
    return Skill(
        skill_id=f"agent::{capability}::{domain_operator}::{signal}",
        skill_type=SkillType.AGENT_RECOVERY,
        name=template["name"],
        trigger_dimensions=dimensions,
        capability_targets=capability_targets,
        failure_types=failure_types,
        domain_operators=domain_operators,
        procedure=template["procedure"],
        success_criteria=template["success_criteria"],
        failure_cases=template["failure_cases"],
        source_records=[str(row.get("sample_id")) for row in rows if row.get("sample_id")],
        status=status,
        metrics={
            "evidence_count": evidence_count,
            "avg_failure_score": avg_score,
            "signal": signal,
            "primary_capability": capability,
        },
    )


def _wm_skill_from_group(
    capability: str,
    domain_operator: str,
    label: str,
    rows: list[dict[str, Any]],
) -> Skill:
    dimensions = _unique(row.get("dimension") for row in rows)
    capability_targets = _unique(
        cap for row in rows for cap in _as_list(row.get("capability_targets"))
    ) or [capability]
    failure_types = _unique(
        failure for row in rows for failure in _as_list(row.get("failure_types"))
    )
    domain_operators = _unique(row.get("domain_operator") for row in rows) or [domain_operator]
    template = _wm_template(
        primary_dimension=dimensions[0] if dimensions else domain_operator,
        capability=capability,
        domain_operator=domain_operator,
        label=label,
        capability_targets=capability_targets,
    )
    validity_scores = [row.get("validity_score") for row in rows]
    diagnosticities = [row.get("diagnosticity") for row in rows]
    evidence_count = len(rows)
    status = SkillStatus.ACTIVE if label != "invalid_or_unfair" else SkillStatus.DEPRECATED
    if label == "high_value_challenge" and evidence_count >= 2:
        status = SkillStatus.PROMOTED
    return Skill(
        skill_id=f"spec::{capability}::{domain_operator}::{label}",
        skill_type=SkillType.INTERVENTION_SPEC,
        name=template["name"],
        trigger_dimensions=dimensions,
        capability_targets=capability_targets,
        failure_types=failure_types,
        domain_operators=domain_operators,
        procedure=template["procedure"],
        success_criteria=template["success_criteria"],
        failure_cases=template["failure_cases"],
        source_records=[str(row.get("sample_id")) for row in rows if row.get("sample_id")],
        status=status,
        metrics={
            "evidence_count": evidence_count,
            "avg_validity": _avg(validity_scores),
            "avg_diagnosticity": _avg(diagnosticities),
            "label": label,
            "primary_capability": capability,
        },
    )


def _agent_template(
    *,
    primary_dimension: str,
    capability: str,
    domain_operator: str,
    signal: str,
    capability_targets: list[str],
) -> dict[str, list[str] | str]:
    base = {
        "name": f"Recover {capability} for {domain_operator}",
        "procedure": [
            "Inspect the latest environment observations and tool results before constructing the final plan.",
            "Check whether any selected resource is now unavailable, contradicted, or price-shifted.",
            "Replace invalid resources with available alternatives from the provided data.",
            "Recompute affected constraints such as date, city, budget, and daily itinerary consistency.",
            *expected_recovery_actions(capability_targets),
        ],
        "success_criteria": [
            "The final plan does not reuse cancelled, closed, fully booked, or contradicted resources.",
            "The response briefly acknowledges the relevant current environment fact when useful.",
            "The output remains in the required benchmark format.",
        ],
        "failure_cases": [
            "Do not keep the original resource after it has been cancelled or closed.",
            "Do not invent resources that are absent from the provided environment state.",
        ],
    }
    if primary_dimension == "flight_cancelled":
        base["name"] = "Flight cancellation recovery"
        base["procedure"] = [
            "Identify the cancelled flight number and remove it from consideration.",
            "Choose an available flight matching the required origin, destination, and travel day.",
            "Update the affected day transportation while preserving the rest of the itinerary where possible.",
        ]
    elif primary_dimension == "restaurant_closed":
        base["name"] = "Restaurant closure substitution"
        base["procedure"] = [
            "Identify the closed restaurant name and meal slot.",
            "Select another available restaurant in the same city for the same meal when possible.",
            "Keep attractions, accommodation, and transportation unchanged unless they depend on the meal change.",
        ]
    elif primary_dimension == "hotel_full":
        base["name"] = "Accommodation full-booking recovery"
        base["procedure"] = [
            "Identify the accommodation that became fully booked.",
            "Select another available accommodation satisfying city, room, and budget constraints.",
            "Use the replacement consistently across all nights that need lodging.",
        ]
    elif primary_dimension == "attraction_closed":
        base["name"] = "Attraction closure itinerary repair"
        base["procedure"] = [
            "Identify the closed attraction.",
            "Replace it with another available attraction in the same city or remove only that stop.",
            "Keep day order and meal slots coherent after the repair.",
        ]
    elif primary_dimension == "negative_irrelevant":
        base["name"] = "Ignore irrelevant environment noise"
        base["procedure"] = [
            "Check whether the notice affects selected resources or user constraints.",
            "If it is irrelevant, keep the original plan stable.",
            "Avoid unnecessary replanning while still satisfying the required output format.",
        ]
        base["success_criteria"] = [
            "The agent does not overreact to irrelevant changes.",
            "The final answer remains stable and grounded in the provided data.",
        ]
    if signal == "avoid_unavailable_resource":
        base["failure_cases"].append(
            "Never mention an unavailable identifier as if it were still usable."
        )
    elif signal == "ground_response_in_modified_environment":
        base["failure_cases"].append(
            "Do not fabricate a recovery without checking the current data."
        )
    elif signal == "detect_and_replan":
        base["procedure"].insert(
            0, "Identify the current relevant environment fact before choosing a replacement."
        )
    base["procedure"] = _merge_unique(
        [*base["procedure"], *expected_recovery_actions(capability_targets)]
    )
    return base


def _wm_template(
    *,
    primary_dimension: str,
    capability: str,
    domain_operator: str,
    label: str,
    capability_targets: list[str],
) -> dict[str, list[str] | str]:
    base = {
        "name": f"{domain_operator} intervention spec for {capability}",
        "procedure": [
            "Target a resource that appears in the agent's factual trajectory when possible.",
            "Use exact target_ids whose type matches the chosen dimension.",
            "Keep the change local and recoverable with alternatives still available.",
            f"Make the spec diagnose: {', '.join(capability_targets)}.",
        ],
        "success_criteria": [
            "The spec instantiates into a valid intervention under the benchmark data.",
            "The spec can expose whether the agent detects and repairs the current environment state.",
            "The spec does not make the task impossible.",
        ],
        "failure_cases": [
            "Do not use a restaurant name as a flight target, or a flight number as a hotel target.",
            "Do not create global disruptions that cannot be verified by the benchmark adapter.",
        ],
    }
    if primary_dimension == "flight_cancelled":
        base["name"] = "Selected-flight cancellation spec"
        base["procedure"] = [
            "Inspect the factual plan for selected flight numbers.",
            "Cancel one selected flight number as target_ids.",
            "Prefer flights that are central to reaching or returning from the destination.",
        ]
    elif primary_dimension == "restaurant_closed":
        base["name"] = "Selected-restaurant closure spec"
        base["procedure"] = [
            "Inspect the factual plan for restaurant names.",
            "Close one selected restaurant with an exact name target_id.",
            "Keep same-city restaurant alternatives available.",
        ]
    elif primary_dimension == "hotel_full":
        base["name"] = "Selected-accommodation full-booking spec"
        base["procedure"] = [
            "Inspect the factual plan for selected accommodation names.",
            "Mark one selected accommodation as fully booked using the exact name target_id.",
            "Leave other feasible accommodations available.",
        ]
    elif primary_dimension == "negative_irrelevant":
        base["name"] = "Negative-irrelevant control spec"
        base["procedure"] = [
            "Create a realistic change that does not affect selected resources or constraints.",
            "Use this sparingly to test overreaction.",
            "Make the irrelevance verifiable from the trajectory and environment state.",
        ]
        base["success_criteria"] = [
            "A robust agent should keep the plan stable.",
            "The control helps distinguish recovery ability from unnecessary replanning.",
        ]
    if label == "low_value_or_too_easy":
        base["procedure"].append(
            "Increase diagnosticity next time by targeting a selected resource."
        )
    elif label == "invalid_or_unfair":
        base["failure_cases"].append(
            "Retire this pattern until validity and fairness can be guaranteed."
        )
    base["procedure"] = _merge_unique(base["procedure"])
    return base


def _primary_capability(row: dict[str, Any]) -> str:
    capability_targets = _as_list(row.get("capability_targets"))
    if capability_targets:
        return str(capability_targets[0])
    dimension = str(row.get("dimension") or "general")
    return infer_capability_targets(dimension)[0]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _unique(values: Any) -> list[str]:
    seen = set()
    items: list[str] = []
    for value in values:
        if value is None:
            continue
        item = str(value)
        if item and item not in seen:
            seen.add(item)
            items.append(item)
    return items


def _merge_unique(items: list[str]) -> list[str]:
    seen = set()
    merged = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def _diagnosticity(judge_total: float, hard: dict[str, Any]) -> float:
    exposes_failure = 1.0 if judge_total < 3.0 else 0.0
    unavailable = 1.0 if hard.get("uses_unavailable_identifier") else 0.0
    valid_parse = 1.0 if hard.get("parse_success") else 0.0
    return round(0.45 * exposes_failure + 0.35 * unavailable + 0.20 * valid_parse, 4)


def _wm_label(record: EvaluationRecord, diagnosticity: float) -> str:
    validity = float(record.metadata.get("intervention_validity") or 0.0)
    if validity < 0.7:
        return "invalid_or_unfair"
    if diagnosticity >= 0.7:
        return "high_value_challenge"
    if diagnosticity >= 0.4:
        return "medium_value_challenge"
    return "low_value_or_too_easy"


def _avg(values: Any) -> float:
    numeric = []
    for value in values:
        try:
            if value is not None:
                numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return 0.0
    return round(sum(numeric) / len(numeric), 4)
