from __future__ import annotations

import re

from cce_mobility.core.capabilities import (
    expected_recovery_actions,
    infer_capability_targets,
    judge_rubric_for_capabilities,
)
from cce_mobility.core.schema import ExpectedRecoverySpec, InterventionCandidate, TaskSample
from cce_mobility.interventions.spec_builder import DEFAULT_RUBRIC


TP_RUBRICS: dict[str, dict[str, str]] = {
    "flight_cancelled": {
        "proactive_adapt": "Uses a still-available flight and adjusts the rest of the itinerary if cost/timing changes matter.",
        "partial_adapt": "Uses a still-available flight but does not propagate timing or budget consequences.",
        "passive_notify": "Mentions cancellation but does not choose a usable replacement.",
        "ignore": "Shows no evidence of cancellation awareness.",
        "hallucinate": "References a cancelled flight.",
    },
    "flight_price_surge": {
        "proactive_adapt": "Accounts for surged flight prices and adjusts budget-sensitive choices.",
        "partial_adapt": "Acknowledges price changes but only adjusts the flight.",
        "passive_notify": "Mentions price surge without adapting.",
        "ignore": "Keeps the same price-sensitive plan.",
        "hallucinate": "Uses original lower prices that contradict modified info.",
    },
    "hotel_full": {
        "proactive_adapt": "Switches to an available hotel and keeps logistics coherent.",
        "partial_adapt": "Switches hotel but leaves possible inconsistency.",
        "passive_notify": "Mentions unavailability without selecting a usable option.",
        "ignore": "No evidence of fully booked hotel awareness.",
        "hallucinate": "Uses a hotel marked fully booked.",
    },
    "hotel_price_surge": {
        "proactive_adapt": "Accounts for higher lodging cost and adjusts budget-sensitive choices.",
        "partial_adapt": "Acknowledges hotel price change but only partly adjusts.",
        "passive_notify": "Mentions price increase without adapting.",
        "ignore": "No price-awareness.",
        "hallucinate": "Uses original lower hotel prices.",
    },
    "attraction_closed": {
        "proactive_adapt": "Replaces closed attractions and keeps the day coherent.",
        "partial_adapt": "Replaces only part of the affected itinerary.",
        "passive_notify": "Mentions closure without substitute.",
        "ignore": "No closure awareness.",
        "hallucinate": "Uses a closed attraction.",
    },
    "restaurant_closed": {
        "proactive_adapt": "Replaces closed restaurants while respecting budget/cuisine constraints.",
        "partial_adapt": "Replaces restaurants but misses some constraints.",
        "passive_notify": "Mentions closure without substitute.",
        "ignore": "No closure awareness.",
        "hallucinate": "Uses a closed restaurant.",
    },
    "route_disruption": {
        "proactive_adapt": "Adjusts itinerary for route delay/cost increase.",
        "partial_adapt": "Acknowledges disruption but only partially adjusts.",
        "passive_notify": "Mentions disruption without changing plan.",
        "ignore": "No route-disruption awareness.",
        "hallucinate": "Uses original undisrupted duration/cost.",
    },
    "combo_weather": {
        "proactive_adapt": "Handles both weather closures and route disruption coherently.",
        "partial_adapt": "Handles only one affected aspect.",
        "passive_notify": "Mentions weather without adapting itinerary.",
        "ignore": "No weather-event awareness.",
        "hallucinate": "Uses closed/weather-unsafe items.",
    },
    "combo_budget": {
        "proactive_adapt": "Handles combined flight and hotel price pressure with budget-aware replanning.",
        "partial_adapt": "Adjusts only one of flight/hotel or leaves budget risk.",
        "passive_notify": "Mentions price pressure without adapting.",
        "ignore": "No budget-pressure awareness.",
        "hallucinate": "Uses prices contradicted by modified info.",
    },
    "negative_irrelevant": {
        "proactive_adapt": "Failure: overreacts to irrelevant information.",
        "partial_adapt": "Weak failure: makes unnecessary changes.",
        "passive_notify": "Weak failure: mentions irrelevant information.",
        "ignore": "Success: correctly ignores irrelevant information.",
        "hallucinate": "Failure: fabricates relevance for irrelevant information.",
    },
}


class TravelPlannerSpecBuilder:
    """Build TravelPlanner-specific expected recovery specs."""

    def build(self, candidate: InterventionCandidate, sample: TaskSample) -> ExpectedRecoverySpec:
        spec = candidate.spec
        dimension = spec.dimension
        negative = spec.is_negative
        capability_targets = spec.capability_targets or infer_capability_targets(dimension)
        capability_actions = spec.expected_recovery_actions or expected_recovery_actions(
            capability_targets
        )
        changed = [spec.condition_text]
        removed = _extract_removed_like_items(candidate)
        if removed:
            changed.append(f"Unavailable/changed identifiers: {', '.join(removed[:8])}")

        if negative:
            must_do = ["Ignore the irrelevant notice and preserve the original travel objective."]
            must_not_do = [
                "Do not change route, flight, hotel, or itinerary solely due to irrelevant noise."
            ]
            must_detect: list[str] = []
            acceptable = ["A normal plan that does not overreact to the irrelevant notice."]
        else:
            must_detect = [dimension]
            must_do = [
                "Detect the modified travel information.",
                "Avoid unavailable, cancelled, unsafe, or contradicted options.",
                "Recover with a feasible alternative or explain constraints clearly.",
            ]
            must_not_do = [
                "Do not use identifiers marked cancelled, closed, fully booked, or unsafe.",
                "Do not keep the original plan if the intervention invalidates it.",
            ]
            acceptable = [
                "Substitute affected items with available options.",
                "Adjust budget, timing, or itinerary constraints when the intervention cascades.",
            ]
        must_do = _merge_unique([*must_do, *capability_actions])
        capability_rubric = judge_rubric_for_capabilities(capability_targets)

        return ExpectedRecoverySpec(
            intervention_id=spec.intervention_id,
            sample_id=sample.sample_id,
            capability_targets=capability_targets,
            recovery_contract={
                "capability_targets": capability_targets,
                "expected_actions": capability_actions,
                "diagnostic_goal": spec.diagnostic_goal,
                "validity_constraints": spec.validity_constraints,
                "domain_operator": spec.domain_operator,
                "target_entity": spec.target_entity,
                "removed_or_changed": removed,
            },
            summary=f"{dimension}: {spec.condition_text}",
            changed_facts=changed,
            must_detect=must_detect,
            must_do=must_do,
            must_not_do=must_not_do,
            acceptable_recoveries=acceptable,
            invalid_failures=[
                "Ignores a relevant intervention.",
                "Hallucinates unavailable options.",
                "Overreacts to a negative/no-op intervention.",
            ],
            rubric={**TP_RUBRICS.get(dimension, DEFAULT_RUBRIC), **capability_rubric},
            metadata={
                "dimension": dimension,
                "domain": spec.domain,
                "domain_operator": spec.domain_operator,
                "capability_targets": capability_targets,
                "severity": spec.severity,
                "removed_or_changed": removed,
            },
        )


def _extract_removed_like_items(candidate: InterventionCandidate) -> list[str]:
    explicit = candidate.spec.metadata.get("unavailable_ids", [])
    if isinstance(explicit, list):
        explicit_ids = [str(item) for item in explicit if item]
    else:
        explicit_ids = []

    records = candidate.spec.metadata.get("intervention_records", [])
    found: list[str] = explicit_ids
    for rec in records if isinstance(records, list) else []:
        metadata = rec.get("metadata", {})
        if isinstance(metadata, dict):
            found.extend(str(item) for item in metadata.get("unavailable_ids", []) if item)
    if found:
        return sorted(set(x for x in found if x))

    for rec in records if isinstance(records, list) else []:
        text = f"{rec.get('original_text', '')} {rec.get('modified_text', '')}"
        found.extend(re.findall(r"F\d{5,}", text))
        found.extend(re.findall(r"'NAME': '([^']+)'", text))
        found.extend(re.findall(r"'Name': '([^']+)'", text))
    return sorted(set(x for x in found if x))


def _merge_unique(items: list[str]) -> list[str]:
    seen = set()
    merged = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            merged.append(item)
    return merged
