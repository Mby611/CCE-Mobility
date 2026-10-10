from __future__ import annotations

from enum import Enum
from typing import Any

from cce_mobility.core.schema import BenchmarkName


class CapabilityTarget(str, Enum):
    CHANGE_DETECTION = "change_detection"
    AVOID_INVALID_RESOURCE = "avoid_invalid_resource"
    LOCAL_SUBSTITUTION = "local_substitution"
    CASCADING_REPLAN = "cascading_replan"
    CONSTRAINT_VERIFICATION = "constraint_verification"
    USER_COMMUNICATION = "user_communication"
    OVERREACTION_CONTROL = "overreaction_control"
    TOOL_ROBUSTNESS = "tool_robustness"


class FailureType(str, Enum):
    MISSED_CHANGE_DETECTION = "missed_change_detection"
    USED_INVALID_RESOURCE = "used_invalid_resource"
    NO_REPLANNING = "no_replanning"
    CASCADE_IGNORED = "cascade_ignored"
    CONSTRAINT_NOT_RECHECKED = "constraint_not_rechecked"
    POOR_USER_COMMUNICATION = "poor_user_communication"
    OVERREACTED_TO_IRRELEVANT_NOISE = "overreacted_to_irrelevant_noise"
    HALLUCINATED_RECOVERY = "hallucinated_recovery"
    LOW_DIAGNOSTICITY = "low_diagnosticity"


DIMENSION_CAPABILITY_MAP: dict[str, list[CapabilityTarget]] = {
    "flight_cancelled": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "flight_price_surge": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "hotel_full": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.LOCAL_SUBSTITUTION,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
    ],
    "hotel_price_surge": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
        CapabilityTarget.CASCADING_REPLAN,
    ],
    "restaurant_closed": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.LOCAL_SUBSTITUTION,
    ],
    "attraction_closed": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.LOCAL_SUBSTITUTION,
        CapabilityTarget.CASCADING_REPLAN,
    ],
    "route_disruption": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "combo_weather": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "combo_budget": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
    ],
    "negative_irrelevant": [
        CapabilityTarget.OVERREACTION_CONTROL,
    ],
    "road_blocked": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "road_closed": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "congestion": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
    ],
    "congestion_spike": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "eta_shift": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "poi_closed": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.LOCAL_SUBSTITUTION,
    ],
    "parking_unavailable": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.LOCAL_SUBSTITUTION,
    ],
    "charging_unavailable": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.CASCADING_REPLAN,
    ],
    "weather_degraded": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "user_preference_change": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "pickup_point_unreachable": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.LOCAL_SUBSTITUTION,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "parking_full": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.LOCAL_SUBSTITUTION,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
    ],
    "charging_pile_unavailable": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.AVOID_INVALID_RESOURCE,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
    ],
    "restriction_policy_changed": [
        CapabilityTarget.CHANGE_DETECTION,
        CapabilityTarget.CASCADING_REPLAN,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "tool_empty_result": [
        CapabilityTarget.TOOL_ROBUSTNESS,
        CapabilityTarget.LOCAL_SUBSTITUTION,
        CapabilityTarget.USER_COMMUNICATION,
    ],
    "tool_conflicting_info": [
        CapabilityTarget.TOOL_ROBUSTNESS,
        CapabilityTarget.CONSTRAINT_VERIFICATION,
        CapabilityTarget.USER_COMMUNICATION,
    ],
}


DIMENSION_OPERATOR_LABELS: dict[str, str] = {
    "flight_cancelled": "selected_transport_unavailable",
    "flight_price_surge": "transport_cost_changed",
    "hotel_full": "selected_lodging_unavailable",
    "hotel_price_surge": "lodging_cost_changed",
    "restaurant_closed": "selected_poi_unavailable",
    "attraction_closed": "selected_poi_unavailable",
    "route_disruption": "route_state_changed",
    "combo_weather": "compound_weather_disruption",
    "combo_budget": "compound_budget_pressure",
    "negative_irrelevant": "irrelevant_environment_noise",
    "road_blocked": "route_state_changed",
    "road_closed": "route_state_changed",
    "congestion": "route_state_changed",
    "congestion_spike": "route_state_changed",
    "eta_shift": "schedule_state_changed",
    "poi_closed": "selected_poi_unavailable",
    "parking_unavailable": "selected_parking_unavailable",
    "charging_unavailable": "selected_charging_unavailable",
    "weather_degraded": "weather_state_changed",
    "user_preference_change": "user_goal_changed",
    "pickup_point_unreachable": "pickup_location_unavailable",
    "parking_full": "selected_parking_unavailable",
    "charging_pile_unavailable": "selected_charging_unavailable",
    "restriction_policy_changed": "traffic_policy_changed",
    "tool_empty_result": "tool_result_missing",
    "tool_conflicting_info": "tool_result_conflict",
}


CAPABILITY_RECOVERY_ACTIONS: dict[CapabilityTarget, list[str]] = {
    CapabilityTarget.CHANGE_DETECTION: [
        "Explicitly detect the changed environment fact before committing to a new plan.",
    ],
    CapabilityTarget.AVOID_INVALID_RESOURCE: [
        "Do not reuse resources marked unavailable, cancelled, closed, unsafe, or contradicted.",
    ],
    CapabilityTarget.LOCAL_SUBSTITUTION: [
        "Substitute the affected resource with an available same-role alternative when possible.",
    ],
    CapabilityTarget.CASCADING_REPLAN: [
        "Propagate the change to dependent route, schedule, budget, and itinerary decisions.",
    ],
    CapabilityTarget.CONSTRAINT_VERIFICATION: [
        "Re-check hard constraints such as time, budget, location, capacity, and user preferences.",
    ],
    CapabilityTarget.USER_COMMUNICATION: [
        "Explain the relevant current environment fact and the recovery choice clearly to the user.",
    ],
    CapabilityTarget.OVERREACTION_CONTROL: [
        "Keep the plan stable when the notice is irrelevant to selected resources and constraints.",
    ],
    CapabilityTarget.TOOL_ROBUSTNESS: [
        "Handle empty or conflicting tool results by verifying evidence or asking a targeted clarification.",
    ],
}


CAPABILITY_JUDGE_RUBRIC: dict[CapabilityTarget, str] = {
    CapabilityTarget.CHANGE_DETECTION: "Did the agent notice the relevant environment change?",
    CapabilityTarget.AVOID_INVALID_RESOURCE: "Did the agent avoid unavailable or contradicted resources?",
    CapabilityTarget.LOCAL_SUBSTITUTION: "Did the agent find a feasible same-role replacement?",
    CapabilityTarget.CASCADING_REPLAN: "Did the agent propagate the change through dependent decisions?",
    CapabilityTarget.CONSTRAINT_VERIFICATION: "Did the agent re-check hard constraints after the change?",
    CapabilityTarget.USER_COMMUNICATION: "Did the agent communicate the change and recovery clearly?",
    CapabilityTarget.OVERREACTION_CONTROL: "Did the agent avoid unnecessary replanning for irrelevant noise?",
    CapabilityTarget.TOOL_ROBUSTNESS: "Did the agent handle missing or conflicting tool evidence robustly?",
}


def infer_capability_targets(dimension: str) -> list[str]:
    targets = DIMENSION_CAPABILITY_MAP.get(dimension)
    if targets:
        return [target.value for target in targets]
    if dimension == "negative_irrelevant":
        return [CapabilityTarget.OVERREACTION_CONTROL.value]
    if any(token in dimension for token in ("closed", "full", "unavailable", "cancelled")):
        return [
            CapabilityTarget.CHANGE_DETECTION.value,
            CapabilityTarget.AVOID_INVALID_RESOURCE.value,
            CapabilityTarget.LOCAL_SUBSTITUTION.value,
        ]
    if any(token in dimension for token in ("route", "road", "congestion", "eta")):
        return [
            CapabilityTarget.CHANGE_DETECTION.value,
            CapabilityTarget.CASCADING_REPLAN.value,
            CapabilityTarget.CONSTRAINT_VERIFICATION.value,
        ]
    if any(token in dimension for token in ("price", "budget", "cost")):
        return [
            CapabilityTarget.CHANGE_DETECTION.value,
            CapabilityTarget.CONSTRAINT_VERIFICATION.value,
        ]
    if "tool" in dimension:
        return [CapabilityTarget.TOOL_ROBUSTNESS.value]
    return [CapabilityTarget.CHANGE_DETECTION.value, CapabilityTarget.CASCADING_REPLAN.value]


def infer_domain_operator(dimension: str) -> str:
    return DIMENSION_OPERATOR_LABELS.get(dimension, dimension)


def infer_failure_types(
    *,
    dimension: str,
    judge_strategy: str | None = None,
    hard_metrics: dict[str, Any] | None = None,
    score_total: float | None = None,
) -> list[str]:
    hard = hard_metrics or {}
    failures: list[FailureType] = []
    if hard.get("uses_unavailable_identifier"):
        failures.append(FailureType.USED_INVALID_RESOURCE)
    if judge_strategy == "ignore":
        failures.append(FailureType.MISSED_CHANGE_DETECTION)
    if judge_strategy == "passive_notify":
        failures.append(FailureType.NO_REPLANNING)
    if judge_strategy == "hallucinate":
        failures.append(FailureType.HALLUCINATED_RECOVERY)
    if dimension == "negative_irrelevant" and score_total is not None and score_total < 3.0:
        failures.append(FailureType.OVERREACTED_TO_IRRELEVANT_NOISE)
    if any(
        target == CapabilityTarget.CASCADING_REPLAN.value
        for target in infer_capability_targets(dimension)
    ):
        if score_total is not None and score_total < 3.0:
            failures.append(FailureType.CASCADE_IGNORED)
    if not failures and score_total is not None and score_total < 3.0:
        failures.append(FailureType.CONSTRAINT_NOT_RECHECKED)
    return sorted({failure.value for failure in failures})


def expected_recovery_actions(capability_targets: list[str]) -> list[str]:
    actions: list[str] = []
    for target in capability_targets:
        try:
            capability = CapabilityTarget(target)
        except ValueError:
            continue
        for action in CAPABILITY_RECOVERY_ACTIONS.get(capability, []):
            if action not in actions:
                actions.append(action)
    return actions


def judge_rubric_for_capabilities(capability_targets: list[str]) -> dict[str, str]:
    rubric: dict[str, str] = {}
    for target in capability_targets:
        try:
            capability = CapabilityTarget(target)
        except ValueError:
            continue
        rubric[capability.value] = CAPABILITY_JUDGE_RUBRIC[capability]
    return rubric


def benchmark_domain(benchmark: BenchmarkName | str) -> str:
    value = benchmark.value if isinstance(benchmark, BenchmarkName) else str(benchmark)
    if value == BenchmarkName.TRAVEL_PLANNER.value:
        return "travel_itinerary"
    if value == BenchmarkName.TRAVEL_BENCH.value:
        return "travel_assistant"
    if value == BenchmarkName.MOBILITY_BENCH.value:
        return "mobility_navigation"
    return value
