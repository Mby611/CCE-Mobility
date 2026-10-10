from __future__ import annotations

from cce_mobility.core.schema import BenchmarkName


MOBILITY_DIMENSIONS = [
    "road_blocked",
    "road_closed",
    "congestion",
    "congestion_spike",
    "eta_shift",
    "poi_closed",
    "pickup_point_unreachable",
    "parking_unavailable",
    "parking_full",
    "charging_unavailable",
    "charging_pile_unavailable",
    "restriction_policy_changed",
    "weather_degraded",
    "user_preference_change",
    "tool_empty_result",
    "tool_conflicting_info",
    "negative_irrelevant",
]


TRAVELPLANNER_DIMENSIONS = [
    "flight_cancelled",
    "flight_price_surge",
    "hotel_full",
    "hotel_price_surge",
    "attraction_closed",
    "restaurant_closed",
    "route_disruption",
    "combo_weather",
    "combo_budget",
    "negative_irrelevant",
]


def default_dimensions(benchmark: BenchmarkName) -> list[str]:
    if benchmark == BenchmarkName.TRAVEL_PLANNER:
        return list(TRAVELPLANNER_DIMENSIONS)
    return list(MOBILITY_DIMENSIONS)
