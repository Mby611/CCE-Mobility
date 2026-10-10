from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class TPInterventionRecord:
    tool_key: str
    dimension: str
    original_text: str
    modified_text: str
    triggered: bool = True
    metadata: dict[str, Any] | None = None


DIMENSION_TOOL_PATTERNS: dict[str, list[str]] = {
    "flight_cancelled": [r"^Flight from .+ to .+ on"],
    "flight_price_surge": [r"^Flight from .+ to .+ on"],
    "hotel_full": [r"^Accommodations in"],
    "hotel_price_surge": [r"^Accommodations in"],
    "attraction_closed": [r"^Attractions in"],
    "restaurant_closed": [r"^Restaurants in"],
    "route_disruption": [r"^Self-driving from", r"^Taxi from"],
    "combo_weather": [r"^Attractions in", r"^Self-driving from", r"^Taxi from"],
    "combo_budget": [r"^Flight from .+ to .+ on", r"^Accommodations in"],
    "negative_irrelevant": [r"^Flight from .+ to .+ on"],
}


class TravelPlannerRuleWorldModel:
    """Deterministic TravelPlanner ref_info mutator.

    This is a clean-room extraction of the useful rule-based intervention idea
    from the old prototype, with severity parameters exposed through
    `InterventionSpec.params`.
    """

    def __init__(
        self,
        dimension: str,
        severity: float = 1.0,
        seed: int = 42,
        params: dict[str, Any] | None = None,
    ):
        self.dimension = dimension
        self.severity = severity
        self.params = params or {}
        self.rng = random.Random(seed)
        self.records: list[TPInterventionRecord] = []

    def intervene_ref_info(self, ref_info: dict[str, Any]) -> dict[str, Any]:
        modified: dict[str, Any] = {}
        for key, value in ref_info.items():
            modified[key] = self._intervene(key, copy.deepcopy(value))
        return modified

    def _matches(self, key: str) -> bool:
        return any(
            re.search(pattern, key) for pattern in DIMENSION_TOOL_PATTERNS.get(self.dimension, [])
        )

    def _intervene(self, key: str, value: Any) -> Any:
        if not self._matches(key):
            return value

        if self.dimension == "flight_cancelled":
            return self._flight_cancelled(key, value)
        if self.dimension == "flight_price_surge":
            return self._flight_price_surge(key, value)
        if self.dimension == "hotel_full":
            return self._hotel_full(key, value)
        if self.dimension == "hotel_price_surge":
            return self._hotel_price_surge(key, value)
        if self.dimension == "attraction_closed":
            return self._attraction_closed(key, value)
        if self.dimension == "restaurant_closed":
            return self._restaurant_closed(key, value)
        if self.dimension == "route_disruption":
            return self._route_disruption(key, value, factor=max(1.0, 1.0 + self.severity))
        if self.dimension == "combo_weather":
            if re.search(r"^Attractions in", key):
                return self._attraction_closed(key, value, reason="CLOSED DUE TO SEVERE WEATHER")
            return self._route_disruption(key, value, factor=max(2.0, 1.5 + self.severity))
        if self.dimension == "combo_budget":
            if re.search(r"^Flight from", key):
                return self._flight_price_surge(key, value, factor=max(1.2, 1.5 + self.severity))
            return self._hotel_price_surge(key, value, factor=max(1.2, 1.3 + self.severity * 0.7))
        if self.dimension == "negative_irrelevant":
            return self._negative_irrelevant(key, value)
        return value

    def _flight_cancelled(self, key: str, value: Any) -> Any:
        if not isinstance(value, list) or len(value) <= 1:
            self._record(key, value, "skipped", triggered=False)
            return value
        target_ids = _param_list(self.params.get("target_ids"))
        if target_ids:
            cancelled = [
                flight
                for flight in value
                if isinstance(flight, dict) and str(flight.get("Flight Number", "")) in target_ids
            ]
            if not cancelled:
                self._record(
                    key,
                    value,
                    "target flight not present",
                    triggered=False,
                    metadata={"target_ids": target_ids},
                )
                return value
            if len(cancelled) >= len(value):
                cancelled = cancelled[: max(0, len(value) - 1)]
            remaining = [
                flight
                for flight in value
                if not (
                    isinstance(flight, dict)
                    and str(flight.get("Flight Number", ""))
                    in {
                        str(item.get("Flight Number", ""))
                        for item in cancelled
                        if isinstance(item, dict)
                    }
                )
            ]
        else:
            cancel_count = int(self.params.get("cancel_count", max(1, round(self.severity))))
            sorted_flights = sorted(
                value, key=lambda x: x.get("Price", 0) if isinstance(x, dict) else 0
            )
            cancel_count = min(cancel_count, len(sorted_flights) - 1)
            cancelled = sorted_flights[:cancel_count]
            remaining = sorted_flights[cancel_count:]
        notices = [
            {
                "Flight Number": f.get("Flight Number", ""),
                "Status": "CANCELLED",
                "Notice": f"Flight {f.get('Flight Number', '')} has been cancelled",
            }
            for f in cancelled
            if isinstance(f, dict)
        ]
        result = remaining + notices
        self._record(
            key,
            value,
            f"{len(remaining)} flights + {len(notices)} cancelled",
            metadata={
                "unavailable_ids": [
                    f.get("Flight Number", "") for f in cancelled if isinstance(f, dict)
                ],
                "target_ids": target_ids,
            },
        )
        return result

    def _flight_price_surge(self, key: str, value: Any, factor: float | None = None) -> Any:
        if not isinstance(value, list):
            return value
        factor = factor or float(self.params.get("factor", max(1.1, 1.0 + self.severity)))
        result = []
        for flight in value:
            f = dict(flight) if isinstance(flight, dict) else flight
            if isinstance(f, dict) and isinstance(f.get("Price"), (int, float)):
                f["Price"] = int(f["Price"] * factor)
                f["Notice"] = f"Price adjusted by x{factor:.2f} due to demand surge"
            result.append(f)
        self._record(key, value, f"{len(result)} flights, prices x{factor:.2f}")
        return result

    def _hotel_full(self, key: str, value: Any) -> Any:
        if not isinstance(value, list) or len(value) <= 1:
            self._record(key, value, "skipped", triggered=False)
            return value
        target_ids = _param_list(self.params.get("target_ids"))
        if target_ids:
            removed = [
                hotel
                for hotel in value
                if isinstance(hotel, dict) and str(hotel.get("NAME", "")) in target_ids
            ]
            if not removed:
                self._record(
                    key,
                    value,
                    "target hotel not present",
                    triggered=False,
                    metadata={"target_ids": target_ids},
                )
                return value
            if len(removed) >= len(value):
                removed = removed[: max(0, len(value) - 1)]
            removed_names = {
                str(item.get("NAME", "")) for item in removed if isinstance(item, dict)
            }
            remaining = [
                hotel
                for hotel in value
                if not (isinstance(hotel, dict) and str(hotel.get("NAME", "")) in removed_names)
            ]
        else:
            remove_count = int(self.params.get("remove_count", max(1, round(2 * self.severity))))
            sorted_hotels = sorted(
                value,
                key=lambda x: x.get("review rate number", 0) if isinstance(x, dict) else 0,
                reverse=True,
            )
            remove_count = min(remove_count, len(sorted_hotels) - 1)
            removed = sorted_hotels[:remove_count]
            remaining = sorted_hotels[remove_count:]
        notices = [
            {
                "NAME": h.get("NAME", ""),
                "Status": "FULLY BOOKED",
                "Notice": f"{h.get('NAME', '')} - No availability",
            }
            for h in removed
            if isinstance(h, dict)
        ]
        result = remaining + notices
        self._record(
            key,
            value,
            f"{len(remaining)} available + {len(notices)} fully booked",
            metadata={
                "unavailable_ids": [h.get("NAME", "") for h in removed if isinstance(h, dict)],
                "target_ids": target_ids,
            },
        )
        return result

    def _hotel_price_surge(self, key: str, value: Any, factor: float | None = None) -> Any:
        if not isinstance(value, list):
            return value
        factor = factor or float(self.params.get("factor", max(1.1, 1.0 + 0.8 * self.severity)))
        result = []
        for hotel in value:
            h = dict(hotel) if isinstance(hotel, dict) else hotel
            if isinstance(h, dict) and isinstance(h.get("price"), (int, float)):
                h["price"] = round(h["price"] * factor, 2)
                h["Notice"] = f"Accommodation price adjusted by x{factor:.2f}"
            result.append(h)
        self._record(key, value, f"{len(result)} hotels, prices x{factor:.2f}")
        return result

    def _attraction_closed(self, key: str, value: Any, reason: str = "TEMPORARILY CLOSED") -> Any:
        if not isinstance(value, list) or len(value) <= 1:
            return value
        target_ids = _param_list(self.params.get("target_ids"))
        result = list(value)
        closed_ids = []
        if target_ids:
            indices = [
                idx
                for idx, item in enumerate(result)
                if isinstance(item, dict) and str(item.get("Name", "")) in target_ids
            ]
            if not indices:
                self._record(
                    key,
                    value,
                    "target attraction not present",
                    triggered=False,
                    metadata={"target_ids": target_ids},
                )
                return value
        else:
            close_count = int(self.params.get("close_count", max(1, round(2 * self.severity))))
            close_count = min(close_count, len(value))
            indices = self.rng.sample(range(len(result)), close_count)
        for idx in indices:
            if isinstance(result[idx], dict):
                item = dict(result[idx])
                item["Status"] = reason
                closed_ids.append(str(item.get("Name", "")))
                result[idx] = item
        self._record(
            key,
            value,
            f"{len(closed_ids)} attractions marked closed",
            metadata={"unavailable_ids": closed_ids, "target_ids": target_ids},
        )
        return result

    def _restaurant_closed(self, key: str, value: Any) -> Any:
        if not isinstance(value, list) or len(value) <= 1:
            return value
        target_ids = _param_list(self.params.get("target_ids"))
        if target_ids:
            removed = [
                restaurant
                for restaurant in value
                if isinstance(restaurant, dict) and str(restaurant.get("Name", "")) in target_ids
            ]
            if not removed:
                self._record(
                    key,
                    value,
                    "target restaurant not present",
                    triggered=False,
                    metadata={"target_ids": target_ids},
                )
                return value
            if len(removed) >= len(value):
                removed = removed[: max(0, len(value) - 1)]
            removed_names = {
                str(item.get("Name", "")) for item in removed if isinstance(item, dict)
            }
            remaining = [
                restaurant
                for restaurant in value
                if not (
                    isinstance(restaurant, dict)
                    and str(restaurant.get("Name", "")) in removed_names
                )
            ]
        else:
            remove_count = int(self.params.get("remove_count", max(1, round(2 * self.severity))))
            sorted_restaurants = sorted(
                value,
                key=lambda x: x.get("Aggregate Rating", 0) if isinstance(x, dict) else 0,
                reverse=True,
            )
            remove_count = min(remove_count, len(sorted_restaurants) - 1)
            removed = sorted_restaurants[:remove_count]
            remaining = sorted_restaurants[remove_count:]
        notices = [
            {
                "Name": r.get("Name", ""),
                "Status": "CLOSED",
                "Notice": f"{r.get('Name', '')} - Temporarily shut down",
            }
            for r in removed
            if isinstance(r, dict)
        ]
        result = remaining + notices
        self._record(
            key,
            value,
            f"{len(remaining)} available + {len(notices)} closed",
            metadata={
                "unavailable_ids": [r.get("Name", "") for r in removed if isinstance(r, dict)],
                "target_ids": target_ids,
            },
        )
        return result

    def _route_disruption(self, key: str, value: Any, factor: float = 2.0) -> Any:
        if not isinstance(value, str):
            return value
        modified = value
        cost_match = re.search(r"cost: (\d+)", modified)
        if cost_match:
            old_cost = int(cost_match.group(1))
            modified = modified.replace(f"cost: {old_cost}", f"cost: {int(old_cost * factor)}")
        modified += (
            f"\n[WARNING] Severe route disruption. Expected duration approximately x{factor:.1f}."
        )
        self._record(key, value, f"route disruption x{factor:.1f}")
        return modified

    def _negative_irrelevant(self, key: str, value: Any) -> Any:
        notice = "Unrelated notice: minor remote event reported. No impact on this trip."
        if isinstance(value, list):
            result = list(value) + [{"Notice": notice}]
        elif isinstance(value, str):
            result = value + f"\n[INFO] {notice}"
        else:
            result = value
        self._record(key, value, "appended irrelevant notice")
        return result

    def _record(
        self,
        key: str,
        original: Any,
        modified: str,
        triggered: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.records.append(
            TPInterventionRecord(
                tool_key=key,
                dimension=self.dimension,
                original_text=_short(original),
                modified_text=modified,
                triggered=triggered,
                metadata=metadata or {},
            )
        )


def _short(value: Any, limit: int = 240) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _param_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []
