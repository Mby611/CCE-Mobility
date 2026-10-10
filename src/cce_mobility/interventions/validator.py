from __future__ import annotations

from cce_mobility.core.schema import InterventionCandidate, TaskSample


class HeuristicInterventionValidator:
    """Fast validator for local, recoverable, judgeable interventions."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def validate(
        self, candidate: InterventionCandidate, sample: TaskSample
    ) -> InterventionCandidate:
        notes: list[str] = []
        score = 1.0

        if candidate.spec.severity <= 0:
            score -= 0.4
            notes.append("non-positive severity")
        if candidate.spec.severity > 3.0:
            score -= 0.3
            notes.append("severity may be too high")
        if not candidate.recovery_spec:
            score -= 0.2
            notes.append("missing recovery spec")
        if candidate.spec.dimension == "negative_irrelevant" and not candidate.spec.is_negative:
            score -= 0.2
            notes.append("negative dimension not marked negative")
        if not sample.query:
            score -= 0.2
            notes.append("empty query")
        target_issue = _travelplanner_target_issue(candidate, sample)
        if target_issue:
            score -= 0.5
            notes.append(target_issue)

        candidate.validity_score = max(0.0, min(1.0, score))
        candidate.validation_notes = notes
        return candidate

    def is_valid(self, candidate: InterventionCandidate) -> bool:
        return candidate.validity_score >= self.threshold


def _travelplanner_target_issue(candidate: InterventionCandidate, sample: TaskSample) -> str:
    if (
        str(sample.benchmark) != "travelplanner"
        and getattr(sample.benchmark, "value", "") != "travelplanner"
    ):
        return ""
    targets = candidate.spec.params.get("target_ids", [])
    if not targets:
        return ""
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, list):
        return "target_ids must be a list or string"
    targets = [str(item) for item in targets if item]
    if not targets:
        return ""

    ref_info = sample.context.get("ref_info", {})
    dimension = candidate.spec.dimension
    if dimension == "flight_cancelled":
        available = _catalog_values(ref_info, prefix="Flight", key="Flight Number")
    elif dimension == "hotel_full":
        available = _catalog_values(ref_info, prefix="Accommodations", key="NAME")
    elif dimension == "restaurant_closed":
        available = _catalog_values(ref_info, prefix="Restaurants", key="Name")
    elif dimension == "attraction_closed":
        available = _catalog_values(ref_info, prefix="Attractions", key="Name")
    else:
        return ""

    unknown = [target for target in targets if target not in available]
    if unknown:
        return f"target ids do not match {dimension} resources: {', '.join(unknown[:3])}"
    return ""


def _catalog_values(ref_info: dict, *, prefix: str, key: str) -> set[str]:
    values: set[str] = set()
    for tool_key, rows in ref_info.items():
        if not str(tool_key).startswith(prefix) or not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get(key):
                values.add(str(row[key]))
    return values
