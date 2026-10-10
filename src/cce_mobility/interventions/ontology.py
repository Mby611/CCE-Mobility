from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from cce_mobility.core.capabilities import (
    CapabilityTarget,
    expected_recovery_actions,
    infer_capability_targets,
    infer_domain_operator,
)
from cce_mobility.core.llm import ChatClient
from cce_mobility.core.schema import EvaluationRecord


OntologyStatus = Literal["candidate", "active", "promoted", "deprecated"]


class InterventionOntologyEntry(BaseModel):
    """A reusable, benchmark-agnostic intervention operator pattern.

    The entry is intentionally more generic than an intervention dimension. A
    world model may discover that "selected POI unavailable" can surface as
    `restaurant_closed` in TravelPlanner, `poi_closed` in MobilityBench, or a
    resource status change in another observable environment.
    """

    ontology_id: str
    operator: str
    affected_entity_type: str = "resource"
    surface_dimensions: list[str] = Field(default_factory=list)
    capability_targets: list[str] = Field(default_factory=list)
    validity_constraints: list[str] = Field(default_factory=list)
    expected_recovery_actions: list[str] = Field(default_factory=list)
    example_condition_text: str = ""
    evidence_from_badcases: list[str] = Field(default_factory=list)
    status: OntologyStatus = "candidate"
    source: str = "wm_ontology_discovery"
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def prompt_block(self) -> str:
        dimensions = ", ".join(self.surface_dimensions) or "open-set"
        capabilities = ", ".join(self.capability_targets) or "general"
        constraints = "; ".join(self.validity_constraints[:3]) or "local, verifiable, recoverable"
        lines = [
            f"[{self.ontology_id}] {self.operator}",
            f"Affected entity: {self.affected_entity_type}",
            f"Surface dimensions: {dimensions}",
            f"Capability targets: {capabilities}",
            f"Validity constraints: {constraints}",
        ]
        if self.example_condition_text:
            lines.append(f"Example condition: {self.example_condition_text}")
        return "\n".join(lines)


class InterventionOntologyRegistry:
    """JSONL-backed registry for discovered intervention ontology entries."""

    def __init__(self, entries: Iterable[InterventionOntologyEntry] | None = None):
        self._entries: dict[str, InterventionOntologyEntry] = {}
        for entry in entries or []:
            self.add_or_merge(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def all(self) -> list[InterventionOntologyEntry]:
        return list(self._entries.values())

    def active(self) -> list[InterventionOntologyEntry]:
        return [entry for entry in self._entries.values() if entry.status in {"active", "promoted"}]

    def add_or_merge(self, entry: InterventionOntologyEntry) -> InterventionOntologyEntry:
        entry = _normalize_entry(entry)
        existing = self._entries.get(entry.ontology_id)
        if existing is None:
            existing = self._find_merge_target(entry)
        if existing is None:
            self._entries[entry.ontology_id] = entry
            return entry

        old_id = existing.ontology_id
        existing.surface_dimensions = _unique(
            [*existing.surface_dimensions, *entry.surface_dimensions]
        )
        existing.capability_targets = _unique(
            [*existing.capability_targets, *entry.capability_targets]
        )
        existing.validity_constraints = _unique(
            [*existing.validity_constraints, *entry.validity_constraints]
        )
        existing.expected_recovery_actions = _unique(
            [*existing.expected_recovery_actions, *entry.expected_recovery_actions]
        )
        existing.evidence_from_badcases = _unique(
            [*existing.evidence_from_badcases, *entry.evidence_from_badcases]
        )
        if not existing.example_condition_text and entry.example_condition_text:
            existing.example_condition_text = entry.example_condition_text
        existing.metrics = _merge_metrics(existing.metrics, entry.metrics)
        existing.metadata = {
            **existing.metadata,
            **entry.metadata,
            "merged_sources": _unique(
                [
                    existing.source,
                    entry.source,
                    *_as_text_list(existing.metadata.get("merged_sources")),
                    *_as_text_list(entry.metadata.get("merged_sources")),
                ]
            ),
        }
        if (
            existing.source == "seed_dimension_ontology"
            and entry.source != "seed_dimension_ontology"
        ):
            existing.source = entry.source
        if _status_rank(entry.status) > _status_rank(existing.status):
            existing.status = entry.status
        existing = _normalize_entry(existing)
        if existing.ontology_id != old_id:
            self._entries.pop(old_id, None)
            same_id = self._entries.get(existing.ontology_id)
            if same_id is not None and same_id is not existing:
                return self.add_or_merge(existing)
            self._entries[existing.ontology_id] = existing
        return existing

    def _find_merge_target(
        self,
        entry: InterventionOntologyEntry,
    ) -> InterventionOntologyEntry | None:
        key = _ontology_merge_key(entry)
        for existing in self._entries.values():
            if _ontology_merge_key(existing) == key:
                return existing
        return None

    def to_prompt(self, *, limit: int = 8) -> str:
        entries = sorted(
            self.active() or self.all(),
            key=lambda entry: (
                _status_rank(entry.status),
                float(entry.metrics.get("evidence_count", 0)),
            ),
            reverse=True,
        )[:limit]
        if not entries:
            return ""
        return "\n\n".join(
            [
                "Discovered intervention ontology registry:",
                *[entry.prompt_block() for entry in entries],
            ]
        )

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for entry in self.all():
                f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "InterventionOntologyRegistry":
        path = Path(path)
        if not path.exists():
            return cls()
        entries: list[InterventionOntologyEntry] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(InterventionOntologyEntry.model_validate(json.loads(line)))
        return cls(entries)


class InterventionOntologyCanonicalizer:
    """Canonicalize open-set WM proposals into stable operator/capability axes."""

    def canonicalize(
        self,
        proposal: dict[str, Any],
        *,
        fallback_dimension: str = "",
    ) -> InterventionOntologyEntry | None:
        dimension = _normalize_slug(
            str(
                proposal.get("benchmark_dimension")
                or proposal.get("surface_dimension")
                or proposal.get("dimension")
                or fallback_dimension
                or ""
            )
        )
        operator = self.operator(
            str(proposal.get("new_domain_operator") or proposal.get("domain_operator") or ""),
            dimension=dimension,
        )
        if not operator and dimension:
            operator = infer_domain_operator(dimension)
        if not operator:
            return None

        entity = self.entity_type(
            str(proposal.get("affected_entity_type") or proposal.get("entity_type") or ""),
            operator=operator,
        )
        capability_targets = self.capabilities(
            proposal.get("capability_targets") or proposal.get("capability_target"),
            dimension=dimension,
        )
        surface_dimensions = [dimension] if dimension else []
        constraints = _as_text_list(proposal.get("validity_constraints")) or [
            "The change must be local and observable from task evidence.",
            "The task must remain recoverable.",
            "The intervention must not reveal the answer.",
        ]
        expected_actions = _as_text_list(proposal.get("expected_recovery_actions")) or (
            expected_recovery_actions(capability_targets)
        )
        evidence = _as_text_list(proposal.get("evidence_from_badcases"))
        example = str(
            proposal.get("example_condition_text") or proposal.get("condition_text") or ""
        )
        status = "active" if evidence else "candidate"
        ontology_id = _ontology_id(operator, entity)
        return InterventionOntologyEntry(
            ontology_id=ontology_id,
            operator=operator,
            affected_entity_type=entity,
            surface_dimensions=surface_dimensions,
            capability_targets=capability_targets,
            validity_constraints=constraints,
            expected_recovery_actions=expected_actions,
            example_condition_text=example,
            evidence_from_badcases=evidence,
            status=status,
            metrics={"evidence_count": len(evidence)},
            metadata={"raw_proposal": proposal},
        )

    def operator(self, value: str, *, dimension: str = "") -> str:
        return _canonical_operator(value, dimension=dimension)

    def entity_type(self, value: str, *, operator: str = "") -> str:
        normalized = _normalize_slug(value)
        if not normalized or normalized in _GENERIC_ENTITY_TYPES:
            return _entity_from_operator(operator) if operator else "resource"
        return _ENTITY_ALIASES.get(normalized, normalized)

    def capabilities(self, value: Any, *, dimension: str = "") -> list[str]:
        raw_values = _as_text_list(value)
        normalized = [
            _CAPABILITY_ALIASES.get(_normalize_slug(item), _normalize_slug(item))
            for item in raw_values
        ]
        allowed = {target.value for target in CapabilityTarget}
        caps = [item for item in normalized if item in allowed]
        if not caps and dimension:
            caps = infer_capability_targets(dimension)
        return _unique(caps or [CapabilityTarget.CHANGE_DETECTION.value])


ONTOLOGY_DISCOVERY_PROMPT = """You are a world model improving a mobility-agent evaluator.

Discover reusable intervention ontology entries from judged counterfactual traces.
Do not merely repeat benchmark-specific dimensions. Propose generic, reusable
operators that can transfer across TravelPlanner, MobilityBench, and other observable tasks.

Return strict JSON only:
{{
  "entries": [
    {{
      "new_domain_operator": "selected_poi_unavailable",
      "affected_entity_type": "poi",
      "surface_dimension": "poi_closed",
      "capability_targets": ["change_detection", "avoid_invalid_resource"],
      "validity_constraints": ["local", "verifiable", "recoverable"],
      "expected_recovery_actions": ["avoid the unavailable POI", "find a same-role replacement"],
      "example_condition_text": "The selected destination POI is currently closed.",
      "evidence_from_badcases": ["sample id or short evidence"]
    }}
  ]
}}

Judged trace summaries:
{records}
"""


class WMInterventionOntologyDiscoverer:
    """Discover intervention ontology entries from records.

    LLM discovery is used when a real world-model client is available. A
    deterministic heuristic fallback keeps tests and local smoke runs stable.
    """

    def __init__(
        self,
        client: ChatClient | None = None,
        *,
        canonicalizer: InterventionOntologyCanonicalizer | None = None,
    ):
        self.client = client
        self.canonicalizer = canonicalizer or InterventionOntologyCanonicalizer()

    def discover(
        self, records: list[EvaluationRecord], *, limit: int = 8
    ) -> InterventionOntologyRegistry:
        if not records:
            return InterventionOntologyRegistry()
        if self.client is None or self.client.is_placeholder:
            return self._heuristic_discover(records, limit=limit)

        response = self.client.chat(
            system_prompt="You are a strict ontology discovery world model. Output JSON only.",
            user_prompt=ONTOLOGY_DISCOVERY_PROMPT.format(
                records=_records_summary(records, limit=limit),
            ),
            temperature=0.2,
            max_tokens=2048,
        )
        if not response.ok:
            return self._heuristic_discover(records, limit=limit)
        parsed = _parse_json(response.text)
        entries = parsed.get("entries", []) if isinstance(parsed, dict) else []
        registry = InterventionOntologyRegistry()
        for proposal in entries if isinstance(entries, list) else []:
            if not isinstance(proposal, dict):
                continue
            entry = self.canonicalizer.canonicalize(proposal)
            if entry:
                entry.metadata["llm_response"] = response.text
                registry.add_or_merge(entry)
        return registry if len(registry) else self._heuristic_discover(records, limit=limit)

    def _heuristic_discover(
        self,
        records: list[EvaluationRecord],
        *,
        limit: int,
    ) -> InterventionOntologyRegistry:
        grouped: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)
        for record in records:
            if not record.intervention:
                continue
            dimension = record.intervention.dimension
            operator = record.intervention.domain_operator or infer_domain_operator(dimension)
            entity = _entity_from_operator(operator)
            grouped[(operator, entity)].append(record)

        registry = InterventionOntologyRegistry()
        for (operator, entity), items in sorted(
            grouped.items(),
            key=lambda pair: len(pair[1]),
            reverse=True,
        )[:limit]:
            dimensions = _unique(
                record.intervention.dimension for record in items if record.intervention
            )
            caps = _unique(
                cap
                for record in items
                if record.intervention
                for cap in (
                    record.intervention.capability_targets
                    or infer_capability_targets(record.intervention.dimension)
                )
            )
            evidence = _unique(record.sample.sample_id for record in items)[:12]
            entry = InterventionOntologyEntry(
                ontology_id=_ontology_id(operator, entity),
                operator=operator,
                affected_entity_type=entity,
                surface_dimensions=dimensions,
                capability_targets=caps,
                validity_constraints=[
                    "The changed fact must be grounded in a selected or queried task resource.",
                    "At least one feasible recovery path or clarification should remain.",
                    "The intervention must be visible as a current environment update, not evaluator jargon.",
                ],
                expected_recovery_actions=expected_recovery_actions(caps),
                example_condition_text=_example_condition(items),
                evidence_from_badcases=evidence,
                status="active" if evidence else "candidate",
                source="heuristic_ontology_discovery",
                metrics={"evidence_count": len(evidence), "record_count": len(items)},
            )
            registry.add_or_merge(entry)
        return registry


def registry_from_seed_dimensions(dimensions: list[str]) -> InterventionOntologyRegistry:
    canonicalizer = InterventionOntologyCanonicalizer()
    registry = InterventionOntologyRegistry()
    for dimension in dimensions:
        entry = canonicalizer.canonicalize(
            {
                "surface_dimension": dimension,
                "domain_operator": infer_domain_operator(dimension),
                "capability_targets": infer_capability_targets(dimension),
                "validity_constraints": [
                    "The change must be local and verifiable.",
                    "The task must remain recoverable.",
                ],
            },
            fallback_dimension=dimension,
        )
        if entry:
            entry.source = "seed_dimension_ontology"
            registry.add_or_merge(entry)
    return registry


def _ontology_id(operator: str, entity: str) -> str:
    raw = f"{operator}:{entity}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"ontology::{operator}::{entity}::{digest}"


def _normalize_entry(entry: InterventionOntologyEntry) -> InterventionOntologyEntry:
    dimension = entry.surface_dimensions[0] if entry.surface_dimensions else ""
    entry.operator = _canonical_operator(entry.operator, dimension=dimension)
    entry.affected_entity_type = _canonical_entity_type(
        entry.affected_entity_type,
        operator=entry.operator,
    )
    entry.surface_dimensions = _unique(
        _normalize_slug(dimension) for dimension in entry.surface_dimensions
    )
    entry.capability_targets = _canonical_capabilities(entry.capability_targets)
    entry.validity_constraints = _unique(entry.validity_constraints)
    entry.expected_recovery_actions = _unique(entry.expected_recovery_actions)
    entry.evidence_from_badcases = _unique(entry.evidence_from_badcases)
    entry.ontology_id = _ontology_id(entry.operator, entry.affected_entity_type)
    if entry.evidence_from_badcases and "evidence_count" not in entry.metrics:
        entry.metrics["evidence_count"] = len(entry.evidence_from_badcases)
    return entry


def _ontology_merge_key(entry: InterventionOntologyEntry) -> tuple[str, str]:
    normalized = _normalize_entry(entry.model_copy(deep=True))
    return normalized.operator, normalized.affected_entity_type


def _canonical_operator(value: str, *, dimension: str = "") -> str:
    normalized = _normalize_slug(value)
    if not normalized:
        return ""
    if normalized in _BROAD_RESOURCE_ALIASES and dimension:
        return infer_domain_operator(dimension)
    return _OPERATOR_ALIASES.get(normalized, normalized)


def _canonical_entity_type(value: str, *, operator: str = "") -> str:
    normalized = _normalize_slug(value)
    if not normalized or normalized in _GENERIC_ENTITY_TYPES:
        return _entity_from_operator(operator) if operator else "resource"
    return _ENTITY_ALIASES.get(normalized, normalized)


def _canonical_capabilities(values: Iterable[Any]) -> list[str]:
    allowed = {target.value for target in CapabilityTarget}
    normalized = [
        _CAPABILITY_ALIASES.get(_normalize_slug(str(item)), _normalize_slug(str(item)))
        for item in values
    ]
    return _unique(item for item in normalized if item in allowed)


def _normalize_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _merge_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
            merged[key] = max(merged[key], value)
        else:
            merged[key] = value
    return merged


def _status_rank(status: str) -> int:
    return {
        "deprecated": 0,
        "candidate": 1,
        "active": 2,
        "promoted": 3,
    }.get(status, 1)


def _entity_from_operator(operator: str) -> str:
    if "transport" in operator:
        return "transport"
    if "lodging" in operator or "hotel" in operator:
        return "lodging"
    if "poi" in operator or "restaurant" in operator or "attraction" in operator:
        return "poi"
    if "route" in operator or "traffic" in operator:
        return "route"
    if "parking" in operator:
        return "parking"
    if "charging" in operator:
        return "charging"
    if "tool" in operator:
        return "tool_result"
    if "weather" in operator:
        return "weather"
    if "noise" in operator:
        return "irrelevant_context"
    return "resource"


def _example_condition(records: list[EvaluationRecord]) -> str:
    for record in records:
        if record.intervention and record.intervention.condition_text:
            return record.intervention.condition_text
    return ""


def _records_summary(records: list[EvaluationRecord], *, limit: int) -> str:
    lines: list[str] = []
    for record in records[:limit]:
        intervention = record.intervention
        score = record.counterfactual_score
        if not intervention:
            continue
        lines.append(
            "; ".join(
                [
                    f"sample={record.sample.sample_id}",
                    f"dimension={intervention.dimension}",
                    f"operator={intervention.domain_operator}",
                    f"capabilities={intervention.capability_targets}",
                    f"score={score.total if score else 'NA'}",
                    f"condition={intervention.condition_text[:240]}",
                ]
            )
        )
    return "\n".join(lines) or "(none)"


def _parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


_BROAD_RESOURCE_ALIASES = {
    "selected_resource_unavailable",
    "resource_unavailable",
    "selected_resource_closed",
    "resource_closed",
}

_GENERIC_ENTITY_TYPES = {
    "resource",
    "selected_resource",
    "task_resource",
    "entity",
    "object",
    "environment",
    "environment_object",
    "state",
    "context",
}

_OPERATOR_ALIASES = {
    "selected_accommodation_unavailable": "selected_lodging_unavailable",
    "selected_hotel_unavailable": "selected_lodging_unavailable",
    "selected_lodging_full": "selected_lodging_unavailable",
    "selected_restaurant_unavailable": "selected_poi_unavailable",
    "selected_restaurant_closed": "selected_poi_unavailable",
    "selected_attraction_unavailable": "selected_poi_unavailable",
    "selected_attraction_closed": "selected_poi_unavailable",
    "poi_unavailable": "selected_poi_unavailable",
    "poi_closed": "selected_poi_unavailable",
    "road_blocked": "route_state_changed",
    "road_closed": "route_state_changed",
    "route_disruption": "route_state_changed",
    "traffic_congestion": "route_state_changed",
    "empty_tool_result": "tool_result_missing",
    "tool_empty_result": "tool_result_missing",
    "conflicting_tool_result": "tool_result_conflict",
    "negative_irrelevant": "irrelevant_environment_noise",
    "irrelevant_noise": "irrelevant_environment_noise",
}

_ENTITY_ALIASES = {
    "restaurant": "poi",
    "attraction": "poi",
    "destination": "poi",
    "hotel": "lodging",
    "accommodation": "lodging",
    "flight": "transport",
    "train": "transport",
    "road": "route",
    "traffic": "route",
    "tool": "tool_result",
}

_CAPABILITY_ALIASES = {
    "constraint_satisfaction": "constraint_verification",
    "constraint_following": "constraint_verification",
    "dynamic_replanning": "cascading_replan",
    "replanning": "cascading_replan",
    "noise_resilience": "overreaction_control",
    "tool_reliability": "tool_robustness",
}
