from __future__ import annotations

from cce_mobility.core.schema import (
    BenchmarkName,
    EvaluationRecord,
    InterventionSpec,
    JudgeScore,
    RecoveryStrategy,
    TaskSample,
    VulnerabilityPoint,
)
from cce_mobility.core.llm import LLMResponse
from cce_mobility.interventions.generator import OntologyAwareLLMInterventionGenerator
from cce_mobility.interventions.ontology import (
    InterventionOntologyCanonicalizer,
    InterventionOntologyEntry,
    InterventionOntologyRegistry,
    WMInterventionOntologyDiscoverer,
    registry_from_seed_dimensions,
)
from cce_mobility.methods.factory import create_method
from cce_mobility.methods.skill_evolution import (
    OntologyDiscoveryPhaseSwitchingCoEvolutionMethod,
)


def test_canonicalizer_maps_surface_dimension_to_generic_operator() -> None:
    canonicalizer = InterventionOntologyCanonicalizer()

    entry = canonicalizer.canonicalize(
        {
            "surface_dimension": "restaurant_closed",
            "domain_operator": "selected_restaurant_closed",
            "affected_entity_type": "restaurant",
            "capability_targets": ["dynamic_replanning", "avoid_invalid_resource"],
            "condition_text": "The selected restaurant is closed.",
            "evidence_from_badcases": ["sample_1"],
        }
    )

    assert entry is not None
    assert entry.operator == "selected_poi_unavailable"
    assert entry.affected_entity_type == "poi"
    assert entry.surface_dimensions == ["restaurant_closed"]
    assert "cascading_replan" in entry.capability_targets
    assert "avoid_invalid_resource" in entry.capability_targets
    assert entry.status == "active"


def test_registry_merges_same_operator_entity(tmp_path) -> None:
    canonicalizer = InterventionOntologyCanonicalizer()
    first = canonicalizer.canonicalize(
        {
            "surface_dimension": "restaurant_closed",
            "domain_operator": "selected_restaurant_closed",
            "affected_entity_type": "restaurant",
            "evidence_from_badcases": ["s1"],
        }
    )
    second = canonicalizer.canonicalize(
        {
            "surface_dimension": "attraction_closed",
            "domain_operator": "selected_attraction_closed",
            "affected_entity_type": "attraction",
            "evidence_from_badcases": ["s2"],
        }
    )
    assert first is not None and second is not None

    registry = InterventionOntologyRegistry([first, second])

    assert len(registry) == 1
    [entry] = registry.all()
    assert entry.operator == "selected_poi_unavailable"
    assert set(entry.surface_dimensions) == {"restaurant_closed", "attraction_closed"}
    assert set(entry.evidence_from_badcases) == {"s1", "s2"}

    path = tmp_path / "ontology.jsonl"
    registry.save_jsonl(path)
    loaded = InterventionOntologyRegistry.load_jsonl(path)
    assert len(loaded) == 1
    assert loaded.all()[0].operator == "selected_poi_unavailable"


def test_registry_merges_seed_resource_with_discovered_specific_entity() -> None:
    seed = InterventionOntologyEntry(
        ontology_id="legacy_seed_id",
        operator="selected_poi_unavailable",
        affected_entity_type="resource",
        surface_dimensions=["poi_closed"],
        capability_targets=["change_detection"],
        source="seed_dimension_ontology",
    )
    discovered = InterventionOntologyEntry(
        ontology_id="legacy_discovered_id",
        operator="selected_poi_unavailable",
        affected_entity_type="poi",
        surface_dimensions=["restaurant_closed"],
        capability_targets=["avoid_invalid_resource"],
        evidence_from_badcases=["case_1"],
        status="active",
        source="heuristic_ontology_discovery",
        metrics={"evidence_count": 1},
    )

    registry = InterventionOntologyRegistry([seed, discovered])

    assert len(registry) == 1
    [entry] = registry.all()
    assert entry.ontology_id.startswith("ontology::selected_poi_unavailable::poi::")
    assert entry.affected_entity_type == "poi"
    assert entry.status == "active"
    assert entry.source == "heuristic_ontology_discovery"
    assert set(entry.surface_dimensions) == {"poi_closed", "restaurant_closed"}
    assert set(entry.capability_targets) == {
        "change_detection",
        "avoid_invalid_resource",
    }
    assert entry.evidence_from_badcases == ["case_1"]
    assert set(entry.metadata["merged_sources"]) == {
        "seed_dimension_ontology",
        "heuristic_ontology_discovery",
    }


def test_seed_dimensions_create_generic_registry() -> None:
    registry = registry_from_seed_dimensions(["poi_closed", "road_blocked", "tool_empty_result"])
    operators = {entry.operator for entry in registry.all()}

    assert operators == {
        "selected_poi_unavailable",
        "route_state_changed",
        "tool_result_missing",
    }


def test_heuristic_discoverer_builds_entries_from_records() -> None:
    record = _charging_failure_record()

    registry = WMInterventionOntologyDiscoverer().discover([record])

    assert len(registry) == 1
    [entry] = registry.all()
    assert entry.operator == "selected_charging_unavailable"
    assert entry.affected_entity_type == "charging"
    assert entry.surface_dimensions == ["charging_pile_unavailable"]
    assert entry.evidence_from_badcases == ["mb_1"]


def test_phase_switching_ontology_method_discovers_and_saves(tmp_path) -> None:
    method = create_method("ontology_phase_switching")
    assert isinstance(method, OntologyDiscoveryPhaseSwitchingCoEvolutionMethod)
    method.configure(
        {
            "start_phase": "wm",
            "min_wm_rounds": 1,
            "max_wm_rounds": 1,
            "seed_ontology_dimensions": ["poi_closed"],
        }
    )
    method.before_iteration([])
    method.observe([_charging_failure_record()])

    metrics = method.report_metrics()
    assert metrics["wm_intervention_ontology_count"] >= 2
    assert metrics["wm_intervention_ontology_update_count"] == 1
    assert "selected_charging_unavailable" in method.world_model_guidance()

    method.save_extra_artifacts(tmp_path, iteration=1)
    assert (tmp_path / "intervention_ontology_iteration_1.jsonl").exists()


def test_ontology_aware_generator_maps_open_surface_dimension() -> None:
    client = _FakeClient(
        """
        {
          "candidates": [
            {
              "capability_target": "change_detection",
              "domain_operator": "selected_poi_unavailable",
              "affected_entity_type": "poi",
              "surface_dimension": "selected_destination_temporarily_closed",
              "benchmark_dimension": "poi_closed",
              "selected_ontology_id": "ontology::selected_poi_unavailable::poi::abc",
              "severity": 1.0,
              "rationale": "The trace depends on this destination POI.",
              "mapping_rationale": "A destination closure maps to poi_closed.",
              "condition_text": "The selected destination is temporarily closed.",
              "target_ids": ["POI A"]
            }
          ]
        }
        """
    )
    generator = OntologyAwareLLMInterventionGenerator(
        client,
        dimensions=["poi_closed", "negative_irrelevant"],
    )
    sample = TaskSample(
        sample_id="mb_2",
        benchmark=BenchmarkName.MOBILITY_BENCH,
        query="Navigate me to POI A.",
    )
    candidates = generator.generate(
        sample,
        trajectory=None,
        points=[VulnerabilityPoint(sample_id="mb_2", step=0, reason="selected POI")],
    )

    assert len(candidates) == 1
    spec = candidates[0].spec
    assert spec.dimension == "poi_closed"
    assert spec.domain_operator == "selected_poi_unavailable"
    assert spec.metadata["ontology_aware_generation"] is True
    assert spec.metadata["llm_wm_surface_dimension"] == "selected_destination_temporarily_closed"
    assert spec.metadata["llm_wm_selected_ontology_id"].startswith("ontology::")


def _charging_failure_record() -> EvaluationRecord:
    sample = TaskSample(
        sample_id="mb_1",
        benchmark=BenchmarkName.MOBILITY_BENCH,
        query="Find a nearby charging station.",
    )
    intervention = InterventionSpec(
        intervention_id="int_1",
        sample_id="mb_1",
        benchmark=BenchmarkName.MOBILITY_BENCH,
        dimension="charging_pile_unavailable",
        domain_operator="selected_charging_unavailable",
        capability_targets=[
            "change_detection",
            "avoid_invalid_resource",
            "cascading_replan",
        ],
        intervention_type="availability",
        condition_text="The selected charging pile is occupied.",
    )
    return EvaluationRecord(
        sample=sample,
        intervention=intervention,
        counterfactual_score=JudgeScore(
            sample_id="mb_1",
            intervention_id="int_1",
            total=1.5,
            strategy=RecoveryStrategy.IGNORE,
        ),
        metadata={
            "intervention_validity": 1.0,
            "counterfactual_hard_metrics": {},
        },
    )


class _FakeClient:
    def __init__(self, text: str):
        self.text = text
        self.is_placeholder = False

    def chat(self, **kwargs) -> LLMResponse:
        return LLMResponse(ok=True, text=self.text)
