from cce_mobility.core.schema import (
    BenchmarkName,
    InterventionCandidate,
    InterventionSpec,
    TaskSample,
)
from cce_mobility.interventions.travelplanner_spec import TravelPlannerSpecBuilder


def test_spec_builder_prefers_structured_unavailable_ids():
    candidate = InterventionCandidate(
        spec=InterventionSpec(
            intervention_id="int_test",
            sample_id="tp_0",
            benchmark=BenchmarkName.TRAVEL_PLANNER,
            dimension="flight_cancelled",
            intervention_type="state_change",
            metadata={
                "intervention_records": [
                    {
                        "original_text": "[{'Flight Number': 'F_OK'}, {'Flight Number': 'F_BAD'}]",
                        "modified_text": "1 flights + 1 cancelled",
                        "metadata": {"unavailable_ids": ["F_BAD"]},
                    }
                ],
                "unavailable_ids": ["F_BAD"],
            },
        )
    )
    sample = TaskSample(
        sample_id="tp_0",
        benchmark=BenchmarkName.TRAVEL_PLANNER,
        query="Plan a trip.",
    )

    spec = TravelPlannerSpecBuilder().build(candidate, sample)

    assert spec.metadata["removed_or_changed"] == ["F_BAD"]
    assert "F_OK" not in " ".join(spec.changed_facts)
    assert "avoid_invalid_resource" in spec.capability_targets
    assert "expected_actions" in spec.recovery_contract
