from cce_mobility.benchmarks.travelplanner import TravelPlannerAdapter
from cce_mobility.core.config import BenchmarkConfig
from cce_mobility.core.llm import LLMResponse
from cce_mobility.core.schema import BenchmarkName, InterventionSpec, TaskSample
from cce_mobility.skills import Skill, SkillRegistry, SkillStatus, SkillType


class FakeRetrievalClient:
    is_placeholder = False

    def __init__(self):
        self.prompts = []

    def chat(self, *, system_prompt, user_prompt, temperature=None, max_tokens=None):
        self.prompts.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if "selected_skill_ids" in user_prompt:
            if "supporting_observable_evidence" in user_prompt:
                return LLMResponse(
                    ok=True,
                    text='{"selected_skill_ids":["skill_transport"],"ignored_skill_ids":["skill_food"],"selection_rationale":{"skill_transport":"The current observation says transport is unavailable."},"supporting_observable_evidence":{"skill_transport":"ENVIRONMENT_UPDATE says selected flight is cancelled."},"trigger_match_reason":{"skill_transport":"The selected transport option is no longer available."},"ignored_skill_reasons":{"skill_food":"No restaurant or meal resource is invalidated."},"no_skill_needed":false}',
                )
            return LLMResponse(
                ok=True,
                text='{"selected_skill_ids":["skill_transport"],"ignored_skill_ids":[],"selection_rationale":{"skill_transport":"A selected transport option is unavailable."},"no_skill_needed":false}',
            )
        return LLMResponse(ok=True, text="Travel Plan:\nDay 1:\nTransportation: alternative")


class FakeFoodRetrievalClient:
    is_placeholder = False

    def __init__(self):
        self.prompts = []

    def chat(self, *, system_prompt, user_prompt, temperature=None, max_tokens=None):
        self.prompts.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if "selected_skill_ids" in user_prompt:
            return LLMResponse(
                ok=True,
                text='{"selected_skill_ids":["skill_food"],"ignored_skill_ids":[],"selection_rationale":{"skill_food":"Only restaurant guidance is visible."},"no_skill_needed":false}',
            )
        return LLMResponse(ok=True, text="Travel Plan:\nDay 1:\nDinner: alternative")


def test_travelplanner_runs_agent_owned_skill_retrieval_before_planning():
    adapter = TravelPlannerAdapter(BenchmarkConfig(name="travelplanner"))
    client = FakeRetrievalClient()
    adapter.set_chat_client(client)
    adapter.set_agent_skill_registry(
        SkillRegistry(
            [
                Skill(
                    skill_id="skill_transport",
                    skill_type=SkillType.AGENT_RECOVERY,
                    name="Recover unavailable transport",
                    trigger_dimensions=["flight_cancelled"],
                    capability_targets=["change_detection", "cascading_replan"],
                    domain_operators=["selected_transport_unavailable"],
                    procedure=["Replace unavailable transport with a valid alternative."],
                    success_criteria=["Do not use unavailable transport identifiers."],
                    status=SkillStatus.ACTIVE,
                )
            ]
        )
    )
    sample = TaskSample(
        sample_id="tp_0",
        benchmark=BenchmarkName.TRAVEL_PLANNER,
        query="Plan a trip.",
        context={
            "ref_info": {
                "Flight from A to B on 2022-03-01": [{"Flight Number": "F1", "Price": 100}]
            },
            "query_data": {"org": "A", "dest": "B", "days": 1},
        },
    )
    trajectory = adapter.run_agent(sample, agent_id="agent_skill_only_iter_1")
    retrieval = trajectory.raw["skill_retrieval"]
    assert retrieval["selected_skill_ids"] == ["skill_transport"]
    assert retrieval["rendered_skill_ids"] == ["skill_transport"]
    assert retrieval["source"] == "llm"
    assert len(client.prompts) == 2
    assert "Available recovery guidance index" in client.prompts[0]["user_prompt"]
    assert "Relevant recovery guidance selected" in client.prompts[1]["user_prompt"]
    assert "counterfactual" not in client.prompts[1]["user_prompt"].lower()
    assert "intervention" not in client.prompts[1]["user_prompt"].lower()


def test_travelplanner_agent_owned_retrieval_can_filter_domain_operators():
    adapter = TravelPlannerAdapter(
        BenchmarkConfig(
            name="travelplanner",
            extra={
                "agent_owned_retrieval": True,
                "agent_retrieval_max_skills": 2,
                "agent_retrieval_exclude_domain_operators": ["selected_transport_unavailable"],
            },
        )
    )
    client = FakeFoodRetrievalClient()
    adapter.set_chat_client(client)
    adapter.set_agent_skill_registry(
        SkillRegistry(
            [
                Skill(
                    skill_id="skill_transport",
                    skill_type=SkillType.AGENT_RECOVERY,
                    name="Recover unavailable transport",
                    trigger_dimensions=["selected_resource_unavailable"],
                    capability_targets=["change_detection", "cascading_replan"],
                    domain_operators=["selected_transport_unavailable"],
                    procedure=["Replace unavailable transport with a valid alternative."],
                    success_criteria=["Do not use unavailable transport identifiers."],
                    status=SkillStatus.ACTIVE,
                ),
                Skill(
                    skill_id="skill_food",
                    skill_type=SkillType.AGENT_RECOVERY,
                    name="Recover unavailable restaurant",
                    trigger_dimensions=["selected_resource_unavailable"],
                    capability_targets=["local_substitution"],
                    domain_operators=["selected_poi_unavailable"],
                    procedure=["Replace unavailable restaurants with valid alternatives."],
                    success_criteria=["Do not use unavailable restaurant identifiers."],
                    status=SkillStatus.ACTIVE,
                ),
            ]
        )
    )
    sample = TaskSample(
        sample_id="tp_0",
        benchmark=BenchmarkName.TRAVEL_PLANNER,
        query="Plan a trip.",
        context={
            "ref_info": {
                "Flight from A to B on 2022-03-01": [{"Flight Number": "F1", "Price": 100}]
            },
            "query_data": {"org": "A", "dest": "B", "days": 1},
        },
    )
    intervention = InterventionSpec(
        intervention_id="int_1",
        sample_id="tp_0",
        benchmark=BenchmarkName.TRAVEL_PLANNER,
        dimension="flight_cancelled",
        intervention_type="availability",
        condition_text="selected flight is cancelled",
    )
    trajectory = adapter.run_agent(sample, agent_id="agent", intervention=intervention)
    retrieval_prompt = client.prompts[0]["user_prompt"]
    retrieval = trajectory.raw["skill_retrieval"]
    assert "skill_transport" not in retrieval_prompt
    assert "skill_food" in retrieval_prompt
    assert retrieval["selected_skill_ids"] == ["skill_food"]
    assert retrieval["available_skill_count"] == 1
    assert retrieval["unfiltered_available_skill_count"] == 2
    assert retrieval["excluded_domain_operators"] == ["selected_transport_unavailable"]


def test_travelplanner_environment_update_sanitizes_evaluator_terms():
    adapter = TravelPlannerAdapter(BenchmarkConfig(name="travelplanner"))
    adapter.set_chat_client(FakeRetrievalClient())
    sample = TaskSample(
        sample_id="tp_0",
        benchmark=BenchmarkName.TRAVEL_PLANNER,
        query="Plan a trip.",
        context={
            "ref_info": {
                "Flight from A to B on 2022-03-01": [{"Flight Number": "F1", "Price": 100}]
            },
            "query_data": {"org": "A", "dest": "B", "days": 1},
        },
    )
    intervention = InterventionSpec(
        intervention_id="int_1",
        sample_id="tp_0",
        benchmark=BenchmarkName.TRAVEL_PLANNER,
        dimension="flight_cancelled",
        intervention_type="availability",
        condition_text="counterfactual intervention: selected flight is cancelled",
    )
    trajectory = adapter.run_agent(sample, agent_id="agent", intervention=intervention)
    observation = trajectory.events[0].observation.lower()
    assert "[environment_update]" in observation
    assert "counterfactual" not in observation
    assert "intervention" not in observation
