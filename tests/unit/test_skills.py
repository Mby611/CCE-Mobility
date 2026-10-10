import json
from cce_mobility.core.schema import (
    BenchmarkName,
    EvaluationRecord,
    InterventionSpec,
    JudgeScore,
    RecoveryStrategy,
    TaskSample,
)
from cce_mobility.methods.skill_evolution import SkillEvolutionMethod
from cce_mobility.skills import (
    Skill,
    SkillRegistry,
    SkillStatus,
    SkillType,
    agent_self_retrieval_prompt,
    mine_skills_from_records,
)
from cce_mobility.skills.retrieval import (
    parse_retrieval_decision,
    render_agent_skill_index,
    render_selected_agent_skill_blocks,
)


def _failure_record() -> EvaluationRecord:
    return EvaluationRecord(
        sample=TaskSample(
            sample_id="tp_0", benchmark=BenchmarkName.TRAVEL_PLANNER, query="Plan a trip."
        ),
        intervention=InterventionSpec(
            intervention_id="int_1",
            sample_id="tp_0",
            benchmark=BenchmarkName.TRAVEL_PLANNER,
            dimension="flight_cancelled",
            intervention_type="availability",
            params={"target_ids": ["F123456"]},
        ),
        counterfactual_score=JudgeScore(
            sample_id="tp_0", intervention_id="int_1", total=1.0, strategy=RecoveryStrategy.IGNORE
        ),
        metadata={
            "intervention_validity": 1.0,
            "counterfactual_hard_metrics": {
                "parse_success": True,
                "uses_unavailable_identifier": True,
            },
        },
    )


def test_mine_skills_from_failure_record_builds_agent_and_wm_skills():
    registry = mine_skills_from_records([_failure_record()])
    agent_prompt = registry.to_prompt(SkillType.AGENT_RECOVERY)
    wm_prompt = registry.to_prompt(SkillType.INTERVENTION_SPEC)
    assert "Flight cancellation recovery" in agent_prompt
    assert "Capability targets:" in agent_prompt
    assert "Never mention an unavailable identifier" in agent_prompt
    assert "Selected-flight cancellation spec" in wm_prompt


def test_skill_registry_round_trip(tmp_path):
    registry = mine_skills_from_records([_failure_record()])
    path = tmp_path / "skills.jsonl"
    registry.save_jsonl(path)
    loaded = SkillRegistry.load_jsonl(path)
    assert len(loaded) == len(registry)
    assert "flight_cancelled" in loaded.to_prompt(SkillType.AGENT_RECOVERY)


def test_skill_registry_filtered_load(tmp_path):
    registry = mine_skills_from_records([_failure_record()])
    path = tmp_path / "skills.jsonl"
    registry.save_jsonl(path)
    loaded = SkillRegistry.load_jsonl_filtered(path, skill_types={SkillType.AGENT_RECOVERY})
    assert len(loaded.active(SkillType.AGENT_RECOVERY)) == 1
    assert loaded.active(SkillType.INTERVENTION_SPEC) == []


def test_skill_registry_filters_by_capability_target():
    registry = mine_skills_from_records([_failure_record()])
    skills = registry.active(
        SkillType.AGENT_RECOVERY, capability_targets=["avoid_invalid_resource"]
    )
    assert len(skills) == 1
    assert "avoid_invalid_resource" in skills[0].capability_targets


def test_agent_skill_prompt_exposes_self_retrieval_catalog():
    registry = mine_skills_from_records([_failure_record()])
    prompt = agent_self_retrieval_prompt(registry)
    assert "self-retrieval" in prompt
    assert "Select only catalog skills" in prompt
    assert "Agent recovery skill memory catalog" in prompt
    assert "Flight cancellation recovery" in prompt


def test_agent_skill_prompt_hides_evaluator_only_counterfactual_terms():
    registry = SkillRegistry(
        [
            Skill(
                skill_id="legacy_recovery",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Legacy recovery",
                procedure=[
                    "Read the counterfactual notice before planning.",
                    "Do not reuse resources marked unavailable by the intervention.",
                    "Ground the recovery in the modified environment state.",
                ],
                success_criteria=["Repair the changed world."],
                status=SkillStatus.ACTIVE,
            )
        ]
    )
    prompt = agent_self_retrieval_prompt(registry)
    lowered = prompt.lower()
    assert "counterfactual" not in lowered
    assert "intervention" not in lowered
    assert "modified environment" not in lowered
    assert "latest environment observation" in prompt
    assert "current environment state" in prompt


def test_agent_owned_retrieval_renders_index_and_selected_blocks_without_evaluator_terms():
    registry = SkillRegistry(
        [
            Skill(
                skill_id="legacy_recovery",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Legacy counterfactual recovery",
                trigger_dimensions=["flight_cancelled"],
                capability_targets=["change_detection"],
                domain_operators=["selected_transport_unavailable"],
                procedure=["Read the counterfactual notice before planning."],
                success_criteria=["Avoid intervention resources."],
                status=SkillStatus.ACTIVE,
            )
        ]
    )
    skills = registry.active(SkillType.AGENT_RECOVERY)
    index = render_agent_skill_index(skills)
    blocks = render_selected_agent_skill_blocks(registry, ["legacy_recovery"])
    combined = f"{index}\n{blocks}".lower()
    assert "legacy_recovery" in index
    assert "counterfactual" not in combined
    assert "intervention" not in combined
    assert "latest environment observation" in blocks


def test_parse_retrieval_decision_filters_unknown_and_limits_selected_ids():
    decision = parse_retrieval_decision(
        '{"selected_skill_ids":["s1","missing","s2"],"ignored_skill_ids":["s3"],"selection_rationale":{"s1":"matches current state"},"no_skill_needed":false}',
        available_skill_ids={"s1", "s2", "s3"},
        max_skills=1,
    )
    assert decision.selected_skill_ids == ["s1"]
    assert decision.ignored_skill_ids == ["s3"]
    assert decision.no_skill_needed is False
    assert decision.parse_error == ""


def test_parse_retrieval_decision_keeps_agent_evidence_fields():
    decision = parse_retrieval_decision(
        json.dumps(
            {
                "selected_skill_ids": ["s1"],
                "ignored_skill_ids": ["s2"],
                "selection_rationale": {"s1": "The tool output is empty."},
                "supporting_observable_evidence": {
                    "s1": "Current tool result says no result was found."
                },
                "trigger_match_reason": {"s1": "This matches the tool-result-missing trigger."},
                "ignored_skill_reasons": {"s2": "No selected location is marked unavailable."},
                "no_skill_needed": False,
            }
        ),
        available_skill_ids={"s1", "s2"},
        max_skills=2,
    )
    assert decision.selected_skill_ids == ["s1"]
    assert decision.ignored_skill_ids == ["s2"]
    assert decision.supporting_observable_evidence["s1"].startswith("Current tool")
    assert "tool-result-missing" in decision.trigger_match_reason["s1"]
    assert decision.ignored_skill_reasons["s2"].startswith("No selected location")


def test_skill_evolution_method_emits_guidance_after_observe():
    method = SkillEvolutionMethod()
    method.before_iteration([])
    assert method.agent_guidance() == ""
    assert method.world_model_guidance() == ""
    method.observe([_failure_record()])
    assert "skill memory catalog" in method.agent_guidance()
    assert "self-retrieval" in method.agent_guidance()
    assert "Flight cancellation recovery" in method.agent_guidance()
    assert "spec memory catalog" in method.world_model_guidance()
    assert "self-retrieval" in method.world_model_guidance()
    assert method.report_metrics()["agent_recovery_skill_count"] == 1
    assert method.report_metrics()["wm_intervention_spec_count"] == 1


def test_skill_evolution_method_can_save_load_and_freeze(tmp_path):
    method = SkillEvolutionMethod()
    method.observe([_failure_record()])
    path = tmp_path / "memory.jsonl"
    method.save_registry(path)
    loaded = SkillEvolutionMethod()
    loaded.load_registry(path)
    loaded.freeze()
    loaded.observe([_failure_record()])
    assert loaded.report_metrics()["agent_recovery_skill_count"] == 1
    assert loaded.report_metrics()["wm_intervention_spec_count"] == 1
