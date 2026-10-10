import json

from cce_mobility.skills.lifecycle import (
    gate_memory_with_paired_outcome_evidence,
    gate_memory_with_usage_evidence,
    select_lifecycle_memory,
)
from cce_mobility.skills.registry import SkillRegistry
from cce_mobility.skills.retrieval import render_selected_agent_skill_blocks
from cce_mobility.skills.schema import Skill, SkillStatus, SkillType


def test_select_lifecycle_memory_outputs_gated_registry(tmp_path):
    memory = tmp_path / "memory.jsonl"
    registry = SkillRegistry(
        [
            Skill(
                skill_id="agent::change_detection::selected_poi_unavailable::repair",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Repair closed POI",
                trigger_dimensions=["restaurant_closed"],
                capability_targets=["change_detection", "local_substitution"],
                domain_operators=["selected_poi_unavailable"],
                status=SkillStatus.ACTIVE,
                metrics={"evidence_count": 3},
            ),
            Skill(
                skill_id="agent::change_detection::selected_transport_unavailable::repair",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Repair cancelled transport",
                trigger_dimensions=["flight_cancelled"],
                capability_targets=["change_detection", "cascading_replan"],
                domain_operators=["selected_transport_unavailable"],
                status=SkillStatus.ACTIVE,
                metrics={"evidence_count": 3},
            ),
            Skill(
                skill_id="spec::change_detection::selected_poi_unavailable::high_value",
                skill_type=SkillType.INTERVENTION_SPEC,
                name="Close selected POI",
                trigger_dimensions=["restaurant_closed"],
                capability_targets=["change_detection"],
                domain_operators=["selected_poi_unavailable"],
                status=SkillStatus.ACTIVE,
                metrics={"evidence_count": 1, "label": "high_value_challenge"},
            ),
        ]
    )
    registry.save_jsonl(memory)

    baseline_records = tmp_path / "baseline_records.jsonl"
    target_records = tmp_path / "target_records.jsonl"
    _write_jsonl(
        baseline_records,
        [
            _record("tp_0", "int_0", "restaurant_closed", 1.0),
            _record("tp_1", "int_1", "flight_cancelled", 2.0),
        ],
    )
    _write_jsonl(
        target_records,
        [
            _record("tp_0", "int_0", "restaurant_closed", 1.5),
            _record("tp_1", "int_1", "flight_cancelled", 1.5),
        ],
    )

    matrix_config = tmp_path / "configs" / "matrix.json"
    matrix_config.parent.mkdir()
    matrix_config.write_text(
        json.dumps(
            {
                "runs": [
                    {"name": "no_evolution", "overrides": {"method": "no_evolution"}},
                    {
                        "name": "ours_checkpoint",
                        "overrides": {
                            "extra": {
                                "load_skill_memory_path": str(memory),
                            }
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    matrix_report = tmp_path / "matrix_report.json"
    matrix_report.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "name": "no_evolution",
                        "method": "no_evolution",
                        "metrics": {"counterfactual_recovery_avg": 1.5},
                        "records_path": str(baseline_records),
                    },
                    {
                        "name": "ours_checkpoint",
                        "method": "ours_skill_spec_coevolution",
                        "metrics": {"counterfactual_recovery_avg": 1.6},
                        "records_path": str(target_records),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "selected.jsonl"
    summary = select_lifecycle_memory(
        matrix_config_path=matrix_config,
        matrix_report_path=matrix_report,
        output_path=output,
    )
    selected = SkillRegistry.load_jsonl(output)
    by_id = {skill.skill_id: skill for skill in selected.all()}

    assert summary["selected_run"] == "ours_checkpoint"
    assert summary["selected_vs_baseline_delta"] == 0.1
    assert output.exists()
    assert output.with_suffix(".summary.json").exists()
    assert output.with_suffix(".summary.md").exists()
    assert (
        by_id["agent::change_detection::selected_poi_unavailable::repair"].status
        == SkillStatus.PROMOTED
    )
    assert (
        by_id["agent::change_detection::selected_transport_unavailable::repair"].status
        == SkillStatus.ACTIVE
    )
    assert (
        by_id["spec::change_detection::selected_poi_unavailable::high_value"].status
        == SkillStatus.PROMOTED
    )


def test_gate_memory_with_usage_evidence_adds_skill_level_lifecycle_metadata(tmp_path):
    memory = tmp_path / "memory.jsonl"
    registry = SkillRegistry(
        [
            Skill(
                skill_id="skill_positive",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Positive skill",
                trigger_dimensions=["flight_cancelled"],
                status=SkillStatus.ACTIVE,
            ),
            Skill(
                skill_id="skill_negative",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Negative skill",
                trigger_dimensions=["restaurant_closed"],
                status=SkillStatus.ACTIVE,
            ),
            Skill(
                skill_id="skill_mixed",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Mixed skill",
                trigger_dimensions=["restaurant_closed", "attraction_closed"],
                status=SkillStatus.ACTIVE,
            ),
            Skill(
                skill_id="spec_kept",
                skill_type=SkillType.INTERVENTION_SPEC,
                name="WM spec",
                status=SkillStatus.PROMOTED,
            ),
        ]
    )
    registry.save_jsonl(memory)
    usage_summary = tmp_path / "usage_summary.json"
    usage_summary.write_text(
        json.dumps(
            {
                "by_skill": [
                    {
                        "skill_id": "skill_positive",
                        "selected_count": 10,
                        "mean_delta": 0.2,
                        "helpful_rate": 0.8,
                        "harmful_rate": 0.1,
                        "low_score_rate_delta": -0.2,
                        "unavailable_usage_delta": -0.1,
                    },
                    {
                        "skill_id": "skill_negative",
                        "selected_count": 8,
                        "mean_delta": -0.2,
                        "helpful_rate": 0.1,
                        "harmful_rate": 0.8,
                        "low_score_rate_delta": 0.2,
                        "unavailable_usage_delta": 0.1,
                    },
                    {
                        "skill_id": "skill_mixed",
                        "selected_count": 12,
                        "mean_delta": 0.0,
                        "helpful_rate": 0.4,
                        "harmful_rate": 0.4,
                        "low_score_rate_delta": 0.0,
                        "unavailable_usage_delta": 0.0,
                    },
                ],
                "by_skill_dimension": [
                    {
                        "skill_id": "skill_mixed",
                        "dimension": "attraction_closed",
                        "selected_count": 6,
                        "mean_delta": 0.3,
                        "helpful_rate": 0.8,
                        "harmful_rate": 0.1,
                    },
                    {
                        "skill_id": "skill_mixed",
                        "dimension": "restaurant_closed",
                        "selected_count": 6,
                        "mean_delta": -0.3,
                        "helpful_rate": 0.1,
                        "harmful_rate": 0.8,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "usage_gated.jsonl"
    summary = gate_memory_with_usage_evidence(
        memory_path=memory,
        usage_summary_path=usage_summary,
        output_path=output,
        min_support=5,
    )
    selected = SkillRegistry.load_jsonl(output)
    by_id = {skill.skill_id: skill for skill in selected.all()}

    assert summary["status_counts"] == {"active": 1, "deprecated": 1, "promoted": 2}
    assert by_id["skill_positive"].status == SkillStatus.PROMOTED
    assert by_id["skill_negative"].status == SkillStatus.DEPRECATED
    assert by_id["skill_mixed"].status == SkillStatus.ACTIVE
    assert by_id["spec_kept"].status == SkillStatus.PROMOTED
    assert by_id["skill_positive"].metrics["usage_mean_delta"] == 0.2
    mixed_gate = by_id["skill_mixed"].metadata["usage_lifecycle_gate"]
    assert mixed_gate["positive_dimension_evidence"][0]["dimension"] == "attraction_closed"
    assert mixed_gate["negative_dimension_evidence"][0]["dimension"] == "restaurant_closed"
    assert output.with_suffix(".usage_gate.summary.json").exists()
    assert output.with_suffix(".usage_gate.summary.md").exists()


def test_usage_risk_gate_repairs_positive_skill_with_unavailable_risk(tmp_path):
    memory = tmp_path / "memory.jsonl"
    registry = SkillRegistry(
        [
            Skill(
                skill_id="skill_resource",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Recover unavailable selected resource",
                trigger_dimensions=["resource_unavailable"],
                capability_targets=["avoid_invalid_resource", "local_substitution"],
                domain_operators=["selected_resource_unavailable"],
                procedure=["Find a suitable replacement."],
                status=SkillStatus.ACTIVE,
            )
        ]
    )
    registry.save_jsonl(memory)
    usage_summary = tmp_path / "usage_summary.json"
    usage_summary.write_text(
        json.dumps(
            {
                "by_skill": [
                    {
                        "skill_id": "skill_resource",
                        "selected_count": 20,
                        "mean_delta": 0.12,
                        "helpful_rate": 0.4,
                        "harmful_rate": 0.2,
                        "low_score_rate_delta": -0.05,
                        "unavailable_usage_delta": 0.02,
                        "final_pass_delta": -0.02,
                    }
                ],
                "by_skill_dimension": [],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "usage_risk_gated.jsonl"
    gate_memory_with_usage_evidence(
        memory_path=memory,
        usage_summary_path=usage_summary,
        output_path=output,
        min_support=5,
        risk_aware=True,
        unavailable_usage_delta_repair_threshold=0.01,
    )
    selected = SkillRegistry.load_jsonl(output)
    skill = selected.all()[0]

    assert skill.status == SkillStatus.ACTIVE
    gate = skill.metadata["usage_lifecycle_gate"]
    assert gate["decision_reason"] == "active_agent_skill_usage_risk_repaired"
    assert gate["repair_action"] == "usage_risk_repair"
    assert skill.metrics["usage_lifecycle_score"] == 0.124
    prompt_text = "\n".join([*skill.procedure, *skill.success_criteria, *skill.failure_cases])
    assert "verify that it is available" in prompt_text
    for forbidden in ["flight", "hotel", "restaurant", "travelplanner"]:
        assert forbidden not in prompt_text.lower()


def test_usage_risk_gate_adds_observable_boundary_for_mixed_dimension_evidence(tmp_path):
    memory = tmp_path / "memory.jsonl"
    registry = SkillRegistry(
        [
            Skill(
                skill_id="skill_resource_boundary",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Recover changed selected resource",
                trigger_dimensions=["resource_state_changed"],
                capability_targets=["change_detection"],
                domain_operators=["selected_resource_unavailable"],
                procedure=["Check whether the selected resource changed."],
                status=SkillStatus.ACTIVE,
            )
        ]
    )
    registry.save_jsonl(memory)
    usage_summary = tmp_path / "usage_summary.json"
    usage_summary.write_text(
        json.dumps(
            {
                "by_skill": [
                    {
                        "skill_id": "skill_resource_boundary",
                        "selected_count": 30,
                        "mean_delta": 0.02,
                        "helpful_rate": 0.4,
                        "harmful_rate": 0.4,
                        "low_score_rate_delta": 0.0,
                        "unavailable_usage_delta": -0.01,
                        "final_pass_delta": 0.0,
                    }
                ],
                "by_skill_dimension": [
                    {
                        "skill_id": "skill_resource_boundary",
                        "dimension": "resource_unavailable",
                        "selected_count": 12,
                        "mean_delta": 1.0,
                        "helpful_rate": 0.8,
                        "harmful_rate": 0.0,
                        "low_score_rate_delta": -0.4,
                        "unavailable_usage_delta": -0.1,
                        "final_pass_delta": 0.1,
                    },
                    {
                        "skill_id": "skill_resource_boundary",
                        "dimension": "unrelated_environment_update",
                        "selected_count": 10,
                        "mean_delta": -0.5,
                        "helpful_rate": 0.1,
                        "harmful_rate": 0.6,
                        "low_score_rate_delta": 0.2,
                        "unavailable_usage_delta": 0.0,
                        "final_pass_delta": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "usage_risk_gated.jsonl"
    gate_memory_with_usage_evidence(
        memory_path=memory,
        usage_summary_path=usage_summary,
        output_path=output,
        min_support=5,
        risk_aware=True,
    )
    selected = SkillRegistry.load_jsonl(output)
    skill = selected.all()[0]

    assert skill.status == SkillStatus.ACTIVE
    gate = skill.metadata["usage_lifecycle_gate"]
    assert gate["decision_reason"] == "active_agent_skill_usage_risk_repaired"
    assert gate["negative_dimension_evidence"][0]["dimension"] == ("unrelated_environment_update")
    prompt_text = "\n".join([*skill.procedure, *skill.success_criteria, *skill.failure_cases])
    assert "only when the current observable state explicitly" in prompt_text
    assert "unrelated to the selected resource" in prompt_text
    for forbidden in ["flight", "hotel", "restaurant", "travelplanner"]:
        assert forbidden not in prompt_text.lower()


def test_usage_gate_preserves_deprecated_skill_without_usage_support(tmp_path):
    memory = tmp_path / "memory.jsonl"
    registry = SkillRegistry(
        [
            Skill(
                skill_id="skill_deprecated",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Deprecated recovery skill",
                trigger_dimensions=["resource_unavailable"],
                domain_operators=["selected_resource_unavailable"],
                procedure=["Old behavior."],
                status=SkillStatus.DEPRECATED,
            )
        ]
    )
    registry.save_jsonl(memory)
    usage_summary = tmp_path / "usage_summary.json"
    usage_summary.write_text(
        json.dumps({"by_skill": [], "by_skill_dimension": []}),
        encoding="utf-8",
    )

    output = tmp_path / "usage_gated.jsonl"
    gate_memory_with_usage_evidence(
        memory_path=memory,
        usage_summary_path=usage_summary,
        output_path=output,
        min_support=5,
        risk_aware=True,
    )
    selected = SkillRegistry.load_jsonl(output)
    skill = selected.all()[0]

    assert skill.status == SkillStatus.DEPRECATED
    assert skill.metadata["usage_lifecycle_gate"]["original_status"] == "deprecated"
    assert skill.metadata["usage_lifecycle_gate"]["decision_reason"] == (
        "kept_agent_skill_existing_status_insufficient_usage_support"
    )
    assert render_selected_agent_skill_blocks(selected, ["skill_deprecated"]) == ""


def test_gate_memory_with_paired_outcome_evidence_repairs_conservative_skills(tmp_path):
    memory = tmp_path / "memory.jsonl"
    registry = SkillRegistry(
        [
            Skill(
                skill_id="skill_tool",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Recover empty tool result",
                trigger_dimensions=["tool_empty_result"],
                capability_targets=["tool_robustness"],
                domain_operators=["tool_result_unavailable"],
                procedure=["Check whether the tool result is empty."],
                status=SkillStatus.ACTIVE,
            ),
            Skill(
                skill_id="skill_road",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Recover blocked road",
                trigger_dimensions=["road_blocked"],
                capability_targets=["cascading_replan"],
                domain_operators=["route_state_changed"],
                procedure=["Notice the road change."],
                status=SkillStatus.ACTIVE,
            ),
            Skill(
                skill_id="spec_kept",
                skill_type=SkillType.INTERVENTION_SPEC,
                name="WM spec",
                status=SkillStatus.PROMOTED,
            ),
        ]
    )
    registry.save_jsonl(memory)

    baseline_records = tmp_path / "baseline_records.jsonl"
    target_records = tmp_path / "target_records.jsonl"
    _write_jsonl(
        baseline_records,
        [
            _rich_record(
                "mb_0",
                "int_0",
                "tool_empty_result",
                "tool_result_unavailable",
                ["tool_robustness"],
                3.0,
                "partial_adapt",
                True,
            ),
            _rich_record(
                "mb_1",
                "int_1",
                "tool_empty_result",
                "tool_result_unavailable",
                ["tool_robustness"],
                3.0,
                "partial_adapt",
                True,
            ),
            _rich_record(
                "mb_2",
                "int_2",
                "road_blocked",
                "route_state_changed",
                ["cascading_replan"],
                3.2,
                "proactive_adapt",
                True,
            ),
            _rich_record(
                "mb_3",
                "int_3",
                "road_blocked",
                "route_state_changed",
                ["cascading_replan"],
                3.1,
                "proactive_adapt",
                True,
            ),
        ],
    )
    _write_jsonl(
        target_records,
        [
            _rich_record(
                "mb_0",
                "int_0",
                "tool_empty_result",
                "tool_result_unavailable",
                ["tool_robustness"],
                3.5,
                "proactive_adapt",
                True,
            ),
            _rich_record(
                "mb_1",
                "int_1",
                "tool_empty_result",
                "tool_result_unavailable",
                ["tool_robustness"],
                3.4,
                "proactive_adapt",
                True,
            ),
            _rich_record(
                "mb_2",
                "int_2",
                "road_blocked",
                "route_state_changed",
                ["cascading_replan"],
                2.6,
                "passive_notify",
                True,
            ),
            _rich_record(
                "mb_3",
                "int_3",
                "road_blocked",
                "route_state_changed",
                ["cascading_replan"],
                2.7,
                "passive_notify",
                True,
            ),
        ],
    )

    output = tmp_path / "paired_gated.jsonl"
    summary = gate_memory_with_paired_outcome_evidence(
        memory_path=memory,
        baseline_records_path=baseline_records,
        target_records_path=target_records,
        output_path=output,
        min_support=2,
        passive_notify_delta_threshold=0.5,
    )
    selected = SkillRegistry.load_jsonl(output)
    by_id = {skill.skill_id: skill for skill in selected.all()}

    assert summary["overall"]["support"] == 4
    assert by_id["skill_tool"].status == SkillStatus.PROMOTED
    assert by_id["skill_road"].status == SkillStatus.DEPRECATED
    assert by_id["spec_kept"].status == SkillStatus.PROMOTED
    assert by_id["skill_tool"].metrics["paired_mean_delta"] == 0.45
    assert any(
        "goes beyond passive notification" in item for item in by_id["skill_tool"].success_criteria
    )
    assert by_id["skill_road"].metadata["paired_lifecycle_gate"]["decision_reason"] == (
        "deprecated_agent_skill_negative_or_conservative_paired_outcome"
    )
    assert output.with_suffix(".paired_gate.summary.json").exists()
    assert output.with_suffix(".paired_gate.summary.md").exists()


def test_paired_outcome_deprecate_only_preserves_non_harmful_skills(tmp_path):
    memory = tmp_path / "memory.jsonl"
    registry = SkillRegistry(
        [
            Skill(
                skill_id="skill_tool",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Recover empty tool result",
                trigger_dimensions=["tool_empty_result"],
                capability_targets=["tool_robustness"],
                domain_operators=["tool_result_unavailable"],
                procedure=["Check whether the tool result is empty."],
                status=SkillStatus.ACTIVE,
            ),
            Skill(
                skill_id="skill_road",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Recover blocked road",
                trigger_dimensions=["road_blocked"],
                capability_targets=["cascading_replan"],
                domain_operators=["route_state_changed"],
                procedure=["Notice the road change."],
                status=SkillStatus.ACTIVE,
            ),
        ]
    )
    registry.save_jsonl(memory)

    baseline_records = tmp_path / "baseline_records.jsonl"
    target_records = tmp_path / "target_records.jsonl"
    _write_jsonl(
        baseline_records,
        [
            _rich_record(
                "mb_0",
                "int_0",
                "tool_empty_result",
                "tool_result_unavailable",
                ["tool_robustness"],
                3.0,
                "partial_adapt",
                True,
            ),
            _rich_record(
                "mb_1",
                "int_1",
                "tool_empty_result",
                "tool_result_unavailable",
                ["tool_robustness"],
                3.0,
                "partial_adapt",
                True,
            ),
            _rich_record(
                "mb_2",
                "int_2",
                "road_blocked",
                "route_state_changed",
                ["cascading_replan"],
                3.2,
                "proactive_adapt",
                True,
            ),
            _rich_record(
                "mb_3",
                "int_3",
                "road_blocked",
                "route_state_changed",
                ["cascading_replan"],
                3.1,
                "proactive_adapt",
                True,
            ),
        ],
    )
    _write_jsonl(
        target_records,
        [
            _rich_record(
                "mb_0",
                "int_0",
                "tool_empty_result",
                "tool_result_unavailable",
                ["tool_robustness"],
                3.5,
                "proactive_adapt",
                True,
            ),
            _rich_record(
                "mb_1",
                "int_1",
                "tool_empty_result",
                "tool_result_unavailable",
                ["tool_robustness"],
                3.4,
                "proactive_adapt",
                True,
            ),
            _rich_record(
                "mb_2",
                "int_2",
                "road_blocked",
                "route_state_changed",
                ["cascading_replan"],
                2.6,
                "passive_notify",
                True,
            ),
            _rich_record(
                "mb_3",
                "int_3",
                "road_blocked",
                "route_state_changed",
                ["cascading_replan"],
                2.7,
                "passive_notify",
                True,
            ),
        ],
    )

    output = tmp_path / "paired_deprecate_only.jsonl"
    summary = gate_memory_with_paired_outcome_evidence(
        memory_path=memory,
        baseline_records_path=baseline_records,
        target_records_path=target_records,
        output_path=output,
        min_support=2,
        passive_notify_delta_threshold=0.5,
        deprecate_only=True,
    )
    selected = SkillRegistry.load_jsonl(output)
    by_id = {skill.skill_id: skill for skill in selected.all()}

    assert summary["thresholds"]["deprecate_only"] is True
    assert by_id["skill_tool"].status == SkillStatus.ACTIVE
    assert by_id["skill_tool"].procedure == ["Check whether the tool result is empty."]
    assert by_id["skill_tool"].metadata["paired_lifecycle_gate"]["decision_reason"] == (
        "kept_agent_skill_deprecate_only_no_negative_paired_outcome"
    )
    assert by_id["skill_road"].status == SkillStatus.DEPRECATED


def test_verification_constrained_repairs_slight_unavailable_risk(tmp_path):
    memory = tmp_path / "memory.jsonl"
    registry = SkillRegistry(
        [
            Skill(
                skill_id="skill_resource",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Recover unavailable resource",
                trigger_dimensions=["resource_unavailable"],
                capability_targets=["avoid_invalid_resource", "local_substitution"],
                domain_operators=["selected_resource_unavailable"],
                procedure=["Find a suitable replacement."],
                status=SkillStatus.ACTIVE,
            )
        ]
    )
    registry.save_jsonl(memory)

    baseline_records = tmp_path / "baseline_records.jsonl"
    target_records = tmp_path / "target_records.jsonl"
    _write_jsonl(
        baseline_records,
        [
            _rich_record(
                f"mb_{idx}",
                f"int_{idx}",
                "resource_unavailable",
                "selected_resource_unavailable",
                ["avoid_invalid_resource"],
                2.0,
                "partial_adapt",
                True,
            )
            for idx in range(10)
        ],
    )
    _write_jsonl(
        target_records,
        [
            _rich_record(
                f"mb_{idx}",
                f"int_{idx}",
                "resource_unavailable",
                "selected_resource_unavailable",
                ["avoid_invalid_resource"],
                2.4,
                "proactive_adapt",
                True,
                uses_unavailable_identifier=(idx == 0),
                unavailable_id_count=1 if idx == 0 else 0,
            )
            for idx in range(10)
        ],
    )

    output = tmp_path / "paired_constrained.jsonl"
    summary = gate_memory_with_paired_outcome_evidence(
        memory_path=memory,
        baseline_records_path=baseline_records,
        target_records_path=target_records,
        output_path=output,
        min_support=5,
        repair_mode="verification_constrained",
    )
    selected = SkillRegistry.load_jsonl(output)
    skill = selected.all()[0]

    assert summary["repair_mode_counts"]["verification_repair"] == 1
    assert skill.status == SkillStatus.ACTIVE
    assert skill.metadata["paired_lifecycle_gate"]["repair_action"] == "verification_repair"
    assert skill.metrics["paired_invalid_resource_rate_delta"] == 0.1
    assert any("verify that it is available" in item for item in skill.procedure)
    prompt_text = "\n".join(
        [
            *skill.procedure,
            *skill.success_criteria,
            *skill.failure_cases,
        ]
    ).lower()
    for forbidden in ["flight", "hotel", "restaurant", "travelplanner"]:
        assert forbidden not in prompt_text


def test_verification_constrained_deprecates_severe_unavailable_risk(tmp_path):
    memory = tmp_path / "memory.jsonl"
    registry = SkillRegistry(
        [
            Skill(
                skill_id="skill_harmful",
                skill_type=SkillType.AGENT_RECOVERY,
                name="Harmful resource recovery",
                trigger_dimensions=["resource_unavailable"],
                capability_targets=["avoid_invalid_resource"],
                domain_operators=["selected_resource_unavailable"],
                status=SkillStatus.ACTIVE,
            )
        ]
    )
    registry.save_jsonl(memory)

    baseline_records = tmp_path / "baseline_records.jsonl"
    target_records = tmp_path / "target_records.jsonl"
    _write_jsonl(
        baseline_records,
        [
            _rich_record(
                f"mb_{idx}",
                f"int_{idx}",
                "resource_unavailable",
                "selected_resource_unavailable",
                ["avoid_invalid_resource"],
                2.5,
                "partial_adapt",
                True,
            )
            for idx in range(10)
        ],
    )
    _write_jsonl(
        target_records,
        [
            _rich_record(
                f"mb_{idx}",
                f"int_{idx}",
                "resource_unavailable",
                "selected_resource_unavailable",
                ["avoid_invalid_resource"],
                1.5,
                "partial_adapt",
                False,
                uses_unavailable_identifier=idx < 6,
                unavailable_id_count=1 if idx < 6 else 0,
            )
            for idx in range(10)
        ],
    )

    output = tmp_path / "paired_constrained_severe.jsonl"
    gate_memory_with_paired_outcome_evidence(
        memory_path=memory,
        baseline_records_path=baseline_records,
        target_records_path=target_records,
        output_path=output,
        min_support=5,
        repair_mode="verification_constrained",
    )
    selected = SkillRegistry.load_jsonl(output)
    skill = selected.all()[0]

    assert skill.status == SkillStatus.DEPRECATED
    assert skill.metadata["paired_lifecycle_gate"]["repair_action"] == "deprecate"
    assert render_selected_agent_skill_blocks(selected, ["skill_harmful"]) == ""


def _record(sample_id: str, intervention_id: str, dimension: str, score: float) -> dict:
    return {
        "sample": {"sample_id": sample_id},
        "intervention": {
            "intervention_id": intervention_id,
            "dimension": dimension,
        },
        "counterfactual_score": {"total": score},
    }


def _rich_record(
    sample_id: str,
    intervention_id: str,
    dimension: str,
    domain_operator: str,
    capability_targets: list[str],
    score: float,
    strategy: str,
    final_pass: bool,
    uses_unavailable_identifier: bool = False,
    unavailable_id_count: int = 0,
) -> dict:
    return {
        "sample": {"sample_id": sample_id},
        "intervention": {
            "intervention_id": intervention_id,
            "dimension": dimension,
            "domain_operator": domain_operator,
            "capability_targets": capability_targets,
        },
        "counterfactual": {"final_response": "response"},
        "counterfactual_score": {
            "total": score,
            "strategy": strategy,
            "rationale": "judge rationale",
        },
        "metadata": {
            "counterfactual_hard_metrics": {
                "final_pass": final_pass,
                "uses_unavailable_identifier": uses_unavailable_identifier,
                "unavailable_id_count": unavailable_id_count,
            }
        },
    }


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
