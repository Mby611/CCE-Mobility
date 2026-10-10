from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cce_mobility.skills.registry import SkillRegistry
from cce_mobility.skills.schema import Skill, SkillStatus, SkillType


DEFAULT_OBJECTIVE = "counterfactual_recovery_avg"
PAIRED_REPAIR_MODES = {"full", "deprecate_only", "verification_constrained"}


def select_lifecycle_memory(
    *,
    matrix_config_path: Path,
    matrix_report_path: Path,
    output_path: Path,
    baseline_name: str | None = None,
    objective: str = DEFAULT_OBJECTIVE,
    promote_delta_threshold: float = 0.02,
    deprecate_delta_threshold: float = -0.1,
) -> dict[str, Any]:
    """Select and gate a skill/spec memory from checkpoint-validation results."""

    matrix_config = _read_json(matrix_config_path)
    matrix_report = _read_json(matrix_report_path)
    run_to_memory = _run_memory_paths(matrix_config, matrix_config_path)
    runs = matrix_report.get("runs", [])
    baseline = _find_baseline_run(runs, baseline_name)
    best = _select_best_checkpoint(runs, run_to_memory, objective)
    memory_path = run_to_memory[best["name"]]

    baseline_records = _read_jsonl(Path(baseline["records_path"]))
    best_records = _read_jsonl(Path(best["records_path"]))
    dimension_deltas = _paired_dimension_deltas(baseline_records, best_records)

    registry = SkillRegistry.load_jsonl(memory_path)
    gated = SkillRegistry()
    decisions = []
    for skill in registry.all():
        updated, reason = _gate_skill(
            skill,
            dimension_deltas=dimension_deltas,
            selected_run=best["name"],
            selected_memory_path=memory_path,
            objective=objective,
            promote_delta_threshold=promote_delta_threshold,
            deprecate_delta_threshold=deprecate_delta_threshold,
        )
        gated.add_or_merge(updated)
        decisions.append(
            {
                "skill_id": updated.skill_id,
                "skill_type": updated.skill_type.value,
                "status": updated.status.value,
                "reason": reason,
                "trigger_dimensions": updated.trigger_dimensions,
                "capability_targets": updated.capability_targets,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gated.save_jsonl(output_path)
    summary = _summary(
        baseline=baseline,
        best=best,
        memory_path=memory_path,
        output_path=output_path,
        objective=objective,
        dimension_deltas=dimension_deltas,
        gated=gated,
        decisions=decisions,
    )
    _write_summary(output_path, summary)
    return summary


def gate_memory_with_usage_evidence(
    *,
    memory_path: Path,
    usage_summary_path: Path,
    output_path: Path,
    promote_delta_threshold: float = 0.05,
    deprecate_delta_threshold: float = -0.05,
    min_support: int = 5,
    risk_aware: bool = False,
    unavailable_usage_delta_repair_threshold: float = 0.01,
    unavailable_usage_delta_deprecate_threshold: float = 0.05,
    low_score_delta_deprecate_threshold: float = 0.08,
) -> dict[str, Any]:
    """Gate a registry using agent-owned retrieval usage evidence.

    Unlike checkpoint-level lifecycle selection, this uses selected-skill paired
    deltas. The evidence is attached to each skill, and dimensions with harmful
    evidence are recorded separately so later retrieval/lifecycle versions can
    use dimension-aware policies without runner-side gating.
    """

    registry = SkillRegistry.load_jsonl(memory_path)
    usage_summary = _read_json(usage_summary_path)
    skill_evidence = {
        row["skill_id"]: row for row in usage_summary.get("by_skill", []) if row.get("skill_id")
    }
    skill_dimension_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usage_summary.get("by_skill_dimension", []):
        skill_id = row.get("skill_id")
        if skill_id:
            skill_dimension_evidence[str(skill_id)].append(row)

    gated = SkillRegistry()
    decisions = []
    for skill in registry.all():
        updated, reason = _gate_skill_with_usage(
            skill,
            skill_evidence=skill_evidence.get(skill.skill_id),
            dimension_evidence=skill_dimension_evidence.get(skill.skill_id, []),
            usage_summary_path=usage_summary_path,
            promote_delta_threshold=promote_delta_threshold,
            deprecate_delta_threshold=deprecate_delta_threshold,
            min_support=min_support,
            risk_aware=risk_aware,
            unavailable_usage_delta_repair_threshold=(unavailable_usage_delta_repair_threshold),
            unavailable_usage_delta_deprecate_threshold=(
                unavailable_usage_delta_deprecate_threshold
            ),
            low_score_delta_deprecate_threshold=low_score_delta_deprecate_threshold,
        )
        gated.add_or_merge(updated)
        decisions.append(
            {
                "skill_id": updated.skill_id,
                "skill_type": updated.skill_type.value,
                "status": updated.status.value,
                "reason": reason,
                "trigger_dimensions": updated.trigger_dimensions,
                "domain_operators": updated.domain_operators,
                "usage_selected_count": updated.metrics.get("usage_selected_count"),
                "usage_mean_delta": updated.metrics.get("usage_mean_delta"),
                "usage_lifecycle_score": updated.metrics.get("usage_lifecycle_score"),
                "usage_repair_action": updated.metadata.get("usage_lifecycle_gate", {}).get(
                    "repair_action"
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gated.save_jsonl(output_path)
    summary = _usage_gate_summary(
        memory_path=memory_path,
        usage_summary_path=usage_summary_path,
        output_path=output_path,
        promote_delta_threshold=promote_delta_threshold,
        deprecate_delta_threshold=deprecate_delta_threshold,
        min_support=min_support,
        risk_aware=risk_aware,
        unavailable_usage_delta_repair_threshold=unavailable_usage_delta_repair_threshold,
        unavailable_usage_delta_deprecate_threshold=(unavailable_usage_delta_deprecate_threshold),
        low_score_delta_deprecate_threshold=low_score_delta_deprecate_threshold,
        gated=gated,
        decisions=decisions,
    )
    _write_usage_gate_summary(output_path, summary)
    return summary


def gate_memory_with_paired_outcome_evidence(
    *,
    memory_path: Path,
    baseline_records_path: Path,
    target_records_path: Path,
    output_path: Path,
    promote_delta_threshold: float = 0.05,
    deprecate_delta_threshold: float = -0.05,
    min_support: int = 5,
    passive_notify_delta_threshold: float = 0.15,
    low_score_delta_threshold: float = 0.05,
    deprecate_only: bool = False,
    repair_mode: str = "full",
    unavailable_usage_delta_repair_threshold: float = 0.01,
    unavailable_usage_delta_deprecate_threshold: float = 0.05,
    lifecycle_score_promote_threshold: float = 0.05,
    lifecycle_score_deprecate_threshold: float = -0.08,
) -> dict[str, Any]:
    """Gate and repair agent skills using strict paired evaluation evidence.

    This is designed for frozen-intervention comparisons where the baseline and
    target runs share identical sample/intervention ids. It treats conservative
    "only notify, no concrete recovery" behavior as an explicit lifecycle risk.
    """

    if deprecate_only:
        repair_mode = "deprecate_only"
    if repair_mode not in PAIRED_REPAIR_MODES:
        raise ValueError(
            f"Unsupported repair_mode: {repair_mode}. "
            f"Expected one of {sorted(PAIRED_REPAIR_MODES)}."
        )

    registry = SkillRegistry.load_jsonl(memory_path)
    baseline_records = _read_jsonl(baseline_records_path)
    target_records = _read_jsonl(target_records_path)
    paired_rows = _paired_outcome_rows(baseline_records, target_records)

    gated = SkillRegistry()
    decisions = []
    for skill in registry.all():
        updated, reason = _gate_skill_with_paired_outcomes(
            skill,
            paired_rows=paired_rows,
            baseline_records_path=baseline_records_path,
            target_records_path=target_records_path,
            promote_delta_threshold=promote_delta_threshold,
            deprecate_delta_threshold=deprecate_delta_threshold,
            min_support=min_support,
            passive_notify_delta_threshold=passive_notify_delta_threshold,
            low_score_delta_threshold=low_score_delta_threshold,
            deprecate_only=deprecate_only,
            repair_mode=repair_mode,
            unavailable_usage_delta_repair_threshold=unavailable_usage_delta_repair_threshold,
            unavailable_usage_delta_deprecate_threshold=unavailable_usage_delta_deprecate_threshold,
            lifecycle_score_promote_threshold=lifecycle_score_promote_threshold,
            lifecycle_score_deprecate_threshold=lifecycle_score_deprecate_threshold,
        )
        gated.add_or_merge(updated)
        decisions.append(
            {
                "skill_id": updated.skill_id,
                "skill_type": updated.skill_type.value,
                "status": updated.status.value,
                "reason": reason,
                "trigger_dimensions": updated.trigger_dimensions,
                "domain_operators": updated.domain_operators,
                "paired_support": updated.metrics.get("paired_support"),
                "paired_mean_delta": updated.metrics.get("paired_mean_delta"),
                "paired_passive_notify_rate_delta": updated.metrics.get(
                    "paired_passive_notify_rate_delta"
                ),
                "paired_invalid_resource_rate_delta": updated.metrics.get(
                    "paired_invalid_resource_rate_delta"
                ),
                "paired_lifecycle_score": updated.metrics.get("paired_lifecycle_score"),
                "paired_repair_action": updated.metadata.get("paired_lifecycle_gate", {}).get(
                    "repair_action"
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gated.save_jsonl(output_path)
    summary = _paired_gate_summary(
        memory_path=memory_path,
        baseline_records_path=baseline_records_path,
        target_records_path=target_records_path,
        output_path=output_path,
        promote_delta_threshold=promote_delta_threshold,
        deprecate_delta_threshold=deprecate_delta_threshold,
        min_support=min_support,
        passive_notify_delta_threshold=passive_notify_delta_threshold,
        low_score_delta_threshold=low_score_delta_threshold,
        deprecate_only=deprecate_only,
        repair_mode=repair_mode,
        unavailable_usage_delta_repair_threshold=unavailable_usage_delta_repair_threshold,
        unavailable_usage_delta_deprecate_threshold=unavailable_usage_delta_deprecate_threshold,
        lifecycle_score_promote_threshold=lifecycle_score_promote_threshold,
        lifecycle_score_deprecate_threshold=lifecycle_score_deprecate_threshold,
        paired_rows=paired_rows,
        gated=gated,
        decisions=decisions,
    )
    _write_paired_gate_summary(output_path, summary)
    return summary


def _gate_skill_with_paired_outcomes(
    skill: Skill,
    *,
    paired_rows: list[dict[str, Any]],
    baseline_records_path: Path,
    target_records_path: Path,
    promote_delta_threshold: float,
    deprecate_delta_threshold: float,
    min_support: int,
    passive_notify_delta_threshold: float,
    low_score_delta_threshold: float,
    deprecate_only: bool,
    repair_mode: str,
    unavailable_usage_delta_repair_threshold: float,
    unavailable_usage_delta_deprecate_threshold: float,
    lifecycle_score_promote_threshold: float,
    lifecycle_score_deprecate_threshold: float,
) -> tuple[Skill, str]:
    updated = skill.model_copy(deep=True)
    effective_repair_mode = "deprecate_only" if deprecate_only else repair_mode
    if updated.skill_type != SkillType.AGENT_RECOVERY:
        updated.metadata = {
            **updated.metadata,
            "paired_lifecycle_gate": {
                "baseline_records_path": str(baseline_records_path),
                "target_records_path": str(target_records_path),
                "repair_mode": effective_repair_mode,
                "repair_action": "none",
                "decision_reason": "kept_non_agent_skill_without_paired_gate",
            },
        }
        return updated, "kept_non_agent_skill_without_paired_gate"

    rows = [row for row in paired_rows if _skill_matches_paired_row(updated, row)]
    stats = _summarize_paired_outcome_rows(rows)
    negative_dimensions = [
        row
        for row in _dimension_stats(rows)
        if row["support"] >= min_support
        and (
            row["mean_delta"] <= deprecate_delta_threshold
            or row["passive_notify_rate_delta"] >= passive_notify_delta_threshold
            or row["low_score_rate_delta"] >= low_score_delta_threshold
            or row.get("invalid_resource_rate_delta", 0.0)
            >= unavailable_usage_delta_deprecate_threshold
        )
    ]
    positive_dimensions = [
        row
        for row in _dimension_stats(rows)
        if row["support"] >= min_support
        and (
            row["mean_delta"] >= promote_delta_threshold
            or row.get("lifecycle_score", 0.0) >= lifecycle_score_promote_threshold
        )
        and row.get("invalid_resource_rate_delta", 0.0)
        < unavailable_usage_delta_deprecate_threshold
    ]

    support = int(stats["support"])
    mean_delta = float(stats["mean_delta"])
    helpful_rate = float(stats["helpful_rate"])
    harmful_rate = float(stats["harmful_rate"])
    passive_notify_delta = float(stats["passive_notify_rate_delta"])
    low_score_delta = float(stats["low_score_rate_delta"])
    final_pass_delta = float(stats["final_pass_rate_delta"])
    invalid_resource_delta = float(stats.get("invalid_resource_rate_delta") or 0.0)
    unavailable_count_delta = float(stats.get("unavailable_id_count_delta") or 0.0)
    baseline_invalid_rate = float(stats.get("baseline_invalid_resource_rate") or 0.0)
    target_invalid_rate = float(stats.get("target_invalid_resource_rate") or 0.0)
    lifecycle_score = float(stats.get("lifecycle_score") or 0.0)
    invalid_usage_doubled = (
        baseline_invalid_rate > 0
        and target_invalid_rate >= 2 * baseline_invalid_rate
        and invalid_resource_delta >= unavailable_usage_delta_repair_threshold
    )
    severe_safety_risk = support >= min_support and (
        invalid_resource_delta >= unavailable_usage_delta_deprecate_threshold
        or invalid_usage_doubled
    )
    verification_risk = support >= min_support and (
        invalid_resource_delta >= unavailable_usage_delta_repair_threshold
        or unavailable_count_delta > 0
    )
    negative_outcome = mean_delta <= deprecate_delta_threshold and harmful_rate >= helpful_rate
    conservative_outcome = (
        passive_notify_delta >= passive_notify_delta_threshold
        and mean_delta <= 0
        and (harmful_rate >= 0.25 or low_score_delta >= low_score_delta_threshold)
    )

    reason = "active_agent_skill_insufficient_paired_support"
    repair_action = "none"
    if support >= min_support:
        if effective_repair_mode == "verification_constrained":
            should_deprecate = (
                severe_safety_risk and lifecycle_score <= lifecycle_score_deprecate_threshold
            ) or (
                (negative_outcome or conservative_outcome)
                and lifecycle_score <= lifecycle_score_deprecate_threshold
            )
        else:
            should_deprecate = negative_outcome or conservative_outcome or severe_safety_risk

        if effective_repair_mode == "deprecate_only" and should_deprecate:
            updated.status = SkillStatus.DEPRECATED
            reason = "deprecated_agent_skill_negative_or_conservative_paired_outcome"
            repair_action = "deprecate"
        elif effective_repair_mode == "deprecate_only":
            reason = "kept_agent_skill_deprecate_only_no_negative_paired_outcome"
            repair_action = "keep"
        elif effective_repair_mode == "verification_constrained":
            if should_deprecate:
                updated.status = SkillStatus.DEPRECATED
                reason = "deprecated_agent_skill_severe_paired_risk"
                repair_action = "deprecate"
            elif verification_risk and lifecycle_score > lifecycle_score_deprecate_threshold:
                updated.status = SkillStatus.ACTIVE
                reason = "verification_repaired_agent_skill_invalid_resource_risk"
                repair_action = "verification_repair"
            elif (
                lifecycle_score >= lifecycle_score_promote_threshold
                and helpful_rate >= harmful_rate
                and not severe_safety_risk
            ):
                updated.status = SkillStatus.PROMOTED
                reason = "promoted_agent_skill_positive_lifecycle_score"
                repair_action = "promote"
            elif positive_dimensions and negative_dimensions:
                updated.status = SkillStatus.ACTIVE
                reason = "active_agent_skill_mixed_paired_evidence"
                repair_action = "keep"
            elif negative_dimensions:
                updated.status = SkillStatus.ACTIVE
                reason = "active_agent_skill_negative_evidence_soft_kept"
                repair_action = "keep"
            else:
                updated.status = SkillStatus.ACTIVE
                reason = "active_agent_skill_neutral_paired_outcome"
                repair_action = "keep"
        elif (
            mean_delta >= promote_delta_threshold
            and helpful_rate > harmful_rate
            and passive_notify_delta <= 0
            and final_pass_delta >= 0
        ):
            updated.status = SkillStatus.PROMOTED
            reason = "promoted_agent_skill_positive_paired_outcome"
            repair_action = "promote"
        elif should_deprecate:
            updated.status = SkillStatus.DEPRECATED
            reason = "deprecated_agent_skill_negative_or_conservative_paired_outcome"
            repair_action = "deprecate"
        elif positive_dimensions and negative_dimensions:
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_mixed_paired_dimension_outcomes"
            repair_action = "keep"
        elif negative_dimensions:
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_has_negative_paired_dimensions"
            repair_action = "keep"
        elif positive_dimensions:
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_positive_dimension_but_global_not_promoted"
            repair_action = "keep"
        else:
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_neutral_paired_outcome"
            repair_action = "keep"
    else:
        updated.status = SkillStatus.ACTIVE
        repair_action = "keep"

    if (
        updated.status != SkillStatus.DEPRECATED
        and effective_repair_mode == "verification_constrained"
        and repair_action == "verification_repair"
    ):
        _reinforce_verification_constrained_contract(updated)
    elif updated.status != SkillStatus.DEPRECATED and effective_repair_mode == "full":
        _reinforce_complete_recovery_contract(updated)

    updated.metrics = {
        **updated.metrics,
        "paired_support": support,
        "paired_mean_delta": round(mean_delta, 4),
        "paired_helpful_rate": round(helpful_rate, 4),
        "paired_harmful_rate": round(harmful_rate, 4),
        "paired_low_score_rate_delta": round(low_score_delta, 4),
        "paired_passive_notify_rate_delta": round(passive_notify_delta, 4),
        "paired_final_pass_rate_delta": round(final_pass_delta, 4),
        "paired_invalid_resource_rate_delta": round(invalid_resource_delta, 4),
        "paired_unavailable_id_count_delta": round(unavailable_count_delta, 4),
        "paired_lifecycle_score": round(lifecycle_score, 4),
    }
    updated.metadata = {
        **updated.metadata,
        "paired_lifecycle_gate": {
            "baseline_records_path": str(baseline_records_path),
            "target_records_path": str(target_records_path),
            "paired_stats": stats,
            "positive_dimension_evidence": positive_dimensions,
            "negative_dimension_evidence": negative_dimensions,
            "repair_mode": effective_repair_mode,
            "repair_action": repair_action,
            "severe_safety_risk": severe_safety_risk,
            "verification_risk": verification_risk,
            "thresholds": {
                "promote_delta_threshold": promote_delta_threshold,
                "deprecate_delta_threshold": deprecate_delta_threshold,
                "min_support": min_support,
                "passive_notify_delta_threshold": passive_notify_delta_threshold,
                "low_score_delta_threshold": low_score_delta_threshold,
                "deprecate_only": deprecate_only,
                "repair_mode": effective_repair_mode,
                "unavailable_usage_delta_repair_threshold": (
                    unavailable_usage_delta_repair_threshold
                ),
                "unavailable_usage_delta_deprecate_threshold": (
                    unavailable_usage_delta_deprecate_threshold
                ),
                "lifecycle_score_promote_threshold": lifecycle_score_promote_threshold,
                "lifecycle_score_deprecate_threshold": lifecycle_score_deprecate_threshold,
            },
            "decision_reason": reason,
        },
    }
    return updated, reason


def _reinforce_complete_recovery_contract(skill: Skill) -> None:
    skill.procedure = _merge_unique(
        [
            *skill.procedure,
            "After detecting the current environment change, diagnose which route, POI, tool result, or user constraint is affected.",
            "Give at least one concrete feasible next action: a replacement POI, route/workaround, alternate query/tool use, or a specific clarification question when required data is missing.",
            "Verify the replacement against the user's original goal and current constraints before finalizing.",
        ]
    )
    dimension_operator_text = " ".join([*skill.trigger_dimensions, *skill.domain_operators])
    if (
        "poi_closed" in dimension_operator_text
        or "selected_poi_unavailable" in dimension_operator_text
    ):
        skill.procedure = _merge_unique(
            [
                *skill.procedure,
                "For an unavailable POI, stop recommending that POI as usable and suggest a nearby or same-intent alternative when possible.",
            ]
        )
    if (
        "road_blocked" in dimension_operator_text
        or "route_state_changed" in dimension_operator_text
    ):
        skill.procedure = _merge_unique(
            [
                *skill.procedure,
                "For a blocked road or route-state change, replan around the affected segment and communicate the impact on route choice or ETA.",
            ]
        )
    skill.success_criteria = _merge_unique(
        [
            *skill.success_criteria,
            "The response goes beyond passive notification and contains an actionable recovery path or an explicit information request.",
        ]
    )
    skill.failure_cases = _merge_unique(
        [
            *skill.failure_cases,
            "Do not stop after merely saying the environment changed; complete the recovery step.",
        ]
    )


def _reinforce_verification_constrained_contract(skill: Skill) -> None:
    skill.procedure = _merge_unique(
        [
            *skill.procedure,
            "Before recommending a replacement resource, verify that it is available in the current observation or tool result.",
            "If availability is not verified, request updated information or ask a targeted clarification instead of inventing a replacement.",
            "Distinguish confirmed alternatives from unverified possibilities.",
        ]
    )
    skill.success_criteria = _merge_unique(
        [
            *skill.success_criteria,
            "The response uses only verified available resources as confirmed recommendations.",
        ]
    )
    skill.failure_cases = _merge_unique(
        [
            *skill.failure_cases,
            "Do not present an unverified resource as a confirmed replacement.",
        ]
    )


def _reinforce_observable_trigger_boundary(skill: Skill) -> None:
    skill.procedure = _merge_unique(
        [
            *skill.procedure,
            "Use this recovery skill only when the current observable state explicitly says a previously selected resource is unavailable, invalid, changed, or missing.",
            "If the new observation is unrelated to the selected resource, keep the original plan and state that no recovery action is needed.",
            "Do not reinterpret unrelated updates as evidence that a selected resource failed.",
        ]
    )
    skill.success_criteria = _merge_unique(
        [
            *skill.success_criteria,
            "The response grounds skill use in an explicit observable trigger from the current task state.",
        ]
    )
    skill.failure_cases = _merge_unique(
        [
            *skill.failure_cases,
            "Do not use this skill when the current observation does not invalidate or change a selected resource.",
        ]
    )


def _gate_skill_with_usage(
    skill: Skill,
    *,
    skill_evidence: dict[str, Any] | None,
    dimension_evidence: list[dict[str, Any]],
    usage_summary_path: Path,
    promote_delta_threshold: float,
    deprecate_delta_threshold: float,
    min_support: int,
    risk_aware: bool,
    unavailable_usage_delta_repair_threshold: float,
    unavailable_usage_delta_deprecate_threshold: float,
    low_score_delta_deprecate_threshold: float,
) -> tuple[Skill, str]:
    updated = skill.model_copy(deep=True)
    if updated.skill_type != SkillType.AGENT_RECOVERY:
        updated.metadata = {
            **updated.metadata,
            "usage_lifecycle_gate": {
                "usage_summary_path": str(usage_summary_path),
                "decision_reason": "kept_non_agent_skill_without_usage_gate",
            },
        }
        return updated, "kept_non_agent_skill_without_usage_gate"

    support = int((skill_evidence or {}).get("selected_count") or 0)
    mean_delta = float((skill_evidence or {}).get("mean_delta") or 0.0)
    helpful_rate = float((skill_evidence or {}).get("helpful_rate") or 0.0)
    harmful_rate = float((skill_evidence or {}).get("harmful_rate") or 0.0)
    low_score_delta = float((skill_evidence or {}).get("low_score_rate_delta") or 0.0)
    unavailable_delta = float((skill_evidence or {}).get("unavailable_usage_delta") or 0.0)
    final_pass_delta = float((skill_evidence or {}).get("final_pass_delta") or 0.0)
    lifecycle_score = (
        mean_delta - 0.4 * low_score_delta - 0.6 * unavailable_delta + 0.2 * final_pass_delta
    )

    positive_dimensions = []
    negative_dimensions = []
    for row in dimension_evidence:
        dimension = str(row.get("dimension") or "unknown")
        row_support = int(row.get("selected_count") or 0)
        row_delta = float(row.get("mean_delta") or 0.0)
        row_harmful = float(row.get("harmful_rate") or 0.0)
        row_low_score_delta = float(row.get("low_score_rate_delta") or 0.0)
        row_unavailable_delta = float(row.get("unavailable_usage_delta") or 0.0)
        row_evidence = {
            "dimension": dimension,
            "selected_count": row_support,
            "mean_delta": row_delta,
            "helpful_rate": row.get("helpful_rate"),
            "harmful_rate": row.get("harmful_rate"),
            "low_score_rate_delta": row.get("low_score_rate_delta"),
            "unavailable_usage_delta": row.get("unavailable_usage_delta"),
            "final_pass_delta": row.get("final_pass_delta"),
        }
        if row_support >= min_support and row_delta >= promote_delta_threshold:
            positive_dimensions.append(row_evidence)
        negative_dimension = (
            row_support >= min_support
            and row_delta <= deprecate_delta_threshold
            and row_harmful >= 0.5
        )
        if risk_aware and row_support >= min_support:
            negative_dimension = negative_dimension or (
                row_low_score_delta >= low_score_delta_deprecate_threshold
                or row_unavailable_delta >= unavailable_usage_delta_deprecate_threshold
            )
        if negative_dimension:
            negative_dimensions.append(row_evidence)

    original_status = updated.status
    reason = "kept_agent_skill_existing_status_insufficient_usage_support"
    repair_action = "none"
    if support >= min_support:
        verification_risk = unavailable_delta >= unavailable_usage_delta_repair_threshold
        severe_unavailable_risk = (
            unavailable_delta >= unavailable_usage_delta_deprecate_threshold
            and harmful_rate >= helpful_rate
        )
        severe_low_score_risk = (
            low_score_delta >= low_score_delta_deprecate_threshold
            and harmful_rate >= helpful_rate
            and mean_delta <= promote_delta_threshold
        )
        negative_usage_outcome = mean_delta <= deprecate_delta_threshold and harmful_rate >= 0.5
        if risk_aware and (
            severe_unavailable_risk or severe_low_score_risk or negative_usage_outcome
        ):
            updated.status = SkillStatus.DEPRECATED
            reason = "deprecated_agent_skill_usage_risk"
            repair_action = "deprecate"
        elif risk_aware and (verification_risk or negative_dimensions):
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_usage_risk_repaired"
            repair_action = "usage_risk_repair"
        elif mean_delta >= promote_delta_threshold and helpful_rate > harmful_rate:
            updated.status = SkillStatus.PROMOTED
            reason = "promoted_agent_skill_positive_usage_delta"
            repair_action = "promote"
        elif mean_delta <= deprecate_delta_threshold and harmful_rate >= 0.5:
            updated.status = SkillStatus.DEPRECATED
            reason = "deprecated_agent_skill_negative_usage_delta"
            repair_action = "deprecate"
        elif positive_dimensions and negative_dimensions:
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_mixed_dimension_usage_evidence"
            repair_action = "keep"
        elif positive_dimensions:
            if mean_delta > 0 and helpful_rate >= harmful_rate:
                updated.status = SkillStatus.PROMOTED
                reason = "promoted_agent_skill_positive_dimension_usage_evidence"
                repair_action = "promote"
            else:
                updated.status = SkillStatus.ACTIVE
                reason = "active_agent_skill_positive_dimension_but_global_nonpositive"
                repair_action = "keep"
        elif negative_dimensions:
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_negative_dimension_usage_evidence"
            repair_action = "keep"
        else:
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_neutral_usage_evidence"
            repair_action = "keep"
    else:
        updated.status = original_status
        repair_action = "keep"

    if risk_aware and updated.status != SkillStatus.DEPRECATED and support >= min_support:
        if unavailable_delta >= unavailable_usage_delta_repair_threshold:
            _reinforce_verification_constrained_contract(updated)
        if negative_dimensions:
            _reinforce_observable_trigger_boundary(updated)

    updated.metrics = {
        **updated.metrics,
        "usage_selected_count": support,
        "usage_mean_delta": round(mean_delta, 4),
        "usage_helpful_rate": round(helpful_rate, 4),
        "usage_harmful_rate": round(harmful_rate, 4),
        "usage_low_score_rate_delta": round(low_score_delta, 4),
        "usage_unavailable_usage_delta": round(unavailable_delta, 4),
        "usage_final_pass_delta": round(final_pass_delta, 4),
        "usage_lifecycle_score": round(lifecycle_score, 4),
    }
    updated.metadata = {
        **updated.metadata,
        "usage_lifecycle_gate": {
            "usage_summary_path": str(usage_summary_path),
            "skill_evidence": skill_evidence or {},
            "positive_dimension_evidence": positive_dimensions,
            "negative_dimension_evidence": negative_dimensions,
            "all_dimension_evidence": dimension_evidence,
            "repair_action": repair_action,
            "risk_aware": risk_aware,
            "original_status": original_status.value,
            "thresholds": {
                "promote_delta_threshold": promote_delta_threshold,
                "deprecate_delta_threshold": deprecate_delta_threshold,
                "min_support": min_support,
                "risk_aware": risk_aware,
                "unavailable_usage_delta_repair_threshold": (
                    unavailable_usage_delta_repair_threshold
                ),
                "unavailable_usage_delta_deprecate_threshold": (
                    unavailable_usage_delta_deprecate_threshold
                ),
                "low_score_delta_deprecate_threshold": low_score_delta_deprecate_threshold,
            },
            "decision_reason": reason,
        },
    }
    return updated, reason


def _usage_gate_summary(
    *,
    memory_path: Path,
    usage_summary_path: Path,
    output_path: Path,
    promote_delta_threshold: float,
    deprecate_delta_threshold: float,
    min_support: int,
    risk_aware: bool,
    unavailable_usage_delta_repair_threshold: float,
    unavailable_usage_delta_deprecate_threshold: float,
    low_score_delta_deprecate_threshold: float,
    gated: SkillRegistry,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(skill.status.value for skill in gated.all())
    type_status_counts = Counter(
        f"{skill.skill_type.value}:{skill.status.value}" for skill in gated.all()
    )
    repair_action_counts = Counter(
        str(decision.get("usage_repair_action") or "none") for decision in decisions
    )
    return {
        "input_memory_path": str(memory_path),
        "usage_summary_path": str(usage_summary_path),
        "output_path": str(output_path),
        "thresholds": {
            "promote_delta_threshold": promote_delta_threshold,
            "deprecate_delta_threshold": deprecate_delta_threshold,
            "min_support": min_support,
            "risk_aware": risk_aware,
            "unavailable_usage_delta_repair_threshold": (unavailable_usage_delta_repair_threshold),
            "unavailable_usage_delta_deprecate_threshold": (
                unavailable_usage_delta_deprecate_threshold
            ),
            "low_score_delta_deprecate_threshold": low_score_delta_deprecate_threshold,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "type_status_counts": dict(sorted(type_status_counts.items())),
        "repair_action_counts": dict(sorted(repair_action_counts.items())),
        "decisions": decisions,
    }


def _write_usage_gate_summary(output_path: Path, summary: dict[str, Any]) -> None:
    output_path.with_suffix(".usage_gate.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Usage-Gated Lifecycle Summary",
        "",
        f"- input_memory_path: `{summary['input_memory_path']}`",
        f"- usage_summary_path: `{summary['usage_summary_path']}`",
        f"- output_path: `{summary['output_path']}`",
        f"- thresholds: `{summary['thresholds']}`",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "| --- | --- |",
    ]
    for status, count in summary["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Repair Action Counts",
            "",
            "| repair_action | count |",
            "| --- | --- |",
        ]
    )
    for action, count in summary.get("repair_action_counts", {}).items():
        lines.append(f"| {action} | {count} |")
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| skill_id | type | status | repair_action | selected_count | mean_delta | lifecycle_score | reason |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for decision in summary["decisions"]:
        lines.append(
            f"| {decision['skill_id']} | {decision['skill_type']} | "
            f"{decision['status']} | {decision.get('usage_repair_action', '')} | "
            f"{decision.get('usage_selected_count', '')} | "
            f"{decision.get('usage_mean_delta', '')} | "
            f"{decision.get('usage_lifecycle_score', '')} | {decision['reason']} |"
        )
    output_path.with_suffix(".usage_gate.summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _paired_outcome_rows(
    baseline_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = {}
    for record in baseline_records:
        key = _record_key(record)
        if key:
            baseline[key] = record

    rows = []
    for target_record in target_records:
        key = _record_key(target_record)
        if not key or key not in baseline:
            continue
        baseline_record = baseline[key]
        left_score = _score(baseline_record)
        right_score = _score(target_record)
        if left_score is None or right_score is None:
            continue
        intervention = target_record.get("intervention") or {}
        capabilities = [str(item) for item in intervention.get("capability_targets") or []]
        left_strategy = _strategy(baseline_record)
        right_strategy = _strategy(target_record)
        left_resource = _resource_validity(baseline_record)
        right_resource = _resource_validity(target_record)
        rows.append(
            {
                "sample_id": key[0],
                "intervention_id": key[1],
                "dimension": str(intervention.get("dimension") or "unknown"),
                "domain_operator": str(intervention.get("domain_operator") or ""),
                "capabilities": capabilities,
                "baseline_score": left_score,
                "target_score": right_score,
                "delta": round(right_score - left_score, 4),
                "baseline_strategy": left_strategy,
                "target_strategy": right_strategy,
                "baseline_low_score": left_score < 2.0,
                "target_low_score": right_score < 2.0,
                "baseline_final_pass": _final_pass(baseline_record),
                "target_final_pass": _final_pass(target_record),
                "baseline_passive_notify": left_strategy == "passive_notify",
                "target_passive_notify": right_strategy == "passive_notify",
                "baseline_uses_invalid_resource": left_resource["uses_invalid_resource"],
                "target_uses_invalid_resource": right_resource["uses_invalid_resource"],
                "baseline_unavailable_id_count": left_resource["unavailable_id_count"],
                "target_unavailable_id_count": right_resource["unavailable_id_count"],
                "baseline_missing_resource_metric": left_resource["missing_resource_metric"],
                "target_missing_resource_metric": right_resource["missing_resource_metric"],
            }
        )
    for row in rows:
        row["low_score_delta"] = int(row["target_low_score"]) - int(row["baseline_low_score"])
        row["final_pass_delta"] = int(row["target_final_pass"]) - int(row["baseline_final_pass"])
        row["passive_notify_delta"] = int(row["target_passive_notify"]) - int(
            row["baseline_passive_notify"]
        )
        row["invalid_resource_delta"] = int(row["target_uses_invalid_resource"]) - int(
            row["baseline_uses_invalid_resource"]
        )
        row["unavailable_id_count_delta"] = int(row["target_unavailable_id_count"]) - int(
            row["baseline_unavailable_id_count"]
        )
    return rows


def _summarize_paired_outcome_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "support": 0,
            "mean_delta": 0.0,
            "improved": 0,
            "worse": 0,
            "same": 0,
            "helpful_rate": 0.0,
            "harmful_rate": 0.0,
            "low_score_rate_delta": 0.0,
            "passive_notify_rate_delta": 0.0,
            "final_pass_rate_delta": 0.0,
            "baseline_invalid_resource_rate": 0.0,
            "target_invalid_resource_rate": 0.0,
            "invalid_resource_rate_delta": 0.0,
            "unavailable_id_count_delta": 0.0,
            "missing_resource_validity_metric_count": 0,
            "lifecycle_score": 0.0,
        }
    deltas = [float(row["delta"]) for row in rows]
    low_score_delta = _avg_list([int(row["low_score_delta"]) for row in rows])
    final_pass_delta = _avg_list([int(row["final_pass_delta"]) for row in rows])
    invalid_delta = _avg_list([int(row["invalid_resource_delta"]) for row in rows])
    lifecycle_score = (
        _avg_list(deltas) - 0.4 * low_score_delta - 0.6 * invalid_delta + 0.2 * final_pass_delta
    )
    return {
        "support": len(rows),
        "mean_delta": _avg_list(deltas),
        "improved": sum(1 for delta in deltas if delta > 0),
        "worse": sum(1 for delta in deltas if delta < 0),
        "same": sum(1 for delta in deltas if delta == 0),
        "helpful_rate": _rate(delta > 0 for delta in deltas),
        "harmful_rate": _rate(delta < 0 for delta in deltas),
        "low_score_rate_delta": low_score_delta,
        "passive_notify_rate_delta": _avg_list([int(row["passive_notify_delta"]) for row in rows]),
        "final_pass_rate_delta": final_pass_delta,
        "baseline_invalid_resource_rate": _rate(
            bool(row["baseline_uses_invalid_resource"]) for row in rows
        ),
        "target_invalid_resource_rate": _rate(
            bool(row["target_uses_invalid_resource"]) for row in rows
        ),
        "invalid_resource_rate_delta": invalid_delta,
        "unavailable_id_count_delta": _avg_list(
            [int(row["unavailable_id_count_delta"]) for row in rows]
        ),
        "missing_resource_validity_metric_count": sum(
            1
            for row in rows
            if row["baseline_missing_resource_metric"] or row["target_missing_resource_metric"]
        ),
        "lifecycle_score": round(lifecycle_score, 4),
    }


def _dimension_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dimension") or "unknown")].append(row)
    return [
        {"dimension": dimension, **_summarize_paired_outcome_rows(group_rows)}
        for dimension, group_rows in sorted(grouped.items())
    ]


def _operator_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("domain_operator") or "unknown")].append(row)
    return [
        {"domain_operator": operator, **_summarize_paired_outcome_rows(group_rows)}
        for operator, group_rows in sorted(grouped.items())
    ]


def _skill_matches_paired_row(skill: Skill, row: dict[str, Any]) -> bool:
    trigger_dimensions = set(skill.trigger_dimensions)
    domain_operators = set(skill.domain_operators)
    capability_targets = set(skill.capability_targets)
    if not trigger_dimensions and not domain_operators and not capability_targets:
        return True
    if row["dimension"] in trigger_dimensions:
        return True
    if row["domain_operator"] and row["domain_operator"] in domain_operators:
        return True
    if capability_targets and capability_targets.intersection(set(row.get("capabilities") or [])):
        return True
    return False


def _paired_gate_summary(
    *,
    memory_path: Path,
    baseline_records_path: Path,
    target_records_path: Path,
    output_path: Path,
    promote_delta_threshold: float,
    deprecate_delta_threshold: float,
    min_support: int,
    passive_notify_delta_threshold: float,
    low_score_delta_threshold: float,
    deprecate_only: bool,
    repair_mode: str,
    unavailable_usage_delta_repair_threshold: float,
    unavailable_usage_delta_deprecate_threshold: float,
    lifecycle_score_promote_threshold: float,
    lifecycle_score_deprecate_threshold: float,
    paired_rows: list[dict[str, Any]],
    gated: SkillRegistry,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(skill.status.value for skill in gated.all())
    type_status_counts = Counter(
        f"{skill.skill_type.value}:{skill.status.value}" for skill in gated.all()
    )
    repair_mode_counts = Counter(
        str(row.get("paired_repair_action") or "none") for row in decisions
    )
    return {
        "input_memory_path": str(memory_path),
        "baseline_records_path": str(baseline_records_path),
        "target_records_path": str(target_records_path),
        "output_path": str(output_path),
        "thresholds": {
            "promote_delta_threshold": promote_delta_threshold,
            "deprecate_delta_threshold": deprecate_delta_threshold,
            "min_support": min_support,
            "passive_notify_delta_threshold": passive_notify_delta_threshold,
            "low_score_delta_threshold": low_score_delta_threshold,
            "deprecate_only": deprecate_only,
            "repair_mode": repair_mode,
            "unavailable_usage_delta_repair_threshold": (unavailable_usage_delta_repair_threshold),
            "unavailable_usage_delta_deprecate_threshold": (
                unavailable_usage_delta_deprecate_threshold
            ),
            "lifecycle_score_promote_threshold": lifecycle_score_promote_threshold,
            "lifecycle_score_deprecate_threshold": lifecycle_score_deprecate_threshold,
        },
        "overall": _summarize_paired_outcome_rows(paired_rows),
        "by_dimension": _dimension_stats(paired_rows),
        "by_domain_operator": _operator_stats(paired_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "type_status_counts": dict(sorted(type_status_counts.items())),
        "repair_mode_counts": dict(sorted(repair_mode_counts.items())),
        "decisions": decisions,
    }


def _write_paired_gate_summary(output_path: Path, summary: dict[str, Any]) -> None:
    output_path.with_suffix(".paired_gate.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Paired-Evidence Lifecycle Summary",
        "",
        f"- input_memory_path: `{summary['input_memory_path']}`",
        f"- baseline_records_path: `{summary['baseline_records_path']}`",
        f"- target_records_path: `{summary['target_records_path']}`",
        f"- output_path: `{summary['output_path']}`",
        f"- thresholds: `{summary['thresholds']}`",
        "",
        "## Overall Paired Outcome",
        "",
        "| support | mean_delta | lifecycle_score | improved | worse | same | low_score_delta | invalid_resource_delta | final_pass_delta |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {summary['overall']['support']} | {summary['overall']['mean_delta']} | "
            f"{summary['overall']['lifecycle_score']} | "
            f"{summary['overall']['improved']} | {summary['overall']['worse']} | "
            f"{summary['overall']['same']} | "
            f"{summary['overall']['low_score_rate_delta']} | "
            f"{summary['overall']['invalid_resource_rate_delta']} | "
            f"{summary['overall']['final_pass_rate_delta']} |"
        ),
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "| --- | --- |",
    ]
    for status, count in summary["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Repair Actions",
            "",
            "| repair_action | count |",
            "| --- | --- |",
        ]
    )
    for action, count in summary.get("repair_mode_counts", {}).items():
        lines.append(f"| {action} | {count} |")
    lines.extend(
        [
            "",
            "## By Dimension",
            "",
            "| dimension | support | mean_delta | lifecycle_score | invalid_resource_delta | improved | worse | passive_notify_delta |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["by_dimension"]:
        lines.append(
            f"| {row['dimension']} | {row['support']} | {row['mean_delta']} | "
            f"{row['lifecycle_score']} | {row['invalid_resource_rate_delta']} | "
            f"{row['improved']} | {row['worse']} | {row['passive_notify_rate_delta']} |"
        )
    lines.extend(
        [
            "",
            "## By Domain Operator",
            "",
            "| domain_operator | support | mean_delta | lifecycle_score | invalid_resource_delta | improved | worse |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary.get("by_domain_operator", []):
        lines.append(
            f"| {row['domain_operator']} | {row['support']} | {row['mean_delta']} | "
            f"{row['lifecycle_score']} | {row['invalid_resource_rate_delta']} | "
            f"{row['improved']} | {row['worse']} |"
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| skill_id | type | status | repair_action | support | mean_delta | lifecycle_score | invalid_resource_delta | reason |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for decision in summary["decisions"]:
        lines.append(
            f"| {decision['skill_id']} | {decision['skill_type']} | "
            f"{decision['status']} | {decision.get('paired_repair_action', '')} | "
            f"{decision.get('paired_support', '')} | "
            f"{decision.get('paired_mean_delta', '')} | "
            f"{decision.get('paired_lifecycle_score', '')} | "
            f"{decision.get('paired_invalid_resource_rate_delta', '')} | "
            f"{decision['reason']} |"
        )
    output_path.with_suffix(".paired_gate.summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _resource_validity(record: dict[str, Any]) -> dict[str, Any]:
    hard = record.get("metadata", {}).get("counterfactual_hard_metrics", {}) or {}
    bool_keys = (
        "uses_unavailable_identifier",
        "invalid_resource_usage",
        "uses_invalid_resource",
    )
    count_keys = (
        "unavailable_id_count",
        "invalid_resource_count",
    )
    has_metric = any(key in hard for key in (*bool_keys, *count_keys))
    unavailable_id_count = 0
    for key in count_keys:
        try:
            unavailable_id_count += int(hard.get(key) or 0)
        except (TypeError, ValueError):
            continue
    uses_invalid = any(bool(hard.get(key)) for key in bool_keys)
    return {
        "uses_invalid_resource": uses_invalid,
        "unavailable_id_count": unavailable_id_count,
        "missing_resource_metric": not has_metric,
    }


def _strategy(record: dict[str, Any]) -> str:
    score = record.get("counterfactual_score") or {}
    value = score.get("strategy")
    return str(value) if value else ""


def _final_pass(record: dict[str, Any]) -> bool:
    hard = record.get("metadata", {}).get("counterfactual_hard_metrics", {}) or {}
    if "final_pass" in hard:
        return bool(hard.get("final_pass"))
    return _score(record) is not None


def _avg_list(values: list[float | int]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _rate(values) -> float:
    items = list(values)
    return round(sum(1 for item in items if item) / len(items), 4) if items else 0.0


def _merge_unique(items: list[str]) -> list[str]:
    seen = set()
    merged = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def _run_memory_paths(matrix_config: dict[str, Any], matrix_config_path: Path) -> dict[str, Path]:
    paths = {}
    project_root = matrix_config_path.resolve().parents[1]
    for run in matrix_config.get("runs", []):
        extra = run.get("overrides", {}).get("extra", {})
        raw_path = extra.get("load_skill_memory_path") or extra.get("load_registry_path")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        paths[run["name"]] = path
    return paths


def _find_baseline_run(runs: list[dict[str, Any]], baseline_name: str | None) -> dict[str, Any]:
    if baseline_name:
        for run in runs:
            if run.get("name") == baseline_name:
                return run
        raise ValueError(f"Baseline run not found: {baseline_name}")
    for run in runs:
        name = str(run.get("name", "")).lower()
        method = str(run.get("method", "")).lower()
        if "no_evolution" in name or method == "no_evolution":
            return run
    raise ValueError("Could not infer no_evolution baseline run.")


def _select_best_checkpoint(
    runs: list[dict[str, Any]],
    run_to_memory: dict[str, Path],
    objective: str,
) -> dict[str, Any]:
    candidates = [run for run in runs if run.get("name") in run_to_memory]
    if not candidates:
        raise ValueError("No checkpoint runs with load_skill_memory_path were found.")
    return max(candidates, key=lambda run: _rank_tuple(run, objective))


def _rank_tuple(run: dict[str, Any], objective: str) -> tuple[float, float, float, float, float]:
    metrics = run.get("metrics", {})
    return (
        float(metrics.get(objective) or 0.0),
        -float(metrics.get("counterfactual_low_score_rate") or 0.0),
        float(metrics.get("counterfactual_final_pass_rate") or 0.0),
        -float(metrics.get("counterfactual_unavailable_usage_rate") or 0.0),
        -float(metrics.get("counterfactual_llm_error_rate") or 0.0),
    )


def _paired_dimension_deltas(
    baseline_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    baseline = {}
    for record in baseline_records:
        key = _record_key(record)
        score = _score(record)
        if key and score is not None:
            baseline[key] = score

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in target_records:
        key = _record_key(record)
        score = _score(record)
        if not key or score is None or key not in baseline:
            continue
        intervention = record.get("intervention") or {}
        dimension = str(intervention.get("dimension") or "unknown")
        grouped[dimension].append(score - baseline[key])

    return {
        dimension: {
            "count": len(deltas),
            "mean_delta": round(sum(deltas) / len(deltas), 4),
            "improved": sum(1 for delta in deltas if delta > 0),
            "worse": sum(1 for delta in deltas if delta < 0),
            "same": sum(1 for delta in deltas if delta == 0),
        }
        for dimension, deltas in sorted(grouped.items())
        if deltas
    }


def _gate_skill(
    skill: Skill,
    *,
    dimension_deltas: dict[str, dict[str, Any]],
    selected_run: str,
    selected_memory_path: Path,
    objective: str,
    promote_delta_threshold: float,
    deprecate_delta_threshold: float,
) -> tuple[Skill, str]:
    updated = skill.model_copy(deep=True)
    evidence = int(float(updated.metrics.get("evidence_count") or 0))
    delta_rows = [
        dimension_deltas[dimension]
        for dimension in updated.trigger_dimensions
        if dimension in dimension_deltas
    ]
    deltas = [float(row["mean_delta"]) for row in delta_rows]
    gate_count = sum(int(row.get("count") or 0) for row in delta_rows)

    reason = "kept_active_by_default"
    if updated.skill_type == SkillType.AGENT_RECOVERY:
        if deltas and max(deltas) < deprecate_delta_threshold and gate_count >= 5:
            updated.status = SkillStatus.DEPRECATED
            reason = "deprecated_agent_skill_severe_negative_gate_delta"
        elif deltas and max(deltas) >= promote_delta_threshold and evidence >= 2:
            updated.status = SkillStatus.PROMOTED
            reason = "promoted_agent_skill_positive_gate_delta"
        elif deltas and max(deltas) < 0:
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_negative_but_not_severe"
        elif evidence >= 3:
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_sufficient_train_evidence"
        else:
            updated.status = SkillStatus.ACTIVE
            reason = "active_agent_skill_insufficient_gate_evidence_for_change"
    elif updated.skill_type == SkillType.INTERVENTION_SPEC:
        label = str(updated.metrics.get("label") or "")
        validity = float(updated.metrics.get("avg_validity") or 1.0)
        diagnosticity = float(updated.metrics.get("avg_diagnosticity") or 0.0)
        if validity < 0.7:
            updated.status = SkillStatus.DEPRECATED
            reason = "deprecated_wm_spec_low_validity"
        elif label == "low_value_or_too_easy" and evidence <= 1:
            updated.status = SkillStatus.DEPRECATED
            reason = "deprecated_wm_spec_low_value_low_evidence"
        elif label == "high_value_challenge" or diagnosticity >= 0.7 or evidence >= 3:
            updated.status = SkillStatus.PROMOTED
            reason = "promoted_wm_spec_diagnostic_or_sufficient_evidence"
        else:
            updated.status = SkillStatus.ACTIVE
            reason = "active_wm_spec_kept_for_coverage"

    updated.metadata = {
        **updated.metadata,
        "lifecycle_gate": {
            "selected_run": selected_run,
            "selected_memory_path": str(selected_memory_path),
            "objective": objective,
            "trigger_dimension_gate_deltas": {
                dimension: dimension_deltas[dimension]
                for dimension in updated.trigger_dimensions
                if dimension in dimension_deltas
            },
            "decision_reason": reason,
        },
    }
    return updated, reason


def _summary(
    *,
    baseline: dict[str, Any],
    best: dict[str, Any],
    memory_path: Path,
    output_path: Path,
    objective: str,
    dimension_deltas: dict[str, dict[str, Any]],
    gated: SkillRegistry,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_metric = float(baseline.get("metrics", {}).get(objective) or 0.0)
    best_metric = float(best.get("metrics", {}).get(objective) or 0.0)
    status_counts = Counter(skill.status.value for skill in gated.all())
    type_status_counts = Counter(
        f"{skill.skill_type.value}:{skill.status.value}" for skill in gated.all()
    )
    return {
        "objective": objective,
        "baseline_run": baseline.get("name"),
        "selected_run": best.get("name"),
        "selected_memory_path": str(memory_path),
        "output_path": str(output_path),
        "baseline_objective": baseline_metric,
        "selected_objective": best_metric,
        "selected_vs_baseline_delta": round(best_metric - baseline_metric, 4),
        "dimension_deltas": dimension_deltas,
        "status_counts": dict(sorted(status_counts.items())),
        "type_status_counts": dict(sorted(type_status_counts.items())),
        "decisions": decisions,
    }


def _write_summary(output_path: Path, summary: dict[str, Any]) -> None:
    summary_json = output_path.with_suffix(".summary.json")
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Lifecycle Selection Summary",
        "",
        f"- objective: `{summary['objective']}`",
        f"- baseline_run: `{summary['baseline_run']}`",
        f"- selected_run: `{summary['selected_run']}`",
        f"- selected_vs_baseline_delta: `{summary['selected_vs_baseline_delta']}`",
        f"- output_path: `{summary['output_path']}`",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "| --- | --- |",
    ]
    for status, count in summary["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Dimension Deltas",
            "",
            "| dimension | count | mean_delta | improved | worse | same |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for dimension, row in summary["dimension_deltas"].items():
        lines.append(
            f"| {dimension} | {row['count']} | {row['mean_delta']} | "
            f"{row['improved']} | {row['worse']} | {row['same']} |"
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| skill_id | type | status | reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for decision in summary["decisions"]:
        lines.append(
            f"| {decision['skill_id']} | {decision['skill_type']} | "
            f"{decision['status']} | {decision['reason']} |"
        )
    output_path.with_suffix(".summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _record_key(record: dict[str, Any]) -> tuple[str, str] | None:
    if not record.get("counterfactual_score") or not record.get("intervention"):
        return None
    sample = record.get("sample") or {}
    intervention = record.get("intervention") or {}
    sample_id = sample.get("sample_id")
    intervention_id = intervention.get("intervention_id")
    if not sample_id or not intervention_id:
        return None
    return (str(sample_id), str(intervention_id))


def _score(record: dict[str, Any]) -> float | None:
    score = (record.get("counterfactual_score") or {}).get("total")
    return float(score) if isinstance(score, (int, float)) else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
