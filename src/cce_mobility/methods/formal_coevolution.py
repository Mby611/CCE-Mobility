from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from cce_mobility.core.llm import ChatClient
from cce_mobility.core.schema import EvaluationRecord
from cce_mobility.interventions.ontology import (
    InterventionOntologyRegistry,
    WMInterventionOntologyDiscoverer,
)
from cce_mobility.skills import Skill, SkillRegistry, SkillType, mine_skills_from_records


FORMAL_TRAINING_PROTOCOL = "mobilitybench-paired-residual-coevolution-v1"
FORMAL_METHOD_VERSION = "cce_mobility_paired_residual_v1"


def paired_key(record: EvaluationRecord) -> tuple[str, str]:
    intervention_id = record.intervention.intervention_id if record.intervention else ""
    return record.sample.sample_id, intervention_id


def recovery_success(record: EvaluationRecord, *, threshold: float = 3.0) -> bool:
    score = record.counterfactual_score
    if score is None or score.total < threshold:
        return False
    hard = record.metadata.get("counterfactual_hard_metrics", {})
    if hard and not bool(hard.get("final_pass", False)):
        return False
    return True


def classify_paired_outcomes(
    pre_records: list[EvaluationRecord],
    post_records: list[EvaluationRecord],
    *,
    threshold: float = 3.0,
) -> dict[str, Any]:
    pre = _index(pre_records)
    post = _index(post_records)
    if set(pre) != set(post):
        missing_post = sorted(set(pre) - set(post))
        missing_pre = sorted(set(post) - set(pre))
        raise ValueError(
            "Paired pre/post keys differ: "
            f"missing_post={missing_post[:5]}, missing_pre={missing_pre[:5]}"
        )

    buckets: dict[str, list[EvaluationRecord]] = {
        "failure_to_failure": [],
        "failure_to_success": [],
        "success_to_failure": [],
        "success_to_success": [],
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(pre):
        before = pre[key]
        after = post[key]
        before_success = recovery_success(before, threshold=threshold)
        after_success = recovery_success(after, threshold=threshold)
        transition = (
            "success_to_success"
            if before_success and after_success
            else "success_to_failure"
            if before_success
            else "failure_to_success"
            if after_success
            else "failure_to_failure"
        )
        buckets[transition].append(after)
        rows.append(
            {
                "task_id": key[0],
                "intervention_id": key[1],
                "transition": transition,
                "pre_score": _score(before),
                "post_score": _score(after),
                "score_delta": round(_score(after) - _score(before), 6),
                "pre_final_pass": _final_pass(before),
                "post_final_pass": _final_pass(after),
            }
        )
    return {
        "counts": {name: len(records) for name, records in buckets.items()},
        "rates": {
            name: round(len(records) / len(rows), 6) if rows else 0.0
            for name, records in buckets.items()
        },
        "rows": rows,
        "persistent_failure_records": buckets["failure_to_failure"],
        "agent_regression_records": buckets["success_to_failure"],
    }


def curation_decision(
    pre_records: list[EvaluationRecord],
    post_records: list[EvaluationRecord],
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    outcomes = classify_paired_outcomes(pre_records, post_records)
    pre_index = _index(pre_records)
    post_index = _index(post_records)
    deltas = [_score(post_index[key]) - _score(pre_index[key]) for key in pre_index]
    mean_delta = statistics.fmean(deltas) if deltas else 0.0
    low_delta = _rate(post_records, _low_score) - _rate(pre_records, _low_score)
    unavailable_delta = _rate(post_records, _uses_unavailable) - _rate(
        pre_records, _uses_unavailable
    )
    final_pass_delta = _rate(post_records, _final_pass) - _rate(pre_records, _final_pass)
    regression_rate = outcomes["rates"]["success_to_failure"]
    score = (
        mean_delta
        - float(thresholds.get("low_score_weight", 0.4)) * low_delta
        - float(thresholds.get("unavailable_weight", 0.6)) * unavailable_delta
        + float(thresholds.get("final_pass_weight", 0.2)) * final_pass_delta
    )
    checks = {
        "lifecycle_score": score >= float(thresholds.get("min_lifecycle_score", 0.0)),
        "mean_delta": mean_delta >= float(thresholds.get("min_mean_delta", -0.05)),
        "low_score_delta": low_delta <= float(thresholds.get("max_low_score_delta", 0.05)),
        "unavailable_delta": unavailable_delta
        <= float(thresholds.get("max_unavailable_delta", 0.02)),
        "final_pass_delta": final_pass_delta
        >= float(thresholds.get("min_final_pass_delta", -0.05)),
        "regression_rate": regression_rate
        <= float(thresholds.get("max_success_to_failure_rate", 0.10)),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "lifecycle_score": round(score, 6),
        "mean_recovery_delta": round(mean_delta, 6),
        "low_score_rate_delta": round(low_delta, 6),
        "unavailable_usage_rate_delta": round(unavailable_delta, 6),
        "final_pass_rate_delta": round(final_pass_delta, 6),
        "transition_counts": outcomes["counts"],
        "transition_rates": outcomes["rates"],
    }


def build_agent_candidate_memory(
    current_memory_path: Path,
    pre_records: list[EvaluationRecord],
    output_path: Path,
) -> dict[str, Any]:
    current = SkillRegistry.load_jsonl(current_memory_path)
    before_ids = {skill.skill_id for skill in current.all()}
    mined = mine_skills_from_records(pre_records)
    for skill in mined.all():
        if skill.skill_type == SkillType.AGENT_RECOVERY:
            current.add_or_merge(_formalize_agent_skill(skill))
    current.save_jsonl(output_path)
    after_ids = {skill.skill_id for skill in current.all()}
    return {
        "input_skill_count": len(before_ids),
        "candidate_skill_count": len(after_ids),
        "new_skill_ids": sorted(after_ids - before_ids),
        "memory_changed": output_path.read_bytes() != current_memory_path.read_bytes(),
    }


def update_world_from_persistent_failures(
    *,
    persistent_records: list[EvaluationRecord],
    current_memory_path: Path,
    current_ontology_path: Path,
    output_memory_path: Path,
    output_ontology_path: Path,
    world_client: ChatClient | None,
    ontology_limit: int = 8,
) -> dict[str, Any]:
    """Update only from final F-to-F records; regressions are never accepted here."""

    memory = SkillRegistry.load_jsonl(current_memory_path)
    ontology = InterventionOntologyRegistry.load_jsonl(current_ontology_path)
    before_memory = {skill.skill_id for skill in memory.all()}
    before_ontology = {entry.ontology_id for entry in ontology.all()}

    mined = mine_skills_from_records(persistent_records)
    for skill in mined.all():
        if skill.skill_type == SkillType.INTERVENTION_SPEC:
            memory.add_or_merge(_formalize_world_spec(skill))
    discovered = WMInterventionOntologyDiscoverer(world_client).discover(
        persistent_records,
        limit=ontology_limit,
    )
    for entry in discovered.all():
        ontology.add_or_merge(entry)

    memory.save_jsonl(output_memory_path)
    ontology.save_jsonl(output_ontology_path)
    after_memory = {skill.skill_id for skill in memory.all()}
    after_ontology = {entry.ontology_id for entry in ontology.all()}
    return {
        "persistent_failure_count": len(persistent_records),
        "input_world_spec_count": len(before_memory),
        "output_world_spec_count": len(after_memory),
        "new_world_spec_ids": sorted(after_memory - before_memory),
        "input_ontology_count": len(before_ontology),
        "output_ontology_count": len(after_ontology),
        "new_ontology_ids": sorted(after_ontology - before_ontology),
        "world_memory_changed": output_memory_path.read_bytes() != current_memory_path.read_bytes(),
        "ontology_changed": output_ontology_path.read_bytes() != current_ontology_path.read_bytes(),
    }


def _formalize_agent_skill(skill: Skill) -> Skill:
    """Remove legacy benchmark-specific prose from formal recovery memories."""

    normalized = skill.model_copy(deep=True)
    normalized.procedure = [
        "Inspect the latest observable tool results before changing the current decision.",
        "Identify the invalidated resource or state and any downstream decisions that depend on it.",
        "Use available tools to obtain current evidence when the existing observation is insufficient.",
        "Choose only a replacement whose availability and relevant constraints are confirmed by observable evidence.",
        "Revise affected decisions while preserving choices that remain valid.",
        "Report the changed fact and the evidence-grounded recovery without exposing internal evaluation metadata.",
    ]
    normalized.success_criteria = [
        "The recovery is grounded only in the current observable history and tool results.",
        "Invalidated resources are not reused and unverified replacements are not presented as confirmed.",
        "The revised result remains consistent with the user request and required output format.",
    ]
    normalized.failure_cases = [
        "Do not ignore an observable change that invalidates the current decision.",
        "Do not invent a replacement, availability claim, or tool result.",
        "Do not revise unrelated valid decisions without evidence that they are affected.",
    ]
    normalized.metadata = {
        **normalized.metadata,
        "formal_template": "benchmark_agnostic_recovery_v1",
    }
    return normalized


def _formalize_world_spec(skill: Skill) -> Skill:
    """Keep formal World memories benchmark-agnostic and tool-observable."""

    normalized = skill.model_copy(deep=True)
    normalized.procedure = [
        "Target a typed resource or environment state evidenced in the factual tool trajectory.",
        "Bind the intervention to a concrete tool interaction and preserve all unrelated observations.",
        "Keep the local change observable, valid, and recoverable through the available tools.",
        "Prefer challenges that expose a documented persistent recovery failure over merely cosmetic changes.",
    ]
    normalized.success_criteria = [
        "The intervention can be applied consistently to the trigger and subsequent matching tool results.",
        "The changed state is visible through normal observations without revealing evaluator-only metadata.",
        "At least one feasible recovery path remains and the challenge diagnoses the targeted capability.",
    ]
    normalized.failure_cases = [
        "Do not target an entity whose type is incompatible with the selected intervention operator.",
        "Do not create an unverifiable, global, impossible, or answer-leaking change.",
        "Do not infer a World update from an Agent regression; use persistent failure evidence only.",
    ]
    normalized.metadata = {
        **normalized.metadata,
        "formal_template": "benchmark_agnostic_world_spec_v1",
    }
    return normalized


def _index(records: list[EvaluationRecord]) -> dict[tuple[str, str], EvaluationRecord]:
    indexed: dict[tuple[str, str], EvaluationRecord] = {}
    for record in records:
        key = paired_key(record)
        if not key[1]:
            raise ValueError(f"Record lacks intervention id: {key[0]}")
        if key in indexed:
            raise ValueError(f"Duplicate paired record: {key}")
        indexed[key] = record
    return indexed


def _score(record: EvaluationRecord) -> float:
    return float(record.counterfactual_score.total) if record.counterfactual_score else 0.0


def _low_score(record: EvaluationRecord) -> bool:
    return _score(record) < 2.0


def _uses_unavailable(record: EvaluationRecord) -> bool:
    return bool(
        record.metadata.get("counterfactual_hard_metrics", {}).get("uses_unavailable_identifier")
    )


def _final_pass(record: EvaluationRecord) -> bool:
    return bool(record.metadata.get("counterfactual_hard_metrics", {}).get("final_pass"))


def _rate(records: list[EvaluationRecord], predicate) -> float:  # type: ignore[no-untyped-def]
    return sum(bool(predicate(record)) for record in records) / len(records) if records else 0.0
