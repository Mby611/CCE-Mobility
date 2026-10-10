from __future__ import annotations
from collections import Counter
from typing import Any
from cce_mobility.core.capabilities import infer_capability_targets, infer_domain_operator
from cce_mobility.core.schema import EvaluationRecord


def aggregate_records(records: list[EvaluationRecord]) -> dict[str, Any]:
    """Aggregate recovery-centered metrics."""
    if not records:
        return {}
    factual_records = _unique_factual_records(records)
    factual_scores = [r.factual_score.total for r in factual_records if r.factual_score is not None]
    cf_scores = [
        r.counterfactual_score.total for r in records if r.counterfactual_score is not None
    ]
    strategies = Counter(
        (
            r.counterfactual_score.strategy.value
            for r in records
            if r.counterfactual_score is not None
        )
    )
    factual_judge_avg = sum(factual_scores) / len(factual_scores) if factual_scores else 0.0
    cf_recovery_avg = sum(cf_scores) / len(cf_scores) if cf_scores else 0.0
    metrics = {
        "count": len(records),
        "sample_count": len(factual_records),
        "counterfactual_count": len(cf_scores),
        "factual_judge_avg": round(factual_judge_avg, 4),
        "counterfactual_recovery_avg": round(cf_recovery_avg, 4),
        "counterfactual_low_score_rate": _low_score_rate(cf_scores),
        "recovery_strategy_distribution": dict(strategies),
        "factual_avg": round(factual_judge_avg, 4),
        "counterfactual_avg": round(cf_recovery_avg, 4),
        "strategy_distribution": dict(strategies),
    }
    metrics.update(_aggregate_hard_metrics(records, factual_records))
    metrics.update(_aggregate_lifecycle_metrics(records))
    metrics.update(_aggregate_prediction_metrics(records))
    metrics.update(_aggregate_agent_retrieval_metrics(records))
    return metrics


def add_experiment_coverage_metrics(
    metrics: dict[str, Any], *, sample_count: int, iterations: int, selected_budget: int
) -> dict[str, Any]:
    """Add expected-vs-realized counterfactual coverage metrics.

    LLM world-model proposals can be filtered out by validation/frontier
    selection. Coverage makes those missing intervention slots visible so that
    recovery averages are not over-interpreted.
    """
    expected = sample_count * iterations * selected_budget
    actual = int(metrics.get("counterfactual_count") or 0)
    metrics["expected_counterfactual_slots"] = expected
    metrics["counterfactual_coverage_rate"] = round(actual / expected, 4) if expected else 0.0
    return metrics


def _aggregate_hard_metrics(
    records: list[EvaluationRecord], factual_records: list[EvaluationRecord]
) -> dict[str, Any]:
    factual = [r.metadata.get("factual_hard_metrics", {}) for r in factual_records]
    counterfactual = [r.metadata.get("counterfactual_hard_metrics", {}) for r in records]
    factual = [item for item in factual if item]
    counterfactual = [item for item in counterfactual if item]
    metrics: dict[str, Any] = {}
    rates: dict[str, float] = {}
    for prefix, values in [("factual", factual), ("counterfactual", counterfactual)]:
        if not values:
            continue
        metrics[f"{prefix}_llm_error_rate"] = _bool_rate(values, "llm_error")
        for key in ["parse_success", "commonsense_pass", "hard_pass", "final_pass"]:
            metric_key = f"{prefix}_{key}_rate"
            metrics[metric_key] = _bool_rate(values, key)
            rates[metric_key] = metrics[metric_key]
        metrics[f"{prefix}_avg_estimated_cost"] = _avg(values, "estimated_cost")
    if factual and counterfactual:
        for key in ["commonsense_pass", "hard_pass", "final_pass"]:
            factual_key = f"factual_{key}_rate"
            counterfactual_key = f"counterfactual_{key}_rate"
            if factual_key in rates and counterfactual_key in rates:
                metrics[f"{key}_rate_delta"] = round(
                    rates[counterfactual_key] - rates[factual_key], 4
                )
        metrics["estimated_cost_delta"] = round(
            metrics.get("counterfactual_avg_estimated_cost", 0.0)
            - metrics.get("factual_avg_estimated_cost", 0.0),
            4,
        )
    cf_with_unavailable = [
        item for item in counterfactual if int(item.get("unavailable_id_count") or 0) > 0
    ]
    if cf_with_unavailable:
        metrics["counterfactual_unavailable_usage_rate"] = _bool_rate(
            cf_with_unavailable, "uses_unavailable_identifier"
        )
        metrics["counterfactual_avg_unavailable_id_count"] = _avg(
            cf_with_unavailable, "unavailable_id_count"
        )
    return metrics


def _unique_factual_records(records: list[EvaluationRecord]) -> list[EvaluationRecord]:
    by_sample: dict[str, EvaluationRecord] = {}
    for record in records:
        by_sample.setdefault(record.sample.sample_id, record)
    return list(by_sample.values())


def _bool_rate(values: list[dict[str, Any]], key: str) -> float:
    present = [bool(item[key]) for item in values if key in item and item[key] is not None]
    if not present:
        return 0.0
    return round(sum((1 for item in present if item)) / len(present), 4)


def _avg(values: list[dict[str, Any]], key: str) -> float:
    nums = [
        float(item[key]) for item in values if key in item and isinstance(item[key], (int, float))
    ]
    if not nums:
        return 0.0
    return round(sum(nums) / len(nums), 4)


def _low_score_rate(values: list[float], threshold: float = 2.0) -> float:
    if not values:
        return 0.0
    return round(sum((1 for value in values if value < threshold)) / len(values), 4)


def _aggregate_lifecycle_metrics(records: list[EvaluationRecord]) -> dict[str, Any]:
    cf_records = [record for record in records if record.counterfactual_score is not None]
    if not cf_records:
        return {}
    capabilities = {
        capability
        for record in cf_records
        if record.intervention is not None
        for capability in record.intervention.capability_targets
        or infer_capability_targets(record.intervention.dimension)
    }
    operators = {
        record.intervention.domain_operator or infer_domain_operator(record.intervention.dimension)
        for record in cf_records
        if record.intervention is not None
    }
    dimensions = {
        record.intervention.dimension for record in cf_records if record.intervention is not None
    }
    wm_labels = [_wm_label(record) for record in cf_records]
    return {
        "capability_coverage": len(capabilities),
        "domain_operator_coverage": len(operators),
        "dimension_coverage": len(dimensions),
        "high_value_wm_challenge_rate": _label_rate(wm_labels, "high_value_challenge"),
        "invalid_or_unfair_wm_challenge_rate": _label_rate(wm_labels, "invalid_or_unfair"),
    }


def _wm_label(record: EvaluationRecord) -> str:
    if "wm_label" in record.metadata:
        return str(record.metadata["wm_label"])
    score = record.counterfactual_score.total if record.counterfactual_score else 0.0
    validity = float(record.metadata.get("intervention_validity") or 0.0)
    hard = record.metadata.get("counterfactual_hard_metrics", {})
    if validity < 0.7:
        return "invalid_or_unfair"
    diagnosticity = 0.0
    if score < 3.0:
        diagnosticity += 0.45
    if hard.get("uses_unavailable_identifier"):
        diagnosticity += 0.35
    if hard.get("parse_success"):
        diagnosticity += 0.2
    if diagnosticity >= 0.7:
        return "high_value_challenge"
    if diagnosticity >= 0.4:
        return "medium_value_challenge"
    return "low_value_or_too_easy"


def _label_rate(labels: list[str], target: str) -> float:
    if not labels:
        return 0.0
    return round(sum((1 for label in labels if label == target)) / len(labels), 4)


def _aggregate_prediction_metrics(records: list[EvaluationRecord]) -> dict[str, Any]:
    rows = [
        record.metadata.get("wm_prediction_metrics", {})
        for record in records
        if record.metadata.get("wm_prediction_metrics")
    ]
    if not rows:
        return {}
    return {
        "wm_prediction_count": len(rows),
        "wm_score_bin_accuracy": _bool_rate(rows, "score_bin_hit"),
        "wm_failure_mode_hit_rate": _bool_rate(rows, "failure_mode_hit"),
        "wm_failure_mode_precision": round(
            sum((float(row.get("failure_mode_precision") or 0.0) for row in rows)) / len(rows), 4
        ),
        "wm_failure_mode_recall": round(
            sum((float(row.get("failure_mode_recall") or 0.0) for row in rows)) / len(rows), 4
        ),
        "wm_prediction_avg_confidence": round(
            sum((float(row.get("confidence") or 0.0) for row in rows)) / len(rows), 4
        ),
    }


def _aggregate_agent_retrieval_metrics(records: list[EvaluationRecord]) -> dict[str, Any]:
    enabled = [
        item
        for record in records
        if record.counterfactual is not None
        if (item := _record_agent_retrieval(record)) and _retrieval_enabled(item)
    ]
    if not enabled:
        return {}
    counts = [len(item.get("selected_skill_ids") or []) for item in enabled]
    selected = sorted(
        {str(skill_id) for item in enabled for skill_id in item.get("selected_skill_ids") or []}
    )
    available = [int(item.get("available_skill_count") or 0) for item in enabled]
    return {
        "agent_owned_retrieval_enabled": True,
        "agent_owned_retrieval_case_count": len(enabled),
        "agent_owned_retrieval_selected_case_count": sum(count > 0 for count in counts),
        "agent_owned_retrieval_no_skill_rate": round(
            sum(bool(item.get("no_skill_needed")) for item in enabled) / len(enabled), 4
        ),
        "agent_owned_retrieval_avg_selected_skills": round(sum(counts) / len(enabled), 4),
        "agent_owned_retrieval_avg_available_skills": round(sum(available) / len(enabled), 4),
        "agent_owned_retrieval_unique_selected_skill_count": len(selected),
        "agent_owned_retrieval_unique_selected_skill_ids": selected,
    }


def _record_agent_retrieval(record: EvaluationRecord) -> dict[str, Any]:
    metadata_retrieval = record.metadata.get("agent_skill_retrieval") or {}
    if metadata_retrieval:
        return metadata_retrieval
    trajectory = record.counterfactual or record.factual
    if trajectory is None:
        return {}
    return trajectory.raw.get("skill_retrieval") or {}


def _retrieval_enabled(item: dict[str, Any]) -> bool:
    if "enabled" in item:
        return bool(item.get("enabled"))
    return bool(
        item.get("selected_skill_ids") or item.get("ignored_skill_ids") or item.get("source")
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
