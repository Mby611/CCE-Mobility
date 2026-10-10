from __future__ import annotations

from cce_mobility.core.capabilities import (
    expected_recovery_actions,
    infer_capability_targets,
    judge_rubric_for_capabilities,
)
from cce_mobility.core.schema import ExpectedRecoverySpec, InterventionCandidate, TaskSample


DEFAULT_RUBRIC = {
    "proactive_adapt": "Detects the change, verifies constraints, and produces an executable alternative.",
    "partial_adapt": "Detects the change and makes a partial adjustment, but leaves some risk or inconsistency.",
    "passive_notify": "Mentions the change but does not provide a usable recovery.",
    "ignore": "Does not show evidence of noticing or acting on the change.",
    "hallucinate": "Uses unavailable or contradicted information, or fabricates a recovery.",
}


class RuleSpecBuilder:
    """Build a benchmark-agnostic expected recovery spec."""

    def build(self, candidate: InterventionCandidate, sample: TaskSample) -> ExpectedRecoverySpec:
        spec = candidate.spec
        negative = spec.is_negative
        capability_targets = spec.capability_targets or infer_capability_targets(spec.dimension)
        capability_actions = spec.expected_recovery_actions or expected_recovery_actions(
            capability_targets
        )
        if negative:
            must_do = [
                "Keep the original task focus and avoid changing the plan for irrelevant noise."
            ]
            must_not_do = ["Do not overreact to irrelevant environmental information."]
            acceptable = [
                "Answer the user normally while optionally ignoring the irrelevant notice."
            ]
        else:
            must_do = [
                "Notice the changed environment condition.",
                "Re-check affected constraints before committing to a plan.",
                "Provide a feasible alternative or explicitly ask for clarification if needed.",
            ]
            must_not_do = [
                "Do not recommend an option marked unavailable or unsafe.",
                "Do not silently keep the original plan when it is invalidated.",
            ]
            acceptable = [
                "Replan around the affected route, POI, schedule, budget, or tool result.",
                "Communicate the change and provide a user-actionable recovery.",
            ]
        must_do = _merge_unique([*must_do, *capability_actions])
        capability_rubric = judge_rubric_for_capabilities(capability_targets)

        return ExpectedRecoverySpec(
            intervention_id=spec.intervention_id,
            sample_id=sample.sample_id,
            capability_targets=capability_targets,
            recovery_contract={
                "capability_targets": capability_targets,
                "expected_actions": capability_actions,
                "diagnostic_goal": spec.diagnostic_goal,
                "validity_constraints": spec.validity_constraints,
            },
            summary=spec.condition_text,
            changed_facts=[spec.condition_text],
            must_detect=[spec.dimension] if not negative else [],
            must_do=must_do,
            must_not_do=must_not_do,
            acceptable_recoveries=acceptable,
            invalid_failures=[
                "Ignores a task-relevant intervention.",
                "Hallucinates unavailable options.",
                "Overreacts to a negative/no-op intervention.",
            ],
            rubric={**DEFAULT_RUBRIC, **capability_rubric},
            metadata={
                "dimension": spec.dimension,
                "domain": spec.domain,
                "domain_operator": spec.domain_operator,
                "severity": spec.severity,
            },
        )


def _merge_unique(items: list[str]) -> list[str]:
    seen = set()
    merged = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            merged.append(item)
    return merged
