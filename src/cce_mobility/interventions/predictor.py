from __future__ import annotations

from cce_mobility.core.capabilities import FailureType
from cce_mobility.core.schema import (
    AgentTrajectory,
    InterventionCandidate,
    TaskSample,
    WMPredictionSpec,
)


class HeuristicWMPredictiveCritic:
    """Lightweight centralized critic for structured WM predictions.

    The critic predicts which recovery failure modes a counterfactual
    intervention is likely to expose before the agent performs the
    counterfactual rollout.
    """

    def predict(
        self,
        *,
        sample: TaskSample,
        factual: AgentTrajectory,
        candidate: InterventionCandidate,
        world_model_guidance: str = "",
    ) -> WMPredictionSpec:
        spec = candidate.spec
        capabilities = spec.capability_targets
        failures = _failure_modes_from_capabilities(capabilities, spec.dimension)
        score_bin = _score_bin_from_failures(failures, spec.is_negative)
        confidence = _confidence(spec.source, candidate.validity_score, world_model_guidance)
        return WMPredictionSpec(
            intervention_id=spec.intervention_id,
            sample_id=sample.sample_id,
            predicted_failure_modes=failures,
            predicted_recovery_score_bin=score_bin,
            affected_capabilities=capabilities,
            confidence=confidence,
            rationale=(
                "Prediction based on intervention capability targets, domain operator, "
                "validity, and learned WM spec guidance."
            ),
            metadata={
                "dimension": spec.dimension,
                "domain_operator": spec.domain_operator,
                "intervention_source": spec.source,
                "has_world_model_guidance": bool(world_model_guidance),
            },
        )


def _failure_modes_from_capabilities(capabilities: list[str], dimension: str) -> list[str]:
    failures: list[str] = []
    if "change_detection" in capabilities:
        failures.append(FailureType.MISSED_CHANGE_DETECTION.value)
    if "avoid_invalid_resource" in capabilities:
        failures.append(FailureType.USED_INVALID_RESOURCE.value)
    if "cascading_replan" in capabilities:
        failures.append(FailureType.CASCADE_IGNORED.value)
    if "constraint_verification" in capabilities:
        failures.append(FailureType.CONSTRAINT_NOT_RECHECKED.value)
    if "user_communication" in capabilities:
        failures.append(FailureType.POOR_USER_COMMUNICATION.value)
    if "overreaction_control" in capabilities or dimension == "negative_irrelevant":
        failures.append(FailureType.OVERREACTED_TO_IRRELEVANT_NOISE.value)
    if "tool_robustness" in capabilities:
        failures.append(FailureType.HALLUCINATED_RECOVERY.value)
    return sorted(set(failures)) or [FailureType.LOW_DIAGNOSTICITY.value]


def _score_bin_from_failures(failures: list[str], is_negative: bool) -> str:
    if is_negative:
        return "medium"
    hard_failures = {
        FailureType.USED_INVALID_RESOURCE.value,
        FailureType.CASCADE_IGNORED.value,
        FailureType.HALLUCINATED_RECOVERY.value,
    }
    if hard_failures & set(failures):
        return "low"
    if len(failures) >= 2:
        return "medium"
    return "high"


def _confidence(source: str, validity_score: float, guidance: str) -> float:
    confidence = 0.45
    if source == "llm_world_model":
        confidence += 0.15
    if validity_score >= 0.9:
        confidence += 0.15
    elif validity_score >= 0.7:
        confidence += 0.05
    if guidance:
        confidence += 0.1
    return round(max(0.0, min(1.0, confidence)), 3)
