from __future__ import annotations

from cce_mobility.core.schema import InterventionCandidate


class FrontierSelector:
    """Select interventions near the agent competence frontier."""

    def __init__(self, low: float = 0.4, high: float = 0.6):
        self.low = low
        self.high = high

    def select(
        self, candidates: list[InterventionCandidate], budget: int
    ) -> list[InterventionCandidate]:
        scored = []
        for cand in candidates:
            pass_rate = cand.estimated_pass_rate
            if pass_rate is None:
                # Before probing exists, use validity and moderate severity as proxy.
                pass_rate = max(0.0, min(1.0, 1.0 - cand.spec.severity / 3.0))
            frontier_value = 1.0 - min(1.0, abs(pass_rate - 0.5) * 2.0)
            diagnosticity = cand.diagnosticity if cand.diagnosticity is not None else 0.0
            utility = (
                0.45 * frontier_value
                + 0.25 * cand.validity_score
                + 0.20 * diagnosticity
                + 0.10 * _novelty_proxy(cand)
            )
            cand.utility = round(utility, 4)
            scored.append(cand)
        scored.sort(key=lambda c: c.utility or 0.0, reverse=True)
        return scored[:budget]


def _novelty_proxy(candidate: InterventionCandidate) -> float:
    # Placeholder: future implementation should use pool history.
    return 1.0 if candidate.spec.source == "grid" else 0.5
