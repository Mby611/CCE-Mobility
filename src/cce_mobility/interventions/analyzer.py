from __future__ import annotations

from cce_mobility.core.schema import AgentTrajectory, TaskSample, VulnerabilityPoint


class HeuristicTrajectoryAnalyzer:
    """Find simple intervention points from a trajectory.

    For sole-planning benchmarks this returns a pre-plan point. For interactive
    traces it scores tool observations that contain route, POI, price, weather,
    or failure signals.
    """

    KEYWORDS = (
        "route",
        "traffic",
        "flight",
        "hotel",
        "restaurant",
        "attraction",
        "weather",
        "price",
        "budget",
        "error",
        "unavailable",
    )

    def find_points(
        self, sample: TaskSample, trajectory: AgentTrajectory
    ) -> list[VulnerabilityPoint]:
        points: list[VulnerabilityPoint] = []
        if not trajectory.events:
            return [
                VulnerabilityPoint(
                    sample_id=sample.sample_id,
                    step=0,
                    reason="No trajectory events; intervene before planning.",
                    score=0.5,
                )
            ]

        for event in trajectory.events:
            text = f"{event.tool_name} {event.observation} {event.action_text}".lower()
            hits = [kw for kw in self.KEYWORDS if kw in text]
            if hits:
                points.append(
                    VulnerabilityPoint(
                        sample_id=sample.sample_id,
                        step=event.step,
                        reason=f"Mobility-critical signal(s): {', '.join(hits[:4])}",
                        target="tool_result",
                        evidence=text[:300],
                        score=min(1.0, 0.2 + 0.15 * len(hits)),
                    )
                )

        if not points:
            points.append(
                VulnerabilityPoint(
                    sample_id=sample.sample_id,
                    step=0,
                    reason="Default pre-plan vulnerability point.",
                    score=0.3,
                )
            )
        return sorted(points, key=lambda p: p.score, reverse=True)
