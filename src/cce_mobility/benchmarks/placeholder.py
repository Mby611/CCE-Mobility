from __future__ import annotations

from cce_mobility.core.config import BenchmarkConfig
from cce_mobility.core.schema import AgentTrajectory, BenchmarkName, InterventionSpec, TaskSample


class PlaceholderBenchmarkAdapter:
    """Explicit placeholder for benchmarks whose adapters are not wired yet."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.name = config.name

    def load_samples(self, limit: int | None = None) -> list[TaskSample]:
        enum_name = BenchmarkName(self.config.name)
        n = limit or self.config.limit or 1
        return [
            TaskSample(
                sample_id=f"{self.config.name}_{i}",
                benchmark=enum_name,
                query=f"Placeholder query {i}",
                metadata={"adapter_status": "placeholder"},
            )
            for i in range(n)
        ]

    def run_agent(
        self,
        sample: TaskSample,
        *,
        agent_id: str,
        intervention: InterventionSpec | None = None,
    ) -> AgentTrajectory:
        suffix = ""
        if intervention is not None:
            suffix = f" Current environment update: {intervention.condition_text}"
        return AgentTrajectory(
            sample_id=sample.sample_id,
            run_mode="counterfactual" if intervention else "factual",
            final_response=f"[placeholder agent={agent_id}] {sample.query}.{suffix}",
        )
