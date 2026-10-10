from __future__ import annotations

from cce_mobility.core.schema import EvaluationRecord


class NoEvolutionMethod:
    name = "no_evolution"

    def __init__(self, agent_id_value: str = "base_agent"):
        self.agent_id_value = agent_id_value

    def before_iteration(self, records: list[EvaluationRecord]) -> None:
        return None

    def agent_id(self) -> str:
        return self.agent_id_value

    def observe(self, records: list[EvaluationRecord]) -> None:
        return None
