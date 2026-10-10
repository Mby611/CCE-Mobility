from __future__ import annotations

from typing import Protocol

from cce_mobility.core.schema import (
    AgentTrajectory,
    EvaluationRecord,
    ExpectedRecoverySpec,
    FactualRecoveryContext,
    InterventionCandidate,
    InterventionSpec,
    JudgeScore,
    TaskSample,
    VulnerabilityPoint,
)


class BenchmarkAdapter(Protocol):
    name: str

    def load_samples(self, limit: int | None = None) -> list[TaskSample]: ...

    def run_agent(
        self,
        sample: TaskSample,
        *,
        agent_id: str,
        intervention: InterventionSpec | None = None,
        recovery_context: FactualRecoveryContext | None = None,
    ) -> AgentTrajectory: ...


class TrajectoryAnalyzer(Protocol):
    def find_points(
        self, sample: TaskSample, trajectory: AgentTrajectory
    ) -> list[VulnerabilityPoint]: ...


class InterventionGenerator(Protocol):
    def generate(
        self,
        sample: TaskSample,
        trajectory: AgentTrajectory | None,
        points: list[VulnerabilityPoint],
    ) -> list[InterventionCandidate]: ...


class InterventionValidator(Protocol):
    def validate(
        self, candidate: InterventionCandidate, sample: TaskSample
    ) -> InterventionCandidate: ...


class SpecBuilder(Protocol):
    def build(
        self, candidate: InterventionCandidate, sample: TaskSample
    ) -> ExpectedRecoverySpec: ...


class RecoveryJudge(Protocol):
    def score(self, record: EvaluationRecord) -> JudgeScore: ...


class EvolutionMethod(Protocol):
    name: str

    def before_iteration(self, records: list[EvaluationRecord]) -> None: ...

    def agent_id(self) -> str: ...

    def observe(self, records: list[EvaluationRecord]) -> None: ...


class FrontierSelector(Protocol):
    def select(
        self, candidates: list[InterventionCandidate], budget: int
    ) -> list[InterventionCandidate]: ...
