from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SkillType(str, Enum):
    AGENT_RECOVERY = "agent_recovery"
    INTERVENTION_SPEC = "intervention_spec"
    # Backward-compatible alias for earlier code/tests. In the paper narrative,
    # the world model evolves an intervention spec library rather than skills.
    WM_CHALLENGE = "intervention_spec"


class SkillStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    PROMOTED = "promoted"
    DEPRECATED = "deprecated"


class Skill(BaseModel):
    """A reusable recovery or challenge procedure with lifecycle metadata."""

    skill_id: str
    skill_type: SkillType
    name: str
    trigger_dimensions: list[str] = Field(default_factory=list)
    capability_targets: list[str] = Field(default_factory=list)
    failure_types: list[str] = Field(default_factory=list)
    domain_operators: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    failure_cases: list[str] = Field(default_factory=list)
    source_records: list[str] = Field(default_factory=list)
    status: SkillStatus = SkillStatus.CANDIDATE
    version: int = 1
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def prompt_block(self) -> str:
        trigger = ", ".join(self.trigger_dimensions) or "general"
        capabilities = ", ".join(self.capability_targets) or "general"
        operators = ", ".join(self.domain_operators) or "general"
        lines = [
            f"[{self.skill_id} v{self.version}] {self.name}",
            f"Trigger dimensions: {trigger}",
            f"Capability targets: {capabilities}",
            f"Domain operators: {operators}",
        ]
        if self.failure_types:
            lines.append("Failure types:")
            lines.extend(f"- {failure}" for failure in self.failure_types)
        if self.procedure:
            lines.append("Procedure:")
            lines.extend(f"- {step}" for step in self.procedure)
        if self.success_criteria:
            lines.append("Success criteria:")
            lines.extend(f"- {criterion}" for criterion in self.success_criteria)
        if self.failure_cases:
            lines.append("Avoid:")
            lines.extend(f"- {case}" for case in self.failure_cases)
        return "\n".join(lines)
