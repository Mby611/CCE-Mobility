from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from cce_mobility.skills.schema import Skill, SkillStatus, SkillType


class SkillRegistry:
    """Small JSONL-backed skill registry for experiment-time skill selection."""

    def __init__(self, skills: Iterable[Skill] | None = None):
        self._skills: dict[str, Skill] = {}
        for skill in skills or []:
            self.add_or_merge(skill)

    def __len__(self) -> int:
        return len(self._skills)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def extend(self, skills: Iterable[Skill]) -> None:
        for skill in skills:
            self.add_or_merge(skill)

    def add_or_merge(self, skill: Skill) -> Skill:
        existing = self._skills.get(skill.skill_id)
        if existing is None:
            self._skills[skill.skill_id] = skill
            return skill

        existing.source_records = sorted(set(existing.source_records) | set(skill.source_records))
        existing.failure_cases = _merge_lines(existing.failure_cases, skill.failure_cases)
        existing.procedure = _merge_lines(existing.procedure, skill.procedure)
        existing.success_criteria = _merge_lines(existing.success_criteria, skill.success_criteria)
        existing.trigger_dimensions = sorted(
            set(existing.trigger_dimensions) | set(skill.trigger_dimensions)
        )
        existing.capability_targets = sorted(
            set(existing.capability_targets) | set(skill.capability_targets)
        )
        existing.failure_types = sorted(set(existing.failure_types) | set(skill.failure_types))
        existing.domain_operators = sorted(
            set(existing.domain_operators) | set(skill.domain_operators)
        )
        existing.metrics = _merge_metrics(existing.metrics, skill.metrics)
        if _status_rank(skill.status) > _status_rank(existing.status):
            existing.status = skill.status
        return existing

    def active(
        self,
        skill_type: SkillType | str,
        *,
        dimension: str | None = None,
        capability_targets: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Skill]:
        normalized = SkillType(skill_type)
        requested_caps = set(capability_targets or [])
        candidates = [
            skill
            for skill in self._skills.values()
            if skill.skill_type == normalized
            and skill.status in {SkillStatus.ACTIVE, SkillStatus.PROMOTED}
            and (
                dimension is None
                or not skill.trigger_dimensions
                or dimension in skill.trigger_dimensions
            )
            and (
                not requested_caps
                or not skill.capability_targets
                or bool(requested_caps & set(skill.capability_targets))
            )
        ]
        candidates.sort(
            key=lambda skill: (
                _status_rank(skill.status),
                float(skill.metrics.get("evidence_count", 0)),
                float(skill.metrics.get("success_rate", 0)),
            ),
            reverse=True,
        )
        return candidates[:limit] if limit is not None else candidates

    def to_prompt(
        self,
        skill_type: SkillType | str,
        *,
        dimension: str | None = None,
        capability_targets: list[str] | None = None,
        limit: int = 5,
    ) -> str:
        skills = self.active(
            skill_type,
            dimension=dimension,
            capability_targets=capability_targets,
            limit=limit,
        )
        if not skills:
            return ""
        header = (
            "Active agent recovery skills:"
            if SkillType(skill_type) == SkillType.AGENT_RECOVERY
            else "Active world-model intervention specs:"
        )
        return "\n\n".join([header, *(skill.prompt_block() for skill in skills)])

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for skill in self.all():
                f.write(json.dumps(skill.model_dump(mode="json"), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> "SkillRegistry":
        skills: list[Skill] = []
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    skills.append(Skill.model_validate(json.loads(line)))
        return cls(skills)

    @classmethod
    def load_jsonl_filtered(
        cls,
        path: Path,
        *,
        skill_types: set[SkillType] | None = None,
    ) -> "SkillRegistry":
        registry = cls.load_jsonl(path)
        if not skill_types:
            return registry
        return cls(skill for skill in registry.all() if skill.skill_type in skill_types)


def _merge_lines(left: list[str], right: list[str]) -> list[str]:
    seen = set()
    merged = []
    for item in [*left, *right]:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def _merge_metrics(left: dict, right: dict) -> dict:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
            merged[key] = max(merged[key], value)
        else:
            merged[key] = value
    return merged


def _status_rank(status: SkillStatus) -> int:
    return {
        SkillStatus.DEPRECATED: 0,
        SkillStatus.CANDIDATE: 1,
        SkillStatus.ACTIVE: 2,
        SkillStatus.PROMOTED: 3,
    }[status]
