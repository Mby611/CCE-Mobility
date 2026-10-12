from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from cce_mobility.core.schema import EvaluationRecord
from cce_mobility.skills import SkillRegistry, SkillType, mine_skills_from_records
from cce_mobility.skills.prompting import (
    agent_skill_catalog_prompt,
    agent_self_retrieval_prompt,
    world_model_self_retrieval_prompt,
)
from cce_mobility.skills.retrieval import sanitize_agent_visible_text


class ReflexionMethod:
    """Task-adapted Reflexion baseline with agent-only reflection memory."""

    name = "reflexion"

    def __init__(self):
        self.iteration = 0
        self.reflections: Counter[str] = Counter()
        self.prompt = ""
        self.frozen = False

    def before_iteration(self, records: list[EvaluationRecord]) -> None:
        self.iteration += 1
        self._refresh_prompt()

    def agent_id(self) -> str:
        return f"reflexion_agent_iter_{self.iteration}"

    def observe(self, records: list[EvaluationRecord]) -> None:
        if self.frozen:
            return
        for record in records:
            if _is_recovery_failure(record):
                self.reflections[_reflection(record)] += 1
        self._refresh_prompt()

    def agent_guidance(self) -> str:
        return self.prompt

    def report_metrics(self) -> dict[str, int]:
        return {"reflection_memory_count": len(self.reflections)}

    def freeze(self) -> None:
        self.frozen = True

    def load_registry(self, path: str | Path) -> None:
        self.reflections.clear()
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "reflection":
                    self.reflections[row["text"]] += int(row.get("count", 1))
        self._refresh_prompt()

    def save_registry(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            for text, count in self.reflections.items():
                f.write(
                    json.dumps(
                        {"kind": "reflection", "text": text, "count": count}, ensure_ascii=False
                    )
                    + "\n"
                )

    def _refresh_prompt(self) -> None:
        if not self.reflections:
            self.prompt = ""
            return
        lines = [
            "Reflections from prior planning and recovery failures:",
            "Use these lessons to avoid repeating the same recovery mistakes.",
        ]
        lines.extend((f"- {item} ({count})" for item, count in self.reflections.most_common(6)))
        self.prompt = "\n".join(lines)


class VoyagerMethod:
    """Task-adapted Voyager baseline with an agent-only persistent skill library."""

    name = "voyager"

    def __init__(self):
        self.iteration = 0
        self.registry = SkillRegistry()
        self.prompt = ""
        self.frozen = False

    def before_iteration(self, records: list[EvaluationRecord]) -> None:
        self.iteration += 1
        self._refresh_prompt()

    def agent_id(self) -> str:
        return f"voyager_agent_iter_{self.iteration}"

    def observe(self, records: list[EvaluationRecord]) -> None:
        if self.frozen:
            return
        mined = mine_skills_from_records(records)
        for skill in mined.all():
            if skill.skill_type == SkillType.AGENT_RECOVERY:
                self.registry.add_or_merge(skill)
        self._refresh_prompt()

    def agent_guidance(self) -> str:
        return self.prompt

    def agent_skill_registry(self) -> SkillRegistry:
        return self.registry

    def report_metrics(self) -> dict[str, int]:
        return {"voyager_skill_count": len(self.registry.active(SkillType.AGENT_RECOVERY))}

    def freeze(self) -> None:
        self.frozen = True

    def load_registry(self, path: str | Path) -> None:
        registry = SkillRegistry.load_jsonl_filtered(
            Path(path), skill_types={SkillType.AGENT_RECOVERY}
        )
        self.registry.extend(registry.all())
        self._refresh_prompt()

    def save_registry(self, path: str | Path) -> None:
        skills = [
            skill for skill in self.registry.all() if skill.skill_type == SkillType.AGENT_RECOVERY
        ]
        SkillRegistry(skills).save_jsonl(Path(path))

    def _refresh_prompt(self) -> None:
        skills = agent_skill_catalog_prompt(self.registry, limit=6)
        if not skills:
            self.prompt = ""
            return
        self.prompt = "\n".join(
            [
                "Retrieved reusable recovery skills from prior episodes:",
                "Apply a matching skill when the current observable task state resembles a prior case.",
                skills,
            ]
        )


class OfficialSkillOptArtifactMethod:
    """Frozen markdown skill exported by Microsoft's official SkillOpt trainer."""

    name = "official_skillopt"

    def __init__(self):
        self.iteration = 0
        self.prompt = ""
        self.frozen = True

    def before_iteration(self, records: list[EvaluationRecord]) -> None:
        self.iteration += 1

    def agent_id(self) -> str:
        return f"official_skillopt_agent_iter_{self.iteration}"

    def observe(self, records: list[EvaluationRecord]) -> None:
        return

    def agent_guidance(self) -> str:
        return self.prompt

    def report_metrics(self) -> dict[str, int]:
        return {
            "official_skillopt_artifact_loaded": int(bool(self.prompt)),
            "official_skillopt_artifact_chars": len(self.prompt),
        }

    def freeze(self) -> None:
        self.frozen = True

    def load_registry(self, path: str | Path) -> None:
        raw = Path(path).read_text(encoding="utf-8")
        self.prompt = sanitize_agent_visible_text(raw).strip()

    def save_registry(self, path: str | Path) -> None:
        Path(path).write_text(self.prompt + "\n", encoding="utf-8")


class AgentSkillOnlyMethod(VoyagerMethod):
    """Ablation: our agent recovery skill library without WM spec evolution."""

    name = "agent_skill_only"

    def agent_id(self) -> str:
        return f"agent_skill_only_iter_{self.iteration}"

    def report_metrics(self) -> dict[str, int]:
        return {
            "agent_recovery_skill_count": len(self.registry.active(SkillType.AGENT_RECOVERY)),
            "wm_intervention_spec_count": 0,
        }

    def _refresh_prompt(self) -> None:
        self.prompt = agent_self_retrieval_prompt(self.registry, limit=8)


class WMSpecOnlyMethod:
    """Ablation: WM intervention spec library without agent skill evolution."""

    name = "wm_spec_only"

    def __init__(self):
        self.iteration = 0
        self.registry = SkillRegistry()
        self.prompt = ""
        self.frozen = False

    def before_iteration(self, records: list[EvaluationRecord]) -> None:
        self.iteration += 1
        self._refresh_prompt()

    def agent_id(self) -> str:
        return f"wm_spec_only_agent_iter_{self.iteration}"

    def observe(self, records: list[EvaluationRecord]) -> None:
        if self.frozen:
            return
        mined = mine_skills_from_records(records)
        for skill in mined.all():
            if skill.skill_type == SkillType.INTERVENTION_SPEC:
                self.registry.add_or_merge(skill)
        self._refresh_prompt()

    def agent_guidance(self) -> str:
        return ""

    def world_model_guidance(self) -> str:
        return self.prompt

    def report_metrics(self) -> dict[str, int]:
        return {
            "agent_recovery_skill_count": 0,
            "wm_intervention_spec_count": len(self.registry.active(SkillType.INTERVENTION_SPEC)),
        }

    def freeze(self) -> None:
        self.frozen = True

    def load_registry(self, path: str | Path) -> None:
        registry = SkillRegistry.load_jsonl_filtered(
            Path(path), skill_types={SkillType.INTERVENTION_SPEC}
        )
        self.registry.extend(registry.all())
        self._refresh_prompt()

    def save_registry(self, path: str | Path) -> None:
        skills = [
            skill
            for skill in self.registry.all()
            if skill.skill_type == SkillType.INTERVENTION_SPEC
        ]
        SkillRegistry(skills).save_jsonl(Path(path))

    def _refresh_prompt(self) -> None:
        self.prompt = world_model_self_retrieval_prompt(self.registry, limit=8)


def _is_recovery_failure(record: EvaluationRecord) -> bool:
    if record.intervention is None or record.counterfactual_score is None:
        return False
    score = record.counterfactual_score
    hard = record.metadata.get("counterfactual_hard_metrics", {})
    return (
        score.total < 3.0
        or score.strategy.value in {"ignore", "hallucinate", "passive_notify"}
        or bool(hard.get("uses_unavailable_identifier"))
    )


def _reflection(record: EvaluationRecord) -> str:
    assert record.intervention is not None
    signal = _training_signal(record)
    return f"For {record.intervention.dimension}, {signal}."


def _training_signal(record: EvaluationRecord) -> str:
    score = record.counterfactual_score
    hard = record.metadata.get("counterfactual_hard_metrics", {})
    if hard.get("uses_unavailable_identifier"):
        return "do not reuse resources currently marked unavailable"
    if score and score.strategy.value == "ignore":
        return "identify relevant environment updates before replanning"
    if score and score.strategy.value == "hallucinate":
        return "ground the recovery in the current environment state"
    return "improve recovery quality, communication, and constraint checking"
