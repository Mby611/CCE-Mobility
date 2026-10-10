from __future__ import annotations
from pathlib import Path
from typing import Any
from cce_mobility.core.capabilities import infer_failure_types
from cce_mobility.core.schema import EvaluationRecord
from cce_mobility.interventions.ontology import (
    InterventionOntologyRegistry,
    WMInterventionOntologyDiscoverer,
    registry_from_seed_dimensions,
)
from cce_mobility.skills import SkillRegistry, SkillType, mine_skills_from_records
from cce_mobility.skills.prompting import (
    agent_self_retrieval_prompt,
    world_model_self_retrieval_prompt,
)


class SkillEvolutionMethod:
    """Judge-anchored recovery-skill/spec lifecycle for co-evolution.

    Iteration 1 starts without learned skills. After each iteration, failed
    recoveries become agent recovery skills, while diagnostic interventions
    become world-model intervention specs. The runner injects active skills into
    the agent prompt and active specs into the world-model prompt next round.
    """

    name = "skill_evolution"

    def __init__(self):
        self.iteration = 0
        self.registry = SkillRegistry()
        self.observed_records: list[EvaluationRecord] = []
        self.agent_prompt = ""
        self.world_model_prompt = ""
        self.frozen = False

    def before_iteration(self, records: list[EvaluationRecord]) -> None:
        self.iteration += 1
        self._refresh_prompts()

    def agent_id(self) -> str:
        return f"skill_agent_iter_{self.iteration}"

    def observe(self, records: list[EvaluationRecord]) -> None:
        if self.frozen:
            return
        self.observed_records.extend(records)
        mined = mine_skills_from_records(records)
        for skill in mined.all():
            self.registry.add_or_merge(skill)
        self._refresh_prompts()

    def agent_guidance(self) -> str:
        return self.agent_prompt

    def agent_skill_registry(self) -> SkillRegistry:
        return SkillRegistry(
            (skill for skill in self.registry.all() if skill.skill_type == SkillType.AGENT_RECOVERY)
        )

    def world_model_guidance(self) -> str:
        return self.world_model_prompt

    def report_metrics(self) -> dict[str, int]:
        agent_skills = self.registry.active(SkillType.AGENT_RECOVERY)
        wm_specs = self.registry.active(SkillType.INTERVENTION_SPEC)
        return {
            "skill_spec_total_count": len(self.registry),
            "agent_recovery_skill_count": len(agent_skills),
            "wm_intervention_spec_count": len(wm_specs),
        }

    def freeze(self) -> None:
        self.frozen = True

    def load_registry(self, path: str | Path) -> None:
        self.registry.extend(SkillRegistry.load_jsonl(Path(path)).all())
        self._refresh_prompts()

    def save_registry(self, path: str | Path) -> None:
        self.registry.save_jsonl(Path(path))

    def _refresh_prompts(self) -> None:
        self.agent_prompt = agent_self_retrieval_prompt(self.registry, limit=8)
        self.world_model_prompt = world_model_self_retrieval_prompt(self.registry, limit=8)


class AdaptivePhaseSwitchingCoEvolutionMethod:
    """Convergence-gated co-evolution with alternating frozen phases.

    Agent phases keep the world-model spec library fixed and update only
    recovery skills. WM phases keep the agent skill library fixed and update
    only intervention specs. Phase switches are triggered by plateau signals or
    conservative max-round guards, so the method does not depend on a fixed
    K-step WM update window.
    """

    name = "adaptive_phase_switching_coevolution"

    def __init__(self):
        self.iteration = 0
        self.phase = "agent"
        self.registry = SkillRegistry()
        self.observed_records: list[EvaluationRecord] = []
        self.agent_prompt = ""
        self.world_model_prompt = ""
        self.frozen = False
        self.min_agent_rounds = 2
        self.max_agent_rounds = 6
        self.min_wm_rounds = 1
        self.max_wm_rounds = 4
        self.patience = 2
        self.agent_epsilon_recovery = 0.02
        self.agent_epsilon_low_score = 0.01
        self.agent_epsilon_unavailable = 0.01
        self.agent_mastery_recovery = 3.0
        self.agent_mastery_low_score = 0.1
        self.wm_epsilon_new_patterns = 2
        self.wm_epsilon_new_coverage = 1
        self.wm_epsilon_prediction = 0.02
        self.max_invalid_or_unfair_rate = 0.2
        self.max_llm_error_rate = 0.1
        self.max_cycles = 3
        self.max_total_rounds = 18
        self.stop_patience_cycles = 2
        self.phase_round = 0
        self.phase_bad_rounds = 0
        self.phase_best: dict[str, float] = {}
        self.phase_seen_patterns: set[tuple[str, str, str]] = set()
        self.phase_seen_coverage: set[tuple[str, str]] = set()
        self.last_agent_end_reason = ""
        self.last_wm_end_reason = ""
        self.converged_cycle_count = 0
        self.cycles_completed = 0
        self.phase_switch_count = 0
        self.agent_update_count = 0
        self.wm_update_count = 0
        self.stop_requested = False
        self.last_switch_reason = ""
        self.phase_history: list[str] = []

    def configure(self, extra: dict[str, Any]) -> None:
        self.phase = str(extra.get("start_phase", self.phase)).lower()
        if self.phase not in {"agent", "wm"}:
            self.phase = "agent"
        self.min_agent_rounds = max(1, int(extra.get("min_agent_rounds", self.min_agent_rounds)))
        self.max_agent_rounds = max(
            self.min_agent_rounds, int(extra.get("max_agent_rounds", self.max_agent_rounds))
        )
        self.min_wm_rounds = max(1, int(extra.get("min_wm_rounds", self.min_wm_rounds)))
        self.max_wm_rounds = max(
            self.min_wm_rounds, int(extra.get("max_wm_rounds", self.max_wm_rounds))
        )
        self.patience = max(1, int(extra.get("patience", self.patience)))
        self.agent_epsilon_recovery = float(
            extra.get("agent_epsilon_recovery", self.agent_epsilon_recovery)
        )
        self.agent_epsilon_low_score = float(
            extra.get("agent_epsilon_low_score", self.agent_epsilon_low_score)
        )
        self.agent_epsilon_unavailable = float(
            extra.get("agent_epsilon_unavailable", self.agent_epsilon_unavailable)
        )
        self.agent_mastery_recovery = float(
            extra.get("agent_mastery_recovery", self.agent_mastery_recovery)
        )
        self.agent_mastery_low_score = float(
            extra.get("agent_mastery_low_score", self.agent_mastery_low_score)
        )
        self.wm_epsilon_new_patterns = int(
            extra.get("wm_epsilon_new_patterns", self.wm_epsilon_new_patterns)
        )
        self.wm_epsilon_new_coverage = int(
            extra.get("wm_epsilon_new_coverage", self.wm_epsilon_new_coverage)
        )
        self.wm_epsilon_prediction = float(
            extra.get("wm_epsilon_prediction", self.wm_epsilon_prediction)
        )
        self.max_invalid_or_unfair_rate = float(
            extra.get("max_invalid_or_unfair_rate", self.max_invalid_or_unfair_rate)
        )
        self.max_llm_error_rate = float(extra.get("max_llm_error_rate", self.max_llm_error_rate))
        self.max_cycles = max(1, int(extra.get("max_cycles", self.max_cycles)))
        self.max_total_rounds = max(1, int(extra.get("max_total_rounds", self.max_total_rounds)))
        self.stop_patience_cycles = max(
            1, int(extra.get("stop_patience_cycles", self.stop_patience_cycles))
        )
        self._refresh_prompts()

    def before_iteration(self, records: list[EvaluationRecord]) -> None:
        self.iteration += 1
        self.phase_round += 1
        self.phase_history.append(f"{self.iteration}:{self.phase}")
        self._refresh_prompts()

    def agent_id(self) -> str:
        return f"phase_switch_{self.phase}_iter_{self.iteration}"

    def observe(self, records: list[EvaluationRecord]) -> None:
        if self.frozen:
            return
        self.observed_records.extend(records)
        if self.phase == "agent":
            self._update_agent_skills(records)
            self._maybe_switch_from_agent(records)
        else:
            self._update_wm_specs(records)
            self._maybe_switch_from_wm(records)
        if self.iteration >= self.max_total_rounds:
            self.stop_requested = True
            self.last_switch_reason = self.last_switch_reason or "max_total_rounds"
        self._refresh_prompts()

    def agent_guidance(self) -> str:
        return self.agent_prompt

    def agent_skill_registry(self) -> SkillRegistry:
        return SkillRegistry(
            (skill for skill in self.registry.all() if skill.skill_type == SkillType.AGENT_RECOVERY)
        )

    def world_model_guidance(self) -> str:
        return self.world_model_prompt

    def report_metrics(self) -> dict[str, int | float | str | bool]:
        agent_skills = self.registry.active(SkillType.AGENT_RECOVERY)
        wm_specs = self.registry.active(SkillType.INTERVENTION_SPEC)
        return {
            "skill_spec_total_count": len(self.registry),
            "agent_recovery_skill_count": len(agent_skills),
            "wm_intervention_spec_count": len(wm_specs),
            "phase_switch_current_phase": self.phase,
            "phase_switch_phase_round": self.phase_round,
            "phase_switch_count": self.phase_switch_count,
            "phase_switch_cycles_completed": self.cycles_completed,
            "phase_switch_converged_cycle_count": self.converged_cycle_count,
            "phase_switch_stop_requested": self.stop_requested,
            "phase_switch_last_reason": self.last_switch_reason,
            "phase_switch_agent_update_count": self.agent_update_count,
            "phase_switch_wm_update_count": self.wm_update_count,
            "phase_switch_min_agent_rounds": self.min_agent_rounds,
            "phase_switch_max_agent_rounds": self.max_agent_rounds,
            "phase_switch_min_wm_rounds": self.min_wm_rounds,
            "phase_switch_max_wm_rounds": self.max_wm_rounds,
            "phase_switch_max_cycles": self.max_cycles,
            "phase_switch_max_total_rounds": self.max_total_rounds,
            "phase_switch_history": ",".join(self.phase_history[-24:]),
        }

    def should_stop(self) -> bool:
        return self.stop_requested

    def freeze(self) -> None:
        self.frozen = True

    def load_registry(self, path: str | Path) -> None:
        self.registry.extend(SkillRegistry.load_jsonl(Path(path)).all())
        self._refresh_prompts()

    def save_registry(self, path: str | Path) -> None:
        self.registry.save_jsonl(Path(path))

    def _update_agent_skills(self, records: list[EvaluationRecord]) -> None:
        mined = mine_skills_from_records(records)
        updated = 0
        for skill in mined.all():
            if skill.skill_type == SkillType.AGENT_RECOVERY:
                self.registry.add_or_merge(skill)
                updated += 1
        if updated:
            self.agent_update_count += 1

    def _update_wm_specs(self, records: list[EvaluationRecord]) -> None:
        mined = mine_skills_from_records(_residual_failure_records(records) or records)
        updated = 0
        for skill in mined.all():
            if skill.skill_type == SkillType.INTERVENTION_SPEC:
                self.registry.add_or_merge(skill)
                updated += 1
        if updated:
            self.wm_update_count += 1

    def _maybe_switch_from_agent(self, records: list[EvaluationRecord]) -> None:
        metrics = _phase_agent_metrics(records)
        significant = self._agent_made_progress(metrics)
        if significant:
            self.phase_bad_rounds = 0
        else:
            self.phase_bad_rounds += 1
        reason = ""
        if self.phase_round >= self.max_agent_rounds:
            reason = "agent_max_rounds"
        elif self.phase_round >= self.min_agent_rounds and self.phase_bad_rounds >= self.patience:
            if (
                metrics["recovery_avg"] >= self.agent_mastery_recovery
                and metrics["low_score_rate"] <= self.agent_mastery_low_score
            ):
                reason = "agent_mastered_plateau"
            else:
                reason = "agent_stalled_plateau"
        if reason:
            self.last_agent_end_reason = reason
            self._switch_phase("wm", reason)

    def _maybe_switch_from_wm(self, records: list[EvaluationRecord]) -> None:
        metrics = _phase_wm_metrics(records)
        if self.phase_round >= self.min_wm_rounds and (
            metrics["invalid_or_unfair_rate"] > self.max_invalid_or_unfair_rate
            or metrics["llm_error_rate"] > self.max_llm_error_rate
        ):
            reason = "wm_quality_guard"
            self.last_wm_end_reason = reason
            self._complete_cycle(reason)
            self._switch_phase("agent", reason)
            return
        significant = self._wm_made_progress(metrics)
        if significant:
            self.phase_bad_rounds = 0
        else:
            self.phase_bad_rounds += 1
        reason = ""
        if self.phase_round >= self.max_wm_rounds:
            reason = "wm_max_rounds"
        elif self.phase_round >= self.min_wm_rounds and self.phase_bad_rounds >= self.patience:
            reason = "wm_plateau"
        if reason:
            self.last_wm_end_reason = reason
            self._complete_cycle(reason)
            self._switch_phase("agent", reason)

    def _agent_made_progress(self, metrics: dict[str, float]) -> bool:
        if not self.phase_best:
            self.phase_best = dict(metrics)
            return True
        recovery_gain = metrics["recovery_avg"] - self.phase_best.get("recovery_avg", 0.0)
        low_gain = self.phase_best.get("low_score_rate", 1.0) - metrics["low_score_rate"]
        unavailable_gain = (
            self.phase_best.get("unavailable_usage_rate", 1.0) - metrics["unavailable_usage_rate"]
        )
        significant = (
            recovery_gain >= self.agent_epsilon_recovery
            or low_gain >= self.agent_epsilon_low_score
            or unavailable_gain >= self.agent_epsilon_unavailable
        )
        if significant:
            self.phase_best["recovery_avg"] = max(
                self.phase_best.get("recovery_avg", 0.0), metrics["recovery_avg"]
            )
            self.phase_best["low_score_rate"] = min(
                self.phase_best.get("low_score_rate", 1.0), metrics["low_score_rate"]
            )
            self.phase_best["unavailable_usage_rate"] = min(
                self.phase_best.get("unavailable_usage_rate", 1.0),
                metrics["unavailable_usage_rate"],
            )
        return significant

    def _wm_made_progress(self, metrics: dict[str, float | set]) -> bool:
        patterns = metrics["high_value_patterns"]
        coverage = metrics["coverage"]
        assert isinstance(patterns, set) and isinstance(coverage, set)
        new_patterns = patterns - self.phase_seen_patterns
        new_coverage = coverage - self.phase_seen_coverage
        prediction_gain = float(metrics["failure_mode_hit_rate"]) - self.phase_best.get(
            "failure_mode_hit_rate", 0.0
        )
        significant = (
            len(new_patterns) >= self.wm_epsilon_new_patterns
            or len(new_coverage) >= self.wm_epsilon_new_coverage
            or prediction_gain >= self.wm_epsilon_prediction
        )
        self.phase_seen_patterns |= patterns
        self.phase_seen_coverage |= coverage
        self.phase_best["failure_mode_hit_rate"] = max(
            self.phase_best.get("failure_mode_hit_rate", 0.0),
            float(metrics["failure_mode_hit_rate"]),
        )
        return significant

    def _switch_phase(self, next_phase: str, reason: str) -> None:
        self.phase = next_phase
        self.phase_round = 0
        self.phase_bad_rounds = 0
        self.phase_best = {}
        self.phase_seen_patterns = set()
        self.phase_seen_coverage = set()
        self.phase_switch_count += 1
        self.last_switch_reason = reason

    def _complete_cycle(self, reason: str) -> None:
        self.cycles_completed += 1
        if "plateau" in self.last_agent_end_reason and "plateau" in reason:
            self.converged_cycle_count += 1
        else:
            self.converged_cycle_count = 0
        if (
            self.cycles_completed >= self.max_cycles
            or self.converged_cycle_count >= self.stop_patience_cycles
        ):
            self.stop_requested = True

    def _refresh_prompts(self) -> None:
        self.agent_prompt = agent_self_retrieval_prompt(self.registry, limit=8)
        self.world_model_prompt = world_model_self_retrieval_prompt(self.registry, limit=8)


class OntologyDiscoveryPhaseSwitchingCoEvolutionMethod(AdaptivePhaseSwitchingCoEvolutionMethod):
    """Adaptive phase-switching with WM-side intervention ontology discovery.

    This is the v2 method line. It keeps the v1 update boundary intact:

    - agent phase updates only agent recovery skills;
    - WM phase updates intervention specs and intervention ontology entries;
    - validation/test can freeze both skill/spec memory and ontology memory.
    """

    name = "ontology_discovery_phase_switching_coevolution"

    def __init__(self):
        super().__init__()
        self.ontology_registry = InterventionOntologyRegistry()
        self.ontology_discoverer = WMInterventionOntologyDiscoverer()
        self.ontology_update_count = 0
        self.seed_ontology_enabled = True
        self.ontology_prompt_limit = 8

    def configure(self, extra: dict[str, Any]) -> None:
        super().configure(extra)
        self.seed_ontology_enabled = bool(
            extra.get("seed_ontology_enabled", self.seed_ontology_enabled)
        )
        self.ontology_prompt_limit = max(
            1, int(extra.get("ontology_prompt_limit", self.ontology_prompt_limit))
        )
        seed_dimensions = extra.get("seed_ontology_dimensions") or extra.get(
            "ontology_seed_dimensions"
        )
        if seed_dimensions and self.seed_ontology_enabled:
            self.seed_intervention_ontology(seed_dimensions)
        load_path = extra.get("load_intervention_ontology_path")
        if load_path:
            self.ontology_registry = InterventionOntologyRegistry.load_jsonl(load_path)
        self._refresh_prompts()

    def seed_intervention_ontology(self, dimensions: list[str] | None) -> None:
        if not dimensions or not self.seed_ontology_enabled or len(self.ontology_registry):
            return
        self.ontology_registry = registry_from_seed_dimensions(dimensions)

    def intervention_ontology_registry(self) -> InterventionOntologyRegistry:
        return self.ontology_registry

    def world_model_guidance(self) -> str:
        parts = [world_model_self_retrieval_prompt(self.registry, limit=8)]
        ontology_prompt = self.ontology_registry.to_prompt(limit=self.ontology_prompt_limit)
        if ontology_prompt:
            parts.extend(
                [
                    "World-model ontology discovery guidance:",
                    "Use the discovered ontology to reason about generic operators first, then map to executable benchmark dimensions.",
                    ontology_prompt,
                ]
            )
        return "\n\n".join((part for part in parts if part))

    def report_metrics(self) -> dict[str, int | float | str | bool]:
        metrics = dict(super().report_metrics())
        entries = self.ontology_registry.active() or self.ontology_registry.all()
        metrics.update(
            {
                "wm_intervention_ontology_count": len(self.ontology_registry),
                "wm_intervention_ontology_active_count": len(entries),
                "wm_intervention_ontology_operator_coverage": len(
                    {entry.operator for entry in entries}
                ),
                "wm_intervention_ontology_surface_dimension_coverage": len(
                    {dimension for entry in entries for dimension in entry.surface_dimensions}
                ),
                "wm_intervention_ontology_update_count": self.ontology_update_count,
            }
        )
        return metrics

    def save_extra_artifacts(self, output_dir: str | Path, *, iteration: int | None) -> None:
        path = Path(output_dir) / (
            f"intervention_ontology_iteration_{iteration}.jsonl"
            if iteration is not None
            else "intervention_ontology_final.jsonl"
        )
        self.ontology_registry.save_jsonl(path)

    def _update_wm_specs(self, records: list[EvaluationRecord]) -> None:
        before = len(self.ontology_registry)
        super()._update_wm_specs(records)
        discovery_records = _residual_failure_records(records) or records
        discovered = self.ontology_discoverer.discover(discovery_records)
        for entry in discovered.all():
            self.ontology_registry.add_or_merge(entry)
        if len(self.ontology_registry) > before or len(discovered):
            self.ontology_update_count += 1

    def _refresh_prompts(self) -> None:
        self.agent_prompt = agent_self_retrieval_prompt(self.registry, limit=8)
        self.world_model_prompt = self.world_model_guidance()


def _residual_failure_records(records: list[EvaluationRecord]) -> list[EvaluationRecord]:
    residual = []
    for record in records:
        if not record.intervention or not record.counterfactual_score:
            continue
        score = record.counterfactual_score
        hard = record.metadata.get("counterfactual_hard_metrics", {})
        if (
            score.total < 3.0
            or score.strategy.value in {"ignore", "hallucinate", "passive_notify"}
            or bool(hard.get("uses_unavailable_identifier"))
        ):
            residual.append(record)
    return residual


def _phase_agent_metrics(records: list[EvaluationRecord]) -> dict[str, float]:
    cf_records = [record for record in records if record.counterfactual_score is not None]
    scores = [float(record.counterfactual_score.total) for record in cf_records]
    hard_rows = [record.metadata.get("counterfactual_hard_metrics", {}) for record in cf_records]
    unavailable_rows = [
        row
        for row in hard_rows
        if int(row.get("unavailable_id_count") or 0) > 0 or "uses_unavailable_identifier" in row
    ]
    return {
        "recovery_avg": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "low_score_rate": _rate([score < 2.0 for score in scores]),
        "unavailable_usage_rate": _rate(
            [bool(row.get("uses_unavailable_identifier")) for row in unavailable_rows]
        ),
    }


def _phase_wm_metrics(records: list[EvaluationRecord]) -> dict[str, float | set]:
    cf_records = [record for record in records if record.counterfactual_score is not None]
    patterns: set[tuple[str, str, str]] = set()
    coverage: set[tuple[str, str]] = set()
    invalid_flags = []
    llm_error_flags = []
    failure_hits = []
    for record in cf_records:
        if not record.intervention or not record.counterfactual_score:
            continue
        hard = record.metadata.get("counterfactual_hard_metrics", {})
        validity = float(record.metadata.get("intervention_validity") or 0.0)
        invalid_flags.append(validity < 0.7)
        llm_error_flags.append(bool(hard.get("llm_error")))
        dimension = record.intervention.dimension
        operator = record.intervention.domain_operator or "unknown_operator"
        coverage.add(("dimension", dimension))
        coverage.add(("operator", operator))
        for capability in record.intervention.capability_targets:
            coverage.add(("capability", capability))
        failures = infer_failure_types(
            dimension=dimension,
            judge_strategy=record.counterfactual_score.strategy.value,
            hard_metrics=hard,
            score_total=record.counterfactual_score.total,
        )
        if record.counterfactual_score.total < 3.0 or hard.get("uses_unavailable_identifier"):
            for failure in failures or ["low_recovery"]:
                patterns.add((dimension, operator, failure))
        prediction = record.metadata.get("wm_prediction_metrics", {})
        if prediction:
            failure_hits.append(bool(prediction.get("failure_mode_hit")))
    return {
        "high_value_patterns": patterns,
        "coverage": coverage,
        "invalid_or_unfair_rate": _rate(invalid_flags),
        "llm_error_rate": _rate(llm_error_flags),
        "failure_mode_hit_rate": _rate(failure_hits),
    }


def _rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return round(sum((1 for value in values if value)) / len(values), 4)
