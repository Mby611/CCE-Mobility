from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any
from cce_mobility.skills.registry import SkillRegistry
from cce_mobility.skills.schema import Skill, SkillStatus, SkillType

AGENT_RETRIEVAL_SYSTEM_PROMPT = "You select reusable planning guidance for a travel assistant from the current observable task state. Return only valid JSON."
AGENT_RETRIEVAL_USER_PROMPT = "Current task:\n{query}\n\nCurrent observable environment state:\n{environment_observation}\n\nAvailable recovery guidance index:\n{skill_index}\n\nSelect at most {max_skills} relevant guidance items.\nChoose an item only if the current observable state supports its trigger.\nIf no guidance is relevant, return no_skill_needed=true.\n\nReturn JSON with exactly these keys:\nselected_skill_ids, ignored_skill_ids, selection_rationale, no_skill_needed.\n"


@dataclass
class SkillRetrievalDecision:
    selected_skill_ids: list[str] = field(default_factory=list)
    ignored_skill_ids: list[str] = field(default_factory=list)
    selection_rationale: dict[str, str] = field(default_factory=dict)
    supporting_observable_evidence: dict[str, str] = field(default_factory=dict)
    trigger_match_reason: dict[str, str] = field(default_factory=dict)
    ignored_skill_reasons: dict[str, str] = field(default_factory=dict)
    no_skill_needed: bool = False
    source: str = "none"
    raw_response: str = ""
    parse_error: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "selected_skill_ids": self.selected_skill_ids,
            "ignored_skill_ids": self.ignored_skill_ids,
            "selection_rationale": self.selection_rationale,
            "supporting_observable_evidence": self.supporting_observable_evidence,
            "trigger_match_reason": self.trigger_match_reason,
            "ignored_skill_reasons": self.ignored_skill_reasons,
            "no_skill_needed": self.no_skill_needed,
            "source": self.source,
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
        }


def active_agent_skills(
    registry: SkillRegistry,
    *,
    limit: int = 8,
    exclude_dimensions: list[str] | set[str] | tuple[str, ...] | str | None = None,
    exclude_domain_operators: list[str] | set[str] | tuple[str, ...] | str | None = None,
) -> list[Skill]:
    skills = registry.active(SkillType.AGENT_RECOVERY, limit=None)
    filtered = filter_agent_skills(
        skills,
        exclude_dimensions=exclude_dimensions,
        exclude_domain_operators=exclude_domain_operators,
    )
    return filtered[:limit]


def filter_agent_skills(
    skills: list[Skill],
    *,
    exclude_dimensions: list[str] | set[str] | tuple[str, ...] | str | None = None,
    exclude_domain_operators: list[str] | set[str] | tuple[str, ...] | str | None = None,
) -> list[Skill]:
    excluded_dimensions = normalize_filter_values(exclude_dimensions)
    excluded_domain_operators = normalize_filter_values(exclude_domain_operators)
    if not excluded_dimensions and (not excluded_domain_operators):
        return skills
    kept = []
    for skill in skills:
        if excluded_dimensions & set(skill.trigger_dimensions):
            continue
        if excluded_domain_operators & set(skill.domain_operators):
            continue
        kept.append(skill)
    return kept


def normalize_filter_values(
    values: list[str] | set[str] | tuple[str, ...] | str | None,
) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        raw_items = values.split(",")
    else:
        raw_items = list(values)
    return {str(item).strip() for item in raw_items if str(item).strip()}


def render_agent_skill_index(skills: list[Skill]) -> str:
    """Render compact agent-visible skill metadata for self-retrieval."""
    lines: list[str] = []
    for skill in skills:
        trigger = _join(skill.trigger_dimensions)
        capabilities = _join(skill.capability_targets)
        operators = _join(skill.domain_operators)
        procedure = _first_line(skill.procedure)
        criteria = _first_line(skill.success_criteria)
        lines.extend(
            [
                f"- skill_id: {skill.skill_id}",
                f"  title: {_sanitize_agent_visible_text(skill.name)}",
                f"  observable_trigger: {_trigger_summary(trigger, operators)}",
                f"  capability_targets: {capabilities}",
                f"  domain_operators: {operators}",
                f"  short_procedure: {_sanitize_agent_visible_text(procedure)}",
                f"  success_signal: {_sanitize_agent_visible_text(criteria)}",
            ]
        )
    return "\n".join(lines)


def render_selected_agent_skill_blocks(
    registry: SkillRegistry, selected_skill_ids: list[str]
) -> str:
    """Render full prompt blocks only for agent-selected skills."""
    by_id = {
        skill.skill_id: skill
        for skill in registry.all()
        if skill.skill_type == SkillType.AGENT_RECOVERY
        and skill.status in {SkillStatus.ACTIVE, SkillStatus.PROMOTED}
    }
    blocks = []
    for skill_id in selected_skill_ids:
        skill = by_id.get(skill_id)
        if skill is not None:
            blocks.append(_sanitize_agent_visible_text(skill.prompt_block()))
    if not blocks:
        return ""
    return "\n\n".join(
        ["Relevant recovery guidance selected from prior planning experience:", *blocks]
    )


def build_agent_retrieval_prompt(
    *, query: str, environment_observation: str, skill_index: str, max_skills: int
) -> str:
    return AGENT_RETRIEVAL_USER_PROMPT.format(
        query=query,
        environment_observation=environment_observation,
        skill_index=skill_index,
        max_skills=max_skills,
    )


def parse_retrieval_decision(
    text: str, *, available_skill_ids: set[str], max_skills: int, source: str = "llm"
) -> SkillRetrievalDecision:
    raw = text.strip()
    try:
        payload = _extract_json(raw)
        selected = _clean_ids(payload.get("selected_skill_ids"), available_skill_ids, max_skills)
        ignored = _clean_ids(payload.get("ignored_skill_ids"), available_skill_ids, None)
        rationale = _clean_text_map(payload.get("selection_rationale"), available_skill_ids)
        evidence = _clean_text_map(payload.get("supporting_observable_evidence"), set(selected))
        trigger_reason = _clean_text_map(payload.get("trigger_match_reason"), set(selected))
        ignored_reasons = _clean_text_map(payload.get("ignored_skill_reasons"), available_skill_ids)
        return SkillRetrievalDecision(
            selected_skill_ids=selected,
            ignored_skill_ids=[skill_id for skill_id in ignored if skill_id not in selected],
            selection_rationale=rationale,
            supporting_observable_evidence=evidence,
            trigger_match_reason=trigger_reason,
            ignored_skill_reasons={
                skill_id: reason
                for skill_id, reason in ignored_reasons.items()
                if skill_id not in selected
            },
            no_skill_needed=bool(payload.get("no_skill_needed")) or not selected,
            source=source,
            raw_response=raw,
        )
    except Exception as exc:
        return SkillRetrievalDecision(
            no_skill_needed=True, source=source, raw_response=raw, parse_error=str(exc)
        )


def heuristic_retrieval_decision(
    *, query: str, environment_observation: str, skills: list[Skill], max_skills: int
) -> SkillRetrievalDecision:
    """Local fallback that uses observable text overlap, not benchmark labels."""
    observable = f"{query}\n{environment_observation}".lower()
    scored: list[tuple[int, Skill]] = []
    for skill in skills:
        terms = _observable_terms(skill)
        score = sum((1 for term in terms if term and term in observable))
        if score:
            scored.append((score, skill))
    scored.sort(
        key=lambda item: (
            item[0],
            item[1].status == SkillStatus.PROMOTED,
            float(item[1].metrics.get("evidence_count", 0)),
        ),
        reverse=True,
    )
    selected = [skill.skill_id for _, skill in scored[:max_skills]]
    selected_set = set(selected)
    evidence = {
        skill.skill_id: _matched_observable_evidence(skill, observable)
        for _, skill in scored[:max_skills]
    }
    return SkillRetrievalDecision(
        selected_skill_ids=selected,
        ignored_skill_ids=[
            skill.skill_id for skill in skills if skill.skill_id not in selected_set
        ],
        selection_rationale={
            skill_id: "Selected because its observable trigger terms appear in the current task state."
            for skill_id in selected
        },
        supporting_observable_evidence=evidence,
        trigger_match_reason={
            skill_id: "The current observable task state contains terms matching this guidance trigger."
            for skill_id in selected
        },
        ignored_skill_reasons={
            skill.skill_id: "No observable trigger terms for this guidance were found in the current task state."
            for skill in skills
            if skill.skill_id not in selected_set
        },
        no_skill_needed=not selected,
        source="heuristic",
    )


def _extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    fenced = re.search("```(?:json)?\\s*(\\{.*?\\})\\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    if not candidate.strip().startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("retrieval response is not a JSON object")
    return parsed


def _clean_ids(value: Any, available_skill_ids: set[str], limit: int | None) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        skill_id = str(item)
        if skill_id in available_skill_ids and skill_id not in cleaned:
            cleaned.append(skill_id)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _clean_text_map(value: Any, available_skill_ids: set[str]) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, str] = {}
    for skill_id, reason in value.items():
        key = str(skill_id)
        if key in available_skill_ids:
            text = _sanitize_agent_visible_text(str(reason)).strip()
            if text:
                cleaned[key] = text[:600]
    return cleaned


def _matched_observable_evidence(skill: Skill, observable: str) -> str:
    matched = [term for term in sorted(_observable_terms(skill)) if term and term in observable]
    if not matched:
        return ""
    return "Matched observable terms: " + ", ".join(matched[:5])


def _observable_terms(skill: Skill) -> set[str]:
    raw_terms = [
        *skill.capability_targets,
        *skill.domain_operators,
        *skill.trigger_dimensions,
        skill.name,
        *skill.procedure[:2],
        *skill.success_criteria[:1],
    ]
    terms: set[str] = set()
    replacements = {
        "selected_transport_unavailable": ["flight", "transport", "cancelled", "unavailable"],
        "selected_poi_unavailable": ["restaurant", "attraction", "closed", "unavailable"],
        "selected_lodging_unavailable": ["hotel", "accommodation", "full", "unavailable"],
        "irrelevant_environment_noise": ["irrelevant", "noise", "unrelated"],
        "avoid_invalid_resource": ["unavailable", "cancelled", "closed", "full"],
        "cascading_replan": ["replan", "arrival", "schedule", "budget"],
        "change_detection": ["update", "unavailable", "cancelled", "closed", "full"],
        "local_substitution": ["alternative", "replace", "substitute"],
        "overreaction_control": ["irrelevant", "unrelated", "noise"],
        "tool_result_missing": [
            "tool result missing",
            "tool result is missing",
            "tool result empty",
            "empty result",
            "missing result",
            "no result",
            "工具结果为空",
            "工具结果缺失",
            "搜索结果为空",
            "检索结果为空",
            "返回空结果",
            "未找到结果",
            "没有找到结果",
            "未查询到",
        ],
        "tool_robustness": [
            "tool result missing",
            "tool result empty",
            "empty result",
            "missing result",
            "工具结果为空",
            "搜索结果为空",
            "检索结果为空",
            "返回空结果",
            "未找到结果",
            "未查询到",
        ],
        "user_communication": ["explain", "communicate", "tell user"],
    }
    for term in raw_terms:
        normalized = str(term).replace("_", " ").replace("-", " ").lower()
        if normalized:
            terms.add(normalized)
        for mapped in replacements.get(str(term), []):
            terms.add(mapped)
    return terms


def _trigger_summary(trigger: str, operators: str) -> str:
    parts = [part for part in [trigger, operators] if part and part != "general"]
    if not parts:
        return "Use only when the current observable task state matches this guidance."
    return _sanitize_agent_visible_text(
        "Use when current observations or tool results indicate: " + "; ".join(parts)
    )


def _join(items: list[str]) -> str:
    return ", ".join(items) if items else "general"


def _first_line(items: list[str]) -> str:
    return items[0] if items else "Use observe-diagnose-replan-verify-communicate recovery."


def _sanitize_agent_visible_text(text: str) -> str:
    replacements = {
        "counterfactual notice": "latest environment observation",
        "counterfactual change": "current environment update",
        "counterfactual failures": "planning and recovery failures",
        "counterfactual failure": "planning and recovery failure",
        "counterfactual noise": "environment noise",
        "current counterfactual": "current observable task state",
        "under any counterfactual change": "under the current task state",
        "counterfactual": "environment update",
        "intervention notice": "latest environment observation",
        "intervention": "environment update",
        "evaluator": "system",
        "judge": "review",
        "benchmark": "task set",
        "posthoc": "after-action",
        "modified environment state": "current environment state",
        "modified data": "current data",
        "changed world": "current environment state",
        "changed travel state": "current travel state",
        "changed travel fact": "current travel fact",
        "changed fact": "current environment fact",
    }
    sanitized = text
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)
        sanitized = sanitized.replace(old.capitalize(), new.capitalize())
    return sanitized


sanitize_agent_visible_text = _sanitize_agent_visible_text
