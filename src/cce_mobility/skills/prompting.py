from __future__ import annotations

from cce_mobility.skills.registry import SkillRegistry
from cce_mobility.skills.retrieval import sanitize_agent_visible_text
from cce_mobility.skills.schema import SkillType


AGENT_RETRIEVAL_BOUNDARY_NONE = "none"
OBSERVABLE_MOBILITY_BOUNDARY = "observable_mobility"


def agent_skill_catalog_prompt(registry: SkillRegistry, *, limit: int = 8) -> str:
    """Render agent recovery skills without exposing evaluator-only framing."""

    return sanitize_agent_visible_text(registry.to_prompt(SkillType.AGENT_RECOVERY, limit=limit))


def agent_self_retrieval_prompt(
    registry: SkillRegistry,
    *,
    limit: int = 8,
    boundary_mode: str = AGENT_RETRIEVAL_BOUNDARY_NONE,
) -> str:
    """Render agent recovery skills as a memory catalog plus retrieval protocol.

    The runner does not preselect skills for a scenario. Instead, the agent
    receives a small structured catalog and must decide which entries match the
    current observable task state.
    """

    catalog = agent_skill_catalog_prompt(registry, limit=limit)
    if not catalog:
        return ""
    prompt = "\n".join(
        [
            "Agent recovery skill memory catalog:",
            "Before finalizing a plan, perform self-retrieval over the current task state:",
            "1. Inspect the user request, current environment observations, and latest tool results.",
            "2. Infer the relevant capability targets, domain operator, and affected resource type.",
            "3. Select only catalog skills whose metadata matches that situation.",
            "4. Apply selected skills as recovery procedures; ignore unrelated skills.",
            "5. If no skill matches, reason from first principles: observe, diagnose, replan, verify, and communicate.",
            catalog,
        ]
    )
    boundary = agent_retrieval_boundary_prompt(boundary_mode)
    if boundary:
        prompt = f"{prompt}\n\n{boundary}"
    return sanitize_agent_visible_text(prompt)


def agent_retrieval_boundary_prompt(boundary_mode: str) -> str:
    """Render optional agent-owned retrieval boundaries for official workflows."""

    normalized = (boundary_mode or AGENT_RETRIEVAL_BOUNDARY_NONE).strip().lower()
    if normalized in {"", AGENT_RETRIEVAL_BOUNDARY_NONE}:
        return ""
    if normalized != OBSERVABLE_MOBILITY_BOUNDARY:
        raise ValueError(f"unknown agent retrieval boundary mode: {boundary_mode}")

    return "\n".join(
        [
            "Applicability boundary for recovery skill use:",
            "Use recovery guidance when the current observable state shows a tool failure, empty result, incomplete result, unavailable selected resource, or a need to distinguish confirmed alternatives from unverified possibilities.",
            "Be conservative when the user asks for route distance, route time, route properties, or preference-constrained planning. In these cases, preserve the original route or preference constraints before applying any fallback.",
            "Do not substitute a different place, route, or resource unless its availability and relevance are confirmed by the current observation or tool result.",
            "If a recovery skill is not directly supported by observable evidence, ignore it and solve from the current task state.",
            "Keep this retrieval check internal; final answers should focus on the user's travel task.",
        ]
    )


def world_model_self_retrieval_prompt(registry: SkillRegistry, *, limit: int = 8) -> str:
    """Render WM intervention specs as a diagnostic spec catalog."""

    catalog = registry.to_prompt(SkillType.INTERVENTION_SPEC, limit=limit)
    if not catalog:
        return ""
    return "\n".join(
        [
            "World-model intervention spec memory catalog:",
            "When proposing counterfactuals, perform self-retrieval:",
            "1. Inspect the factual trajectory and identify the agent's likely weak capability targets.",
            "2. Select catalog specs whose capability targets and domain operators match that weakness.",
            "3. Instantiate the selected spec with exact benchmark-grounded target_ids.",
            "4. Keep the intervention local, verifiable, recoverable, and non-leaking.",
            "5. If no spec matches, propose a new local diagnostic spec and mark it through metadata/rationale.",
            catalog,
        ]
    )
