from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from cce_mobility.core.schema import (
    AgentTrajectory,
    AgentVisibleHistoryEvent,
    FactualContextArtifact,
    FactualRecoveryContext,
    RunMode,
    TaskSample,
)


def build_factual_recovery_context(
    sample: TaskSample,
    factual: AgentTrajectory,
) -> FactualRecoveryContext:
    """Build a deterministic, agent-visible factual prefix for recovery."""

    return _build_context_from_fields(
        sample_id=sample.sample_id,
        goal=sample.query,
        factual=factual,
    )


def validate_factual_recovery_context(
    sample: TaskSample,
    context: FactualRecoveryContext,
) -> None:
    """Reject cross-sample, changed-goal, or mutated factual prefixes."""

    if context.sample_id != sample.sample_id:
        raise ValueError(
            "Recovery context sample does not match current sample: "
            f"{context.sample_id!r} != {sample.sample_id!r}"
        )
    if context.goal != sample.query:
        raise ValueError("Recovery context must preserve the original user goal")
    payload = _hash_payload(
        schema_version=context.schema_version,
        sample_id=context.sample_id,
        goal=context.goal,
        factual_plan=context.factual_plan,
        tool_history=context.tool_history,
    )
    if context.prefix_hash != _stable_hash(payload):
        raise ValueError("Recovery context prefix hash does not match its visible content")


def render_factual_recovery_context(context: FactualRecoveryContext) -> str:
    """Render only the frozen fields an agent previously observed or produced."""

    lines = [
        "Existing planning context to continue from:",
        "Original user request (unchanged):",
        context.goal,
        "",
        "Previously produced plan:",
        context.factual_plan or "(no prior plan was produced)",
        "",
        "Prior visible tool interactions:",
    ]
    if not context.tool_history:
        lines.append("(no prior tool interactions were recorded)")
    for event in context.tool_history:
        lines.append(f"Step {event.step} - {event.tool_name or 'agent action'}")
        if event.parameters:
            lines.append(
                "Parameters: "
                + json.dumps(event.parameters, ensure_ascii=False, sort_keys=True, default=str)
            )
        if event.action_text and event.action_text.strip() != context.factual_plan.strip():
            lines.append(f"Action: {event.action_text}")
        if event.observation:
            lines.append(f"Observation: {event.observation}")
        if not event.success:
            lines.append("Outcome: unsuccessful")
    lines.extend(
        [
            "",
            "Continue from this existing context. Preserve still-valid choices and revise "
            "only choices invalidated by current observable evidence or their dependencies.",
        ]
    )
    return "\n".join(lines).strip()


def build_factual_context_artifact(
    *,
    iteration: int,
    sample: TaskSample,
    factual: AgentTrajectory,
    recovery_context: FactualRecoveryContext,
) -> FactualContextArtifact:
    """Bind a full factual trajectory to its frozen, agent-visible prefix."""

    validate_factual_recovery_context(sample, recovery_context)
    rebuilt_context = build_factual_recovery_context(sample, factual)
    if rebuilt_context != recovery_context:
        raise ValueError("Factual trajectory does not match the supplied recovery context")
    payload = _artifact_hash_payload(
        iteration=iteration,
        sample_id=sample.sample_id,
        goal=sample.query,
        factual=factual,
        recovery_context=recovery_context,
        prefix_hash=recovery_context.prefix_hash,
    )
    return FactualContextArtifact(
        **payload,
        artifact_hash=_stable_hash(payload),
    )


def validate_factual_context_artifact(
    artifact: FactualContextArtifact,
    *,
    sample: TaskSample | None = None,
    iteration: int | None = None,
) -> None:
    """Validate artifact integrity and, when supplied, its requested lookup key."""

    if artifact.sample_id != artifact.factual.sample_id:
        raise ValueError("Frozen factual artifact sample does not match its trajectory")
    if artifact.sample_id != artifact.recovery_context.sample_id:
        raise ValueError("Frozen factual artifact sample does not match its recovery context")
    if artifact.goal != artifact.recovery_context.goal:
        raise ValueError("Frozen factual artifact goal does not match its recovery context")
    if artifact.prefix_hash != artifact.recovery_context.prefix_hash:
        raise ValueError("Frozen factual artifact prefix hash is inconsistent")

    reconstructed = _build_context_from_fields(
        sample_id=artifact.sample_id,
        goal=artifact.goal,
        factual=artifact.factual,
    )
    if reconstructed != artifact.recovery_context:
        raise ValueError("Frozen factual trajectory does not match its recovery context")

    payload = _artifact_hash_payload(
        iteration=artifact.iteration,
        sample_id=artifact.sample_id,
        goal=artifact.goal,
        factual=artifact.factual,
        recovery_context=artifact.recovery_context,
        prefix_hash=artifact.prefix_hash,
    )
    if artifact.artifact_hash != _stable_hash(payload):
        raise ValueError("Frozen factual artifact hash does not match its content")

    if iteration is not None and artifact.iteration != iteration:
        raise ValueError(
            "Frozen factual artifact iteration does not match requested iteration: "
            f"{artifact.iteration} != {iteration}"
        )
    if sample is not None:
        if artifact.sample_id != sample.sample_id:
            raise ValueError(
                "Frozen factual artifact sample does not match requested sample: "
                f"{artifact.sample_id!r} != {sample.sample_id!r}"
            )
        if artifact.goal != sample.query:
            raise ValueError("Frozen factual artifact goal does not match current sample goal")
        validate_factual_recovery_context(sample, artifact.recovery_context)


def load_factual_context_artifacts(
    path: str | Path,
) -> dict[tuple[int, str], FactualContextArtifact]:
    """Strictly load a versioned JSONL factual-context artifact set."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Frozen factual context artifact not found: {source}")
    artifacts: dict[tuple[int, str], FactualContextArtifact] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                artifact = FactualContextArtifact.model_validate_json(line)
            except (ValidationError, ValueError) as exc:
                raise ValueError(
                    f"Invalid frozen factual context artifact at {source}:{line_number}: {exc}"
                ) from exc
            try:
                validate_factual_context_artifact(artifact)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid frozen factual context artifact at {source}:{line_number}: {exc}"
                ) from exc
            key = (artifact.iteration, artifact.sample_id)
            if key in artifacts:
                raise ValueError(
                    "Duplicate frozen factual context key "
                    f"iteration={artifact.iteration}, sample_id={artifact.sample_id!r} "
                    f"in {source}"
                )
            artifacts[key] = artifact
    return artifacts


def append_factual_context_artifact(
    path: str | Path,
    artifact: FactualContextArtifact,
) -> None:
    validate_factual_context_artifact(artifact)
    target = Path(path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(artifact.model_dump_json() + "\n")


def _hash_payload(
    *,
    schema_version: str,
    sample_id: str,
    goal: str,
    factual_plan: str,
    tool_history: list[AgentVisibleHistoryEvent],
) -> dict:
    return {
        "schema_version": schema_version,
        "sample_id": sample_id,
        "goal": goal,
        "factual_plan": factual_plan,
        "tool_history": [event.model_dump(mode="json") for event in tool_history],
    }


def _build_context_from_fields(
    *,
    sample_id: str,
    goal: str,
    factual: AgentTrajectory,
) -> FactualRecoveryContext:
    if factual.sample_id != sample_id:
        raise ValueError(
            "Factual trajectory sample does not match recovery sample: "
            f"{factual.sample_id!r} != {sample_id!r}"
        )
    if factual.run_mode != RunMode.FACTUAL:
        raise ValueError("Recovery context must be built from a factual trajectory")
    history = [
        AgentVisibleHistoryEvent(
            step=event.step,
            tool_name=event.tool_name,
            parameters=event.parameters,
            observation=event.observation,
            action_text=event.action_text,
            success=event.success,
        )
        for event in factual.events
        if event.metadata.get("agent_visible") is not False
    ]
    payload = _hash_payload(
        schema_version="factual-recovery-context-v1",
        sample_id=sample_id,
        goal=goal,
        factual_plan=factual.final_response,
        tool_history=history,
    )
    return FactualRecoveryContext(**payload, prefix_hash=_stable_hash(payload))


def _artifact_hash_payload(
    *,
    iteration: int,
    sample_id: str,
    goal: str,
    factual: AgentTrajectory,
    recovery_context: FactualRecoveryContext,
    prefix_hash: str,
) -> dict:
    return {
        "schema_version": "factual-context-artifact-v1",
        "iteration": iteration,
        "sample_id": sample_id,
        "goal": goal,
        "factual": factual.model_dump(mode="json"),
        "recovery_context": recovery_context.model_dump(mode="json"),
        "prefix_hash": prefix_hash,
    }


def _stable_hash(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
