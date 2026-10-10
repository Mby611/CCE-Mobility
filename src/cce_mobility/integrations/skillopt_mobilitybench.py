from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from skillopt.datasets.base import BatchSpec, SplitDataLoader
from skillopt.envs.base import EnvAdapter
from skillopt.model import chat_target
from skillopt.prompts import load_prompt

from cce_mobility.core.config import ModelConfig
from cce_mobility.core.llm import ChatClient
from cce_mobility.core.schema import (
    AgentTrajectory,
    EvaluationRecord,
    ExpectedRecoverySpec,
    InterventionSpec,
    RunMode,
    TaskSample,
    ToolEvent,
)
from cce_mobility.judges.spec_llm import SpecGroundedLLMJudge
from cce_mobility.skills.retrieval import sanitize_agent_visible_text


MOBILITY_ANALYST_CONSTRAINTS = """

## MobilityBench recovery constraints
- Use scored rollouts, hard constraints, environment updates, and recovery contracts as evidence.
- Do not encode sample IDs, exact resource IDs, benchmark answers, or evaluator terminology.
- Generalize around observable-state checking, invalid-resource avoidance,
  constraint propagation, substitution, communication, and overreaction control.
"""


RECOVERY_SYSTEM_PROMPT = """You are a mobility assistant operating in a changing task environment.

Use only the current user request, the previous task-state summary, the latest
environment update, and any reusable recovery guidance. Detect changed task
conditions, avoid invalidated information, and give a concise recovery response
or targeted clarification to the user."""


class CCEMobilityBenchDataLoader(SplitDataLoader):
    pass


class CCEMobilityBenchSkillOptAdapter(EnvAdapter):
    """Official SkillOpt environment backed by official MobilityBench traces."""

    def __init__(
        self,
        split_dir: str = "",
        split_mode: str = "split_dir",
        split_seed: int = 42,
        seed: int = 42,
        limit: int = 0,
        workers: int = 1,
        analyst_workers: int = 1,
        failure_only: bool = False,
        minibatch_size: int = 4,
        edit_budget: int = 4,
        max_completion_tokens: int = 2048,
        judge_model: str = "",
        judge_chat_url: str = "",
        judge_api_key_env: str = "CCE_JUDGE_API_KEY",
        judge_call_delay_seconds: float = 0.5,
        judge_api_max_retries: int = 5,
        judge_api_retry_backoff_seconds: float = 10.0,
        target_call_delay_seconds: float = 0.5,
    ) -> None:
        if not judge_model or not judge_chat_url:
            raise ValueError("SkillOpt MB requires judge_model and judge_chat_url in its config")
        self.workers = max(1, int(workers))
        self.analyst_workers = max(1, int(analyst_workers))
        self.failure_only = bool(failure_only)
        self.minibatch_size = int(minibatch_size)
        self.edit_budget = int(edit_budget)
        self.max_completion_tokens = int(max_completion_tokens)
        self.target_call_delay_seconds = max(0.0, float(target_call_delay_seconds))
        self.dataloader = CCEMobilityBenchDataLoader(
            split_dir=split_dir,
            split_mode=split_mode,
            split_seed=split_seed,
            seed=seed,
            limit=limit,
        )
        judge_client = ChatClient(
            ModelConfig(
                model=judge_model,
                provider="openai",
                chat_url=judge_chat_url,
                api_key_env=judge_api_key_env,
                temperature=0.0,
                max_tokens=2048,
                extra={
                    "enable_thinking": False,
                    "api_call_delay_seconds": float(judge_call_delay_seconds),
                    "api_max_retries": int(judge_api_max_retries),
                    "api_retry_backoff_seconds": float(judge_api_retry_backoff_seconds),
                },
            )
        )
        self.judge = SpecGroundedLLMJudge(judge_client)

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        return list(self.dataloader.build_train_batch(batch_size, seed).payload or [])

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        return list(self.dataloader.build_eval_batch(env_num, split, seed).payload or [])

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict[str, Any]]:
        items = list(env_manager or [])
        results = self._rollout_items(items, skill_content)
        if results and all(
            str(row.get("predicted_answer", "")).startswith("[LLM_ERROR]") for row in results
        ):
            raise RuntimeError(
                "All SkillOpt target rollouts failed; refusing to optimize invalid "
                f"data. Examples: {_failure_examples(results, 'predicted_answer')}"
            )
        if results and all(
            str(row.get("judge_rationale", "")).startswith("LLM judge failed:") for row in results
        ):
            raise RuntimeError(
                "All CCE judge calls failed; refusing to optimize fallback-only "
                f"scores. Examples: {_failure_examples(results, 'judge_rationale')}"
            )
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in results:
            _write_prediction_artifacts(output_dir, row)
        (output_dir / "rollouts.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return results

    def _rollout_items(
        self,
        items: list[dict[str, Any]],
        skill_content: str,
    ) -> list[dict[str, Any]]:
        if self.workers <= 1 or len(items) <= 1:
            return [self._rollout_one(item, skill_content) for item in items]
        ordered: list[dict[str, Any] | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._rollout_one, item, skill_content): index
                for index, item in enumerate(items)
            }
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        return [row for row in ordered if row is not None]

    def get_task_types(self) -> list[str]:
        values = []
        for item in self.dataloader.train_items + self.dataloader.val_items:
            task_type = str(item.get("task_type") or "mobility_recovery")
            if task_type not in values:
                values.append(task_type)
        return values or ["mobility_recovery"]

    def get_error_minibatch_prompt(self) -> str:
        return load_prompt("analyst_error") + MOBILITY_ANALYST_CONSTRAINTS

    def get_success_minibatch_prompt(self) -> str:
        return load_prompt("analyst_success") + MOBILITY_ANALYST_CONSTRAINTS

    def build_reference_text(self, item: dict) -> str:
        recovery = item.get("recovery_spec") or {}
        return json.dumps(
            {
                "summary": recovery.get("summary"),
                "must_do": recovery.get("must_do", []),
                "must_not_do": recovery.get("must_not_do", []),
                "rubric": recovery.get("rubric", {}),
            },
            ensure_ascii=False,
        )

    def _rollout_one(self, item: dict[str, Any], skill_content: str) -> dict[str, Any]:
        sample = TaskSample.model_validate(item["sample"])
        factual = AgentTrajectory.model_validate(item["factual"])
        intervention = InterventionSpec.model_validate(item["intervention"])
        recovery_spec = ExpectedRecoverySpec.model_validate(item["recovery_spec"])

        counterfactual_text, target_usage = _generate_recovery_response(
            query=sample.query,
            factual=factual,
            intervention=intervention,
            skill_content=skill_content,
            max_completion_tokens=self.max_completion_tokens,
            call_delay_seconds=self.target_call_delay_seconds,
        )
        counterfactual = _trajectory(
            sample=sample,
            factual=factual,
            intervention=intervention,
            text=counterfactual_text,
            usage=target_usage,
        )
        hard_metrics = _posthoc_hard_metrics(counterfactual, intervention)
        record = EvaluationRecord(
            sample=sample,
            factual=factual,
            counterfactual=counterfactual,
            intervention=intervention,
            recovery_spec=recovery_spec,
            metadata={
                **dict(item.get("metadata") or {}),
                "counterfactual_hard_metrics": hard_metrics,
            },
        )
        score = self.judge.score(record)
        llm_error = bool(counterfactual.raw.get("llm_error"))
        parse_success = bool(hard_metrics.get("parse_success"))
        hard_pass = bool(hard_metrics.get("hard_pass"))
        unavailable = bool(hard_metrics.get("uses_unavailable_identifier"))
        hard = int(score.total >= 3.0 and parse_success and hard_pass and not llm_error)
        recovery_quality = max(0.0, min(1.0, float(score.total) / 4.0))
        constraint_quality = (
            sum(
                (
                    float(parse_success),
                    float(bool(hard_metrics.get("commonsense_pass"))),
                    float(hard_pass),
                    float(not unavailable),
                )
            )
            / 4.0
        )
        soft = 0.85 * recovery_quality + 0.15 * constraint_quality

        target_system_prompt, target_user_prompt = _build_recovery_prompts(
            query=sample.query,
            factual=factual,
            intervention=intervention,
            skill_content=skill_content,
        )
        reference_text = self.build_reference_text(item)
        fail_reason = ""
        if not hard:
            fail_reason = (
                f"recovery_score={score.total:.3f}; parse_success={parse_success}; "
                f"hard_pass={hard_pass}; llm_error={llm_error}"
            )

        return {
            "id": str(item["id"]),
            "hard": hard,
            "soft": round(soft, 6),
            "question": sample.query,
            "task_description": sample.query,
            "predicted_answer": counterfactual_text,
            "response": counterfactual_text,
            "factual_answer": factual.final_response,
            "task_type": str(item.get("task_type") or intervention.domain_operator),
            "sample_id": sample.sample_id,
            "intervention_id": intervention.intervention_id,
            "dimension": intervention.dimension,
            "domain_operator": intervention.domain_operator,
            "capability_targets": intervention.capability_targets,
            "environment_update": _agent_visible_update(intervention),
            "judge_total": score.total,
            "judge_strategy": score.strategy.value,
            "judge_rationale": score.rationale,
            "hard_metrics": hard_metrics,
            "fail_reason": fail_reason,
            "n_turns": 1,
            "agent_ok": not llm_error,
            "reference_text": reference_text,
            "target_system_prompt": target_system_prompt,
            "target_user_prompt": target_user_prompt,
            "target_usage": target_usage,
        }


def _generate_recovery_response(
    *,
    query: str,
    factual: AgentTrajectory,
    intervention: InterventionSpec,
    skill_content: str,
    max_completion_tokens: int,
    call_delay_seconds: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    system, user = _build_recovery_prompts(
        query=query,
        factual=factual,
        intervention=intervention,
        skill_content=skill_content,
    )
    try:
        if call_delay_seconds > 0:
            time.sleep(call_delay_seconds)
        text, usage = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            stage="cce_mobilitybench_target",
        )
    except Exception as exc:
        return f"[LLM_ERROR] {exc}", {}
    return str(text).strip(), dict(usage or {})


def _failure_examples(
    rows: list[dict[str, Any]],
    key: str,
    *,
    limit: int = 3,
    width: int = 500,
) -> str:
    examples: list[str] = []
    for row in rows:
        text = str(row.get(key) or "").strip()
        if not text:
            continue
        text = text.replace("\n", " ")
        if key == "judge_rationale":
            text = text.removeprefix("LLM judge failed:").split("; fallback:", 1)[0].strip()
        if text not in examples:
            examples.append(text[:width])
        if len(examples) >= limit:
            break
    return " | ".join(examples) if examples else "(none)"


def _build_recovery_prompts(
    *,
    query: str,
    factual: AgentTrajectory,
    intervention: InterventionSpec,
    skill_content: str,
) -> tuple[str, str]:
    system = RECOVERY_SYSTEM_PROMPT
    safe_skill = sanitize_agent_visible_text(skill_content).strip()
    parts = [
        f"User request:\n{query}",
        f"Previous task-state summary:\n{_trajectory_summary(factual)}",
        f"Current environment update:\n{_agent_visible_update(intervention)}",
    ]
    if safe_skill:
        parts.append(f"Reusable recovery guidance:\n{safe_skill}")
    parts.append(
        "Continue assisting the user under the current environment. Explain the "
        "changed condition and provide a feasible recovery or clarification."
    )
    return system, "\n\n".join(parts)


def _trajectory_summary(trajectory: AgentTrajectory, limit: int = 4000) -> str:
    parts = []
    for event in trajectory.events:
        if event.metadata.get("official_role") == "planner":
            parts.append(
                "Planner:\n"
                f"- intent: {event.parameters.get('intent', '')}\n"
                f"- steps: {event.parameters.get('steps', [])}\n"
                f"- thinking: {event.action_text[:700]}"
            )
        elif event.metadata.get("official_role") == "worker":
            parts.append(
                "Worker/tool:\n"
                f"- tool: {event.tool_name}\n"
                f"- args: {event.parameters}\n"
                f"- observation: {event.observation[:900]}"
            )
        elif event.metadata.get("official_role") == "reporter":
            parts.append(f"Reporter final response:\n{event.action_text[:900]}")
    if not parts:
        parts.append(trajectory.final_response[:1200])
    return "\n\n".join(parts)[:limit]


def _agent_visible_update(intervention: InterventionSpec) -> str:
    text = intervention.condition_text or intervention.diagnostic_goal or intervention.dimension
    return (
        text.replace("counterfactual", "current")
        .replace("Counterfactual", "Current")
        .replace("intervention", "environment update")
        .replace("Intervention", "Environment update")
        .replace("evaluator", "system")
        .replace("Evaluator", "System")
        .replace("judge", "reviewer")
        .replace("Judge", "Reviewer")
        .replace("benchmark", "task set")
        .replace("Benchmark", "Task set")
        .replace("posthoc", "after-action")
        .replace("Posthoc", "After-action")
    )


def _posthoc_hard_metrics(cf: AgentTrajectory, intervention: InterventionSpec) -> dict[str, Any]:
    text = cf.final_response.strip()
    lower = text.lower()
    llm_error = bool(cf.raw.get("llm_error"))
    parse_success = bool(text) and not llm_error
    detects = any(
        token in lower
        for token in [
            "changed",
            "closed",
            "blocked",
            "unavailable",
            "delay",
            "empty",
            "conflict",
            "不可用",
            "关闭",
            "封闭",
            "拥堵",
            "延误",
            "变化",
            "无法",
            "没有结果",
        ]
    )
    recovers = any(
        token in lower
        for token in [
            "alternative",
            "replan",
            "instead",
            "verify",
            "clarification",
            "替代",
            "重新规划",
            "建议",
            "确认",
            "换",
        ]
    )
    negative = intervention.is_negative
    hard_pass = parse_success and ((not negative and (detects or recovers)) or negative)
    return {
        "benchmark": "mobilitybench",
        "run_mode": "counterfactual",
        "adapter_scope": "official_trace_posthoc_environment_update",
        "official_score": None,
        "parse_success": parse_success,
        "commonsense_pass": parse_success,
        "hard_pass": hard_pass,
        "final_pass": hard_pass,
        "uses_unavailable_identifier": False,
        "unavailable_id_count": 0,
        "estimated_cost": 0.0,
        "llm_error": llm_error,
        "llm_error_message": cf.raw.get("llm_error_message", ""),
        "detected_environment_update": detects,
        "showed_recovery": recovers,
        "intervention_dimension": intervention.dimension,
    }


def _trajectory(
    *,
    sample: TaskSample,
    factual: AgentTrajectory,
    intervention: InterventionSpec,
    text: str,
    usage: dict[str, Any],
) -> AgentTrajectory:
    error = text.startswith("[LLM_ERROR]")
    return AgentTrajectory(
        sample_id=sample.sample_id,
        run_mode=RunMode.COUNTERFACTUAL,
        events=[
            *factual.events,
            ToolEvent(
                step=len(factual.events),
                tool_name="official_skillopt_mb_recovery",
                observation=f"[ENVIRONMENT_UPDATE] {_agent_visible_update(intervention)}",
                action_text=text,
                success=not error,
                metadata={
                    "agent_visible": True,
                    "adapter_scope": "official_skillopt_mb",
                    "intervention_id": intervention.intervention_id,
                    "intervention_dimension": intervention.dimension,
                },
            ),
        ],
        final_response=text,
        raw={
            **factual.raw,
            "adapter": "official_skillopt_cce_mobilitybench",
            "adapter_scope": "official_trace_posthoc_environment_update",
            "llm_error": error,
            "llm_error_message": text.removeprefix("[LLM_ERROR]").strip() if error else "",
            "target_usage": usage,
            "intervention_dimension": intervention.dimension,
        },
    )


def _write_prediction_artifacts(output_dir: Path, row: dict[str, Any]) -> None:
    prediction_dir = output_dir / "predictions" / str(row["id"])
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "target_system_prompt.txt").write_text(
        str(row.get("target_system_prompt") or ""),
        encoding="utf-8",
    )
    (prediction_dir / "target_user_prompt.txt").write_text(
        str(row.get("target_user_prompt") or ""),
        encoding="utf-8",
    )
    verification = {
        "judge_total": row.get("judge_total"),
        "judge_strategy": row.get("judge_strategy"),
        "judge_rationale": row.get("judge_rationale"),
        "hard": row.get("hard"),
        "soft": row.get("soft"),
        "hard_metrics": row.get("hard_metrics"),
        "expected_recovery": row.get("reference_text"),
    }
    conversation = [
        {"role": "user", "content": str(row.get("target_user_prompt") or "")},
        {"role": "assistant", "content": str(row.get("predicted_answer") or "")},
        {
            "role": "system",
            "content": "[EVALUATION RESULT]\n" + json.dumps(verification, ensure_ascii=False),
        },
    ]
    (prediction_dir / "conversation.json").write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
