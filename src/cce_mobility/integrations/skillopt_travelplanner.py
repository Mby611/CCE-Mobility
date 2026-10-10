from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from skillopt.datasets.base import BatchSpec, SplitDataLoader
from skillopt.envs.base import EnvAdapter
from skillopt.model import chat_target
from skillopt.prompts import load_prompt

from cce_mobility.benchmarks.travelplanner import (
    TRAVELPLANNER_PLANNER_PROMPT,
    TRAVELPLANNER_SYSTEM_PROMPT,
    TravelPlannerAdapter,
    _ref_info_to_text,
)
from cce_mobility.core.config import BenchmarkConfig, ModelConfig
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

## Mobility-specific constraints
- Use judge scores, hard constraints, environment updates, and recovery contracts as evidence.
- Do not encode sample IDs, exact resource IDs, benchmark answers, or evaluator terminology.
- Generalize around observable-state checking, invalid-resource avoidance,
  constraint propagation, substitution, communication, and overreaction control.
"""


class CCETravelPlannerDataLoader(SplitDataLoader):
    pass


class CCETravelPlannerSkillOptAdapter(EnvAdapter):
    """Official SkillOpt environment backed by CCE TravelPlanner recovery tasks."""

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
        max_completion_tokens: int = 4096,
        judge_model: str = "deepseek-v4-pro",
        judge_chat_url: str = "",
        judge_provider: str = "openai_compatible",
        judge_api_key_env: str = "CCE_JUDGE_API_KEY",
        judge_call_delay_seconds: float = 0.2,
        judge_api_max_retries: int = 5,
        judge_api_retry_backoff_seconds: float = 10.0,
        target_call_delay_seconds: float = 0.2,
    ) -> None:
        self.workers = max(1, int(workers))
        self.analyst_workers = max(1, int(analyst_workers))
        self.failure_only = bool(failure_only)
        self.minibatch_size = int(minibatch_size)
        self.edit_budget = int(edit_budget)
        self.max_completion_tokens = int(max_completion_tokens)
        self.target_call_delay_seconds = max(0.0, float(target_call_delay_seconds))
        self.dataloader = CCETravelPlannerDataLoader(
            split_dir=split_dir,
            split_mode=split_mode,
            split_seed=split_seed,
            seed=seed,
            limit=limit,
        )
        self.benchmark = TravelPlannerAdapter(
            BenchmarkConfig(
                name="travelplanner",
                split="train",
                extra={"planner_mode": "adaptive_rule"},
            )
        )
        judge_client = ChatClient(
            ModelConfig(
                model=judge_model,
                provider=judge_provider,
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
        self._factual_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}

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
        intervention = InterventionSpec.model_validate(item["intervention"])
        recovery_spec = ExpectedRecoverySpec.model_validate(item["recovery_spec"])
        original, effective, intervention_records = self.benchmark._effective_ref_info(
            sample,
            intervention,
        )

        skill_hash = hashlib.sha256(skill_content.encode("utf-8")).hexdigest()[:16]
        factual_key = (skill_hash, sample.sample_id)
        factual_cached = self._factual_cache.get(factual_key)
        if factual_cached is None:
            factual_cached = _generate_plan(
                query=sample.query,
                ref_info=original,
                skill_content=skill_content,
                max_completion_tokens=self.max_completion_tokens,
                call_delay_seconds=self.target_call_delay_seconds,
            )
            self._factual_cache[factual_key] = factual_cached
        factual_text, factual_usage = factual_cached
        counterfactual_text, counterfactual_usage = _generate_plan(
            query=sample.query,
            ref_info=effective,
            skill_content=skill_content,
            max_completion_tokens=self.max_completion_tokens,
            environment_update=intervention.condition_text,
            call_delay_seconds=self.target_call_delay_seconds,
        )
        factual = _trajectory(sample.sample_id, factual_text, RunMode.FACTUAL)
        counterfactual = _trajectory(sample.sample_id, counterfactual_text, RunMode.COUNTERFACTUAL)
        hard_metrics = self.benchmark.evaluate_trajectory(sample, counterfactual, intervention)
        record = EvaluationRecord(
            sample=sample,
            factual=factual,
            counterfactual=counterfactual,
            intervention=intervention,
            recovery_spec=recovery_spec,
            metadata={
                "counterfactual_hard_metrics": hard_metrics,
                "intervention_records": intervention_records,
            },
        )
        score = self.judge.score(record)
        unavailable = bool(hard_metrics.get("uses_unavailable_identifier"))
        llm_error = counterfactual_text.startswith("[LLM_ERROR]")
        parse_success = bool(hard_metrics.get("parse_success"))
        hard = int(score.total >= 3.0 and parse_success and not unavailable and not llm_error)
        recovery_quality = max(0.0, min(1.0, float(score.total) / 4.0))
        constraint_quality = (
            sum(
                (
                    float(parse_success),
                    float(not unavailable),
                    float(bool(hard_metrics.get("commonsense_pass"))),
                    float(bool(hard_metrics.get("hard_pass"))),
                )
            )
            / 4.0
        )
        soft = 0.85 * recovery_quality + 0.15 * constraint_quality

        target_system_prompt, target_user_prompt = _build_plan_prompts(
            query=sample.query,
            ref_info=effective,
            skill_content=skill_content,
            environment_update=intervention.condition_text,
        )
        reference_text = self.build_reference_text(item)
        fail_reason = ""
        if not hard:
            fail_reason = (
                f"recovery_score={score.total:.3f}; parse_success={parse_success}; "
                f"uses_unavailable_identifier={unavailable}; llm_error={llm_error}"
            )

        return {
            "id": str(item["id"]),
            "hard": hard,
            "soft": round(soft, 6),
            "question": sample.query,
            "task_description": sample.query,
            "predicted_answer": counterfactual_text,
            "response": counterfactual_text,
            "factual_answer": factual_text,
            "task_type": str(item.get("task_type") or intervention.domain_operator),
            "sample_id": sample.sample_id,
            "intervention_id": intervention.intervention_id,
            "dimension": intervention.dimension,
            "domain_operator": intervention.domain_operator,
            "capability_targets": intervention.capability_targets,
            "environment_update": intervention.condition_text,
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
            "target_usage": {
                "factual": factual_usage,
                "counterfactual": counterfactual_usage,
            },
        }


def _generate_plan(
    *,
    query: str,
    ref_info: dict[str, Any],
    skill_content: str,
    max_completion_tokens: int,
    environment_update: str = "",
    call_delay_seconds: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    system, user = _build_plan_prompts(
        query=query,
        ref_info=ref_info,
        skill_content=skill_content,
        environment_update=environment_update,
    )
    try:
        if call_delay_seconds > 0:
            time.sleep(call_delay_seconds)
        text, usage = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            stage="cce_travelplanner_target",
        )
    except Exception as exc:  # SkillOpt records failed rollouts for reflection.
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


def _build_plan_prompts(
    *,
    query: str,
    ref_info: dict[str, Any],
    skill_content: str,
    environment_update: str = "",
) -> tuple[str, str]:
    system = TRAVELPLANNER_SYSTEM_PROMPT
    if skill_content.strip():
        system += (
            "\n\nReusable mobility recovery skill:\n"
            + sanitize_agent_visible_text(skill_content).strip()
        )
    user = TRAVELPLANNER_PLANNER_PROMPT.format(
        info_text=_ref_info_to_text(ref_info),
        query=query,
    )
    if environment_update:
        update = "[ENVIRONMENT_UPDATE] " + environment_update.strip()
        user = user.replace("\nTravel Plan:", f"\n\n{update}\n\nTravel Plan:")
    return system, user


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


def _trajectory(sample_id: str, text: str, mode: RunMode) -> AgentTrajectory:
    return AgentTrajectory(
        sample_id=sample_id,
        run_mode=mode,
        events=[ToolEvent(step=0, tool_name="Planner", action_text=text)],
        final_response=text,
        raw={
            "adapter": "official_skillopt_cce_travelplanner",
            "llm_error": text.startswith("[LLM_ERROR]"),
            "llm_error_message": (
                text.removeprefix("[LLM_ERROR]").strip() if text.startswith("[LLM_ERROR]") else ""
            ),
        },
    )
