from __future__ import annotations

from cce_mobility.core.schema import EvaluationRecord, JudgeScore, RecoveryStrategy


class HeuristicRecoveryJudge:
    """Transparent recovery judge used for smoke tests.

    This is not the final paper judge. It makes the pipeline runnable and
    documents the scoring dimensions that an LLM/human judge should follow.
    """

    CHANGE_WORDS = (
        "closed",
        "cancel",
        "unavailable",
        "traffic",
        "delay",
        "surge",
        "warning",
        "blocked",
        "weather",
        "封闭",
        "关闭",
        "停运",
        "取消",
        "不可用",
        "不可达",
        "拥堵",
        "堵车",
        "延误",
        "天气",
        "限行",
        "变更",
    )
    RECOVERY_WORDS = (
        "alternative",
        "instead",
        "replan",
        "adjust",
        "replace",
        "switch",
        "verify",
        "confirm",
        "替代",
        "备选",
        "改走",
        "绕行",
        "重新规划",
        "重规划",
        "调整",
        "更换",
        "避开",
        "核实",
        "确认",
    )

    def score(self, record: EvaluationRecord) -> JudgeScore:
        trajectory = record.counterfactual or record.factual
        text = ""
        if trajectory:
            text = trajectory.final_response.lower()

        intervention_id = record.intervention.intervention_id if record.intervention else None
        negative = bool(record.intervention and record.intervention.is_negative)
        detects = any(w in text for w in self.CHANGE_WORDS)
        hard_metrics = record.metadata.get(
            "counterfactual_hard_metrics" if record.counterfactual else "factual_hard_metrics",
            {},
        )
        implicit_recovery = bool(
            hard_metrics.get("final_pass") and not hard_metrics.get("uses_unavailable_identifier")
        )
        recovers = any(w in text for w in self.RECOVERY_WORDS) or implicit_recovery

        if negative:
            strategy = RecoveryStrategy.IGNORE if not recovers else RecoveryStrategy.PARTIAL_ADAPT
            total = 4.0 if strategy == RecoveryStrategy.IGNORE else 2.0
            return JudgeScore(
                sample_id=record.sample.sample_id,
                intervention_id=intervention_id,
                total=total,
                detection=1.0,
                diagnosis=1.0,
                replanning=0.0 if strategy == RecoveryStrategy.IGNORE else 0.5,
                communication=0.5,
                recovery=1.0 if strategy == RecoveryStrategy.IGNORE else 0.5,
                safety=1.0,
                strategy=strategy,
                rationale="Negative intervention judged by overreaction heuristic.",
            )

        if detects and recovers:
            strategy = RecoveryStrategy.PROACTIVE_ADAPT
            total = 4.0
        elif recovers:
            strategy = RecoveryStrategy.PARTIAL_ADAPT
            total = 3.0
        elif detects:
            strategy = RecoveryStrategy.PASSIVE_NOTIFY
            total = 2.0
        else:
            strategy = RecoveryStrategy.IGNORE
            total = 1.0

        return JudgeScore(
            sample_id=record.sample.sample_id,
            intervention_id=intervention_id,
            total=total,
            detection=1.0 if detects else 0.0,
            diagnosis=1.0 if detects else 0.0,
            replanning=1.0 if recovers else 0.0,
            verification=0.5 if recovers else 0.0,
            communication=0.5 if detects else 0.0,
            recovery=1.0 if recovers else 0.0,
            safety=1.0,
            strategy=strategy,
            rationale="Heuristic recovery score from agent text plus deterministic hard metrics.",
            raw={"hard_metrics_used": bool(hard_metrics), "implicit_recovery": implicit_recovery},
        )
