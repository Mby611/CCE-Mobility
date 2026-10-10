from __future__ import annotations

import json
import re
from typing import Any

from cce_mobility.core.llm import ChatClient
from cce_mobility.core.schema import EvaluationRecord, JudgeScore, RecoveryStrategy
from cce_mobility.judges.recovery import HeuristicRecoveryJudge


JUDGE_PROMPT = """You are evaluating a mobility/travel assistant under a counterfactual world change.

User query:
{query}

Counterfactual intervention:
{intervention_summary}

Changed facts:
{changed_facts}

Expected recovery behavior:
Must detect:
{must_detect}

Must do:
{must_do}

Must not do:
{must_not_do}

Rubric:
{rubric}

Factual response:
{factual_response}

Counterfactual response:
{counterfactual_response}

Score the counterfactual response on these dimensions, each from 0 to 1:
- detection
- diagnosis
- replanning
- verification
- communication
- recovery
- safety

Also classify strategy as exactly one of:
proactive_adapt, partial_adapt, passive_notify, ignore, hallucinate

Output strict JSON only:
{{
  "detection": 0.0,
  "diagnosis": 0.0,
  "replanning": 0.0,
  "verification": 0.0,
  "communication": 0.0,
  "recovery": 0.0,
  "safety": 0.0,
  "strategy": "ignore",
  "rationale": "short explanation"
}}"""


class SpecGroundedLLMJudge:
    """LLM judge grounded by an ExpectedRecoverySpec."""

    def __init__(self, client: ChatClient):
        self.client = client
        self.fallback = HeuristicRecoveryJudge()

    def score(self, record: EvaluationRecord) -> JudgeScore:
        if (
            self.client.is_placeholder
            or record.recovery_spec is None
            or record.counterfactual is None
        ):
            return self.fallback.score(record)

        spec = record.recovery_spec
        intervention = record.intervention
        factual_response = record.factual.final_response if record.factual else ""
        cf_response = record.counterfactual.final_response
        prompt = JUDGE_PROMPT.format(
            query=record.sample.query,
            intervention_summary=spec.summary,
            changed_facts=_bullets(spec.changed_facts),
            must_detect=_bullets(spec.must_detect),
            must_do=_bullets(spec.must_do),
            must_not_do=_bullets(spec.must_not_do),
            rubric=_format_rubric(spec.rubric),
            factual_response=factual_response[:6000],
            counterfactual_response=cf_response[:6000],
        )
        response = self.client.chat(
            system_prompt="You are a strict evaluator. Output only valid JSON.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=2048,
        )
        if not response.ok:
            fallback = self.fallback.score(record)
            fallback.rationale = (
                f"LLM judge failed: {response.error}; fallback: {fallback.rationale}"
            )
            fallback.raw["token_usage"] = _token_usage(response.raw)
            fallback.raw["latency_ms"] = response.latency_ms
            return fallback

        parsed = _parse_json(response.text)
        if parsed is None:
            fallback = self.fallback.score(record)
            fallback.rationale = f"LLM judge JSON parse failed; fallback: {fallback.rationale}"
            fallback.raw["llm_response"] = response.text
            fallback.raw["token_usage"] = _token_usage(response.raw)
            fallback.raw["latency_ms"] = response.latency_ms
            return fallback

        strategy = parsed.get("strategy", "ignore")
        try:
            strategy_enum = RecoveryStrategy(strategy)
        except ValueError:
            strategy_enum = RecoveryStrategy.IGNORE

        dims = {
            key: _unit_float(parsed.get(key, 0.0))
            for key in (
                "detection",
                "diagnosis",
                "replanning",
                "verification",
                "communication",
                "recovery",
                "safety",
            )
        }
        total = 4.0 * sum(dims.values()) / len(dims)
        return JudgeScore(
            sample_id=record.sample.sample_id,
            intervention_id=intervention.intervention_id if intervention else None,
            total=round(total, 3),
            strategy=strategy_enum,
            rationale=str(parsed.get("rationale", "")),
            raw={
                "llm_response": response.text,
                "parsed": parsed,
                "token_usage": _token_usage(response.raw),
                "latency_ms": response.latency_ms,
            },
            **dims,
        )


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


def _format_rubric(rubric: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in rubric.items()) if rubric else "(none)"


def _unit_float(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _parse_json(text: Any) -> dict | None:
    if not text:
        return None
    if isinstance(text, dict):
        for key in ("text", "content"):
            if key in text:
                parsed = _parse_json(text[key])
                if parsed is not None:
                    return parsed
        if not any(
            key in text
            for key in (
                "detection",
                "diagnosis",
                "replanning",
                "verification",
                "communication",
                "recovery",
                "safety",
                "strategy",
            )
        ):
            return None
        return text
    if isinstance(text, list):
        for item in text:
            parsed = _parse_json(item)
            if parsed is not None:
                return parsed
        return None
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False)
        except TypeError:
            text = str(text)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for idx, ch in enumerate(text[start:], start):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end < 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _token_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("total_tokens") or prompt + completion),
    }
