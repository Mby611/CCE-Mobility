from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.core.schema import EvaluationRecord, InterventionCandidate  # noqa: E402
from cce_mobility.integrations.mobilitybench_continuation import (  # noqa: E402
    AGENT_VISIBLE_CONTEXT_KEYS,
    CONTINUATION_PROTOCOL,
    MobilityBenchContinuationCase,
    build_continuation_case,
    cce_continuation_runtime_provenance,
    file_sha256,
    official_mobilitybench_provenance,
    stable_hash,
    validate_continuation_case,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or audit frozen inputs for official MobilityBench continuation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cases = subparsers.add_parser("cases", help="Bind interventions to concrete tool events.")
    cases.add_argument("--factual-records", type=Path, required=True)
    cases.add_argument("--frozen-interventions", type=Path, required=True)
    cases.add_argument("--output", type=Path, required=True)
    cases.add_argument("--expected-count", type=int, default=0)
    cases.add_argument("--max-fallback-trigger-count", type=int, default=0)

    audit = subparsers.add_parser("audit", help="Read-only audit of prepared cases.")
    audit.add_argument("--cases", type=Path, required=True)
    audit.add_argument("--expected-count", type=int, default=0)

    prompts = subparsers.add_parser(
        "prompt-bundle",
        help="Capture official base prompts once; run with the MobilityBench Python environment.",
    )
    prompts.add_argument("--output", type=Path, required=True)
    prompts.add_argument("--framework", default="plan_and_execute")
    args = parser.parse_args()

    if args.command == "cases":
        summary = prepare_cases(
            factual_records_path=args.factual_records,
            frozen_interventions_path=args.frozen_interventions,
            output_path=args.output,
            expected_count=args.expected_count,
            max_fallback_trigger_count=args.max_fallback_trigger_count,
        )
    elif args.command == "audit":
        summary = audit_cases(args.cases, expected_count=args.expected_count)
    else:
        summary = prepare_prompt_bundle(args.output, framework=args.framework)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def prepare_cases(
    *,
    factual_records_path: Path,
    frozen_interventions_path: Path,
    output_path: Path,
    expected_count: int = 0,
    max_fallback_trigger_count: int = 0,
) -> dict[str, Any]:
    factual_records_path = factual_records_path.expanduser().resolve()
    frozen_interventions_path = frozen_interventions_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen continuation cases: {output_path}")
    factual = _load_factual_records(factual_records_path)
    candidates = _load_interventions(frozen_interventions_path)
    cases: list[MobilityBenchContinuationCase] = []
    failures: list[dict[str, str]] = []
    for sample_id, sample_candidates in candidates.items():
        record = factual.get(sample_id)
        if record is None:
            failures.append({"sample_id": sample_id, "error": "missing_factual_record"})
            continue
        for candidate in sample_candidates:
            try:
                cases.append(build_continuation_case(record, candidate))
            except Exception as exc:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "intervention_id": candidate.spec.intervention_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    if failures:
        raise ValueError(
            "Continuation case preparation failed; no partial artifact was written. "
            + json.dumps(failures[:10], ensure_ascii=False)
        )
    _assert_expected_count(len(cases), expected_count)
    _assert_unique([case.pair_key for case in cases], "pair_key")
    _assert_unique([case.case_hash for case in cases], "case_hash")
    fallback_count = sum(case.trigger_resolution == "first_worker_event_fallback" for case in cases)
    if fallback_count > max_fallback_trigger_count:
        raise ValueError(
            "Continuation cases could not be grounded at a concrete intervention-relevant "
            f"worker/tool position: fallback_count={fallback_count}, "
            f"allowed={max_fallback_trigger_count}. No artifact was written."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")
    summary = audit_cases(output_path, expected_count=expected_count)
    summary.update(
        {
            "factual_records_path": str(factual_records_path),
            "factual_records_sha256": file_sha256(factual_records_path),
            "frozen_interventions_path": str(frozen_interventions_path),
            "frozen_interventions_sha256": file_sha256(frozen_interventions_path),
        }
    )
    _write_json(output_path.with_suffix(".audit.json"), summary)
    return summary


def audit_cases(path: Path, *, expected_count: int = 0) -> dict[str, Any]:
    path = path.expanduser().resolve()
    cases = _load_cases(path)
    for case in cases:
        validate_continuation_case(case)
    _assert_expected_count(len(cases), expected_count)
    pair_keys = [case.pair_key for case in cases]
    _assert_unique(pair_keys, "pair_key")
    dimensions = Counter(case.intervention.dimension for case in cases)
    trigger_tools = Counter(case.trigger_tool_name for case in cases)
    resolutions = Counter(case.trigger_resolution for case in cases)
    context_keys = Counter(
        key for case in cases for key in (case.agent_visible.get("context") or {})
    )
    unexpected_context_keys = sorted(set(context_keys) - AGENT_VISIBLE_CONTEXT_KEYS)
    return {
        "protocol": CONTINUATION_PROTOCOL,
        "cases_path": str(path),
        "cases_sha256": file_sha256(path),
        "case_count": len(cases),
        "task_count": len({case.task_id for case in cases}),
        "pair_key_count": len(set(pair_keys)),
        "factual_source_hash_count": len({case.factual_source_hash for case in cases}),
        "intervention_source_hash_count": len({case.intervention_source_hash for case in cases}),
        "recovery_contract_hash_count": len({case.recovery_contract_hash for case in cases}),
        "dimension_counts": dict(sorted(dimensions.items())),
        "trigger_tool_counts": dict(sorted(trigger_tools.items())),
        "trigger_resolution_counts": dict(sorted(resolutions.items())),
        "agent_visible_context_key_counts": dict(sorted(context_keys.items())),
        "agent_visible_context_allowed_keys": sorted(AGENT_VISIBLE_CONTEXT_KEYS),
        "agent_visible_context_unexpected_keys": unexpected_context_keys,
        "agent_payload_privileged_field_count": 0,
        "continuation_mode": "state_reconstruction_continuation",
        "bitwise_graph_resume": False,
    }


def prepare_prompt_bundle(output_path: Path, *, framework: str) -> dict[str, Any]:
    """Capture one immutable official prompt snapshot for all seven curve points."""

    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen prompt bundle: {output_path}")

    # The official YAML references LLM_API_KEY. Populate it inside the process so
    # callers do not have to paste credentials into shell history.
    _load_local_secret_into_process("CCE_MODEL_API_KEY", "LLM_API_KEY")
    try:
        from mobility_bench.agent.prompts import get_system_prompt
    except ImportError as exc:  # pragma: no cover - requires official MB runtime
        raise RuntimeError(
            "mobility_bench is unavailable; run this subcommand with the official "
            "MobilityBench virtual environment and PYTHONPATH including patches/mobilitybench."
        ) from exc

    previous_skill_path = os.environ.pop("CCE_MOBILITYBENCH_SKILL_PROMPT_PATH", None)
    try:
        role_prompts = {
            role: get_system_prompt(framework, role) for role in ("planner", "worker", "reporter")
        }
    finally:
        if previous_skill_path is not None:
            os.environ["CCE_MOBILITYBENCH_SKILL_PROMPT_PATH"] = previous_skill_path
    official_provenance = official_mobilitybench_provenance()
    cce_provenance = cce_continuation_runtime_provenance(PROJECT_ROOT)
    dirty = {
        "official_mobilitybench": official_provenance.get("runtime_dirty"),
        "cce_runtime": cce_provenance.get("runtime_dirty"),
    }
    unknown_commits = {
        "official_mobilitybench": official_provenance.get("git_commit") in {"", "unknown"},
        "cce_runtime": cce_provenance.get("git_commit") in {"", "unknown"},
    }
    if any(value is not False for value in dirty.values()) or any(unknown_commits.values()):
        raise RuntimeError(
            "Refusing to freeze a formal prompt bundle without committed clean runtime code: "
            f"dirty={dirty}, unknown_commits={unknown_commits}"
        )
    payload = {
        "schema_version": "mobilitybench-official-prompt-bundle-v1",
        "framework": framework,
        "official_mobilitybench_provenance": official_provenance,
        "cce_continuation_runtime_provenance": cce_provenance,
        "prompts": role_prompts,
        "role_sha256": {role: stable_hash(prompt) for role, prompt in role_prompts.items()},
        "bundle_sha256": stable_hash(role_prompts),
        "environment_contract": {
            "disable_thinking": os.getenv("CCE_MOBILITYBENCH_DISABLE_THINKING", "1"),
            "force_tool_choice_required": os.getenv(
                "CCE_MOBILITYBENCH_FORCE_TOOL_CHOICE_REQUIRED", "1"
            ),
            "normalize_tool_args": os.getenv("CCE_MOBILITYBENCH_NORMALIZE_TOOL_ARGS", "1"),
            "plan_json_fallback": os.getenv("CCE_MOBILITYBENCH_PLAN_JSON_FALLBACK", "0"),
            "synthetic_plan_fallback": os.getenv("CCE_MOBILITYBENCH_SYNTHETIC_PLAN_FALLBACK", "0"),
        },
    }
    _write_json(output_path, payload)
    return {**payload, "output_path": str(output_path)}


def _load_local_secret_into_process(source_name: str, target_name: str) -> None:
    if os.getenv(target_name):
        return
    try:
        from cce_mobility import local_secrets
    except ImportError:
        return
    value = getattr(local_secrets, source_name, "")
    if value:
        os.environ[target_name] = value


def _load_factual_records(path: Path) -> dict[str, EvaluationRecord]:
    records: dict[str, EvaluationRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = EvaluationRecord.model_validate_json(line)
            sample_id = record.sample.sample_id
            if sample_id in records:
                raise ValueError(f"Duplicate factual sample at {path}:{line_number}: {sample_id}")
            records[sample_id] = record
    return records


def _load_interventions(path: Path) -> dict[str, list[InterventionCandidate]]:
    candidates: dict[str, list[InterventionCandidate]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = InterventionCandidate.model_validate_json(line)
            candidates.setdefault(candidate.spec.sample_id, []).append(candidate)
    return candidates


def _load_cases(path: Path) -> list[MobilityBenchContinuationCase]:
    cases: list[MobilityBenchContinuationCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(MobilityBenchContinuationCase.model_validate_json(line))
    return cases


def _assert_expected_count(actual: int, expected: int) -> None:
    if expected and actual != expected:
        raise ValueError(f"Expected {expected} continuation cases, got {actual}")


def _assert_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        duplicates = [value for value, count in Counter(values).items() if count > 1]
        raise ValueError(f"Duplicate {label}: {duplicates[:10]}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
