from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.core.schema import (  # noqa: E402
    AgentTrajectory,
    BenchmarkName,
    EvaluationRecord,
    RunMode,
    TaskSample,
    ToolEvent,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert official MobilityBench raw_results.json into CCE factual records."
    )
    parser.add_argument("--raw-results", required=True, help="Official raw_results.json path.")
    parser.add_argument("--dataset", required=True, help="Official dataset CSV used by mbench run.")
    parser.add_argument("--output", required=True, help="Output CCE records.jsonl path.")
    parser.add_argument("--model", default="", help="Optional model name override.")
    args = parser.parse_args()

    raw_results_path = Path(args.raw_results)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    rows = json.loads(raw_results_path.read_text(encoding="utf-8"))
    cases = _load_cases(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = []
    missing_cases = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        case = cases.get(case_id)
        if case is None:
            missing_cases.append(case_id)
            case = _fallback_case(case_id)
        sample = _task_sample(case, dataset_path)
        factual = _trajectory_from_official(row, sample.sample_id, model=args.model)
        record = EvaluationRecord(
            sample=sample,
            factual=factual,
            metadata={
                "source": "official_mobilitybench_raw_results",
                "official_case_id": case_id,
                "official_model_name": row.get("model_name"),
                "official_error": row.get("error"),
                "official_execution_time": row.get("execution_time"),
                "official_token_usage": row.get("token_usage") or {},
            },
        )
        converted.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        for record in converted:
            f.write(record.model_dump_json() + "\n")

    summary = {
        "raw_results": str(raw_results_path),
        "dataset": str(dataset_path),
        "output": str(output_path),
        "converted_records": len(converted),
        "missing_case_count": len(missing_cases),
        "missing_case_ids": missing_cases[:20],
        "factual_event_counts": [
            len(record.factual.events) if record.factual else 0 for record in converted
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_cases(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cases = {}
    for idx, row in enumerate(rows):
        case_id = str(row.get("case_id") or row.get("Case ID") or idx)
        cases[case_id] = row
    return cases


def _fallback_case(case_id: str) -> dict[str, Any]:
    return {"case_id": case_id, "query": "", "context": ""}


def _task_sample(case: dict[str, Any], dataset_path: Path) -> TaskSample:
    case_id = str(case.get("case_id") or case.get("Case ID") or "")
    return TaskSample(
        sample_id=f"mb_official_{case_id}",
        benchmark=BenchmarkName.MOBILITY_BENCH,
        query=str(case.get("query") or case.get("user_query") or ""),
        context={
            "observable_state": _parse_or_text(case.get("context")),
            "city": str(case.get("city") or case.get("user_city") or ""),
            "task_scenario": str(case.get("task_scenario") or ""),
            "intent_family": str(case.get("intent_family") or ""),
            "tool_list": _parse_or_text(case.get("tool_list")),
            "route_answer": case.get("route_ans") or case.get("route_info") or "",
            "poi_answer": case.get("poi_result") or case.get("ans") or "",
            "raw": case,
        },
        ground_truth={
            "tool_list": _parse_or_text(case.get("tool_list")),
            "route_ans": case.get("route_ans"),
            "ans": case.get("ans"),
            "route_info": case.get("route_info"),
            "poi_result": case.get("poi_result"),
        },
        metadata={
            "source_path": str(dataset_path),
            "source_case_id": case_id,
            "split": "official_smoke",
            "task_scenario": case.get("task_scenario", ""),
            "intent_family": case.get("intent_family", ""),
        },
    )


def _trajectory_from_official(
    row: dict[str, Any],
    sample_id: str,
    *,
    model: str = "",
) -> AgentTrajectory:
    events: list[ToolEvent] = []
    planner = row.get("planner_output") or {}
    if planner:
        events.append(
            ToolEvent(
                step=len(events),
                tool_name="official_mobilitybench_planner",
                parameters={
                    "intent": planner.get("intent", ""),
                    "steps": planner.get("steps", []),
                },
                observation=str(planner.get("human") or ""),
                action_text=str(planner.get("thinking") or ""),
                success=True,
                metadata={
                    "agent_visible": False,
                    "official_role": "planner",
                    "system": planner.get("system", ""),
                    "raw": planner,
                },
            )
        )

    for worker in row.get("worker_output") or []:
        task_text = str(worker.get("ai_response") or worker.get("task_id") or "")
        tool_calls = worker.get("tool_calls") or []
        tool_results = worker.get("tool_results") or {}
        if not tool_calls:
            events.append(
                ToolEvent(
                    step=len(events),
                    tool_name="official_mobilitybench_worker",
                    parameters={"task_id": worker.get("task_id")},
                    observation=json.dumps(tool_results, ensure_ascii=False),
                    action_text=task_text,
                    success=not bool(worker.get("error")),
                    metadata={
                        "agent_visible": True,
                        "official_role": "worker",
                        "raw": worker,
                    },
                )
            )
            continue
        for call in tool_calls:
            call_id = str(call.get("id") or "")
            events.append(
                ToolEvent(
                    step=len(events),
                    tool_name=str(call.get("name") or "official_tool_call"),
                    parameters=call.get("args") or {},
                    observation=str(tool_results.get(call_id, "")),
                    action_text=task_text,
                    success=not _tool_result_is_error(tool_results.get(call_id)),
                    metadata={
                        "agent_visible": True,
                        "official_role": "worker",
                        "official_task_id": worker.get("task_id"),
                        "official_call_id": call_id,
                        "raw_call": call,
                        "raw_worker": worker,
                    },
                )
            )

    reporter_output = str(row.get("reporter_output") or "")
    events.append(
        ToolEvent(
            step=len(events),
            tool_name="official_mobilitybench_reporter",
            observation="",
            action_text=reporter_output,
            success=not bool(row.get("error")),
            metadata={
                "agent_visible": True,
                "official_role": "reporter",
            },
        )
    )
    return AgentTrajectory(
        sample_id=sample_id,
        run_mode=RunMode.FACTUAL,
        events=events,
        final_response=reporter_output,
        raw={
            "adapter": "official_mobilitybench_plan_and_execute",
            "adapter_scope": "official_mbench_raw_results",
            "agent_id": row.get("model_name") or model,
            "official_case_id": row.get("case_id"),
            "official_model_name": row.get("model_name"),
            "official_error": row.get("error"),
            "official_token_usage": row.get("token_usage") or {},
        },
    )


def _tool_result_is_error(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return False
    lower = text.lower()
    return "error" in lower or "暂不支持" in text or "'status': 1" in text


def _parse_or_text(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return value
    text = str(value)
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


if __name__ == "__main__":
    raise SystemExit(main())
