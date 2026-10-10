from __future__ import annotations

import json
from typing import Any


TOOL_INPUT_ERROR_MARKERS = (
    "输入验证失败",
    "输入参数错误",
    "parse error, cannot convert to number",
)
UNKNOWN_TOOL_MARKERS = (
    "unknown tool:",
    "unknown tool",
)
QUOTA_ERROR_MARKERS = (
    "allocated quota exceeded",
    "throttling.allocationquota",
    "insufficient_quota",
    "you exceeded your current quota",
)


def official_raw_dirty_reasons(row: dict[str, Any]) -> list[str]:
    text = json.dumps(row, ensure_ascii=False)
    normalized = text.lower()
    reasons: list[str] = []
    if row.get("error"):
        reasons.append("row_error")
    if any(marker in normalized for marker in QUOTA_ERROR_MARKERS):
        reasons.append("quota")
    if any(marker in text for marker in TOOL_INPUT_ERROR_MARKERS):
        reasons.append("tool_input_error")
    if any(marker in normalized for marker in UNKNOWN_TOOL_MARKERS):
        reasons.append("unknown_tool")
    if not row.get("planner_output"):
        reasons.append("missing_planner")
    if not str(row.get("reporter_output") or "").strip():
        reasons.append("empty_reporter")
    tool_calls = sum(
        len(worker.get("tool_calls") or [])
        for worker in (row.get("worker_output") or [])
        if isinstance(worker, dict)
    )
    if tool_calls == 0:
        reasons.append("no_worker_tool_call")
    return reasons
