from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cce_mobility.integrations.mobilitybench_official_audit import (
    official_raw_dirty_reasons,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECONCILE_PATH = (
    PROJECT_ROOT
    / "src/cce_mobility/workflows/mobilitybench"
    / "reconcile_mobilitybench_official_run.py"
)
SPEC = importlib.util.spec_from_file_location("reconcile_mb_official", RECONCILE_PATH)
assert SPEC and SPEC.loader
reconcile_mb_official = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reconcile_mb_official)


def _clean_row(case_id: str = "1") -> dict:
    return {
        "case_id": case_id,
        "planner_output": {"steps": [["use tool"]]},
        "worker_output": [{"tool_calls": [{"name": "driving_route"}]}],
        "reporter_output": "done",
    }


def test_official_dirty_audit_catches_tool_input_and_unknown_tool_errors() -> None:
    tool_error = _clean_row()
    tool_error["worker_output"][0]["observation"] = (
        "驾车路线输入验证失败: parse error, cannot convert to number"
    )
    unknown_tool = _clean_row()
    unknown_tool["reporter_output"] = "Unknown tool: invented_route"

    assert "tool_input_error" in official_raw_dirty_reasons(tool_error)
    assert "unknown_tool" in official_raw_dirty_reasons(unknown_tool)


def test_canonicalize_only_copies_complete_clean_raw(tmp_path: Path) -> None:
    source = tmp_path / "source" / "raw_results.json"
    output = tmp_path / "canonical" / "model" / "raw_results.json"
    source.parent.mkdir()
    source.write_text(json.dumps([_clean_row()]), encoding="utf-8")

    result = reconcile_mb_official.canonicalize_clean_raw(
        source_raw=source,
        output_raw=output,
        expected_count=1,
    )

    assert result["byte_identical"] is True
    assert output.read_bytes() == source.read_bytes()
    manifest = json.loads(
        (output.parent.parent / "canonical_source.json").read_text(encoding="utf-8")
    )
    assert manifest["source_raw"]["sha256"] == manifest["canonical_raw"]["sha256"]


def test_canonicalize_rejects_polluted_raw(tmp_path: Path) -> None:
    source = tmp_path / "raw_results.json"
    output = tmp_path / "canonical" / "model" / "raw_results.json"
    row = _clean_row()
    row["reporter_output"] = "Unknown tool: fake_tool"
    source.write_text(json.dumps([row]), encoding="utf-8")

    with pytest.raises(ValueError, match="dirty rows"):
        reconcile_mb_official.canonicalize_clean_raw(
            source_raw=source,
            output_raw=output,
            expected_count=1,
        )

    assert not output.exists()
