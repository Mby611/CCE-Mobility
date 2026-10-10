from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_convert_mobilitybench_official_to_cce(tmp_path: Path) -> None:
    dataset = tmp_path / "official.csv"
    with dataset.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "query", "context", "tool_list"])
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case1",
                "query": "附近加油站",
                "context": "{'city': '北京市'}",
                "tool_list": "[]",
            }
        )

    raw_results = tmp_path / "raw_results.json"
    raw_results.write_text(
        json.dumps(
            [
                {
                    "case_id": "case1",
                    "model_name": "unit-model",
                    "planner_output": {
                        "system": "planner system",
                        "human": "User requirement:\n附近加油站",
                        "steps": [["搜索附近加油站"]],
                        "thinking": "Need nearby POI.",
                        "intent": "nearby_search",
                    },
                    "worker_output": [
                        {
                            "task_id": "task1",
                            "ai_response": "search nearby gas station",
                            "tool_calls": [
                                {
                                    "id": "call1",
                                    "name": "search_around_poi",
                                    "args": {"keywords": "加油站"},
                                }
                            ],
                            "tool_results": {"call1": "{'status': 0, 'data': {'name': 'A'}}"},
                        }
                    ],
                    "reporter_output": "附近有 A 加油站",
                    "token_usage": {"planner": {"total_tokens": 1}},
                    "execution_time": 1.2,
                    "error": None,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "cce_records.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "src/cce_mobility/workflows/mobilitybench"
                / "convert_mobilitybench_official_to_cce.py"
            ),
            "--raw-results",
            str(raw_results),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    record = rows[0]
    assert record["sample"]["sample_id"] == "mb_official_case1"
    assert record["factual"]["run_mode"] == "factual"
    assert [event["tool_name"] for event in record["factual"]["events"]] == [
        "official_mobilitybench_planner",
        "search_around_poi",
        "official_mobilitybench_reporter",
    ]
    assert record["factual"]["events"][1]["parameters"] == {"keywords": "加油站"}
    assert record["factual"]["final_response"] == "附近有 A 加油站"
