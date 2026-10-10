import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(
    "src/cce_mobility/workflows/mobilitybench/compare_mobilitybench_official_case_metrics.py"
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_case_level_compare_reads_official_metric_csvs(tmp_path):
    baseline_dir = tmp_path / "baseline_eval"
    target_dir = tmp_path / "target_eval"
    model = "aihub-qwen35b"
    dataset = tmp_path / "dataset.csv"
    output_json = tmp_path / "case_compare.json"
    output_md = tmp_path / "case_compare.md"
    output_csv = tmp_path / "case_compare.csv"

    _write_csv(
        dataset,
        [
            {
                "case_id": "1",
                "query": "帮我找附近的咖啡店",
                "intent_family": "Basic Information Retrieval",
                "task_scenario": "Nearby Query",
            },
            {
                "case_id": "2",
                "query": "规划路线",
                "intent_family": "Basic Route Planning",
                "task_scenario": "Point-to-Point Planning",
            },
        ],
    )
    _write_csv(
        baseline_dir / model / "tool_evaluation.csv",
        [
            {
                "case_id": "1",
                "metric_name": "tool",
                "score": 0,
                "TS_P": 0.5,
                "TS_R": 0.5,
                "SC": 1.0,
            },
            {
                "case_id": "2",
                "metric_name": "tool",
                "score": 0,
                "TS_P": 1.0,
                "TS_R": 1.0,
                "SC": 1.0,
            },
        ],
    )
    _write_csv(
        target_dir / model / "tool_evaluation.csv",
        [
            {
                "case_id": "1",
                "metric_name": "tool",
                "score": 0,
                "TS_P": 1.0,
                "TS_R": 0.5,
                "SC": 1.0,
            },
            {
                "case_id": "2",
                "metric_name": "tool",
                "score": 0,
                "TS_P": 0.5,
                "TS_R": 1.0,
                "SC": 1.0,
            },
        ],
    )
    _write_csv(
        baseline_dir / model / "efficiency_evaluation.csv",
        [
            {"case_id": "1", "metric_name": "efficiency", "score": 0, "IT": 100, "OT": 20},
            {"case_id": "2", "metric_name": "efficiency", "score": 0, "IT": 100, "OT": 20},
        ],
    )
    _write_csv(
        target_dir / model / "efficiency_evaluation.csv",
        [
            {"case_id": "1", "metric_name": "efficiency", "score": 0, "IT": 90, "OT": 25},
            {"case_id": "2", "metric_name": "efficiency", "score": 0, "IT": 130, "OT": 20},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-eval-dir",
            str(baseline_dir),
            "--target-eval-dir",
            str(target_dir),
            "--dataset",
            str(dataset),
            "--metric",
            "tool",
            "--metric",
            "efficiency",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["case_count"] == 2
    assert data["aggregate"]["improved_cases"] == 1
    assert data["aggregate"]["worse_cases"] == 1
    first = data["all_cases"][0]
    assert first["case_id"] == "2"
    assert first["signed_delta_sum"] < 0
    assert "MobilityBench Official Case-Level" in output_md.read_text(encoding="utf-8")
    assert "Basic Route Planning" in output_csv.read_text(encoding="utf-8")


def _write_raw(path: Path, dirty_ids: set[str]) -> None:
    rows = []
    for case_id in map(str, range(1, 6)):
        rows.append(
            {
                "case_id": case_id,
                "planner_output": {"steps": ["search"]},
                "worker_output": [] if case_id in dirty_ids else [{"tool_calls": [{}]}],
                "reporter_output": "answer",
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_common_clean_compares_only_valid_matched_metric_rows(tmp_path):
    model = "aihub-glm-5-2"
    dataset = tmp_path / "dataset.csv"
    baseline_raw = tmp_path / "baseline_raw.json"
    target_raw = tmp_path / "target_raw.json"
    baseline_eval = tmp_path / "baseline_eval"
    target_eval = tmp_path / "target_eval"
    output_json = tmp_path / "common_clean.json"
    output_md = tmp_path / "common_clean.md"
    _write_csv(
        dataset, [{"case_id": str(i), "intent_family": "Basic Route Planning"} for i in range(1, 6)]
    )
    _write_raw(baseline_raw, {"1", "2"})
    _write_raw(target_raw, {"2", "3"})
    _write_csv(
        baseline_eval / model / "tool_evaluation.csv",
        [{"case_id": str(i), "TS_P": 0.5, "error": ""} for i in range(1, 6)],
    )
    _write_csv(
        target_eval / model / "tool_evaluation.csv",
        [{"case_id": str(i), "TS_P": 0.7, "error": "bad" if i == 5 else ""} for i in range(1, 6)],
    )
    _write_csv(
        baseline_eval / model / "answer_evaluation.csv",
        [{"case_id": str(i), "DR": 0.8 if i == 4 else 0.4} for i in range(1, 6)],
    )
    _write_csv(
        target_eval / model / "answer_evaluation.csv",
        [{"case_id": str(i), "DR": 0.7 if i == 4 else 0.6} for i in range(1, 6)],
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-eval-dir",
            str(baseline_eval),
            "--target-eval-dir",
            str(target_eval),
            "--baseline-raw",
            str(baseline_raw),
            "--target-raw",
            str(target_raw),
            "--common-clean-only",
            "--dataset",
            str(dataset),
            "--model",
            model,
            "--metric",
            "tool",
            "--metric",
            "answer",
            "--bootstrap-samples",
            "100",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["common_clean_audit"] == {
        "baseline_raw": str(baseline_raw),
        "target_raw": str(target_raw),
        "dataset_count": 5,
        "baseline_dirty_count": 2,
        "target_dirty_count": 2,
        "dirty_overlap_count": 1,
        "common_clean_count": 2,
    }
    assert {row["case_id"] for row in report["all_cases"]} == {"4", "5"}
    by_metric = {(row["metric"], row["sub_dimension"]): row for row in report["by_metric"]}
    assert by_metric[("tool", "TS_P")]["paired_count"] == 1
    assert by_metric[("tool", "TS_P")]["mean_delta"] == 0.2
    assert by_metric[("answer", "DR")]["paired_count"] == 2
    assert by_metric[("answer", "DR")]["mean_delta"] == 0.05
    markdown = output_md.read_text(encoding="utf-8")
    assert "common clean: 2" in markdown
    assert "| tool | TS_P | 1 |" in markdown


def test_common_clean_rejects_mismatched_raw_case_ids(tmp_path):
    dataset = tmp_path / "dataset.csv"
    baseline_raw = tmp_path / "baseline_raw.json"
    target_raw = tmp_path / "target_raw.json"
    _write_csv(dataset, [{"case_id": str(i)} for i in range(1, 6)])
    _write_raw(baseline_raw, set())
    _write_raw(target_raw, set())
    target_raw.write_text(
        json.dumps(json.loads(target_raw.read_text(encoding="utf-8"))[:-1]), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-eval-dir",
            str(tmp_path / "base"),
            "--target-eval-dir",
            str(tmp_path / "target"),
            "--baseline-raw",
            str(baseline_raw),
            "--target-raw",
            str(target_raw),
            "--common-clean-only",
            "--dataset",
            str(dataset),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "raw/dataset case IDs differ" in result.stderr
